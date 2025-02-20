import torch
import matplotlib.pyplot as plt

from oscillator import HarmonicOscillator
from model import FCN, TimeRNN
from plot_utils import Plotter
from trainer import HarmonicTrainer

def main(max_iterations):
    # -- 1) Define oscillator and domain
    d, w0 = 4, 30
    oscillator = HarmonicOscillator(d, w0)

    # -- 2) Generate domain data and exact solution
    x_full = torch.linspace(0, 4, 1500).view(-1,1)
    y_exact = oscillator.analytical_solution(x_full).view(-1,1)

    # -- 3) Training data (small number of points from the LHS of the domain)
    x_data = x_full[0:200:10]
    y_data = y_exact[0:200:10]

    # -- 4) Physics points (for enforcing the PDE)
    x_physics = torch.linspace(0, 4, 30).view(-1,1)

    # -- 5) Initialize the NN model and trainer
    model = FCN(N_INPUT=1, N_OUTPUT=1, N_HIDDEN=128, N_LAYERS=4)
    trainer = HarmonicTrainer(model, oscillator, x_data, y_data, x_physics, lr=1e-3)

    # -- 6) Define a callback that saves each plot frame
    def plot_callback(iteration):
        """
        This function is called periodically (in this example, every 100 iterations).
        It saves the current prediction as a PNG frame for later GIF creation.
        """
        with torch.no_grad():
            # Move x_full & y_exact to device only when needed
            x_full_device = x_full.to(trainer.device)
            y_full_device = y_exact.to(trainer.device)

            y_pred = trainer.model(x_full_device).cpu()  # model's prediction
            xp_cpu = trainer.x_physics.detach().cpu()    # physics points (for plotting)

            # Save the figure for this iteration
            Plotter.plot_and_save_frame(
                x_full, y_exact,
                trainer.x_data.cpu(), trainer.y_data.cpu(),
                y_pred, xp_cpu,
                iteration=iteration,
                out_dir="frames"  # folder to store frames
            )

    # -- 7) Train the model (this will generate PNG frames)
    trainer.train(max_iterations=max_iterations, plot_callback=plot_callback)

    # -- 8) After training, create the GIF from all saved frames
    # (If you want to store frames for all 80,000 iterations, you'll end up with many images!
    #  Consider saving a frame every 500 or 1000 iterations to reduce file size.)
    Plotter.create_gif_from_frames(
        frame_dir="frames",
        output_gif=f"gifs/training_RNN_{max_iterations}_3.gif",
        fps=5
    )

if __name__ == "__main__":
    max_ie = [10_000, 25_000, 75_000, 100_000, 150_000, 250_000]
    # for i in max_ie:
    main(max_iterations=max_ie[0])
