import os
import torch
import matplotlib.pyplot as plt
import imageio

class Plotter:
    """
    Handles plotting of training results and saving them for GIF creation.
    """

    @staticmethod
    def plot_and_save_frame(x, y, x_data, y_data, yh, xp=None, iteration=None, out_dir="frames"):
        """
        Creates and saves a single plot comparing the exact solution vs. the model prediction.
        
        Saves the resulting figure to out_dir/frame_<iteration>.png. 
        Later these frames can be combined into a GIF.

        Parameters:
        -----------
        x : torch.Tensor
            Full domain points (for exact / final prediction).
        y : torch.Tensor
            Exact solution over domain.
        x_data : torch.Tensor
            Training domain points.
        y_data : torch.Tensor
            Training data values.
        yh : torch.Tensor
            Model predictions over domain x.
        xp : torch.Tensor, optional
            Points used for the physics constraint (for plotting reference).
        iteration : int, optional
            Current training iteration (used for filename).
        out_dir : str
            Directory to store frame images.
        """
        if iteration is None:
            iteration = 0
        
        # Ensure output directory exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        plt.figure(figsize=(8,4))
        plt.plot(x, y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(x, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Model prediction")
        plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.6, label='Training data')
        
        if xp is not None:
            plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                        label='Physics loss locations')

        l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.ylim(-1.1, 1.1)
        plt.title(f"Training iteration: {iteration}")

        # Save the figure as a PNG image
        filename = os.path.join(out_dir, f"frame_{iteration:05d}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close("all")

    @staticmethod
    def create_gif_from_frames(frame_dir="frames", output_gif="training.gif", fps=5):
        """
        Reads all saved frames in 'frame_dir' and assembles them into an animated GIF.

        Parameters:
        -----------
        frame_dir : str
            Directory containing PNG frames (e.g., "frames/frame_00001.png", etc.).
        output_gif : str
            Output GIF filename.
        fps : int
            Frames per second in the resulting GIF.
        """
        # Collect all .png files in frame_dir in sorted order
        frame_files = sorted(
            [f for f in os.listdir(frame_dir) if f.endswith(".png")]
        )

        # Load images and write to GIF
        images = []
        for f in frame_files:
            file_path = os.path.join(frame_dir, f)
            images.append(imageio.v2.imread(file_path))

        if images:
            imageio.mimsave(output_gif, images, fps=fps)
            print(f"GIF saved as {output_gif}")
        else:
            print(f"No PNG frames found in directory '{frame_dir}'. GIF not created.")
