import torch
import torch.optim as optim
from torch.autograd import grad

class HarmonicTrainer:
    """
    Encapsulates the training loop for a 1D underdamped harmonic oscillator
    using a physics-informed neural network approach.
    
    Attributes:
    -----------
    model : torch.nn.Module
        The neural network model to train.
    oscillator : HarmonicOscillator
        Instance of harmonic oscillator parameters (d, w0) used for the PDE.
    x_data : torch.Tensor
        Training points (e.g., domain points on the LHS).
    y_data : torch.Tensor
        Training values at x_data.
    x_physics : torch.Tensor
        Points in the domain used for enforcing physics constraints.
    device : torch.device
        CPU or GPU device.
    optimizer : torch.optim.Optimizer
        Optimizer used to train the model.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    mu : float
        2*d from the PDE.
    k : float
        w0^2 from the PDE.
    """

    def __init__(self, model, oscillator, x_data, y_data, x_physics, lr=1e-3):
        self.model = model
        self.oscillator = oscillator
        self.x_data = x_data
        self.y_data = y_data
        self.x_physics = x_physics

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # PDE parameters: for equation y'' + mu*y' + k*y = 0,
        # mu = 2*d, k = w0^2
        self.mu = 2 * self.oscillator.d
        self.k = self.oscillator.w0**2

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1000
        )

        # Ensure each tensor is on the device
        self.x_data = self.x_data.to(self.device)
        self.y_data = self.y_data.to(self.device)
        self.x_physics = self.x_physics.to(self.device)
        self.x_physics.requires_grad_(True)

    def physics_loss(self):
        """
        Computes the physics/PDE loss term:
        residual = y'' + mu*y' + k*y.
        """
        y_pred = self.model(self.x_physics)
        dy_dx = grad(
            y_pred,
            self.x_physics,
            torch.ones_like(y_pred).to(self.device),
            create_graph=True
        )[0]
        d2y_dx2 = grad(
            dy_dx,
            self.x_physics,
            torch.ones_like(dy_dx).to(self.device),
            create_graph=True
        )[0]

        residual = d2y_dx2 + self.mu * dy_dx + self.k * y_pred
        # Weighted factor for physics loss can be tuned
        return (1e-4) * torch.mean(residual**2)

    def data_loss(self):
        """
        Computes the MSE loss between model predictions and ground truth data.
        """
        y_pred = self.model(self.x_data)
        return torch.mean((y_pred - self.y_data)**2)

    def train(self, max_iterations, plot_callback):
        """
        Runs the training loop.
        
        plot_callback : callable, optional
            Called to save or display plots (e.g. once every N iterations).
        """
        for i in range(max_iterations):
            self.optimizer.zero_grad()

            loss_data = self.data_loss()
            loss_physics = self.physics_loss()
            total_loss = loss_data + loss_physics
            total_loss.backward()

            self.optimizer.step()
            self.scheduler.step(total_loss.item())  #now disabled

            # Print learning rate & losses periodically
            if (i + 1) % 1000 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Iteration {i+1}, Loss: {total_loss.item():.10f}, LR: {current_lr:.6f}")

            # Call the user-defined callback to save frames or do custom plotting
            if plot_callback is not None:
                # For example, we might only save frames every 100 iterations to avoid huge disk usage
                if (i + 1) % 500 == 0:
                    plot_callback(iteration=(i+1))