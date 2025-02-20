import torch
import numpy as np

class HarmonicOscillator:
    """
    Class for the 1D underdamped harmonic oscillator problem.
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/
    
    Attributes:
    -----------
    d : float
        Damping coefficient.
    w0 : float
        Natural frequency.
    """

    def __init__(self, d: float, w0: float):
        assert d < w0, "Requires d < w0 for an underdamped oscillator."
        self.d = d
        self.w0 = w0

    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution y(x) for the underdamped oscillator.
        
        y(x) = exp(-d*x)*2*A*cos(phi + w*x), 
        where w = sqrt(w0^2 - d^2),
              phi = arctan(-d/w),
              A = 1/(2*cos(phi)).
        
        Parameters:
        -----------
        x : torch.Tensor
            Domain points.
        
        Returns:
        --------
        y : torch.Tensor
            Analytical solution at domain points.
        """
        w = np.sqrt(self.w0**2 - self.d**2)
        phi = np.arctan(-self.d / w)
        A = 1 / (2 * np.cos(phi))

        cos_term = torch.cos(torch.tensor(phi) + w * x)
        exp_term = torch.exp(-self.d * x)
        y = exp_term * 2 * A * cos_term
        return y
