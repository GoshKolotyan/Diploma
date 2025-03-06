import torch
import numpy as np 

def sample_collocation_points(rho_max, z_max, num_points):
    """
    Samples collocation points (rho, z) in [0, rho_max] x [-z_max, z_max]
    Uniformly for PDE residual enforcement.
    Returns Tensors (rho_t, z_t) with requires_grad=True.
    """
    rho_np = np.random.rand(num_points, 1) * rho_max
    z_np   = (np.random.rand(num_points, 1)*2.0 - 1.0)*z_max

    rho_t = torch.tensor(rho_np, dtype=torch.float32, requires_grad=True)
    z_t   = torch.tensor(z_np,   dtype=torch.float32, requires_grad=True)
    return rho_t, z_t



def sample_boundary_points(rho_max, z_max, num_bd=200):
    """
    Samples points on domain boundaries for Dirichlet-like BC: Psi -> 0.
    1) rho = rho_max, z in [-z_max, z_max]
    2) z = ±z_max,    rho in [0, rho_max]
    """
    # 1) Rho boundary
    z_rand = np.random.uniform(-z_max, z_max, size=(num_bd,1))
    rho_bd_rmax = np.full_like(z_rand, fill_value=rho_max)

    # 2) Z boundary (top/bot)
    rho_rand = np.random.uniform(0.0, rho_max, size=(num_bd,1))
    z_bd_top = np.full_like(rho_rand, fill_value= z_max)
    z_bd_bot = np.full_like(rho_rand, fill_value=-z_max)

    # Combine them
    rho_bd = np.vstack([rho_bd_rmax, rho_rand, rho_rand])
    z_bd   = np.vstack([z_rand,       z_bd_top, z_bd_bot])
    return rho_bd, z_bd