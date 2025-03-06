import torch

from physics_loss import schrodinger_pde_residual, boundary_loss
from data_collocation import sample_boundary_points, sample_collocation_points

import torch
import torch.nn as nn
import numpy as np

def train_pinn_for_L(model, L, 
                     rho_max=5.0, # radial domain
                     epochs=5000, 
                     num_coll=2000, 
                     num_bd=200, 
                     num_norm=1000, 
                     lambda_pde=1.0, 
                     lambda_bd=10.0, 
                     lr=1e-3,
                     device=torch.device("cpu")):
    """
    Train the given PINN model for a QW of width L => z in [-L/2, L/2].
    Typical steps:
      1) PDE residual in the interior
      2) Boundary condition (Psi=0 at boundary)
      3) Normalization (∫|Psi|^2 = 1)
      4) Single combined loss, optimize
    Returns the trained model.
    """
    # We interpret z_max = L/2 for the "well" boundary in z-direction
    z_max = L/2.0

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        optimizer.zero_grad()

        # 1) PDE Collocation points
        rho_coll_np, z_coll_np = sample_collocation_points(rho_max, z_max, num_coll)
        rho_coll = rho_coll_np.to(device)
        z_coll   = z_coll_np.to(device)

        # PDE Residual => pde_loss
        pde_res = schrodinger_pde_residual(model, rho_coll, z_coll)  
        pde_loss = torch.mean(pde_res**2)

        # 2) Boundary Loss
        rho_bd_np, z_bd_np = sample_boundary_points(rho_max, z_max, num_bd)
        bd_loss = boundary_loss(model, rho_bd_np, z_bd_np, device)


        # 4) Combine
        total_loss = lambda_pde*pde_loss + lambda_bd*bd_loss 

        # Backprop
        total_loss.backward()
        optimizer.step()

        # Optional print
        if epoch % 1000 == 0 or epoch == epochs:
            E_current = model.get_energy().item()
            print(f"[Epoch {epoch:4d}] PDE: {pde_loss.item():.6f}, "
                  f"BD: {bd_loss.item():.6f}"
                  f"E~{E_current:.4f}, total: {total_loss.item():.6f}")

    return model  # trained