import torch 
from configs import hbar, mstar, omega, e_charge, epsilon_r, z_i
from torch.autograd import grad

def laplacian_cylindrical(psi, rho, z):
    # 1) First derivative wrt rho
    dpsi_drho = torch.autograd.grad(
        psi, rho, grad_outputs=torch.ones_like(psi),
        create_graph=True,   # <--- Key: We want to differentiate again
        retain_graph=True    # <--- Keep graph for repeated usage
    )[0]
    
    # 2) Multiply by rho, then differentiate again
    rho_dpsi_drho = rho * dpsi_drho
    d_rho_dpsi_drho = torch.autograd.grad(
        rho_dpsi_drho, rho, grad_outputs=torch.ones_like(rho_dpsi_drho),
        create_graph=True,   # <--- Must also set create_graph here
        retain_graph=True
    )[0]
    
    term_rho = d_rho_dpsi_drho / (rho + 1e-10)
    
    # 3) Second derivative wrt z
    dpsi_dz = torch.autograd.grad(
        psi, z, grad_outputs=torch.ones_like(psi),
        create_graph=True,   # <--- again
        retain_graph=True
    )[0]
    d2psi_dz2 = torch.autograd.grad(
        dpsi_dz, z, grad_outputs=torch.ones_like(dpsi_dz),
        create_graph=True,
        retain_graph=True
    )[0]
    
    return term_rho + d2psi_dz2

def potential(rho, z):
    """
    Returns V(rho,z) = (1/2) m* ω^2 z^2 - e^2/(ε sqrt(rho^2 + (z - z_i)^2))
    """
    # Parabolic term
    parabolic = 0.5 * mstar * (omega**2) * (z**2)
    
    # Coulomb-like term
    dist = torch.sqrt(rho**2 + (z - z_i)**2 + 1e-12)  # small epsilon to avoid zero
    coulomb = - (e_charge**2) / (epsilon_r * dist)
    
    return parabolic + coulomb

def schrodinger_pde_residual(model, rho, z):
    """
    Compute PDE residual:  H Ψ - E Ψ = 0  =>  (kinetic + potential*Ψ - E Ψ).
    
    model:  PINN_Schrodinger
    rho, z: Tensors with shape (batch_size, 1) and requires_grad=True
    
    Returns a Tensor of shape (batch_size, 1) with the PDE residual at each collocation point.
    """
    # 1) Forward pass
    psi = model(rho, z)         # shape: (batch_size, 1)
    E   = model.get_energy()    # scalar trainable parameter
    
    # 2) Laplacian(psi)
    lap_psi = laplacian_cylindrical(psi, rho, z)  # (batch_size, 1)
    
    # 3) Kinetic term: - (hbar^2 / 2m*) lap_psi
    kinetic = - (hbar**2 / (2.0 * mstar)) * lap_psi
    
    # 4) Potential term: V(rho,z) * psi
    V_ = potential(rho, z)
    pot_term = V_ * psi
    
    # 5) PDE residual: [Kinetic + Potential*Ψ - E*Ψ]
    residual = kinetic + pot_term - E * psi
    
    return residual

def boundary_loss(model, rho_bd, z_bd, device):
    """
    MSE( Psi(rho_bd, z_bd), 0 ) => Enforces wavefunction -> 0
    """
    rho_t = torch.tensor(rho_bd, dtype=torch.float32, requires_grad=False).to(device)
    z_t   = torch.tensor(z_bd,   dtype=torch.float32, requires_grad=False).to(device)
    psi_bd = model(rho_t, z_t)  # shape (N, 1)
    return torch.mean(psi_bd**2)
