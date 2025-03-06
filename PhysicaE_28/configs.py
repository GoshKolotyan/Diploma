import torch 

L_values = [item for item in range(0, 100, 10)]
E0 = 0.200
num_epochs = 5_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Physical constants (example placeholders; use realistic values)

h = 6.62607015e-34  # Planck's constant (J·s)
hbar = 1#h / (2 * torch.pi)  # Reduced Planck's constant (ħ)
mstar = 1.0  # Effective mass (in chosen units)
omega = 1.0  # Parabolic potential well frequency
e_charge = 1#1.602176634e-19  # Elementary charge (C)
epsilon_r = 1.0  # Relative dielectric constant
z_i = 0.0  # Initial z-position (or another relevant coordinate)
