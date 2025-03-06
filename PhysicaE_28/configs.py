import torch 

z_i = 0
L_values = [10, 20, 40, 60, 80, 100]
E0 = 0.200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Physical constants (example placeholders; use realistic values)
hbar = 1.0         # Planck's constant / 2π in chosen units
mstar = 1.0        # effective mass
omega = 1.0        # parabolic well frequency
e_charge = 1.0     # elementary charge
epsilon_r = 1.0    # dielectric constant, etc.
z_i = 0.0   