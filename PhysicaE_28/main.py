import matplotlib.pyplot as plt

import torch
from model import PINN
from training import  train_pinn_for_L
from configs import device, L_values, E0



def example_scan_over_widths():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L_values = [10, 20, 40, 60, 80, 100]  # in Angstrom, for example
    E0 = 0.200  # reference conduction band energy, for instance

    binding_energies = []

    for L in L_values:
        # Build a fresh model
        model = PINN(n_hidden=64, n_layers=4)
        
        print(f"--- Training for QW width L={L} Å ---")
        # We'll define 'train_pinn_for_L' as above
        trained_model = train_pinn_for_L(
            model, L, rho_max=5.0, epochs=5000, num_coll=2000, num_bd=400,
            num_norm=1000, lambda_pde=1.0, lambda_bd=10.0,
            lr=1e-3, device=device
        )

        # Once trained, get final energy
        E_pinn = trained_model.get_energy().item()
        # Binding energy = E0 - E_pinn
        E_bind = E0 - E_pinn
        binding_energies.append(E_bind)
        print(f"  --> E_pinn={E_pinn:.4f}, E_bind={E_bind:.4f}\n")

    # Now you can plot or analyze the binding energies
    # e.g. using matplotlib
    import matplotlib.pyplot as plt
    plt.plot(L_values, binding_energies, marker='o')
    plt.xlabel("QW width L (Å)")
    plt.ylabel("Binding Energy")
    plt.title("Donor Binding Energy vs. QW width (PINN)")
    plt.show()

if __name__ == "__main__":
    example_scan_over_widths()