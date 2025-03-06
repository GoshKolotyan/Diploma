import matplotlib.pyplot as plt

import torch
from model import PINN
from training import train_pinn_for_L
from configs import device, L_values, E0, num_epochs


def example_scan_over_widths(L_values, E0, device, num_epochs):
    binding_energies = []

    for L in L_values:
        # Build a fresh model
        model = PINN(n_hidden=64, n_layers=5)

        print(f"--- Training for QW width L={L} Å ---")
        # We'll define 'train_pinn_for_L' as above
        trained_model = train_pinn_for_L(
            model,
            L,
            rho_max=5.0,
            epochs=num_epochs,
            num_coll=2000,
            num_bd=400,
            num_norm=1000,
            lambda_pde=1.0,
            lambda_bd=10.0,
            lr=1e-3,
            device=device,
        )

        # Once trained, get final energy
        E_pinn = trained_model.get_energy().item()
        # Binding energy = E0 - E_pinn
        E_bind = E0 - E_pinn
        binding_energies.append(abs(E_bind))
        print(f"  --> E_pinn={E_pinn:.4f}, E_bind={E_bind:.4f}\n")

    # Now you can plot or analyze the binding energies
    # e.g. using matplotlib
    import matplotlib.pyplot as plt

    plt.plot(L_values, binding_energies, marker="o")
    plt.xlabel("QW width L (Å)")
    plt.ylabel("Binding Energy")
    plt.title("Donor Binding Energy vs. QW width (PINN)")
    plt.show()


if __name__ == "__main__":
    example_scan_over_widths(L_values=L_values, 
                             E0=E0, 
                             device=device,
                             num_epochs=num_epochs)
