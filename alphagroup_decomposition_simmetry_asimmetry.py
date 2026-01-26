import numpy as np
import matplotlib.pyplot as plt

def analyze_alpha_group_symmetric_antisymmetric():
    theta_range = np.linspace(40, 140, 2000)
    EPS = 1e-9

    sym_energy = []
    asym_energy = []
    total_energy = []

    for theta_deg in theta_range:

        theta = np.radians(theta_deg)
        t = np.tan(theta)
        t = np.clip(t, -1e3, 1e3)
        c = 1.0 / t if abs(t) > EPS else 1e3

        # ===============================
        # REAL GEOMETRIC MATRIX M(θ)
        # ===============================
        M = np.array([
            [1,  -c,  -t,   1],
            [c,  -1,   1,  -c],
            [t,   1,   1,  -t],
            [1,  -t,  -c,   1]
        ], dtype=float)

        # ===============================
        # INTRINSIC DECOMPOSITION
        # ===============================
        M_sym  = 0.5 * (M + M.T)
        M_asym = 0.5 * (M - M.T)

        # ===============================
        # ENERGIES (FROBENIUS)
        # ===============================
        E_sym  = np.sum(M_sym**2)
        E_asym = np.sum(M_asym**2)
        E_tot  = np.sum(M**2)

        sym_energy.append(E_sym)
        asym_energy.append(E_asym)
        total_energy.append(E_tot)

    # ===============================
    # VISUALIZATION
    # ===============================
    plt.figure(figsize=(12, 7))

    plt.plot(theta_range, sym_energy,
             marker='s', markevery=25, linewidth=2,
             label='Symmetric Part (Isotropic)')

    plt.plot(theta_range, asym_energy,
             marker='^', markevery=25, linewidth=2,
             label='Antisymmetric Part (Anisotropic)')

    plt.plot(theta_range, total_energy,
             linestyle='--', color='black', alpha=0.5,
             label='Total Energy')

    plt.axvline(90, color='black', linestyle=':',
                alpha=0.7, label=r'$\theta = 90^\circ$')

    plt.yscale('log')
    plt.xlabel(r'$\theta$ (degrees)')
    plt.ylabel('Energy (Frobenius norm)')
    plt.title('Symmetric / Antisymmetric Decomposition of $M(\\theta)$ — Alpha Group')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_alpha_group_symmetric_antisymmetric()

