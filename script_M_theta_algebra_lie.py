#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoria exata da reconstrução de M(theta) a partir dos geradores da base Alpha
e comparação dinâmica utilizando o operador reconstruído com erro zero.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

# -------------------- Parâmetros --------------------
SEED = 42
STEPS = 100_000
THETA = np.pi / 2 - 0.001
NOISE_SIGMA = 0.05
PLOT_POINTS = 50_000
OUT = Path("resultados_auditoria_M_theta_corrigido")
OUT.mkdir(exist_ok=True)

I = 1j
I4 = np.eye(4, dtype=complex)

# -------------------- Matriz original M(theta) --------------------
def build_M_original(theta, mu=1.0):
    t = np.tan(theta)
    c = 1.0 / t
    return np.array([
        [1, -c, -t, 1],
        [c, 1j, -1, -t],
        [t, -1, mu, -c],
        [1, t, c, 1j * mu],
    ], dtype=complex)

# -------------------- Geradores B1,...,B16 do apêndice A --------------------
B1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
B2 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=float)
B3 = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
B4 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
B5 = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=float)
B6 = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=float)
B7 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=float)
B8 = np.array([[0, -1, -1, 0], [1, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]], dtype=float)
B9 = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
B10 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
B11 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=float)
B12 = np.array([[0, 1, -1, 0], [1, 0, 0, -1], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)
B13 = np.array([[2, 0, 0, 1], [0, 0, 1, 0], [0, -1, -1, 0], [1, 0, 0, 1]], dtype=float)
B14 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, -1, -1, 0], [1, 0, 0, 1]], dtype=float)
B15 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]], dtype=float)
B16 = np.array([[0, 1, 1, 0], [-1, 0, 0, -1], [2, 0, 0, 1], [0, 0, 1, 0]], dtype=float)

def build_M_fix_from_generators():
    return (
        (1.5 - 0.5j) * B1
        + (-1.0 + 1.0j) * B4
        + (-0.5 + 0.5j) * B5
        + B6
        + (1.0 - 1.0j) * B11
        + (0.5 - 0.5j) * B13
        + (-1.0 + 1.0j) * B14
    ).astype(complex)

def build_M_from_generators(theta):
    return build_M_fix_from_generators() + (1.0 / np.tan(theta)) * B2 + np.tan(theta) * B3

# -------------------- Auditoria matricial --------------------
def audit_reconstruction(theta_values):
    rows = []
    print("Auditoria exata da reconstrução M(theta)")
    for theta in theta_values:
        M0 = build_M_original(theta)
        M1 = build_M_from_generators(theta)
        diff = M0 - M1
        fro = float(np.linalg.norm(diff, ord="fro"))
        maxerr = float(np.max(np.abs(diff)))
        rows.append({"theta_rad": float(theta), "frobenius_error": fro})
        print(f"theta: {theta:.5f} | Erro Frobenius: {fro:.2e}")
    return rows

# -------------------- Simulação Dinâmica --------------------
def simulate_with_noise(operator, noise_matrix, initial_state):
    state = initial_state.astype(float).copy()
    state /= np.linalg.norm(state)
    trajectory = np.empty((STEPS, 4), dtype=float)
    for k in range(STEPS):
        candidate = np.real(operator @ state) + noise_matrix[k]
        norm = np.linalg.norm(candidate)
        state = candidate / norm
        trajectory[k] = state
    return trajectory

def plot_trajectory_comparison(traj_original, traj_reconstructed, theta):
    fig = plt.figure(figsize=(14, 6))
    titles = [rf"Trajectory under $M(\theta)$", "Exact reconstruction via generators"]
    trajectories = [traj_original, traj_reconstructed]

    for idx, (traj, title) in enumerate(zip(trajectories, titles), start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")

        # Reduzimos para 11.000 pontos para manter o gráfico limpo e evitar a saturação amarela
        take = np.linspace(0, len(traj) - 1, min(11000, len(traj)), dtype=int)
        xyz = traj[take]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        t = np.linspace(0, 1, len(x))

        # 1. Linha principal (fina e suave em royalblue)
        ax.plot(x, y, z, lw=0.3, alpha=0.8, color='royalblue')

        # 2. Pontos coloridos pelo tempo usando o colormap plasma
        sc = ax.scatter(x, y, z, c=t, cmap=cm.plasma, s=0.5, alpha=0.5)

        # 3. Destaque para início e fim
        ax.scatter(x[0], y[0], z[0], color='lime', label='Start', s=50, edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], color='red', label='End', s=50, edgecolors='black')

        ax.set_xlabel('Dimension 1 — real')
        ax.set_ylabel('Dimension 2 — real')
        ax.set_zlabel('Dimension 3 — real')
        ax.set_title(f'{title}\nθ = {theta:.5f}, {len(traj):,} steps')

        # Barra de cores para o progresso temporal
        cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label('Temporal Progression')
        ax.legend(loc='upper right')

    #fig.suptitle(f"Stochastic Trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "comparacao_trajetorias_original_estilo.png", dpi=180, bbox_inches="tight")
    plt.show()

def main():
    theta_values = [THETA]
    audit_reconstruction(theta_values)

    theta = THETA
    M_original = build_M_original(theta)
    M_reconstructed = build_M_from_generators(theta)

    print("\nErro Frobenius:", np.linalg.norm(M_original - M_reconstructed, ord="fro"))

    rng = np.random.default_rng(SEED)
    initial_state = rng.normal(size=4)
    noise_matrix = NOISE_SIGMA * rng.normal(size=(STEPS, 4))

    traj_original = simulate_with_noise(M_original, noise_matrix, initial_state)
    traj_reconstructed = simulate_with_noise(M_reconstructed, noise_matrix, initial_state)

    plot_trajectory_comparison(traj_original, traj_reconstructed, theta)
    print("Gráfico gerado com o colormap plasma e salvo em:", OUT.resolve())

    # Statistical comparison of the trajectories
    diff = np.asarray(traj_original) - np.asarray(traj_reconstructed)
    abs_diff = np.abs(diff)

    mae = np.mean(abs_diff)
    max_error = np.max(abs_diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    norm_original = np.linalg.norm(traj_original)
    relative_error = (
        np.linalg.norm(diff) / max(norm_original, 1e-15)
    )

    print("\n=== Trajectory comparison statistics ===")
    print(f"Iterations: {STEPS:,}")
    print(f"Mean absolute error (MAE): {mae:.6e}")
    print(f"Maximum absolute error: {max_error:.6e}")
    print(f"Root mean square error (RMSE): {rmse:.6e}")
    print(f"Global relative error: {relative_error:.6e}")

if __name__ == "__main__":
    main()
