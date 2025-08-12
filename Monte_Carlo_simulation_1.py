import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set parameters to handle large plots and simplify path
plt.rcParams['agg.path.chunksize'] = 100000  # Increased chunksize
plt.rcParams['path.simplify_threshold'] = 1.0 # Increased simplify_threshold

# =====================================
# 1. DEFINIÇÃO DA MATRIZ M(θ) DO GRUPO ALPHA
# =====================================
def build_M(theta):
    """Constrói a matriz M(θ) do artigo (Equação 3)."""
    mu = 1.0  # Valor padrão para μ
    i = 1j    # Unidade imaginária

    M = np.array([
        [1, -1/np.tan(theta), -np.tan(theta), 1],
        [1/np.tan(theta), i, -1, -np.tan(theta)],
        [np.tan(theta), -1, mu, -1/np.tan(theta)],
        [1, np.tan(theta), 1/np.tan(theta), i*mu]
    ], dtype=complex)

    return M

# =====================================
# 2. SIMULAÇÃO DE UMA ÚNICA RUN (TRAJETÓRIA)
# =====================================
def simulate_single_run(theta, steps=100000):
    """Simula uma única trajetória longa."""
    np.random.seed(42)  # Para reprodutibilidade
    M = build_M(theta)

    # Estado inicial aleatório em 4D (parte real para simplificar)
    state = np.random.randn(4)
    trajectory = np.zeros((steps, 4))  # Pré-alocação

    for step in range(steps):
        # Aplica M(θ) com ruído estocástico
        noise = 0.05 * np.random.randn(4)  # Ruído reduzido
        state = np.real(M @ state) + noise
        state /= np.linalg.norm(state) + 1e-10  # Normalização
        trajectory[step] = state

    return trajectory

# =====================================
# 3. VISUALIZAÇÃO (PROJEÇÃO 3D)
# =====================================
def plot_single_trajectory(trajectory, theta, dims=(0,1,2)):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = trajectory[:, dims[0]], trajectory[:, dims[1]], trajectory[:, dims[2]]
    t = np.linspace(0, 1, len(x))  # Para gradiente de cor

    # Linha principal (más fina e suave)
    ax.plot(x, y, z, lw=0.3, alpha=0.8, color='royalblue')

    # Pontos coloridos pelo tempo
    sc = ax.scatter(x, y, z, c=t, cmap=cm.plasma, s=0.5, alpha=0.5)

    # Destaque para início/fim
    ax.scatter(x[0], y[0], z[0], color='lime', label='Início', s=50, edgecolors='black')
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='Fim', s=50, edgecolors='black')

    ax.set_xlabel(f'Dimensão {dims[0]+1}')
    ax.set_ylabel(f'Dimensão {dims[1]+1}')
    ax.set_zlabel(f'Dimensão {dims[2]+1}')
    ax.set_title(f'Trajetória Única sob $M(\\theta)$\n$\\theta = {theta:.3f}$, {len(x):,} passos', pad=20)

    # Barra de cores para tempo
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Progresso Temporal')

    ax.legend()
    plt.tight_layout()
    plt.show()

# =====================================
# 4. EXECUÇÃO PARA UM VALOR ESPECÍFICO DE θ
# =====================================
if __name__ == "__main__":
    #theta = np.pi/2 - 0.001  # Valor único de θ (ajuste conforme necessário)
    theta = 0.0001
    #theta = np.pi/4
    steps = 100000   # Número aumentado de passos para trajetória única

    print(f"Simulando 1 trajetória com {steps:,} passos para θ = {theta:.3f}...")
    trajectory = simulate_single_run(theta, steps)

    print("Plotando resultados...")
    plot_single_trajectory(trajectory, theta, dims=(0,1,2))  # Projeta as 3 primeiras dimensões
