import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARÂMETROS UNIVERSAIS (CONFIGURADO PARA 1024 DIMENSÕES REAIS)
# ==========================================================

DIM_REAL = 1024  # Escalado para 1024 dimensões reais (512 dimensões complexas)
N_PASSOS = 200000

NOISE_LEVEL = 0.50

THETAS = [
    1.54,
    1.56,
    1.58
]

CORES = [
    "royalblue",
    "crimson",
    "orange"
]

# Cálculo automático da dimensão complexa correspondente
DIM_COMPLEX = DIM_REAL // 2

if DIM_REAL % 8 != 0:
    print("[AVISO] Para manter a integridade dos blocos 4x4 complexos,")
    print("recomenda-se usar DIM_REAL múltiplo de 8.")

# ==========================================================
# ÁLGEBRA DO GRUPO ALPHA (ESCALÁVEL VIA KRONECKER)
# ==========================================================

class AlphaGroupAlgebraScalable:

    def __init__(self, dim_complex):
        # Matrizes base originais (4x4)
        A_c_4 = np.array([
            [0,-1,0,0],
            [1,0,0,0],
            [0,0,0,-1],
            [0,0,1,0]
        ], dtype=np.complex128)

        A_t_4 = np.array([
            [0,0,-1,0],
            [0,0,0,-1],
            [1,0,0,0],
            [0,1,0,0]
        ], dtype=np.complex128)

        S_4 = np.array([
            [1,0,0,1],
            [0,0,-1,0],
            [0,-1,0,0],
            [1,0,0,0]
        ], dtype=np.complex128)

        B_mu_4 = np.array([
            [1,1,1,1],
            [1,1j,1,1],
            [1,1,1,1],
            [1,1,1,1j]
        ], dtype=np.complex128)

        # Calcula quantas vezes precisamos repetir o bloco 4x4 na diagonal
        n_repeticoes = dim_complex // 4
        I_rep = np.eye(n_repeticoes)

        # Expansão matemática automática via produto de Kronecker
        self.A_c = np.kron(I_rep, A_c_4)
        self.A_t = np.kron(I_rep, A_t_4)
        self.S   = np.kron(I_rep, S_4)
        self.B_mu = np.kron(I_rep, B_mu_4)

    def get_M_theta(self, theta):

        if np.isclose(np.mod(theta, np.pi), 0):
            theta += 1e-12

        cot_theta = 1.0 / np.tan(theta)
        tan_theta = np.tan(theta)

        return (
            self.S
            + cot_theta * self.A_c
            + tan_theta * self.A_t
        ) @ self.B_mu

# ==========================================================
# TRAJETÓRIA EM ALTA DIMENSÃO COM NORMA MÁXIMA
# ==========================================================

def run_trajectory(M_theta, n_steps, dim_complex, noise_level):

    # Vetor de estado inicializado na dimensão configurada
    x = np.full(dim_complex, 0.5, dtype=np.complex128)

    # Normalização por Norma Máxima
    norm = np.max(np.abs(x))
    if norm > 1e-12:
        x /= norm

    # Aloca espaço dinâmico para a saída (Número de colunas = DIM_REAL)
    pts = np.zeros((n_steps, dim_complex * 2))

    for t in range(n_steps):

        # Dinâmica principal
        x = M_theta @ x

        # Injeção de ruído em todas as dimensões do hiper-espaço
        if noise_level > 0:
            noise = noise_level * (
                np.random.randn(dim_complex)
                +
                1j * np.random.randn(dim_complex)
            )
            x = x + noise

        # Normalização por Norma Máxima
        norm = np.max(np.abs(x))
        if norm > 1e-12:
            x /= norm

        # Mapeamento para coordenadas Reais (colunas pares) e Imaginárias (colunas ímpares)
        for k in range(dim_complex):
            pts[t, 2 * k]     = x[k].real
            pts[t, 2 * k + 1] = x[k].imag

    return pts

# ==========================================================
# EXECUÇÃO E GERAÇÃO GRÁFICA
# ==========================================================

alg = AlphaGroupAlgebraScalable(dim_complex=DIM_COMPLEX)

fig = plt.figure(figsize=(12, 10), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

for theta, cor in zip(THETAS, CORES):

    M = alg.get_M_theta(theta)

    # Executa a simulação adaptada para 1024R
    pts_sim = run_trajectory(M, N_PASSOS, DIM_COMPLEX, NOISE_LEVEL)

    # Projeção invariante nos canais Re(x1), Re(x2) e Re(x4)
    X = pts_sim[:, 0]  # Re(x1)
    Y = pts_sim[:, 2]  # Re(x2)
    Z = pts_sim[:, 6]  # Re(x4)

    ax.plot(
        X, Y, Z,
        color=cor,
        linewidth=1.0,
        alpha=0.85,
        label=fr"$\theta={theta:.4f}$"
    )

    ax.scatter(X[0], Y[0], Z[0], color=cor, s=50)

# ==========================================================
# AJUSTES VISUAIS DETALHADOS
# ==========================================================

ax.set_title(
    fr"Universal Angular Dynamic Mapping ({DIM_REAL}D Real)" "\n"
    fr"Max Norm ($L_\infty$) | Projection: Re(x1), Re(x2), Re(x4)" "\n"
    f"Noise={NOISE_LEVEL}",
    fontsize=16,
    fontweight='bold',
    pad=25
)

ax.set_xlabel("Re(x1)", fontsize=12, labelpad=12)
ax.set_ylabel("Re(x2)", fontsize=12, labelpad=12)
ax.set_zlabel("Re(x4)", fontsize=12, labelpad=22)

# Fixação estrita nos limites do hipercubo unitário
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])

ax.view_init(elev=25, azim=-60)
ax.grid(alpha=0.35)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=11)

plt.show()
