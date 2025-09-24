import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import networkx as nx
import pandas as pd
from itertools import combinations
from scipy.sparse.csgraph import connected_components

# ==========================================================
# Núcleo do bloco M(θ, μ) - ORIGINAL
# ==========================================================
def build_M_alpha_block(theta: float, mu: float = 1.0, size: int = 4) -> np.ndarray:
    """
    Constrói um bloco M_alpha de tamanho 'size' x 'size'.
    Para size > 4, a base 4x4 é repetida/encaixada ao longo da diagonal.
    """
    eps = 1e-12
    s, c = np.sin(theta), np.cos(theta)
    # Evitar divisões instáveis
    if abs(s) < eps:
        s = np.sign(s) * eps if s != 0 else eps
    if abs(c) < eps:
        c = np.sign(c) * eps if c != 0 else eps

    tan = np.clip(s / c, -1e6, 1e6)
    cot = np.clip(c / s, -1e6, 1e6)

    base = np.array([
        [1.0, -cot, -tan, 1.0],
        [cot, 0+1j, -1.0, -tan],
        [tan, -1.0,  mu,  -cot],
        [1.0,  tan,  cot,  0+1j*mu]
    ], dtype=np.complex128)

    if size == 4:
        return base

    # Bloco maior: preencher com cópias da base 4x4 na diagonal
    M = np.zeros((size, size), dtype=np.complex128)
    for i in range(0, size, 4):
        b = min(4, size - i)
        M[i:i+b, i:i+b] = base[:b, :b]
    return M

# ==========================================================
# M_alpha de dimensão (n+1)x(n+1) - ORIGINAL
# ==========================================================
def generate_M_alpha_Sn(n: int, theta: float, mu: float = 1.0) -> tuple[np.ndarray, int, int]:
    """
    Gera a matriz M_alpha (n+1)x(n+1) com blocos adaptativos na diagonal.
    Retorna (M, num_blocks, block_size).
    """
    size = n + 1
    num_blocks = ceil(size / 4)
    block_size = ceil(size / num_blocks)

    M = np.zeros((size, size), dtype=np.complex128)
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, size)
        cur_sz = end - start
        M[start:end, start:end] = build_M_alpha_block(theta, mu, cur_sz)

    return M, num_blocks, block_size

# ==========================================================
# MÉTODO ORIGINAL (complementado)
# ==========================================================
def original_betti_graph(M: np.ndarray, threshold: float = 1e-2) -> tuple[int, int, int]:
    """
    Método original baseado em grafo não-direcionado.
    """
    A = (np.abs(M) > threshold).astype(int)
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)

    H0 = nx.number_connected_components(G)
    V = G.number_of_nodes()
    E = G.number_of_edges()
    H1 = max(0, E - V + H0)
    H2 = count_triangles_original(G)

    return H0, H1, H2

def count_triangles_original(G: nx.Graph) -> int:
    """
    Contagem original de triângulos (método não-direcionado).
    """
    triangles = 0
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        for u, v in combinations(neighbors, 2):
            if G.has_edge(u, v):
                triangles += 1
    return triangles // 3

# ==========================================================
# HOMOLOGIA DE CAMINHOS MODIFICADA (conservadora)
# ==========================================================
def conservative_path_homology(M: np.ndarray, threshold: float = 1e-2) -> tuple[int, int, int]:
    """
    Versão conservadora da homologia de caminhos que mantém resultados
    mais próximos do método original, mas ainda considera a natureza direcionada.
    """
    A = (np.abs(M) > threshold).astype(int)
    np.fill_diagonal(A, 0)

    n = A.shape[0]

    # H0: Usar componentes conectadas não-direcionadas (como original)
    A_undirected = np.maximum(A, A.T)  # Tornar simétrico para H0
    H0 = compute_connected_components(A_undirected)

    # H1: Fórmula conservadora que considera apenas ciclos simples
    H1 = compute_conservative_H1(A, H0)

    # H2: Triângulos não-direcionados (como original)
    H2 = compute_undirected_triangles(A)

    return H0, H1, H2

def compute_connected_components(A: np.ndarray) -> int:
    """
    Calcula componentes conectadas não-direcionadas.
    """
    G = nx.from_numpy_array(A)
    return nx.number_connected_components(G)

def compute_conservative_H1(A: np.ndarray, H0: int) -> int:
    """
    Calcula H1 de forma conservadora, considerando apenas a estrutura
    não-direcionada subjacente.
    """
    n = A.shape[0]

    # Criar grafo não-direcionado para cálculo de H1
    A_undirected = np.maximum(A, A.T)
    G = nx.from_numpy_array(A_undirected)

    V = G.number_of_nodes()
    E = G.number_of_edges()

    # Fórmula de Euler para grafos (como método original)
    return max(0, E - V + H0)

def compute_undirected_triangles(A: np.ndarray) -> int:
    """
    Calcula triângulos não-direcionados (equivalente ao método original).
    """
    A_undirected = np.maximum(A, A.T)  # Tornar simétrico
    G = nx.from_numpy_array(A_undirected)
    return count_triangles_original(G)

# ==========================================================
# HOMOLOGIA DE CAMINHOS COMPLETA (para referência)
# ==========================================================
def full_path_homology(M: np.ndarray, threshold: float = 1e-2) -> tuple[int, int, int]:
    """
    Homologia de caminhos completa (valores maiores, para comparação).
    """
    A = (np.abs(M) > threshold).astype(int)
    np.fill_diagonal(A, 0)

    n = A.shape[0]

    # H0: Componentes fortemente conectadas
    H0 = compute_strongly_connected_components(A)

    # H1: Ciclos direcionados completos
    H1 = compute_full_directed_H1(A, H0)

    # H2: Todos os triângulos direcionados
    H2 = compute_all_directed_triangles(A)

    return H0, H1, H2

def compute_strongly_connected_components(A: np.ndarray) -> int:
    """
    Componentes fortemente conectadas (direcionadas).
    """
    n_components, _ = connected_components(A, directed=True, connection='strong')
    return n_components

def compute_full_directed_H1(A: np.ndarray, H0: int) -> int:
    """
    H1 completo considerando toda a estrutura direcionada.
    """
    n = A.shape[0]
    E = np.sum(A > 0)

    # Contar ciclos elementares
    elementary_cycles = count_elementary_cycles(A, max_length=10)

    return max(0, elementary_cycles)

def compute_all_directed_triangles(A: np.ndarray) -> int:
    """
    Conta todos os triângulos direcionados possíveis.
    """
    n = A.shape[0]
    triangle_count = 0

    for i in range(n):
        for j in range(n):
            if A[i, j]:
                for k in range(n):
                    if A[j, k] and A[k, i]:
                        triangle_count += 1

    return triangle_count // 3

def count_elementary_cycles(A: np.ndarray, max_length: int = 10) -> int:
    """
    Conta ciclos elementares (aproximação conservadora).
    """
    n = A.shape[0]
    cycle_count = 0

    # Para ciclos pequenos (2, 3, 4 vértices)
    for i in range(n):
        for j in range(n):
            if A[i, j] and A[j, i]:  # ciclo de tamanho 2
                cycle_count += 1

    # Ciclos de tamanho 3 (já contados em triângulos)
    # Ciclos de tamanho 4+
    # (implementação simplificada para evitar complexidade excessiva)

    return cycle_count // 2  # Cada ciclo contado duas vezes

# ==========================================================
# Tabela comparativa
# ==========================================================
def compute_comparison_table(theta: float, mu: float,
                           spheres: list[int] = [2, 3, 4, 5, 6, 7, 15, 23, 31],
                           threshold: float = 1e-3) -> pd.DataFrame:
    """
    Constrói tabela comparativa com três métodos.
    """
    rows = []
    for n in spheres:
        M, nb, bs = generate_M_alpha_Sn(n, theta, mu)

        # Método original
        H0_orig, H1_orig, H2_orig = original_betti_graph(M, threshold)

        # Método conservador
        H0_cons, H1_cons, H2_cons = conservative_path_homology(M, threshold)

        # Método completo (referência)
        H0_full, H1_full, H2_full = full_path_homology(M, threshold)

        rows.append({
            "S^n": f"S^{n}",
            "n": n,
            "H0_orig": H0_orig,
            "H1_orig": H1_orig,
            "H2_orig": H2_orig,
            "H0_cons": H0_cons,
            "H1_cons": H1_cons,
            "H2_cons": H2_cons,
            "H0_full": H0_full,
            "H1_full": H1_full,
            "H2_full": H2_full,
            "num_blocks": nb,
            "block_size": bs
        })

    df = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    return df

# ==========================================================
# Visualização
# ==========================================================
def plot_comparison(df: pd.DataFrame, title: str = ""):
    """
    Plota comparação entre os métodos.
    """
    spheres = df["n"].tolist()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # H0
    axes[0].plot(spheres, df["H0_orig"], 'o-', label="Original", linewidth=2, markersize=6)
    axes[0].plot(spheres, df["H0_cons"], 's-', label="Conservador", linewidth=2, markersize=6)
    axes[0].plot(spheres, df["H0_full"], '^-', label="Completo", linewidth=2, markersize=6, alpha=0.7)
    axes[0].set_ylabel("H0")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # H1
    axes[1].plot(spheres, df["H1_orig"], 'o-', label="Original", linewidth=2, markersize=6)
    axes[1].plot(spheres, df["H1_cons"], 's-', label="Conservador", linewidth=2, markersize=6)
    axes[1].plot(spheres, df["H1_full"], '^-', label="Completo", linewidth=2, markersize=6, alpha=0.7)
    axes[1].set_ylabel("H1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # H2
    axes[2].plot(spheres, df["H2_orig"], 'o-', label="Original", linewidth=2, markersize=6)
    axes[2].plot(spheres, df["H2_cons"], 's-', label="Conservador", linewidth=2, markersize=6)
    axes[2].plot(spheres, df["H2_full"], '^-', label="Completo", linewidth=2, markersize=6, alpha=0.7)
    axes[2].set_xlabel("Dimensão n (S^n)")
    axes[2].set_ylabel("H2")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title if title else "Comparação de Métodos de Homologia")
    plt.tight_layout()
    plt.show()

# ==========================================================
# Análise estatística
# ==========================================================
def analyze_results(df: pd.DataFrame):
    """
    Analisa os resultados comparativos.
    """
    print("Análise Comparativa:")
    print("=" * 60)

    print("\n1. Similaridade entre Método Original e Conservador:")
    h0_match = np.mean(df["H0_orig"] == df["H0_cons"])
    h1_match = np.mean(df["H1_orig"] == df["H1_cons"])
    h2_match = np.mean(df["H2_orig"] == df["H2_cons"])

    print(f"   H0: {h0_match:.1%} de coincidência")
    print(f"   H1: {h1_match:.1%} de coincidência")
    print(f"   H2: {h2_match:.1%} de coincidência")

    print("\n2. Diferenças entre Método Completo e Original:")
    h0_diff = np.mean(df["H0_full"] / np.maximum(df["H0_orig"], 1))
    h1_diff = np.mean(df["H1_full"] / np.maximum(df["H1_orig"], 1))
    h2_diff = np.mean(df["H2_full"] / np.maximum(df["H2_orig"], 1))

    print(f"   H0: {h0_diff:.1f}x maior (em média)")
    print(f"   H1: {h1_diff:.1f}x maior (em média)")
    print(f"   H2: {h2_diff:.1f}x maior (em média)")

# ==========================================================
# Execução principal
# ==========================================================
if __name__ == "__main__":
    theta = 1.5606
    mu = 1.0
    threshold = 1e-3

    print("Calculando comparação entre métodos de homologia...")
    df = compute_comparison_table(theta, mu, threshold=threshold)

    print("\nResultados:")
    print(df[['S^n', 'n', 'H0_orig', 'H0_cons', 'H0_full',
              'H1_orig', 'H1_cons', 'H1_full',
              'H2_orig', 'H2_cons', 'H2_full']].round(2))

    # Análise
    analyze_results(df)

    # Plot
    plot_comparison(df, title=f"Comparação de Métodos - θ={theta:.4f}")

    # Salvar
    df.to_csv(f"homologia_comparativa_theta{theta:.4f}.csv", index=False)
    print(f"\nResultados salvos em: homologia_comparativa_theta{theta:.4f}.csv")
