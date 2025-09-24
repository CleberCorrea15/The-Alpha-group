import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
"""
Script para análise do operador M_alpha em esferas S^n,
calculando aproximações dos números de Betti (H0, H1, H2)
a partir de um grafo derivado da matriz |M_alpha| > threshold.
Versão melhorada com cálculos mais precisos de homologia.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import networkx as nx
import pandas as pd
from scipy.linalg import null_space
import warnings

# ==========================================================
# Núcleo do bloco M(θ, μ)
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
# M_alpha de dimensão (n+1)x(n+1)
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
# Betti via grafo (versão melhorada)
# ==========================================================
def improved_betti_graph(M: np.ndarray, threshold: float = 1e-2) -> tuple[int, int, int]:
    """
    Versão melhorada do cálculo de números de Betti.
    Usa métodos mais robustos para estimar H1 e H2.
    """
    A = (np.abs(M) > threshold).astype(int)
    np.fill_diagonal(A, 0)  # remover laços
    G = nx.from_numpy_array(A)

    # H0: número de componentes conexas (correto)
    H0 = nx.number_connected_components(G)

    # Para H1: cálculo mais robusto
    # Para grafos conexos: H1 = E - V + 1 (número de ciclos independentes)
    if H0 == 1:
        H1 = max(0, G.number_of_edges() - G.number_of_nodes() + 1)
    else:
        # Para múltiplos componentes: soma sobre componentes
        H1 = 0
        for comp in nx.connected_components(G):
            subgraph = G.subgraph(comp)
            H1 += max(0, subgraph.number_of_edges() - subgraph.number_of_nodes() + 1)

    # H2: difícil de calcular para grafos, geralmente 0 para grafos planares
    # Usamos uma aproximação baseada em cliques de tamanho 3
    H2 = 0
    try:
        # Contar triângulos de forma mais precisa
        triangles = sum(nx.triangles(G).values()) // 3
        H2 = max(0, triangles)
    except:
        pass

    return H0, H1, H2


# ==========================================================
# Cálculo alternativo via álgebra linear
# ==========================================================
def betti_linear_algebra(M: np.ndarray, threshold: float = 1e-2) -> tuple[int, int, int]:
    """
    Alternativa: cálculo de números de Betti via álgebra linear.
    Nota: Esta é uma abordagem experimental.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Matriz de adjacência
        A = (np.abs(M) > threshold).astype(float)
        np.fill_diagonal(A, 0)

        # H0: número de componentes conexas
        G = nx.from_numpy_array(A)
        H0 = nx.number_connected_components(G)

        # Para H1: usar o posto da matriz de adjacência
        # Esta é uma aproximação grosseira
        rank = np.linalg.matrix_rank(A)
        H1 = max(0, np.sum(A > 0) - rank)

        # H2: difícil de calcular, retornar 0
        H2 = 0

    return H0, H1, H2


# ==========================================================
# Tabela e gráfico
# ==========================================================
def compute_table(theta: float, mu: float,
                  spheres: list[int] = [2, 3, 4, 5, 6, 7, 15, 23, 31],
                  threshold: float = 1e-2, method: str = "graph") -> pd.DataFrame:
    """
    Constrói a tabela com H0, H1, H2 e metadados (num_blocks, block_size)
    para as esferas S^n em 'spheres', dados θ e μ.

    Parâmetros:
    - method: "graph" (padrão) ou "linear" para o método de cálculo
    """
    rows = []
    for n in spheres:
        M, nb, bs = generate_M_alpha_Sn(n, theta, mu)

        if method == "graph":
            H0, H1, H2 = improved_betti_graph(M, threshold=threshold)
        else:
            H0, H1, H2 = betti_linear_algebra(M, threshold=threshold)

        rows.append({
            "S^n": f"S^{n}",
            "n": n,
            "H0": H0,
            "H1": H1,
            "H2": H2,
            "num_blocks": nb,
            "block_size": bs
        })

    df = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    return df


def plot_betti_table(df: pd.DataFrame, title: str = ""):
    """
    Plota H0, H1, H2 da tabela retornada por compute_table.
    """
    spheres = df["n"].tolist()
    H0 = df["H0"].tolist()
    H1 = df["H1"].tolist()
    H2 = df["H2"].tolist()

    plt.figure(figsize=(12, 6))
    plt.plot(spheres, H0, 'o-', label="H0")
    plt.plot(spheres, H1, 's-', label="H1")
    plt.plot(spheres, H2, '^-', label="H2")
    plt.xticks(spheres)
    plt.xlabel("Dimensão n (S^n)")
    plt.ylabel("Betti numbers (aprox.)")
    plt.title(title if title else "Betti numbers dinâmicos (H0–H2) para S³–S³¹")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_comparison(df1: pd.DataFrame, df2: pd.DataFrame, title: str = ""):
    """
    Compara dois conjuntos de resultados.
    """
    spheres = df1["n"].tolist()

    plt.figure(figsize=(14, 8))

    # H0
    plt.subplot(2, 2, 1)
    plt.plot(spheres, df1["H0"], 'o-', label="Método Grafo")
    plt.plot(spheres, df2["H0"], 's-', label="Método Linear")
    plt.xticks(spheres)
    plt.xlabel("Dimensão n (S^n)")
    plt.ylabel("H0")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # H1
    plt.subplot(2, 2, 2)
    plt.plot(spheres, df1["H1"], 'o-', label="Método Grafo")
    plt.plot(spheres, df2["H1"], 's-', label="Método Linear")
    plt.xticks(spheres)
    plt.xlabel("Dimensão n (S^n)")
    plt.ylabel("H1")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # H2
    plt.subplot(2, 2, 3)
    plt.plot(spheres, df1["H2"], 'o-', label="Método Grafo")
    plt.plot(spheres, df2["H2"], 's-', label="Método Linear")
    plt.xticks(spheres)
    plt.xlabel("Dimensão n (S^n)")
    plt.ylabel("H2")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle(title if title else "Comparação de métodos de cálculo")
    plt.tight_layout()
    plt.show()


# ==========================================================
# Execução principal
# ==========================================================
if __name__ == "__main__":
    theta = 1.5606        # ~ próximo de π/2
    mu = 1.0
    threshold = 1e-2    # sensibilidade do grafo

    print("Calculando com método de grafo...")
    df_graph = compute_table(theta, mu, spheres=[2, 3, 4, 5, 6, 7, 15, 23, 31],
                           threshold=threshold, method="graph")
    print("Método de Grafo:")
    print(df_graph)

    print("\nCalculando com método de álgebra linear...")
    df_linear = compute_table(theta, mu, spheres=[2, 3, 4, 5, 6, 7, 15, 23, 31],
                            threshold=threshold, method="linear")
    print("Método de Álgebra Linear:")
    print(df_linear)

    # Plot dos resultados
    plot_betti_table(df_graph, title=f"Betti numbers (Método Grafo) — θ={theta:.2f}, μ={mu:.2f}, thr={threshold:g}")
    plot_betti_table(df_linear, title=f"Betti numbers (Método Linear) — θ={theta:.2f}, μ={mu:.2f}, thr={threshold:g}")

    # Comparação entre métodos
    plot_comparison(df_graph, df_linear, title=f"Comparação de métodos — θ={theta:.2f}, μ={mu:.2f}")

    # Análise de sensibilidade ao threshold
    print("\nAnálise de sensibilidade ao threshold:")
    thresholds = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    for thr in thresholds:
        df_temp = compute_table(theta, mu, spheres=[3, 7, 15], threshold=thr, method="graph")
        print(f"Threshold {thr}: S³(H0,H1,H2)=({df_temp.iloc[0]['H0']},{df_temp.iloc[0]['H1']},{df_temp.iloc[0]['H2']})")
