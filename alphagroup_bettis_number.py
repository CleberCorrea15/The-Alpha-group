# -*- coding: utf-8 -*-
"""
Script para análise do operador M_alpha em esferas S^n,
calculando aproximações dos números de Betti (H0, H1, H2)
a partir de um grafo derivado da matriz |M_alpha| > threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import networkx as nx
import pandas as pd

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
# Betti via grafo
# ==========================================================
def betti_graph(M: np.ndarray, threshold: float = 1e-2) -> tuple[int, int, int]:
    """
    Aproxima H0, H1 e H2 a partir do grafo de adjacência |M| > threshold.
    H0: nº de componentes conectadas
    H1: ~ E - V + H0  (característica de Euler para grafos)
    H2: ~ nº de triângulos (faces) do grafo
    """
    A = (np.abs(M) > threshold).astype(int)
    np.fill_diagonal(A, 0)  # remover laços
    G = nx.from_numpy_array(A)

    H0 = nx.number_connected_components(G)
    V = G.number_of_nodes()
    E = G.number_of_edges()
    H1 = max(0, E - V + H0)
    tri_dict = nx.triangles(G)
    H2 = sum(tri_dict.values()) // 3
    return H0, H1, H2


# ==========================================================
# Tabela e gráfico
# ==========================================================
def compute_table(theta: float, mu: float,
                  spheres: list[int] = [3,4,5,6,7,15,23,31],
                  threshold: float = 1e-2) -> pd.DataFrame:
    """
    Constrói a tabela com H0, H1, H2 e metadados (num_blocks, block_size)
    para as esferas S^n em 'spheres', dados θ e μ.
    """
    rows = []
    for n in spheres:
        M, nb, bs = generate_M_alpha_Sn(n, theta, mu)
        H0, H1, H2 = betti_graph(M, threshold=threshold)
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


# ==========================================================
# Execução principal
# ==========================================================
if __name__ == "__main__":
    theta = 1.5606        # ~ próximo de π/2
    mu = 1.0
    threshold = 1e-2    # sensibilidade do grafo

    df = compute_table(theta, mu, spheres=[3,4,5,6,7,15,23,31], threshold=threshold)
    print(df)

    # Exemplo de ajuste manual (se quiser forçar valores experimentais observados):
    # df.loc[df['n'] == 31, ['H0','H1','H2']] = [5, 58, 31]

    plot_betti_table(df, title=f"Betti numbers (H0–H2) — θ={theta:.2f}, μ={mu:.2f}, thr={threshold:g}")
