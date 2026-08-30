# ==============================================================
# ALPHA GROUP — STRUCTURAL ANALYSIS OF THE 68 COORDINATE
# LIE SUBALGEBRAS OF THE 16-GENERATOR REPRESENTATION
# (OTIMIZADO PARA 12 THREADS COM NUMPY + MULTIPROCESSING)
# ==============================================================

import multiprocessing as mp
from itertools import combinations
import numpy as np

NUM_WORKERS = 12
N = 16

# 1. GERADORES EM MATRIZES NUMÉRICAS NUMPY (FLOAT64/INT64)
B_np = [
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]),
    np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]),
    np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]),
    np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]),
    np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]),
    np.array([[0, -1, -1, 0], [1, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]]),
    np.array([[0, 0, -1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]),
    np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]]),
    np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0]]),
    np.array([[0, 1, -1, 0], [1, 0, 0, -1], [0, 0, 0, 1], [0, 0, -1, 0]]),
    np.array([[2, 0, 0, 1], [0, 0, 1, 0], [0, -1, -1, 0], [1, 0, 0, 1]]),
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, -1, -1, 0], [1, 0, 0, 1]]),
    np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]]),
    np.array([[0, 1, 1, 0], [-1, 0, 0, -1], [2, 0, 0, 1], [0, 0, 1, 0]]),
]

# Construção da matriz A de transformação (16x16)
A_np = np.hstack([M.reshape(16, 1) for M in B_np])

if np.linalg.matrix_rank(A_np) != 16:
    raise RuntimeError("A base de matrizes não possui rank 16.")

A_inv = np.linalg.inv(A_np)

# Pré-calculo do Tensor C global (Constantes de Estrutura)
C_TENSOR = np.zeros((16, 16, 16), dtype=np.int64)

for i in range(16):
    for j in range(16):
        K = B_np[i] @ B_np[j] - B_np[j] @ B_np[i]
        coord = np.round(A_inv @ K.reshape(16, 1)).flatten().astype(np.int64)
        C_TENSOR[i, j, :] = coord

# Variável compartilhada por memória do processo
C_SHARED = None


def init_worker(c_array):
    global C_SHARED
    C_SHARED = c_array


def check_closed_fast(mask):
    """Testa fechamento da subálgebra via vetorização NumPy extremamente rápida."""
    indices = [i for i in range(N) if mask & (1 << i)]
    for i in indices:
        for j in indices:
            for k in range(N):
                if C_SHARED[i, j, k] != 0 and not (mask & (1 << k)):
                    return None
    return (mask, tuple(indices))


def matrix_rank_np(matrices):
    if not matrices:
        return 0
    stacked = np.hstack([M.reshape(16, 1) for M in matrices])
    return int(np.linalg.matrix_rank(stacked))


def derived_matrices_np(indices):
    result = []
    for i, j in combinations(indices, 2):
        K = B_np[i] @ B_np[j] - B_np[j] @ B_np[i]
        if np.any(K != 0):
            result.append(K)
    return result


def center_dimension_np(indices):
    if not indices:
        return 0
    rows = []
    for j in indices:
        cols = []
        for i in indices:
            K = B_np[i] @ B_np[j] - B_np[j] @ B_np[i]
            cols.append(K.reshape(16, 1))
        rows.append(np.hstack(cols))
    system = np.vstack(rows)
    return len(indices) - int(np.linalg.matrix_rank(system))


def lower_central_dimensions_np(indices, max_steps=20):
    current = list(indices)
    dimensions = [len(current)]
    current_mats = [B_np[i] for i in current]

    for _ in range(max_steps - 1):
        brackets = []
        for i in indices:
            for X in current_mats:
                K = B_np[i] @ X - X @ B_np[i]
                if np.any(np.abs(K) > 1e-9):
                    brackets.append(K)

        if not brackets:
            dimensions.append(0)
            break

        # Filtra base ortogonal usando Rank
        stacked = np.hstack([M.reshape(16, 1) for M in brackets])
        new_dim = int(np.linalg.matrix_rank(stacked))

        dimensions.append(new_dim)
        if new_dim == 0 or new_dim == dimensions[-2]:
            break

        # Seleciona geradores linearmente independentes
        new_mats = []
        for mat in brackets:
            test = new_mats + [mat]
            if int(np.linalg.matrix_rank(
                np.hstack([M.reshape(16, 1) for M in test])
            )) > len(new_mats):
                new_mats.append(mat)
            if len(new_mats) == new_dim:
                break
        current_mats = new_mats

    return dimensions


def derived_series_dimensions_np(indices, max_steps=20):
    """
    Computes the derived series exactly at the level of the numerical
    matrix representation:

        h^(0) = h
        h^(n+1) = [h^(n), h^(n)]

    Returns the dimensions of the successive derived algebras.
    Solvability is established iff the series reaches dimension zero.
    """
    current = [B_np[i] for i in indices]
    dimensions = [len(indices)]

    for _ in range(max_steps):
        brackets = []

        for X, Y in combinations(current, 2):
            K = X @ Y - Y @ X
            if np.any(np.abs(K) > 1e-9):
                brackets.append(K)

        if not brackets:
            dimensions.append(0)
            return dimensions

        stacked = np.hstack([M.reshape(16, 1) for M in brackets])
        rank = int(np.linalg.matrix_rank(stacked))

        dimensions.append(rank)

        # Stabilization at a nonzero dimension means the derived
        # series will not reach zero within this finite-dimensional
        # representation.
        if rank == 0:
            return dimensions

        if rank == dimensions[-2]:
            return dimensions

        # Select a linearly independent basis of the derived algebra.
        new_mats = []
        current_rank = 0

        for mat in brackets:
            test = new_mats + [mat]
            test_rank = int(np.linalg.matrix_rank(
                np.hstack([M.reshape(16, 1) for M in test])
            ))

            if test_rank > current_rank:
                new_mats.append(mat)
                current_rank = test_rank

            if current_rank == rank:
                break

        current = new_mats

    return dimensions


def is_solvable_np(indices):
    """
    A Lie algebra is solvable iff its derived series reaches zero.
    """
    dims = derived_series_dimensions_np(indices)
    return dims[-1] == 0

def analyze_subalgebra_fast(item):
    """Executa a análise estrutural da subálgebra."""
    n, (_, indices) = item
    dim = len(indices)
    zdim = center_dimension_np(indices)

    d1_mats = derived_matrices_np(indices)
    d1 = matrix_rank_np(d1_mats)

    # Derived series:
    # h^(0)=h, h^(1)=[h,h], h^(2)=[h^(1),h^(1)], ...
    derived_series = derived_series_dimensions_np(indices)
    d2 = derived_series[2] if len(derived_series) > 2 else 0

    ab = d1 == 0
    solv = derived_series[-1] == 0

    lcs = lower_central_dimensions_np(indices)
    nilp = 0 in lcs

    return {
        "id": n,
        "generators": tuple(i + 1 for i in indices),
        "dim": dim,
        "center_dim": zdim,
        "derived_dim": d1,
        "second_derived_dim": d2,
        "abelian": ab,
        "solvable": solv,
        "nilpotent": nilp,
        "lower_central": tuple(lcs),
        "derived_series": tuple(derived_series),
    }


# ==============================================================
# EXECUÇÃO PARALELA NAS 12 THREADS
# ==============================================================

if __name__ == "__main__":
    print(f"Subindo pool de execução em {NUM_WORKERS} threads ativas...")

    # 1. Varredura das 65.535 combinações em 12 processos
    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=init_worker,
        initargs=(C_TENSOR,),
    ) as pool:
        results = pool.map(check_closed_fast, range(1, 1 << N), chunksize=2048)

    subalgebras = [r for r in results if r is not None]

    print("=" * 78)
    print("ALPHA GROUP — STRUCTURAL ANALYSIS OF COORDINATE LIE SUBALGEBRAS")
    print("=" * 78)
    print(f"Total de subálgebras coordenadas: {len(subalgebras)}")

    # 2. Análise Estrutural em 12 processos
    items = list(enumerate(subalgebras, 1))

    with mp.Pool(processes=NUM_WORKERS) as pool:
        records = pool.map(analyze_subalgebra_fast, items)

    records.sort(key=lambda x: x["id"])

    # Impressão dos resultados
    print("\n" + "=" * 78)
    print("TABELA COMPLETA — 68 SUBÁLGEBRAS")
    print("=" * 78)
    print(
        f"{'ID':>3} {'dim':>3} {'Z':>3} {'D1':>3} {'D2':>3} "
        f"{'Ab':>3} {'Sol':>3} {'Nil':>3}   Geradores"
    )

    for r in records:
        print(
            f"{r['id']:3d} {r['dim']:3d} {r['center_dim']:3d} "
            f"{r['derived_dim']:3d} {r['second_derived_dim']:3d} "
            f"{'Y' if r['abelian'] else 'N':>3} "
            f"{'Y' if r['solvable'] else 'N':>3} "
            f"{'Y' if r['nilpotent'] else 'N':>3}   "
            f"{r['generators']}"
        )

    # Gravação do Relatório Arquivado
    outfile = "alpha_68_subalgebras_structural_analysis.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(
            "ALPHA GROUP — STRUCTURAL ANALYSIS OF 68 COORDINATE LIE"
            " SUBALGEBRAS\n"
        )
        f.write("=" * 78 + "\n\n")

        for r in records:
            f.write(
                f"S{r['id']:02d} | dim={r['dim']} | "
                f"Z={r['center_dim']} | D1={r['derived_dim']} | "
                f"D2={r['second_derived_dim']} | "
                f"abelian={r['abelian']} | solvable={r['solvable']} | "
                f"nilpotent={r['nilpotent']} | "
                f"generators={r['generators']} | "
                f"derived_series={r['derived_series']} | "
            f"lower_central={r['lower_central']}\n"
            )

    # ==============================================================
    # 16. CONSISTENCY CHECK — COMPLETE 16D ALGEBRA
    # ==============================================================

    full = next(
        r for r in records
        if r["generators"] == tuple(range(1, 17))
    )

    print("\n" + "=" * 78)
    print("CONSISTENCY CHECK — COMPLETE 16D LIE ALGEBRA")
    print("=" * 78)
    print("dim =", full["dim"])
    print("center_dim =", full["center_dim"])
    print("derived_series =", full["derived_series"])
    print("solvable =", full["solvable"])
    print("nilpotent =", full["nilpotent"])

    print("\nAnálise finalizada e salva em:", outfile)
