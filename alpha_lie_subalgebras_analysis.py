import numpy as np
import scipy.linalg
from itertools import combinations
import sympy as sp
import pandas as pd

# ============================================================
# ANÁLISE DA ÁLGEBRA DE LIE DO GRUPO ALPHA
# ============================================================
# Este script:
# 1. Enumera subálgebras coordenadas nas dimensões selecionadas;
# 2. Classifica-as como abelianas/não-abelianas,
#    resolúveis/não-resolúveis e nilpotentes;
# 3. Apresenta a classificação geométrica dos 16 geradores.
#
# A primeira parte utiliza NumPy/SciPy para a enumeração numérica.
# A segunda parte fornece a definição simbólica dos geradores e
# integra as classes geométricas com estatísticas de interações.
# ============================================================

# ============================================================
# 1. GERADORES EM MATRIZES NUMÉRICAS
# ============================================================

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

gens = [m.astype(float) for m in B_np]


def is_subalgebra(indices):
    """Verifica se o subespaço é fechado sob o comutador."""
    sub_gens = [gens[i] for i in indices]
    M_sub = np.hstack([g.reshape(16, 1) for g in sub_gens])

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            comm = sub_gens[i] @ sub_gens[j] - sub_gens[j] @ sub_gens[i]
            c, _, _, _ = np.linalg.lstsq(
                M_sub, comm.reshape(16, 1), rcond=None
            )

            if not np.allclose(
                M_sub @ c, comm.reshape(16, 1), atol=1e-8
            ):
                return False

    return True


def analyze_subalgebra(indices):
    """Classifica a subálgebra em abeliana, nilpotente e resolúvel."""
    k = len(indices)
    sub_gens = [gens[i] for i in indices]
    M_sub = np.hstack([g.reshape(16, 1) for g in sub_gens])

    # Representação adjunta
    ad_sub = [np.zeros((k, k)) for _ in range(k)]

    for i in range(k):
        for j in range(k):
            comm = sub_gens[i] @ sub_gens[j] - sub_gens[j] @ sub_gens[i]

            c, _, _, _ = np.linalg.lstsq(
                M_sub, comm.reshape(16, 1), rcond=None
            )

            ad_sub[i][:, j] = c.flatten()

    # Abeliana
    is_abelian = all(
        np.allclose(ad_sub[i], 0, atol=1e-8)
        for i in range(k)
    )

    # Nilpotência via representação adjunta
    is_nilpotent = True

    for i in range(k):
        eigvals = np.linalg.eigvals(ad_sub[i])

        if not np.allclose(eigvals, 0, atol=1e-8):
            is_nilpotent = False
            break

    # Série derivada
    curr_basis = M_sub.copy()
    is_solvable = False

    for step in range(k + 2):
        if curr_basis.shape[1] == 0:
            is_solvable = True
            break

        dim_curr = curr_basis.shape[1]
        comm_vecs = []

        for i in range(dim_curr):
            vi = curr_basis[:, i].reshape(4, 4)

            for j in range(i + 1, dim_curr):
                vj = curr_basis[:, j].reshape(4, 4)
                comm = vi @ vj - vj @ vi
                comm_vecs.append(comm.flatten())

        if len(comm_vecs) == 0:
            is_solvable = True
            break

        comm_mat = np.column_stack(comm_vecs)

        q, r, p = scipy.linalg.qr(
            comm_mat, pivoting=True
        )

        rank = np.sum(
            np.abs(np.diag(r)) > 1e-8
        )

        if rank == 0:
            is_solvable = True
            break

        if rank == curr_basis.shape[1]:
            is_solvable = False
            break

        curr_basis = q[:, :rank]

    return {
        "abelian": is_abelian,
        "nilpotent": is_nilpotent,
        "solvable": is_solvable,
    }


# ============================================================
# 2. ENUMERAÇÃO DAS SUBÁLGEBRAS
# ============================================================

target_dims = [1, 2, 3, 4, 5, 8, 16]
summary = {}

for k in target_dims:

    stats = {
        "total": 0,
        "abelian": 0,
        "non_abelian": 0,
        "solvable": 0,
        "non_solvable": 0,
        "nilpotent": 0,
    }

    for cb in combinations(range(16), k):

        if is_subalgebra(cb):

            res = analyze_subalgebra(cb)

            stats["total"] += 1

            if res["abelian"]:
                stats["abelian"] += 1
            else:
                stats["non_abelian"] += 1

            if res["solvable"]:
                stats["solvable"] += 1
            else:
                stats["non_solvable"] += 1

            if res["nilpotent"]:
                stats["nilpotent"] += 1

    summary[k] = stats


print(
    f"{'Dimensão':<10} {'Total':<7} "
    f"{'Abeliana':<10} {'Não-Abeliana':<14} "
    f"{'Resolúvel':<11} {'Não-Resolúvel':<15} "
    f"{'Nilpotente':<10}"
)
print("-" * 80)

tot_total = tot_abel = tot_nabel = 0
tot_solv = tot_nsolv = tot_nilp = 0

for k, st in summary.items():

    print(
        f"{k:<10} {st['total']:<7} "
        f"{st['abelian']:<10} "
        f"{st['non_abelian']:<14} "
        f"{st['solvable']:<11} "
        f"{st['non_solvable']:<15} "
        f"{st['nilpotent']:<10}"
    )

    tot_total += st["total"]
    tot_abel += st["abelian"]
    tot_nabel += st["non_abelian"]
    tot_solv += st["solvable"]
    tot_nsolv += st["non_solvable"]
    tot_nilp += st["nilpotent"]

print("-" * 80)

print(
    f"{'Total':<10} {tot_total:<7} "
    f"{tot_abel:<10} {tot_nabel:<14} "
    f"{tot_solv:<11} {tot_nsolv:<15} "
    f"{tot_nilp:<10}"
)


# ============================================================
# 3. CAMADA SIMBÓLICA: CLASSIFICAÇÃO DOS GERADORES
# ============================================================

class GeneratorSymbolic:

    def __init__(self, name, geo_class, dynamic):
        self.name = name
        self.geo_class = geo_class
        self.dynamic = dynamic

        self.symbolic_matrix = sp.MatrixSymbol(
            name, 4, 4
        )


generators_base = []

# Central
generators_base.append(
    GeneratorSymbolic(
        "E0", "Central", "Simetria"
    )
)

# Compactos
for i in range(1, 4):
    generators_base.append(
        GeneratorSymbolic(
            f"C{i}", "Compacta", "Rotação"
        )
    )

# Não-compactos
for i in range(1, 3):
    generators_base.append(
        GeneratorSymbolic(
            f"NC{i}", "Não-Compacta", "Expansão"
        )
    )

# Nilpotentes
for i in range(1, 3):
    generators_base.append(
        GeneratorSymbolic(
            f"N{i}", "Nilpotente", "Propagação"
        )
    )

# Projetivos
for i in range(1, 9):
    generators_base.append(
        GeneratorSymbolic(
            f"P{i}", "Projetiva", "Projeção"
        )
    )


# ============================================================
# 4. INTERAÇÕES DE LIE POR CLASSE
# ============================================================

class_interactions = {
    "Central": 1,
    "Compacta": 24,
    "Não-Compacta": 16,
    "Nilpotente": 14,
    "Projetiva": 65,
}


df_gen = pd.DataFrame(
    [
        {
            "Gerador": g.name,
            "Classe": g.geo_class,
            "Dinâmica": g.dynamic,
        }
        for g in generators_base
    ]
)

summary_classes = (
    df_gen
    .groupby(["Classe", "Dinâmica"])
    .size()
    .reset_index(name="Qtd Geradores")
)

summary_classes["Interações de Lie"] = (
    summary_classes["Classe"].map(class_interactions)
)

summary_classes["% Interações Total"] = (
    summary_classes["Interações de Lie"] / 120 * 100
).round(2)


print("\n=== ESTRUTURA DOS GERADORES DA ÁLGEBRA DE LIE ===")
print(summary_classes.to_string(index=False))
