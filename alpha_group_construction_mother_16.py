# =============================================================================
# GRUPO ALPHA — CONSTRUÇÃO-MÃE DOS 16 OPERADORES
# =============================================================================
#
# Fonte estrutural:
#   M4(theta) + levantamento 4x4 -> 16x16
#   cinco geradores iniciais
#   fechamento multiplicativo por posto
#   análise posterior pelo colchete de Lie
#
# A matriz original M4(theta) é mantida explicitamente.
#
# =============================================================================

import numpy as np

np.set_printoptions(precision=10, suppress=True)

TOL_RANK = 1e-10
TOL_LIE  = 1e-10


# =============================================================================
# 1. GERADORES ALPHA
# =============================================================================

I4 = np.eye(4)

G_C = np.array([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 0,-1],
    [0,  0, 1, 0]
], dtype=float)

G_T = np.array([
    [0, 0,-1, 0],
    [0, 0, 0,-1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=float)

G_mu = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)

G_const = np.array([
    [1,  0, 0, 1],
    [0,  1, 1, 0],
    [0, -1, 0, 0],
    [1,  0, 0, 0]
], dtype=float)


# =============================================================================
# 2. MATRIZ ORIGINAL M4(theta)
# =============================================================================

def M4_theta(theta):
    """
    Matriz angular original do Grupo Alpha.

        [ 1       -cot      -tan       1 ]
        [ cot       i        -1       -tan]
        [ tan      -1         1       -cot]
        [ 1        tan       cot        i ]
    """

    tan = np.tan(theta)

    if abs(tan) < 1e-14:
        raise ValueError(
            "tan(theta) demasiado próximo de zero."
        )

    cot = 1.0 / tan

    return np.array([
        [1,    -cot, -tan,  1],
        [cot,   1j,  -1,   -tan],
        [tan,  -1,    1,   -cot],
        [1,     tan,  cot,  1j]
    ], dtype=complex)


# =============================================================================
# 3. ESTRUTURA INTERNA
# =============================================================================

GC = G_C
GT = G_T
GMU = G_mu
GCMU = GC @ GMU

INTERNAL = [
    [I4, I4, I4, GCMU],
    [I4, GC, I4, GCMU],
    [I4, I4, GMU, I4],
    [I4, I4, I4, GCMU]
]


# =============================================================================
# 4. LEVANTAMENTO 4x4 -> 16x16
# =============================================================================

def lift_to_16(M4):
    rows = []

    for i in range(4):
        blocks = []

        for j in range(4):
            blocks.append(
                M4[i, j] * INTERNAL[i][j]
            )

        rows.append(
            np.hstack(blocks)
        )

    return np.vstack(rows)


def M_theta_16(theta):
    return lift_to_16(
        M4_theta(theta)
    )


# =============================================================================
# 5. POSTO DE UMA FAMÍLIA DE MATRIZES
# =============================================================================

def rank_of(matrices):

    if not matrices:
        return 0

    A = np.column_stack([
        M.reshape(-1)
        for M in matrices
    ])

    return np.linalg.matrix_rank(
        A,
        tol=TOL_RANK
    )


# =============================================================================
# 6. CONSTRUÇÃO DOS 16 OPERADORES
# =============================================================================
#
# A construção é associativa:
#
#       produto matricial
#              |
#              v
#       aumento de posto
#              |
#              v
#       nova direção
#
# O colchete de Lie NÃO é usado para criar a base.
# Ele será usado somente depois, na seção 8.
# =============================================================================

def build_basis(generators):

    basis = [
        G.copy()
        for G in generators
    ]

    A = np.column_stack([
        M.reshape(16)
        for M in basis
    ])

    rank = np.linalg.matrix_rank(
        A,
        tol=TOL_RANK
    )

    genealogy = []

    while rank < 16:

        curr_len = len(basis)
        added = False

        for i in range(curr_len):

            for j in range(curr_len):

                prod = basis[i] @ basis[j]
                v = prod.reshape(16)

                test_A = np.column_stack([
                    A,
                    v
                ])

                new_rank = np.linalg.matrix_rank(
                    test_A,
                    tol=TOL_RANK
                )

                if new_rank > rank:

                    basis.append(prod.copy())
                    A = test_A
                    rank = new_rank
                    added = True

                    genealogy.append({
                        "new_index": len(basis),
                        "left": i + 1,
                        "right": j + 1,
                        "rank": rank
                    })

                    print(
                        f"B{len(basis):02d} <- "
                        f"B{i+1} B{j+1}    "
                        f"rank = {rank}"
                    )

                    if rank == 16:
                        break

            if rank == 16:
                break

        if not added:
            break

    # Redução final para uma base linearmente independente
    Base = []
    rank_final = 0

    for M in basis:

        candidate = Base + [M]
        new_rank = rank_of(candidate)

        if new_rank > rank_final:
            Base.append(M.copy())
            rank_final = new_rank

        if rank_final == 16:
            break

    return Base, genealogy


# =============================================================================
# 7. CONSTRUÇÃO
# =============================================================================

GENERATORS = [
    I4,
    G_C,
    G_T,
    G_mu,
    G_const
]

print("=" * 80)
print("GRUPO ALPHA — CONSTRUÇÃO DOS 16 OPERADORES")
print("=" * 80)

Base, genealogy = build_basis(GENERATORS)

if len(Base) != 16:
    raise RuntimeError(
        f"A base não atingiu dimensão 16: {len(Base)}"
    )

B = {
    i + 1: Base[i]
    for i in range(16)
}

print("\n✓ Dimensão 16 confirmada.")


# =============================================================================
# 8. MATRIZ ORIGINAL EM UM PONTO DE TESTE
# =============================================================================

theta_test = np.pi / 4

print("\n" + "=" * 80)
print("MATRIZ ORIGINAL M4(theta)")
print("=" * 80)

print(f"\ntheta = pi/4")

M4 = M4_theta(theta_test)

print("\nM4(theta) =")
print(M4)

print("\nM16(theta) =")
print(M_theta_16(theta_test))


# =============================================================================
# 9. GENEALOGIA
# =============================================================================

print("\n" + "=" * 80)
print("GENEALOGIA DA CONSTRUÇÃO")
print("=" * 80)

for g in genealogy:
    print(
        f"B{g['new_index']:02d} <- "
        f"B{g['left']} B{g['right']}    "
        f"rank = {g['rank']}"
    )


# =============================================================================
# 10. OS 16 OPERADORES
# =============================================================================

print("\n" + "=" * 80)
print("B1 ... B16")
print("=" * 80)

for i in range(1, 17):
    print(f"\nB{i} =")
    print(B[i])


# =============================================================================
# 11. RELAÇÕES DE GENEALOGIA IMPORTANTES
# =============================================================================

print("\n" + "=" * 80)
print("RELAÇÕES ESPECÍFICAS")
print("=" * 80)

relations = {
    "B6 = B2 B3":  B[6]  - B[2] @ B[3],
    "B7 = B2 B4":  B[7]  - B[2] @ B[4],
    "B8 = B2 B5":  B[8]  - B[2] @ B[5],
    "B9 = B3 B4":  B[9]  - B[3] @ B[4],
    "B10 = B3 B5": B[10] - B[3] @ B[5],
    "B11 = B4 B5": B[11] - B[4] @ B[5],
    "B12 = B5 B3": B[12] - B[5] @ B[3],
    "B13 = B5 B5": B[13] - B[5] @ B[5],
    "B14 = B2 B10": B[14] - B[2] @ B[10],
    "B15 = B3 B12": B[15] - B[3] @ B[12],
    "B16 = B3 B13": B[16] - B[3] @ B[13],
}

for name, error_matrix in relations.items():
    print(
        f"{name:20s} "
        f"erro = {np.linalg.norm(error_matrix):.3e}"
    )


# =============================================================================
# 12. COLCHETE DE LIE
# =============================================================================

def commutator(X, Y):
    return X @ Y - Y @ X


print("\n" + "=" * 80)
print("ESTRUTURA DE LIE")
print("=" * 80)

commuting = 0
non_commuting = 0

for i in range(1, 17):
    for j in range(i + 1, 17):

        C = commutator(B[i], B[j])

        if np.linalg.norm(C) < TOL_LIE:
            commuting += 1
        else:
            non_commuting += 1

print("\nPares independentes :", 120)
print("Comutativos         :", commuting)
print("Não comutativos     :", non_commuting)


# =============================================================================
# 13. IDENTIDADE DE JACOBI
# =============================================================================

max_jacobi_error = 0.0

for i in range(1, 17):
    for j in range(1, 17):
        for k in range(1, 17):

            X = B[i]
            Y = B[j]
            Z = B[k]

            J = (
                commutator(X, commutator(Y, Z))
                + commutator(Y, commutator(Z, X))
                + commutator(Z, commutator(X, Y))
            )

            error = np.linalg.norm(J)

            max_jacobi_error = max(
                max_jacobi_error,
                error
            )

print(
    "\nErro máximo de Jacobi = "
    f"{max_jacobi_error:.3e}"
)

if max_jacobi_error < 1e-10:
    print("✓ Identidade de Jacobi confirmada.")
else:
    print("⚠ Jacobi requer investigação.")


# =============================================================================
# 14. RESUMO FINAL
# =============================================================================

print("\n" + "=" * 80)
print("RESUMO")
print("=" * 80)

print(f"""
Matriz original M4(theta)       : incluída
Levantamento para 16x16         : incluído
Geradores iniciais              : 5
Dimensão da base                : {len(B)}
Pares independentes             : 120
Pares comutativos               : {commuting}
Pares não comutativos           : {non_commuting}
Erro máximo de Jacobi           : {max_jacobi_error:.3e}

Construção:
    M4(theta)
         |
         v
    estrutura Alpha
         |
         v
    5 geradores
         |
         v
    fechamento multiplicativo
         |
         v
    B1 ... B16
         |
         v
    colchete de Lie
         |
         v
    testes estruturais
""")

print("=" * 80)
print("FIM")
print("=" * 80)
