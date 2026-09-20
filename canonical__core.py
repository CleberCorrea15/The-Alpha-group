
import numpy as np

np.set_printoptions(precision=6, suppress=True)

print("=" * 78)
print("MOINHO 6.2 — NÚCLEO CANÔNICO DO ESPAÇO")
print("=" * 78)

# ============================================================
# 1. COLE AQUI EXATAMENTE AS 16 MATRIZES DO MOINHO 6.1
# ============================================================

B = [
    np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], float
    ),  # B1

    np.array(
        [[0, -1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, -1],
         [0, 0, 1, 0]], float
    ),  # B2

    np.array(
        [[0, 0, -1, 0],
         [0, 0, 0, -1],
         [1, 0, 0, 0],
         [0, 1, 0, 0]], float
    ),  # B3

    np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], float
    ),  # B4

    np.array(
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, -1, 0, 0],
         [1, 0, 0, 0]], float
    ),  # B5

    np.array(
        [[0, 0, 0, 1],
         [0, 0, -1, 0],
         [0, -1, 0, 0],
         [1, 0, 0, 0]], float
    ),  # B6

    np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, -1],
         [0, 0, 1, 0]], float
    ),  # B7

    np.array(
        [[0, -1, -1, 0],
         [1, 0, 0, 1],
         [-1, 0, 0, 0],
         [0, -1, 0, 0]], float
    ),  # B8

    np.array(
        [[0, 0, -1, 0],
         [0, 0, 0, -1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]], float
    ),  # B9

    np.array(
        [[0, 1, 0, 0],
         [-1, 0, 0, 0],
         [1, 0, 0, 1],
         [0, 1, 1, 0]], float
    ),  # B10

    np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, -1, 0, 0],
         [1, 0, 0, 0]], float
    ),  # B11

    np.array(
        [[0, 1, -1, 0],
         [1, 0, 0, -1],
         [0, 0, 0, 1],
         [0, 0, -1, 0]], float
    ),  # B12

    np.array(
        [[2, 0, 0, 1],
         [0, 0, 1, 0],
         [0, -1, -1, 0],
         [1, 0, 0, 1]], float
    ),  # B13

    np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, -1, -1, 0],
         [1, 0, 0, 1]], float
    ),  # B14

    np.array(
        [[0, 0, 0, -1],
         [0, 0, 1, 0],
         [0, 1, -1, 0],
         [1, 0, 0, -1]], float
    ),  # B15

    np.array(
        [[0, 1, 1, 0],
         [-1, 0, 0, -1],
         [2, 0, 0, 1],
         [0, 0, 1, 0]], float
    ),  # B16
]

# B6 = componente da diagonal secundária de M(theta)
# B6 realiza a involução do espaço canônico:
# Ad_B6(X) = B6 X B6
# com B6^2 = I.

if len(B) != 16:
    raise RuntimeError(
        f"Foram carregadas {len(B)} matrizes. "
        "O Moinho 6.2 exige exatamente B1...B16."
    )


# ============================================================
# 2. OPERAÇÕES
# ============================================================

def norm(A):
    return np.linalg.norm(A)


def prod(A, C):
    return A @ C


def comm(A, C):
    return A @ C - C @ A


def anticom(A, C):
    return A @ C + C @ A


def ad(A, C):
    return A @ C @ A


def residual(A, C):
    return norm(A - C)


def identify(A, tol=1e-10):

    """
    Procura se A coincide, até tolerância,
    com algum B_j ou -B_j.
    """

    found = []

    for j, Bj in enumerate(B):

        if residual(A, Bj) < tol:
            found.append(f"B{j+1}")

        if residual(A, -Bj) < tol:
            found.append(f"-B{j+1}")

    return found


# ============================================================
# 3. NÚCLEO CANÔNICO
# ============================================================

core = {
    "B1 = real": B[0],
    "B3 = i": B[2],
    "B4 = mu": B[3],
    "B6": B[5],
    "B9 = i.mu": B[8]
}

print()
print("NÚCLEO CANÔNICO:")
for name in core:
    print(" ", name)


# ============================================================
# 4. TESTE INDIVIDUAL
# ============================================================

print()
print("=" * 78)
print("4. QUADRADOS")
print("=" * 78)

for name, A in core.items():

    R = A @ A

    print()
    print(name)
    print("  ||A²|| =", norm(R))
    print("  identificação:", identify(R))


# ============================================================
# 5. PRODUTOS ENTRE OS ELEMENTOS DO NÚCLEO
# ============================================================

print()
print("=" * 78)
print("5. PRODUTOS DO NÚCLEO")
print("=" * 78)

items = list(core.items())

for i in range(len(items)):

    nameA, A = items[i]

    for j in range(i + 1, len(items)):

        nameC, C = items[j]

        AC = A @ C
        CA = C @ A

        print()
        print(f"{nameA}  ×  {nameC}")

        print("  A C ->", identify(AC))
        print("  C A ->", identify(CA))

        print(
            "  ||AC - CA|| =",
            norm(AC - CA)
        )


# ============================================================
# 6. COMUTADORES
# ============================================================

print()
print("=" * 78)
print("6. COMUTADORES DO NÚCLEO")
print("=" * 78)

for i in range(len(items)):

    nameA, A = items[i]

    for j in range(i + 1, len(items)):

        nameC, C = items[j]

        K = comm(A, C)

        print()
        print(
            f"[{nameA}, {nameC}]"
        )

        print(
            "  norma =",
            norm(K)
        )

        print(
            "  identificação =",
            identify(K)
        )


# ============================================================
# 7. ANTICOMUTADORES
# ============================================================

print()
print("=" * 78)
print("7. ANTICOMUTADORES")
print("=" * 78)

for i in range(len(items)):

    nameA, A = items[i]

    for j in range(i + 1, len(items)):

        nameC, C = items[j]

        K = anticom(A, C)

        print()
        print(
            f"{{{nameA}, {nameC}}}"
        )

        print(
            "  norma =",
            norm(K)
        )

        print(
            "  identificação =",
            identify(K)
        )


# ============================================================
# 8. TESTE ESPECIAL: B3 B4
# ============================================================

print()
print("=" * 78)
print("8. TESTE ESPECIAL i × mu")
print("=" * 78)

B3 = B[2]
B4 = B[3]
B6 = B[5]
B9 = B[8]

P34 = B3 @ B4
P43 = B4 @ B3

print()
print("B3 B4:")
print(P34)

print()
print("Identificação:", identify(P34))

print()
print("B4 B3:")
print(P43)

print()
print("Identificação:", identify(P43))

print()
print(
    "||B3B4 + B4B3|| =",
    norm(P34 + P43)
)

print(
    "||B3B4 - B4B3|| =",
    norm(P34 - P43)
)


# ============================================================
# 9. TESTE B9 = i.mu
# ============================================================

print()
print("=" * 78)
print("9. COMPARAÇÃO B3B4 COM B9")
print("=" * 78)

print(
    "||B3 B4 - B9|| =",
    norm(B3 @ B4 - B9)
)

print(
    "||B4 B3 + B9|| =",
    norm(B4 @ B3 + B9)
)

print(
    "||B9² - B4|| =",
    norm(B9 @ B9 - B4)
)

print("||B3 B4 - B9|| =", np.linalg.norm(B3 @ B4 - B9))
print("||B4 B3 + B9|| =", np.linalg.norm(B4 @ B3 + B9))
# ============================================================
# 10. INVOLUÇÃO Ad_B6
# ============================================================

print()
print("=" * 78)
print("10. AÇÃO DA INVOLUÇÃO Ad_B6")
print("=" * 78)

print(
    "||B6² - I|| =",
    norm(B6 @ B6 - np.eye(4))
)

for name, A in core.items():

    T = B6 @ A @ B6

    print()
    print(name)

    print(
        "  Ad_B6(A):",
        identify(T)
    )

    print(
        "  ||Ad(A)-A|| =",
        norm(T - A)
    )

    print(
        "  ||Ad(A)+A|| =",
        norm(T + A)
    )


# ============================================================
# 11. AÇÃO DE B6 NOS 16 OPERADORES
# ============================================================

print()
print("=" * 78)
print("11. Ad_B6 NOS 16 OPERADORES CANÔNICOS")
print("=" * 78)

for j, Bj in enumerate(B):

    T = B6 @ Bj @ B6

    plus = norm(T - Bj)
    minus = norm(T + Bj)

    if plus < 1e-10:
        status = "+"

    elif minus < 1e-10:
        status = "-"

    else:
        status = "MISTO"

    print(
        f"B{j+1:2d} : {status}"
    )


# ============================================================
# 12. MATRIZ DE MULTIPLICAÇÃO DO NÚCLEO
# ============================================================

print()
print("=" * 78)
print("12. PRODUTOS DO NÚCLEO — RESUMO")
print("=" * 78)

for nameA, A in items:

    for nameC, C in items:

        R = A @ C

        ids = identify(R)

        if ids:

            print(
                f"{nameA:12s} × "
                f"{nameC:12s} -> "
                f"{', '.join(ids)}"
            )


# ============================================================
# 13. TESTE DE FECHAMENTO DO NÚCLEO
# ============================================================

print()
print("=" * 78)
print("13. FECHAMENTO CANÔNICO")
print("=" * 78)

core_names = list(core.keys())
core_matrices = list(core.values())

closed = True
nonclosed = []

for i, A in enumerate(core_matrices):

    for j, C in enumerate(core_matrices):

        R = A @ C

        if not identify(R):

            closed = False

            nonclosed.append(
                (
                    core_names[i],
                    core_names[j],
                    norm(R)
                )
            )

if closed:

    print(
        "O núcleo é fechado, "
        "até sinal e tolerância, "
        "nos operadores canônicos."
    )

else:

    print(
        "O núcleo NÃO é fechado apenas "
        "pelos cinco operadores canônicos."
    )

    print()
    print("Produtos não identificados:")

    for item in nonclosed:

        print(
            f"  {item[0]} × {item[1]} "
            f"(norma={item[2]:.6g})"
        )


# ============================================================
# 14. CONCLUSÃO COMPUTACIONAL
# ============================================================

print()
print("=" * 78)
print("FIM DO MOINHO 6.2")
print("=" * 78)

# ============================================================
# 5A. FECHAMENTO DOS 16 OPERADORES CANÔNICOS
# ============================================================

print()
print("=" * 78)
print("5A. FECHAMENTO DOS 16 OPERADORES CANÔNICOS")
print("=" * 78)

# ------------------------------------------------------------
# 5A.1 — Teste direto:
# B_i B_j = +/- B_k ?
# ------------------------------------------------------------

exact_closed = 0
exact_total = 0
exact_failures = []

exact_relations = []

for i in range(16):

    for j in range(16):

        P = B[i] @ B[j]

        exact_total += 1

        ids = identify(P)

        if ids:

            exact_closed += 1

            exact_relations.append(
                (i + 1, j + 1, ids)
            )

        else:

            exact_failures.append(
                (i + 1, j + 1, norm(P))
            )


print()
print("FECHAMENTO DIRETO")
print(
    f"Produtos testados : {exact_total}"
)

print(
    f"Produtos = +/- Bk : {exact_closed}"
)

print(
    f"Produtos não identificados : "
    f"{len(exact_failures)}"
)

print(
    f"Percentual de fechamento direto : "
    f"{100*exact_closed/exact_total:.2f}%"
)


# ------------------------------------------------------------
# 5A.2 — Mostrar relações canônicas encontradas
# ------------------------------------------------------------

print()
print("RELAÇÕES B_i B_j = +/- B_k")
print("-" * 78)

for i, j, ids in exact_relations:

    print(
        f"B{i} × B{j}  ->  "
        f"{', '.join(ids)}"
    )


# ------------------------------------------------------------
# 5A.3 — Base linear dos 16 operadores
# ------------------------------------------------------------

# Cada matriz 4x4 é transformada em vetor de dimensão 16.

V = np.column_stack([
    Bj.reshape(16)
    for Bj in B
])

rank_B = np.linalg.matrix_rank(V, tol=1e-10)

print()
print("=" * 78)
print("5B. FECHAMENTO LINEAR")
print("=" * 78)

print(
    "Rank dos 16 operadores =",
    rank_B
)


# ------------------------------------------------------------
# 5B.1 — Coordenadas lineares dos produtos
# ------------------------------------------------------------

linear_closed = 0
linear_failures = []

product_coordinates = []

for i in range(16):

    for j in range(16):

        P = B[i] @ B[j]

        p = P.reshape(16)

        coeff, residuals, _, _ = np.linalg.lstsq(
            V,
            p,
            rcond=None
        )

        reconstruction = V @ coeff

        err = np.linalg.norm(
            p - reconstruction
        )

        if err < 1e-10:

            linear_closed += 1

            product_coordinates.append(
                (
                    i + 1,
                    j + 1,
                    coeff,
                    err
                )
            )

        else:

            linear_failures.append(
                (
                    i + 1,
                    j + 1,
                    err
                )
            )


print()
print(
    f"Produtos pertencentes ao span dos 16: "
    f"{linear_closed}/{exact_total}"
)

print(
    f"Produtos fora do span: "
    f"{len(linear_failures)}"
)

print(
    f"Percentual de fechamento linear: "
    f"{100*linear_closed/exact_total:.2f}%"
)


# ------------------------------------------------------------
# 5B.2 — Diagnóstico final
# ------------------------------------------------------------

print()
print("=" * 78)
print("DIAGNÓSTICO DE FECHAMENTO")
print("=" * 78)

if exact_closed == exact_total:

    print(
        "RESULTADO: os 16 B's fecham diretamente "
        "sob multiplicação."
    )

else:

    print(
        "RESULTADO: os 16 B's NÃO fecham diretamente "
        "como conjunto discreto."
    )


if linear_closed == exact_total:

    print(
        "RESULTADO LINEAR: o span dos 16 B's "
        "é fechado sob multiplicação."
    )

else:

    print(
        "RESULTADO LINEAR: o span dos 16 B's "
        "NÃO é fechado sob multiplicação."
    )


# ------------------------------------------------------------
# 5B.3 — Tabela de produtos que não são +/- Bk
# ------------------------------------------------------------

print()
print("PRODUTOS NÃO IDENTIFICADOS COMO +/- Bk")
print("-" * 78)

for i, j, err in exact_failures[:50]:

    print(
        f"B{i} × B{j}   "
        f"||produto|| = {err:.6e}"
    )

if len(exact_failures) > 50:

    print(
        f"... mais {len(exact_failures)-50} produtos."
    )


