import numpy as np
import sympy as sp

# =============================================================================
# ALPHA GROUP — TESTE ALGÉBRICO DOS 16 GERADORES
#
# Objetivo:
#
#   1. Confirmar dim(span(B1,...,B16)) = 16
#   2. Testar fechamento de grau 2:
#
#          Bi Bj ∈ span(B1,...,B16)
#
#      para todos os 256 produtos.
#
#   3. Construir os coeficientes estruturais:
#
#          Bi Bj = Σ_k c[i,j,k] Bk
#
#   4. Procurar automaticamente estruturas:
#
#          X² = -I
#          P² = P
#          XP = -PX
#
#   5. Testar a involução por transposição:
#
#          T² = I
#
# =============================================================================
# B2 and B3 realize the i-type relation X² = -I.
# B4 realizes the idempotent mu-type relation X² = X.
# These are matrix representatives/executors of the abstract
# Alpha-algebraic elements, not literal identifications.


D = 4
N = 16


# =============================================================================
# 1. OS 16 GERADORES
# =============================================================================

B = [

sp.Matrix([
[1,0,0,0],
[0,1,0,0],
[0,0,1,0],
[0,0,0,1]
]),

sp.Matrix([
[0,-1,0,0],
[1,0,0,0],
[0,0,0,-1],
[0,0,1,0]
]),

sp.Matrix([
[0,0,-1,0],
[0,0,0,-1],
[1,0,0,0],
[0,1,0,0]
]),

sp.Matrix([
[0,0,0,0],
[0,0,0,0],
[0,0,1,0],
[0,0,0,1]
]),

sp.Matrix([
[1,0,0,1],
[0,1,1,0],
[0,-1,0,0],
[1,0,0,0]
]),

sp.Matrix([
[0,0,0,1],
[0,0,-1,0],
[0,-1,0,0],
[1,0,0,0]
]),

sp.Matrix([
[0,0,0,0],
[0,0,0,0],
[0,0,0,-1],
[0,0,1,0]
]),

sp.Matrix([
[0,-1,-1,0],
[1,0,0,1],
[-1,0,0,0],
[0,-1,0,0]
]),

sp.Matrix([
[0,0,-1,0],
[0,0,0,-1],
[0,0,0,0],
[0,0,0,0]
]),

sp.Matrix([
[0,1,0,0],
[-1,0,0,0],
[1,0,0,1],
[0,1,1,0]
]),

sp.Matrix([
[0,0,0,0],
[0,0,0,0],
[0,-1,0,0],
[1,0,0,0]
]),

sp.Matrix([
[0,1,-1,0],
[1,0,0,-1],
[0,0,0,1],
[0,0,-1,0]
]),

sp.Matrix([
[2,0,0,1],
[0,0,1,0],
[0,-1,-1,0],
[1,0,0,1]
]),

sp.Matrix([
[1,0,0,0],
[0,1,0,0],
[0,-1,-1,0],
[1,0,0,1]
]),

sp.Matrix([
[0,0,0,-1],
[0,0,1,0],
[0,1,-1,0],
[1,0,0,-1]
]),

sp.Matrix([
[0,1,1,0],
[-1,0,0,-1],
[2,0,0,1],
[0,0,1,0]
])

]


# =============================================================================
# 2. FUNÇÕES AUXILIARES
# =============================================================================

def vec(M):

    return sp.Matrix([
        M[i,j]
        for i in range(D)
        for j in range(D)
    ])


def commutator(A, B):

    return A*B - B*A


def anticommutator(A, B):

    return A*B + B*A


# =============================================================================
# 3. BASE DOS 16 GERADORES
# =============================================================================

print("\n" + "="*78)
print("ALPHA GROUP — ESTRUTURA ALGÉBRICA")
print("="*78)

MB = sp.Matrix.hstack(
    *[vec(M) for M in B]
)

rank_B = MB.rank()

print("\nDimensão span(B1,...,B16) =", rank_B)

if rank_B == 16:

    print(">>> Os 16 geradores são linearmente independentes.")

else:

    print(">>> ATENÇÃO: a base não possui dimensão 16.")


# =============================================================================
# 4. FECHAMENTO DE GRAU 2
# =============================================================================

print("\n" + "="*78)
print("TESTE 1 — FECHAMENTO DE GRAU 2")
print("="*78)

products = []
product_labels = []

for i in range(N):

    for j in range(N):

        P = B[i] * B[j]

        products.append(P)
        product_labels.append((i+1,j+1))


# -------------------------------------------------------------------------
# Posto do espaço gerado pelos produtos
# -------------------------------------------------------------------------

Mprod = sp.Matrix.hstack(
    *[vec(P) for P in products]
)

rank_prod = Mprod.rank()

print(
    "\nDimensão span{Bi Bj} =",
    rank_prod
)


# -------------------------------------------------------------------------
# Verificar fechamento produto por produto
# -------------------------------------------------------------------------

closure_failures = []

structure = np.zeros(
    (N,N,N),
    dtype=object
)

for n,P in enumerate(products):

    i,j = product_labels[n]

    coeff = MB.gauss_jordan_solve(
        vec(P)
    )[0]

    residual = sp.simplify(
        MB*coeff - vec(P)
    )

    if residual != sp.zeros(16,1):

        closure_failures.append(
            (i,j)
        )

    else:

        for k in range(N):

            structure[i-1,j-1,k] = coeff[k]


print(
    "\nNúmero de produtos fora do span:",
    len(closure_failures)
)


if len(closure_failures) == 0:

    print(
        "\n>>> FECHAMENTO DE GRAU 2 CONFIRMADO."
    )

else:

    print(
        "\n>>> O fechamento de grau 2 FALHOU."
    )

    print("\nPrimeiras falhas:")

    for i,j in closure_failures[:20]:

        print(
            f"  B{i} B{j}"
        )


# =============================================================================
# 5. PRODUTOS FUNDAMENTAIS
# =============================================================================

print("\n" + "="*78)
print("TESTE 2 — PRODUTOS FUNDAMENTAIS")
print("="*78)


def show_product(i,j):

    coeff = MB.gauss_jordan_solve(
        vec(B[i-1]*B[j-1])
    )[0]

    terms = []

    for k,c in enumerate(coeff):

        if c != 0:

            if c == 1:

                terms.append(f"B{k+1}")

            elif c == -1:

                terms.append(f"-B{k+1}")

            else:

                terms.append(
                    f"({c}) B{k+1}"
                )

    expression = " + ".join(terms)
    expression = expression.replace(
        "+ -",
        "- "
    )

    print(
        f"B{i} B{j} = {expression}"
    )


# Relações mais importantes
for i,j in [
    (1,1),
    (2,2),
    (2,3),
    (3,2),
    (3,3),
    (3,4),
    (4,3),
    (4,4)
]:

    show_product(i,j)


# =============================================================================
# 6. QUADRADOS DOS 16 GERADORES
# =============================================================================

print("\n" + "="*78)
print("TESTE 3 — QUADRADOS DOS 16 GERADORES")
print("="*78)

I4 = sp.eye(4)

square_minus_identity = []
square_idempotent = []

for k in range(N):

    X = B[k]

    X2 = X*X

    if X2 == -I4:

        square_minus_identity.append(k+1)

    if X2 == X:

        square_idempotent.append(k+1)


print(
    "\nGeradores com B_i² = -I:"
)

print(
    square_minus_identity
)


print(
    "\nGeradores com B_i² = B_i:"
)

print(
    square_idempotent
)


# =============================================================================
# 7. BUSCA DE COMBINAÇÕES X = a Bi + b Bj
# =============================================================================
#
# Procuramos combinações de pares que possam produzir:
#
#       X² = -I
#
#       X² = X
#
# sem assumir previamente qual gerador representa i ou mu.
# =============================================================================

print("\n" + "="*78)
print("TESTE 4 — COMBINAÇÕES LINEARES DE DOIS GERADORES")
print("="*78)

a,b = sp.symbols('a b')


def solve_quadratic_pair(X,Y,target):

    Z = a*X + b*Y

    E = sp.expand(Z*Z - target)

    equations = [
        E[i,j]
        for i in range(D)
        for j in range(D)
        if E[i,j] != 0
    ]

    if not equations:

        return []


    return sp.solve(
        equations,
        [a,b],
        dict=True
    )


solutions_complex = []
solutions_idempotent = []


for i in range(N):

    for j in range(i+1,N):

        sol1 = solve_quadratic_pair(
            B[i],
            B[j],
            -I4
        )

        if sol1:

            solutions_complex.append(
                (i+1,j+1,sol1)
            )


        sol2 = solve_quadratic_pair(
            B[i],
            B[j],
            a*B[i] + b*B[j]
        )

        if sol2:

            solutions_idempotent.append(
                (i+1,j+1,sol2)
            )


print(
    "\nPares capazes de gerar candidatos X² = -I:",
    len(solutions_complex)
)

for item in solutions_complex[:20]:

    print(item)


print(
    "\nPares capazes de gerar candidatos idempotentes:",
    len(solutions_idempotent)
)

for item in solutions_idempotent[:20]:

    print(item)


# =============================================================================
# 8. TESTE DE ANTICOMUTAÇÃO
# =============================================================================
#
# Para cada candidato X²=-I, procuramos P tal que:
#
#       XP + PX = 0
#
# e depois verificamos P²=P.
# =============================================================================

print("\n" + "="*78)
print("TESTE 5 — RELAÇÃO CLIFFORD/ALPHA")
print("="*78)

clifford_candidates = []


# Procurar diretamente entre os 16 geradores
for i in range(N):

    X = B[i]

    if X*X == -I4:

        for j in range(N):

            P = B[j]

            if X*P + P*X == sp.zeros(4):

                clifford_candidates.append(
                    (i+1,j+1,P*P == P)
                )


print(
    "\nPares (X,P) com:"
    "\n    X² = -I"
    "\n    XP + PX = 0"
    "\n"
)

for item in clifford_candidates:

    print(
        f"X = B{item[0]}, "
        f"P = B{item[1]}, "
        f"P²=P? {item[2]}"
    )


# =============================================================================
# 9. INVOLUÇÃO POR TRANSPOSTA
# =============================================================================

print("\n" + "="*78)
print("TESTE 6 — INVOLUÇÃO POR TRANSPOSTA")
print("="*78)

T = sp.zeros(N,N)

closure_T = True

for j in range(N):

    rhs = vec(B[j].T)

    coeff = MB.gauss_jordan_solve(
        rhs
    )[0]

    residual = sp.simplify(
        MB*coeff - rhs
    )

    if residual != sp.zeros(16,1):

        closure_T = False

    else:

        for i in range(N):

            T[i,j] = coeff[i]


if closure_T:

    print(
        "\n>>> A transposição fecha exatamente em g16."
    )

else:

    print(
        "\n>>> A transposição NÃO fecha."
    )


T2 = sp.simplify(T*T)

print(
    "\nT² = I ?",
    T2 == sp.eye(N)
)


# =============================================================================
# 10. DIMENSÕES DOS SETORES DA INVOLUÇÃO
# =============================================================================

plus = (T-sp.eye(N)).nullspace()
minus = (T+sp.eye(N)).nullspace()

print("\n" + "="*78)
print("TESTE 7 — DECOMPOSIÇÃO DA INVOLUÇÃO")
print("="*78)

print(
    "\ndim(g+) =",
    len(plus)
)

print(
    "dim(g-) =",
    len(minus)
)

print(
    "\nTotal =",
    len(plus)+len(minus)
)


# =============================================================================
# 11. RESUMO FINAL
# =============================================================================

print("\n" + "="*78)
print("RESUMO FINAL")
print("="*78)

print(f"""
Dimensão dos geradores          = {rank_B}

Dimensão dos produtos de grau 2 = {rank_prod}

Falhas de fechamento             = {len(closure_failures)}

Fechamento de grau 2             = {
    "SIM" if len(closure_failures)==0 else "NÃO"
}

Geradores com Bi² = -I           = {square_minus_identity}

Geradores com Bi² = Bi           = {square_idempotent}

Transposição fecha               = {
    "SIM" if closure_T else "NÃO"
}

T² = I                            = {
    "SIM" if T2 == sp.eye(N) else "NÃO"
}

dim(g+)                          = {len(plus)}

dim(g-)                          = {len(minus)}
""")

print("="*78)

# =============================================================================
# TESTE DO NÚCLEO ALPHA — {B1, B2, B3, B4}
#
# B1 = 1
# B2, B3 = candidatos a executor de i
# B4 = candidato a executor de mu
# =============================================================================

print("\n" + "="*78)
print("NÚCLEO ALPHA — B1, B2, B3, B4")
print("="*78)

I4 = sp.eye(4)

B1 = B[0]
B2 = B[1]
B3 = B[2]
B4 = B[3]


# =============================================================================
# 1. UNIDADE
# =============================================================================

print("\n--- UNIDADE ---")

print("B1 = I ?", B1 == I4)
print("B1² = B1 ?", B1*B1 == B1)


# =============================================================================
# 2. ASSINATURA DE i
# =============================================================================

print("\n--- EXECUTORES DE i ---")

print("B2² = -1 ?", B2*B2 == -I4)
print("B3² = -1 ?", B3*B3 == -I4)


# =============================================================================
# 3. ASSINATURA DE mu
# =============================================================================

print("\n--- EXECUTOR DE mu ---")

print("B4² = B4 ?", B4*B4 == B4)

print("\nB4 =")
sp.pprint(B4)


# =============================================================================
# 4. PRODUTOS i-mu
# =============================================================================

print("\n" + "-"*78)
print("RELAÇÕES ENTRE OS EXECUTORES")
print("-"*78)

print("\nB2 B4 =")
sp.pprint(B2*B4)

print("\nB4 B2 =")
sp.pprint(B4*B2)

print("\nB3 B4 =")
sp.pprint(B3*B4)

print("\nB4 B3 =")
sp.pprint(B4*B3)


# =============================================================================
# 5. TESTE DE COMUTAÇÃO / ANTICOMUTAÇÃO
# =============================================================================

print("\n" + "-"*78)
print("COMUTAÇÃO / ANTICOMUTAÇÃO")
print("-"*78)

print(
    "\nB2 B4 = - B4 B2 ?",
    B2*B4 == -B4*B2
)

print(
    "B3 B4 = - B4 B3 ?",
    B3*B4 == -B4*B3
)

print(
    "\nB2 B4 = B4 B2 ?",
    B2*B4 == B4*B2
)

print(
    "B3 B4 = B4 B3 ?",
    B3*B4 == B4*B3
)


# =============================================================================
# 6. RELAÇÃO ENTRE B2 E B3
# =============================================================================

print("\n" + "-"*78)
print("RELAÇÃO ENTRE B2 E B3")
print("-"*78)

print("\nB2 B3 =")
sp.pprint(B2*B3)

print("\nB3 B2 =")
sp.pprint(B3*B2)

print(
    "\nB2 B3 = B3 B2 ?",
    B2*B3 == B3*B2
)

print(
    "B2 B3 = -B3 B2 ?",
    B2*B3 == -B3*B2
)


# =============================================================================
# 7. NÚCLEO ALGÉBRICO — TESTE SEM EXIGIR FECHAMENTO EM DIMENSÃO 4
# =============================================================================

print("\n" + "="*78)
print("NÚCLEO ALPHA — B1, B2, B3, B4")
print("="*78)

B1 = B[0]
B2 = B[1]
B3 = B[2]
B4 = B[3]

I4 = sp.eye(4)

print("\n--- IDENTIFICAÇÃO ---")

print("B1 = I       ?", B1 == I4)
print("B2² = -I     ?", B2*B2 == -I4)
print("B3² = -I     ?", B3*B3 == -I4)
print("B4² = B4     ?", B4*B4 == B4)

# -------------------------------------------------------------------------
# Relações fundamentais
# -------------------------------------------------------------------------

print("\n--- RELAÇÕES FUNDAMENTAIS ---")

print("B2 B3 = B3 B2 ?", B2*B3 == B3*B2)
print("B2 B4 = B4 B2 ?", B2*B4 == B4*B2)
print("B3 B4 = B4 B3 ?", B3*B4 == B4*B3)

print("B2 B4 = -B4 B2 ?", B2*B4 == -B4*B2)
print("B3 B4 = -B4 B3 ?", B3*B4 == -B4*B3)

# -------------------------------------------------------------------------
# Produtos do núcleo: mostrar suas coordenadas em g16
# -------------------------------------------------------------------------

print("\n--- PRODUTOS DO NÚCLEO EM g16 ---")

for i in range(4):
    for j in range(4):

        P = B[i] * B[j]

        coeff = MB.gauss_jordan_solve(vec(P))[0]

        termos = []

        for k, c in enumerate(coeff):

            c = sp.simplify(c)

            if c != 0:

                if c == 1:
                    termos.append(f"B{k+1}")

                elif c == -1:
                    termos.append(f"-B{k+1}")

                else:
                    termos.append(f"({c})B{k+1}")

        expr = " + ".join(termos)
        expr = expr.replace("+ -", "- ")

        print(f"B{i+1} B{j+1} = {expr}")

# -------------------------------------------------------------------------
# Dimensão gerada pelo núcleo
# -------------------------------------------------------------------------

nucleo = [B1, B2, B3, B4]

MN = sp.Matrix.hstack(
    *[vec(M) for M in nucleo]
)

print("\nDimensão span{B1,B2,B3,B4} =", MN.rank())

# -------------------------------------------------------------------------
# Dimensão gerada pelo núcleo após produtos de grau 2
# -------------------------------------------------------------------------

nucleo_grau2 = []

for X in nucleo:
    for Y in nucleo:
        nucleo_grau2.append(X*Y)

MN2 = sp.Matrix.hstack(
    *[vec(M) for M in nucleo + nucleo_grau2]
)

print(
    "Dimensão após produtos de grau 2 =",
    MN2.rank()
)

print("\n>>> O núcleo de 4 dimensões não é fechado isoladamente.")
print(">>> Seus produtos propagam-se para o espaço g16.")
print(">>> O fechamento correto é realizado em dimensão 16.")


# =============================================================================
# 8. RESUMO ESTRUTURAL
# =============================================================================

print("\n" + "="*78)
print("RESUMO ESTRUTURAL")
print("="*78)

print("""
B1 = unidade 1

B2² = -I
B3² = -I

B4² = B4

B2 e B3 comutam.

B2 e B4 comutam.

B3 e B4 não comutam.

O subespaço span{B1,B2,B3,B4}
não deve ser forçado a ser uma subálgebra fechada.

O fechamento de grau 2 ocorre no espaço completo
span{B1,...,B16}.
""")
