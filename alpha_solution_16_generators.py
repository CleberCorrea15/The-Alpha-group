# ==============================================================
# ALPHA GROUP — RECONSTRUÇÃO DA SOLUÇÃO DO ARTIGO
# NA BASE DE 16 GERADORES
# ==============================================================

import sympy as sp

sqrt2 = sp.sqrt(2)

# ==============================================================
# 1. OS 16 GERADORES — MESMA BASE DO SCRIPT ORIGINAL
# ==============================================================

B = [
sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),

sp.Matrix([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]),

sp.Matrix([[0,0,-1,0],[0,0,0,-1],[1,0,0,0],[0,1,0,0]]),

sp.Matrix([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]),

sp.Matrix([[1,0,0,1],[0,1,1,0],[0,-1,0,0],[1,0,0,0]]),

sp.Matrix([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]]),

sp.Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,-1],[0,0,1,0]]),

sp.Matrix([[0,-1,-1,0],[1,0,0,1],[-1,0,0,0],[0,-1,0,0]]),

sp.Matrix([[0,0,-1,0],[0,0,0,-1],[0,0,0,0],[0,0,0,0]]),

sp.Matrix([[0,1,0,0],[-1,0,0,0],[1,0,0,1],[0,1,1,0]]),

sp.Matrix([[0,0,0,0],[0,0,0,0],[0,-1,0,0],[1,0,0,0]]),

sp.Matrix([[0,1,-1,0],[1,0,0,-1],[0,0,0,1],[0,0,-1,0]]),

sp.Matrix([[2,0,0,1],[0,0,1,0],[0,-1,-1,0],[1,0,0,1]]),

sp.Matrix([[1,0,0,0],[0,1,0,0],[0,-1,-1,0],[1,0,0,1]]),

sp.Matrix([[0,0,0,-1],[0,0,1,0],[0,1,-1,0],[1,0,0,-1]]),

sp.Matrix([[0,1,1,0],[-1,0,0,-1],[2,0,0,1],[0,0,1,0]])
]

N = len(B)
D = B[0].rows
I4 = sp.eye(D)
TARGET4 = -I4

# --------------------------------------------------------------
# BASE
# --------------------------------------------------------------

B1 = B[0]
B2 = B[1]   # representação dinâmica de i
B4 = B[3]   # representação dinâmica de mu
B7 = B[6]   # representação dinâmica de i*mu

print("=" * 75)
print("ESTRUTURA DE REFERÊNCIA")
print("=" * 75)

print("\nB2²:")
sp.pprint(sp.simplify(B2**2))

print("\nB4²:")
sp.pprint(sp.simplify(B4**2))

print("\n[B2,B4]:")
sp.pprint(sp.simplify(B2*B4 - B4*B2))

# ==============================================================
# 2. EXPRESSÃO DIRETA DO ARTIGO
# ==============================================================

X0 = (
    sqrt2/2 * I4
    - sqrt2/2 * B2
    - B4
    + B2*B4
)

print("\n" + "=" * 75)
print("X0 — SUBSTITUIÇÃO DIRETA")
print("=" * 75)

sp.pprint(sp.simplify(X0))

print("\nX0² + I4:")
sp.pprint(sp.simplify(X0**2 + I4))

# ==============================================================
# 3. DECOMPOSIÇÃO DE X0 NA BASE
# ==============================================================

def decompose_matrix(X, Base):
    """Decomposição exata de uma matriz 4x4 na base B1,...,B16."""
    A = sp.Matrix.hstack(
        *[sp.Matrix(list(M)) for M in Base]
    )
    target = sp.Matrix(list(X))
    sol = sp.linsolve((A, target))
    return sol

print("\n" + "=" * 75)
print("DECOMPOSIÇÃO DE X0 NA BASE")
print("=" * 75)

resultado = decompose_matrix(X0, B)
print(resultado)

# ==============================================================
# 4. MATRIZ GENÉRICA DA ÁLGEBRA
# ==============================================================

c = sp.symbols("c1:17")

X = sum(
    (c[j] * B[j] for j in range(16)),
    sp.zeros(4)
)

print("\n" + "=" * 75)
print("ELEMENTO GENÉRICO")
print("=" * 75)

print("X = c1 B1 + ... + c16 B16")

# ==============================================================
# 5. EQUAÇÃO X² = -I4
# ==============================================================

eq_matrix = sp.expand(X*X + I4)

equacoes = []

for i in range(4):
    for j in range(4):
        equacoes.append(sp.expand(eq_matrix[i,j]))

equacoes = [
    sp.factor(eq)
    for eq in equacoes
    if eq != 0
]

print("\nNúmero de equações independentes encontradas:",
      len(equacoes))

for k, eq in enumerate(equacoes, 1):
    print(f"E{k} =", eq)

# ==============================================================
# 6. VERIFICAÇÃO DO ESPAÇO GERADO PELOS 16 B_i
# ==============================================================

A = sp.Matrix.hstack(
    *[sp.Matrix(list(M)) for M in B]
)

print("\n" + "=" * 75)
print("RANK DA REPRESENTAÇÃO")
print("=" * 75)

print("rank =", A.rank())

# ==============================================================
# 7. SUBÁLGEBRA <B2, B4, B7>
# ==============================================================

print("\n" + "=" * 75)
print("SUBÁLGEBRA <B2, B4, B7>")
print("=" * 75)

print("\nB2²:")
sp.pprint(sp.simplify(B2**2))

print("\nB4²:")
sp.pprint(sp.simplify(B4**2))

print("\nB7²:")
sp.pprint(sp.simplify(B7**2))

print("\nB2 B4:")
sp.pprint(sp.simplify(B2*B4))

print("\nB4 B2:")
sp.pprint(sp.simplify(B4*B2))

print("\nB2 B7:")
sp.pprint(sp.simplify(B2*B7))

print("\nB7 B2:")
sp.pprint(sp.simplify(B7*B2))

print("\nB4 B7:")
sp.pprint(sp.simplify(B4*B7))

print("\nB7 B4:")
sp.pprint(sp.simplify(B7*B4))

# --------------------------------------------------------------
# COMUTADORES
# --------------------------------------------------------------

def comm(A, C):
    return sp.simplify(A*C - C*A)

print("\n" + "=" * 75)
print("COMUTADORES")
print("=" * 75)

print("\n[B2,B4] =")
sp.pprint(comm(B2, B4))

print("\n[B2,B7] =")
sp.pprint(comm(B2, B7))

print("\n[B4,B7] =")
sp.pprint(comm(B4, B7))

# --------------------------------------------------------------
# IDENTIFICAÇÃO EM TERMOS DA BASE
# --------------------------------------------------------------

def decompor(X, Base):
    A = sp.Matrix.hstack(
        *[sp.Matrix(list(M)) for M in Base]
    )
    y = sp.Matrix(list(X))
    sol = sp.linsolve((A, y))

    if sol == sp.EmptySet:
        return None

    return next(iter(sol))

print("\n" + "=" * 75)
print("DECOMPOSIÇÃO DOS PRODUTOS NA BASE")
print("=" * 75)

produtos = {
    "B2*B4": B2*B4,
    "B2*B7": B2*B7,
    "B4*B2": B4*B2,
    "B4*B7": B4*B7,
    "B7*B2": B7*B2,
    "B7*B4": B7*B4,
}

for nome, M in produtos.items():
    print(f"\n{nome} =")
    coef = decompor(sp.simplify(M), B)

    if coef is None:
        print("Não pertence ao span da base.")
    else:
        termos = []
        for j, cc in enumerate(coef):
            if sp.simplify(cc) != 0:
                termos.append(f"({sp.simplify(cc)}) B{j+1}")
        print(" + ".join(termos))

# ==============================================================
# 8. FECHAMENTO DO CONJUNTO {B1,B2,B4,B7}
# ==============================================================

print("\n" + "=" * 75)
print("TESTE DE FECHAMENTO")
print("=" * 75)

sub = [B1, B2, B4, B7]

for A_sub in sub:
    for C_sub in sub:
        M = sp.simplify(A_sub*C_sub)
        coef = decompor(M, B)

        suporte = []
        for k, cc in enumerate(coef):
            if sp.simplify(cc) != 0:
                suporte.append(k+1)

        print(
            f"B{sub.index(A_sub)+1} B{sub.index(C_sub)+1}"
            f" -> B{suporte}"
        )

# ==============================================================
# 9. SOLUÇÃO DO ARTIGO
# ==============================================================

X_artigo = (
    sp.sqrt(2)/2 * B1
    - sp.sqrt(2)/2 * B2
    - B4
    + B7
)

print("\n" + "=" * 75)
print("SOLUÇÃO DO ARTIGO")
print("=" * 75)

sp.pprint(X_artigo)

print("\nX_artigo²:")
sp.pprint(sp.simplify(X_artigo**2))

print("\nX_artigo² + I4:")
sp.pprint(sp.simplify(X_artigo**2 + I4))

# ==============================================================
# 10. POLINÔMIO CARACTERÍSTICO
# ==============================================================

lam = sp.symbols("lambda")

print("\n" + "=" * 75)
print("POLINÔMIO CARACTERÍSTICO")
print("=" * 75)

charpoly = sp.factor(
    X_artigo.charpoly(lam).as_expr()
)

print("det(lambda I - X) =")
sp.pprint(charpoly)

print("\nFatoração:")
sp.pprint(sp.factor(charpoly))

print("\n" + "=" * 75)
print("FIM — RECONSTRUÇÃO DA SOLUÇÃO DO ARTIGO")
print("=" * 75)
