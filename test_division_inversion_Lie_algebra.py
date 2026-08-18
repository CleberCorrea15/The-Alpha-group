
import sympy as sp

# ============================================================
# 1. GERADORES ORIGINAIS DO GRUPO ALPHA
# ============================================================

I4 = sp.eye(4)

G_C = sp.Matrix([
    [0,-1,0,0],
    [1,0,0,0],
    [0,0,0,-1],
    [0,0,1,0]
])

G_T = sp.Matrix([
    [0,0,-1,0],
    [0,0,0,-1],
    [1,0,0,0],
    [0,1,0,0]
])

G_mu = sp.Matrix([
    [0,0,0,0],
    [0,0,1,0],
    [0,0,1,0],
    [0,0,0,1]
])

# Mantido exatamente como no arquivo original
G_mu = sp.Matrix([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,1,0],
    [0,0,0,1]
])

G_const = sp.Matrix([
    [1,0,0,1],
    [0,1,1,0],
    [0,-1,0,0],
    [1,0,0,0]
])

GENERATORS = [I4, G_C, G_T, G_mu, G_const]


# ============================================================
# 2. CONSTRUÇÃO DA BASE DE DIMENSÃO 16
# ============================================================

def matrix_to_columns(matrices):
    return sp.Matrix.hstack(
        *[M.reshape(16, 1) for M in matrices]
    )


def build_basis(generators):
    conjunto = list(generators)
    rank = matrix_to_columns(conjunto).rank()

    while True:
        adicionou = False

        base_atual = [
            M.reshape(4, 4)
            for M in matrix_to_columns(conjunto).columnspace()
        ]

        for A in base_atual:
            for B in base_atual:
                P = A * B
                novo_rank = matrix_to_columns(
                    conjunto + [P]
                ).rank()

                if novo_rank > rank:
                    conjunto.append(P)
                    rank = novo_rank
                    adicionou = True

        if not adicionou:
            break

    return [
        vec.reshape(4, 4)
        for vec in matrix_to_columns(conjunto).columnspace()
    ]


Base = build_basis(GENERATORS)
N = len(Base)

print("=" * 72)
print("TESTE DE INVERSÃO DA DIVISÃO — GRUPO ALPHA")
print("=" * 72)
print(f"Dimensão obtida da base: {N}")

if N != 16:
    raise RuntimeError(
        f"ERRO: a base obtida tem dimensão {N}, não 16."
    )

print("✓ Os 16 geradores foram reconstruídos.\n")


# ============================================================
# 3. DECOMPOSIÇÃO NA BASE B1,...,B16
# ============================================================

Bmat = matrix_to_columns(Base)


def decompose(M):
    """Retorna c tal que M = sum(c_i B_i)."""
    rhs = M.reshape(16, 1)
    sol = sp.linsolve((Bmat, rhs))

    if sol == sp.EmptySet:
        raise ValueError("Matriz fora do espaço gerado pelos 16 B_i.")

    tup = next(iter(sol))
    return [sp.simplify(c) for c in tup]


def reconstruct(coeffs):
    R = sp.zeros(4)
    for c, B in zip(coeffs, Base):
        R += c * B
    return sp.simplify(R)


def print_coordinates(name, coeffs):
    print(f"\n{name} =")
    nonzero = False
    for i, c in enumerate(coeffs, 1):
        if sp.simplify(c) != 0:
            print(f"  {c} * B{i}")
            nonzero = True
    if not nonzero:
        print("  0")


# ============================================================
# 4. ELEMENTOS ALPHA DE TESTE
# ============================================================
#
# Usamos combinações lineares dos geradores originais.
# Isso permite testar a divisão dentro da álgebra.
#
# X e Y são deliberadamente não triviais.
# Y será o divisor.
# ============================================================

X = (
    2*I4
    + 3*G_C
    - 2*G_T
    + G_mu
    + G_const
)

Y = (
    I4
    - 2*G_C
    + G_T
    + 2*G_mu
    - G_const
)

print("=" * 72)
print("ELEMENTOS DE TESTE")
print("=" * 72)

print("\nX =")
sp.pprint(X)

print("\nY =")
sp.pprint(Y)

detY = sp.factor(Y.det())

print(f"\ndet(Y) = {detY}")

if detY == 0:
    raise RuntimeError("Y não é invertível. Escolha outro divisor.")

print("✓ Y é invertível.")


# ============================================================
# 5. DIVISÃO
# ============================================================
#
# Q = X Y^{-1}
#
# Em álgebra matricial, esta é a divisão à direita.
# ============================================================

Yinv = sp.simplify(Y.inv())
Q = sp.simplify(X * Yinv)

print("\n" + "=" * 72)
print("DIVISÃO")
print("=" * 72)

print("\nY^(-1) =")
sp.pprint(Yinv)

print("\nQ = X Y^(-1) =")
sp.pprint(Q)


# ============================================================
# 6. COORDENADAS DOS 16 GERADORES
# ============================================================

cx = decompose(X)
cy = decompose(Y)
cy_inv = decompose(Yinv)
cq = decompose(Q)

print_coordinates("X", cx)
print_coordinates("Y", cy)
print_coordinates("Y^(-1)", cy_inv)
print_coordinates("Q = X Y^(-1)", cq)


# ============================================================
# 7. RECONSTRUÇÃO
# ============================================================

X_rec = reconstruct(cx)
Y_rec = reconstruct(cy)
Yinv_rec = reconstruct(cy_inv)
Q_rec = reconstruct(cq)

err_X = sp.simplify(X_rec - X)
err_Y = sp.simplify(Y_rec - Y)
err_Yinv = sp.simplify(Yinv_rec - Yinv)
err_Q = sp.simplify(Q_rec - Q)

print("\n" + "=" * 72)
print("TESTE DE RECONSTRUÇÃO")
print("=" * 72)

print("X  reconstruído corretamente :", err_X == sp.zeros(4))
print("Y  reconstruído corretamente :", err_Y == sp.zeros(4))
print("Y⁻¹ reconstruído corretamente:", err_Yinv == sp.zeros(4))
print("Q  reconstruído corretamente :", err_Q == sp.zeros(4))


# ============================================================
# 8. TESTE FUNDAMENTAL DA DIVISÃO
# ============================================================
#
# Q Y = X
# ============================================================

XY = sp.simplify(Q * Y)
division_error = sp.simplify(XY - X)

print("\n" + "=" * 72)
print("TESTE FUNDAMENTAL: Q Y = X")
print("=" * 72)

print("Q Y =")
sp.pprint(XY)

print("\nErro QY - X =")
sp.pprint(division_error)

print(
    "\nRESULTADO:",
    "✓ DIVISÃO RECONSTRUÍDA"
    if division_error == sp.zeros(4)
    else "✗ FALHA"
)


# ============================================================
# 9. INVERSÃO DO QUOCIENTE
# ============================================================
#
# Q = X Y⁻¹
#
# Portanto:
#
# Q⁻¹ = Y X⁻¹
#
# e
#
# Q⁻¹ X = Y
#
# ============================================================

Qinv = sp.simplify(Q.inv())
Qinv_coeff = decompose(Qinv)
Qinv_rec = reconstruct(Qinv_coeff)

reverse_error = sp.simplify(Qinv * X - Y)
Q_inverse_reconstruction_error = sp.simplify(Qinv_rec - Qinv)

print("\n" + "=" * 72)
print("INVERSÃO DA DIVISÃO")
print("=" * 72)

print("\nQ^(-1) =")
sp.pprint(Qinv)

print_coordinates("Q^(-1)", Qinv_coeff)

print("\nTeste Q^(-1) X = Y:")

sp.pprint(reverse_error)

print(
    "\nReconstrução de Q⁻¹:",
    "✓ correta"
    if Q_inverse_reconstruction_error == sp.zeros(4)
    else "✗ falha"
)

print(
    "Inversão da divisão:",
    "✓ RECUPERADA"
    if reverse_error == sp.zeros(4)
    else "✗ falha"
)


# ============================================================
# 10. TESTE DUPLO DA INVERSÃO
# ============================================================
#
# (Q⁻¹)⁻¹ = Q
# ============================================================

double_inverse = sp.simplify(Qinv.inv() - Q)

print("\n" + "=" * 72)
print("DUPLA INVERSÃO")
print("=" * 72)

print(
    "(Q⁻¹)⁻¹ = Q :",
    "✓"
    if double_inverse == sp.zeros(4)
    else "✗"
)


# ============================================================
# 11. IDENTIDADE ASSOCIADA À DIVISÃO
# ============================================================
#
# Q Y X⁻¹ = I
#
# Esta é uma forma particularmente limpa de verificar
# que a operação foi realmente invertida.
# ============================================================

identity_test = sp.simplify(Q * Y * X.inv() - I4)

print("\n" + "=" * 72)
print("IDENTIDADE DA DIVISÃO")
print("=" * 72)

print("Q Y X⁻¹ - I =")
sp.pprint(identity_test)

print(
    "\nQ Y X⁻¹ = I :",
    "✓"
    if identity_test == sp.zeros(4)
    else "✗"
)


# ============================================================
# 12. RESUMO FINAL
# ============================================================

tests = {
    "Base com 16 geradores": N == 16,
    "X reconstruído": err_X == sp.zeros(4),
    "Y reconstruído": err_Y == sp.zeros(4),
    "Y^-1 reconstruído": err_Yinv == sp.zeros(4),
    "Q reconstruído": err_Q == sp.zeros(4),
    "Q Y = X": division_error == sp.zeros(4),
    "Q^-1 reconstruído": Q_inverse_reconstruction_error == sp.zeros(4),
    "Q^-1 X = Y": reverse_error == sp.zeros(4),
    "(Q^-1)^-1 = Q": double_inverse == sp.zeros(4),
    "Q Y X^-1 = I": identity_test == sp.zeros(4),
}

print("\n" + "=" * 72)
print("RESUMO DO EXPERIMENTO")
print("=" * 72)

for nome, ok in tests.items():
    print(f"{'✓' if ok else '✗'} {nome}")

if all(tests.values()):
    print("\nCONCLUSÃO COMPUTACIONAL:")
    print("A divisão matricial e sua inversão são")
    print("reconstruídas exatamente nas coordenadas dos 16 geradores.")
    print()
    print("ATENÇÃO:")
    print("Isto demonstra fechamento/reconstrução da operação")
    print("dentro da representação matricial. Não constitui,")
    print("por si só, uma prova de que a operação de divisão")
    print("seja um novo colchete de Lie.")
else:
    print("\nHá pelo menos um teste que falhou.")


# ============================================================
# 13. TESTE OPCIONAL COM VÁRIOS PARES
# ============================================================
#
# Para sair de um único exemplo, usamos combinações inteiras
# dos cinco geradores iniciais.
# ============================================================

print("\n" + "=" * 72)
print("TESTE MULTIPLO")
print("=" * 72)

casos = [
    (
        2*I4 + G_C + 2*G_T + G_mu,
        I4 - G_C + G_T + G_const
    ),
    (
        3*I4 - 2*G_C + G_mu + G_const,
        2*I4 + G_T - G_mu
    ),
    (
        I4 + 2*G_C - G_T + 3*G_mu,
        3*I4 - G_C + 2*G_T + G_const
    ),
    (
        2*I4 + G_C - G_T + 2*G_const,
        I4 + G_C + G_mu - G_const
    ),
]

sucessos = 0

for k, (A, B) in enumerate(casos, 1):

    if B.det() == 0:
        print(f"Caso {k}: divisor singular — ignorado.")
        continue

    R = sp.simplify(A * B.inv())

    # Verifica se R é reconstruível pelos 16 B_i
    cR = decompose(R)
    Rrec = reconstruct(cR)

    ok_rec = sp.simplify(Rrec - R) == sp.zeros(4)
    ok_div = sp.simplify(R * B - A) == sp.zeros(4)

    if ok_rec and ok_div:
        sucessos += 1

    print(
        f"Caso {k}: "
        f"reconstrução={'OK' if ok_rec else 'FALHA'}, "
        f"divisão={'OK' if ok_div else 'FALHA'}"
    )

print(f"\nCasos aprovados: {sucessos}/{len(casos)}")

print("\nFim do teste.")
