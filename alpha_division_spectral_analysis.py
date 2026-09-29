# ============================================================
# SPECTRAL ANALYSIS OF THE CLASSICAL ALPHA DIVISION MATRIX
# Golden-ratio and sqrt(3) polynomial structures
#
# The script performs symbolic analysis of the characteristic
# polynomial of the classical division matrix M(theta), with
# mu = 1, and examines the special point theta = pi/4.
# ============================================================

import sympy as sp

# ============================================================
# MATRIZ CLÁSSICA DA DIVISÃO
# ============================================================

t, x = sp.symbols('t x')
I = sp.I

M = sp.Matrix([
    [1,   -1/t, -t,    1],
    [1/t, I,    -1,   -t],
    [t,   -1,    1,   -1/t],
    [1,    t,    1/t,  I]
])

print("=" * 75)
print("MATRIZ CLÁSSICA M(t), com mu = 1")
print("=" * 75)
print(M)


# ============================================================
# POLINÔMIO CARACTERÍSTICO
# ============================================================

lam = sp.symbols('lambda')

chi = sp.factor(M.charpoly(lam).as_expr())

print("\n" + "=" * 75)
print("POLINÔMIO CARACTERÍSTICO")
print("=" * 75)

print("chi(lambda) =")
print(chi)


# ============================================================
# POLINÔMIO ÁUREO
# ============================================================

golden = lam**2 - lam - 1

print("\n" + "=" * 75)
print("TESTE DO FATOR ÁUREO")
print("=" * 75)

print("Fator procurado:")
print(golden)


# ============================================================
# RESTO DA DIVISÃO POLINOMIAL
# ============================================================

num, den = sp.together(chi).as_numer_denom()

rem = sp.rem(
    num,
    golden,
    domain=sp.QQ.frac_field(t, I)
)

print("\nResto da divisão:")
print(sp.factor(rem))


# ============================================================
# COEFICIENTES DO RESTO
# ============================================================

rem = sp.collect(sp.expand(rem), lam)

print("\nResto expandido:")
print(rem)

coeffs = sp.Poly(rem, lam).all_coeffs()

print("\nCoeficientes que devem desaparecer:")
for c in coeffs:
    print(sp.factor(c))


# ============================================================
# TESTE ESPECIAL t = 1
# theta = pi/4
# ============================================================

print("\n" + "=" * 75)
print("TESTE EM t = 1  <=>  theta = pi/4")
print("=" * 75)

chi_pi4 = sp.factor(chi.subs(t, 1))

print("\nchi(lambda) em t=1:")
print(chi_pi4)

print("\nDivisão por lambda^2-lambda-1:")

q, r = sp.div(
    chi_pi4,
    golden,
    domain=sp.QQ_I
)

print("Quociente:")
print(sp.factor(q))

print("\nResto:")
print(sp.factor(r))


# ============================================================
# RAÍZES DO FATOR ÁUREO
# ============================================================

print("\n" + "=" * 75)
print("RAÍZES ÁUREAS")
print("=" * 75)

roots = sp.solve(golden, lam)

for r in roots:
    print("lambda =", r)
    print("valor numerico =", sp.N(r, 15))


# ============================================================
# VERIFICAÇÃO DA RAIZ DE 3
# ============================================================

sqrt3_poly = 2*lam**2 - 2*lam - 1

print("\n" + "=" * 75)
print("ESTRUTURA ASSOCIADA A sqrt(3)")
print("=" * 75)

print("Polinomio:")
print(sqrt3_poly)

print("Fatoração:")
print(sp.factor(sqrt3_poly))

print("Raízes:")
for r in sp.solve(sqrt3_poly, lam):
    print(r, "=", sp.N(r, 15))
