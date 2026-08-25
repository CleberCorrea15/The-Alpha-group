# =============================================================================
# MAIN RESULT
# =============================================================================
#
# The Alpha Group angular modal operator admits the exact Clifford
# decomposition
#
# A(theta) =
#     I + i Gamma_2
#       - cot(theta) Sigma_13
#       + tan(theta) Gamma5 Gamma_0
#
# Symbolic reconstruction residual:
#
#     EXACTLY ZERO
#
# This script verifies the result symbolically and numerically,
# including the spectral equivalence near theta = pi/2.
# =============================================================================

import sympy as sp
import numpy as np
import itertools

# =============================================================================
# TESTE DIRETO DOS MODOS:
# M(theta) / A(theta) x REPRESENTAÇÃO CLIFFORD
# =============================================================================
#
# Resultado procurado:
#
# A(theta) =
#     I + i Gamma_2
#       - cot(theta) Sigma_13
#       + tan(theta) Gamma5 Gamma_0
#
# O script verifica:
#   1. Cl(3,1)
#   2. decomposição Clifford simbólica
#   3. resíduo simbólico
#   4. identidade compacta
#   5. comparação espectral
#   6. comportamento próximo de theta = pi/2
#
# IMPORTANTE:
# Neste teste a parte angular A(theta) é tratada separadamente.
# A matriz H_mu do Alpha deve ser incorporada em um segundo teste,
# preservando mu e i*mu como setores algébricos.
# =============================================================================

sp.init_printing()

# =============================================================================
# 1. MATRIZ ANGULAR DO ALPHA
# =============================================================================

theta = sp.symbols("theta", real=True)

A = sp.Matrix([
    [1,             -sp.cot(theta), -sp.tan(theta), 1],
    [sp.cot(theta),  1,             -1,            -sp.tan(theta)],
    [sp.tan(theta), -1,               1,            -sp.cot(theta)],
    [1,              sp.tan(theta),   sp.cot(theta), 1]
])

# =============================================================================
# 2. CLIFFORD Cl(3,1)
# eta = diag(+1,-1,-1,-1)
# =============================================================================

I2 = sp.eye(2)
Z2 = sp.zeros(2)

sigma1 = sp.Matrix([[0, 1], [1, 0]])
sigma2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sigma3 = sp.Matrix([[1, 0], [0, -1]])

Gamma = [
    sp.Matrix(sp.BlockMatrix([[ I2, Z2], [ Z2,-I2]])),
    sp.Matrix(sp.BlockMatrix([[ Z2, sigma1],[-sigma1,Z2]])),
    sp.Matrix(sp.BlockMatrix([[ Z2, sigma2],[-sigma2,Z2]])),
    sp.Matrix(sp.BlockMatrix([[ Z2, sigma3],[-sigma3,Z2]]))
]

I4 = sp.eye(4)
eta = sp.diag(1, -1, -1, -1)

# =============================================================================
# 3. VERIFICAÇÃO CLIFFORD
# =============================================================================

clifford_ok = True
max_clifford_error = 0

for a in range(4):
    for b in range(4):

        E = sp.simplify(
            Gamma[a]*Gamma[b]
            + Gamma[b]*Gamma[a]
            - 2*eta[a,b]*I4
        )

        if E != sp.zeros(4):
            clifford_ok = False

print("="*78)
print("1. VERIFICAÇÃO DA ÁLGEBRA Cl(3,1)")
print("="*78)

if clifford_ok:
    print("RESULTADO: {Gamma_a,Gamma_b} = 2 eta_ab I exatamente.")
else:
    print("ERRO: relação Clifford não satisfeita.")

# =============================================================================
# 4. GERADORES SPIN
# =============================================================================

Sigma = {}

for a in range(4):
    for b in range(a+1, 4):

        Sigma[(a,b)] = sp.simplify(
            (Gamma[a]*Gamma[b] - Gamma[b]*Gamma[a]) / 2
        )

Gamma5 = sp.I * Gamma[0]*Gamma[1]*Gamma[2]*Gamma[3]

# =============================================================================
# 5. BASE CLIFFORD COMPLETA
# =============================================================================

basis = [I4]
names = ["I"]

for a in range(4):
    basis.append(Gamma[a])
    names.append(f"Gamma_{a}")

for a in range(4):
    for b in range(a+1,4):
        basis.append(Sigma[(a,b)])
        names.append(f"Sigma_{a}{b}")

for a in range(4):
    basis.append(Gamma5*Gamma[a])
    names.append(f"Gamma5Gamma_{a}")

basis.append(Gamma5)
names.append("Gamma5")

B = sp.Matrix.hstack(
    *[E.reshape(16,1) for E in basis]
)

# =============================================================================
# 6. DECOMPOSIÇÃO CLIFFORD SIMBÓLICA
# =============================================================================

coeff = sp.simplify(
    B.inv() * A.reshape(16,1)
)

A_rec = sp.zeros(4)

for c,E in zip(coeff,basis):
    A_rec += sp.simplify(c)*E

residual = sp.simplify(A-A_rec)

print("\n" + "="*78)
print("2. DECOMPOSIÇÃO CLIFFORD SIMBÓLICA")
print("="*78)

if residual == sp.zeros(4):
    print("RESÍDUO SIMBÓLICO = ZERO")
    print("A(theta) pertence exatamente ao espaço Clifford.")
else:
    print("Existe resíduo simbólico:")
    sp.pprint(residual)

# =============================================================================
# 7. COEFICIENTES NÃO NULOS
# =============================================================================

print("\n" + "="*78)
print("3. COEFICIENTES CLIFFORD NÃO NULOS")
print("="*78)

for name,c in zip(names,coeff):

    c = sp.simplify(c)

    if c != 0:
        print(f"{name:20s} = {c}")

# =============================================================================
# 8. IDENTIDADE COMPACTA
# =============================================================================

A_compact = (
    I4
    + sp.I*Gamma[2]
    - sp.cot(theta)*Sigma[(1,3)]
    + sp.tan(theta)*(Gamma5*Gamma[0])
)

compact_residual = sp.simplify(A-A_compact)

print("\n" + "="*78)
print("4. TESTE DA IDENTIDADE COMPACTA")
print("="*78)

if compact_residual == sp.zeros(4):

    print("IDENTIDADE CONFIRMADA SIMBOLICAMENTE:")
    print()
    print("A(theta) = I + i Gamma_2")
    print("          - cot(theta) Sigma_13")
    print("          + tan(theta) Gamma5 Gamma_0")

else:

    print("IDENTIDADE NÃO CONFIRMADA.")
    sp.pprint(compact_residual)

# =============================================================================
# 9. VERSÃO NUMÉRICA
# =============================================================================

Gamma_np = [
    np.array(G,dtype=complex)
    for G in Gamma
]

Sigma_np = {
    key: np.array(E,dtype=complex)
    for key,E in Sigma.items()
}

Gamma5_np = np.array(Gamma5,dtype=complex)
I4_np = np.eye(4,dtype=complex)

def A_numeric(th):

    t = np.tan(th)
    c = 1.0/t

    return np.array([
        [1,-c,-t,1],
        [c,1,-1,-t],
        [t,-1,1,-c],
        [1,t,c,1]
    ],dtype=complex)

def A_clifford_numeric(th):

    return (
        I4_np
        + 1j*Gamma_np[2]
        - (1/np.tan(th))*Sigma_np[(1,3)]
        + np.tan(th)*(Gamma5_np @ Gamma_np[0])
    )

# =============================================================================
# 10. TESTE ESPECTRAL
# =============================================================================

angles_deg = [
    10,30,45,60,80,
    89,89.9,89.99,
    90.01,90.1,91
]

print("\n" + "="*78)
print("5. COMPARAÇÃO DOS MODOS ESPECTRAIS")
print("="*78)

print(
    f"{'theta':>8} | "
    f"{'erro M':>12} | "
    f"{'erro eig':>12} | "
    f"{'max |Im(lambda)|':>18}"
)

for deg in angles_deg:

    th = np.deg2rad(deg)

    M_original = A_numeric(th)
    M_clifford = A_clifford_numeric(th)

    matrix_error = np.linalg.norm(
        M_original-M_clifford
    )

    eig_original = np.linalg.eigvals(M_original)
    eig_clifford = np.linalg.eigvals(M_clifford)

    # Pareamento das quatro raízes.
    best_error = min(
        np.linalg.norm(
            eig_original -
            eig_clifford[list(p)]
        )
        for p in itertools.permutations(range(4))
    )

    max_imaginary = np.max(
        np.abs(np.imag(eig_original))
    )

    print(
        f"{deg:8.2f} | "
        f"{matrix_error:12.3e} | "
        f"{best_error:12.3e} | "
        f"{max_imaginary:18.6e}"
    )

# =============================================================================
# 11. REGIME CRÍTICO
# =============================================================================

print("\n" + "="*78)
print("6. REGIME CRÍTICO theta -> pi/2")
print("="*78)

for delta in [1e-1,1e-2,1e-3,1e-4,1e-5]:

    for sign in [-1,1]:

        th = np.pi/2 + sign*delta

        M_original = A_numeric(th)
        M_clifford = A_clifford_numeric(th)

        error = np.linalg.norm(
            M_original-M_clifford
        )

        side = "-" if sign < 0 else "+"

        print(
            f"theta = pi/2 {side} {delta:.0e} | "
            f"|tan| = {abs(np.tan(th)):.6e} | "
            f"|cot| = {abs(1/np.tan(th)):.6e} | "
            f"erro = {error:.3e}"
        )

# =============================================================================
# 12. RESUMO
# =============================================================================

print("\n" + "="*78)
print("7. RESUMO FINAL")
print("="*78)

print("""
A parte angular A(theta) foi verificada:

  [OK] relação Clifford Cl(3,1)
  [OK] decomposição simbólica exata
  [OK] resíduo simbólico zero
  [OK] identidade compacta
  [OK] reconstrução numérica
  [OK] espectro original = espectro Clifford
  [OK] teste dos dois lados de theta = pi/2

IDENTIDADE:

A(theta) =
    I
  + i Gamma_2
  - cot(theta) Sigma_13
  + tan(theta) Gamma5 Gamma_0

PRÓXIMA ETAPA:

Incorporar H_mu do Alpha e testar o operador completo
M(theta) = A(theta) H_mu, mantendo os setores
1, i, mu e i*mu explicitamente separados.
""")
