# =============================================================================
# TESTE ESTRUTURAL COMPLETO
# ALPHA 4D -> 16 GERADORES -> CLIFFORD Cl(3,1) -> H_mu -> M(theta)
# =============================================================================
#
# OBJETIVOS
#
# 1. Reconstruir a base Alpha B1,...,B16.
# 2. Confirmar que os 16 geradores são linearmente independentes.
# 3. Identificar o setor Clifford já presente no Alpha.
# 4. Construir o operador dinâmico:
#
#       H_mu = Gamma_0 (Gamma_1 px + Gamma_2 py + Gamma_3 pz)
#              + mu Gamma_0
#
# 5. Construir:
#
#       M(theta) = A(theta) H_mu
#
# 6. Decompor H_mu e M(theta) na base Alpha.
# 7. Decompor H_mu e M(theta) na base Clifford.
# 8. Verificar resíduos simbólicos.
# 9. Comparar os espaços gerados antes/depois de H_mu.
#
# IMPORTANTE:
#
# mu, px, py, pz são tratados como parâmetros escalares nesta representação
# matricial Clifford. Isso NÃO identifica mu escalar com o elemento idempotente
# do Alpha. A identificação algébrica de mu será feita separadamente.
# =============================================================================

import numpy as np
import sympy as sp

from sympy.physics.matrices import mgamma

sp.init_printing()

# =============================================================================
# 1. PARÂMETROS
# =============================================================================

theta = sp.symbols("theta", real=True)

px, py, pz = sp.symbols(
    "p_x p_y p_z",
    real=True
)

mu = sp.symbols(
    "mu",
    real=True
)

# =============================================================================
# 2. BASE ALPHA
# =============================================================================

D = 4
N = 16

B = np.zeros(
    (N,D,D),
    dtype=float
)

B[0] = [
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]

B[1] = [
    [0,-1,0,0],
    [1,0,0,0],
    [0,0,0,-1],
    [0,0,1,0]
]

B[2] = [
    [0,0,-1,0],
    [0,0,0,-1],
    [1,0,0,0],
    [0,1,0,0]
]

B[3] = [
    [0,0,0,0],
    [0,0,0,0],
    [0,0,1,0],
    [0,0,0,1]
]

B[4] = [
    [1,0,0,1],
    [0,1,1,0],
    [0,-1,0,0],
    [1,0,0,0]
]

B[5] = [
    [0,0,0,1],
    [0,0,-1,0],
    [0,-1,0,0],
    [1,0,0,0]
]

B[6] = [
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,-1],
    [0,0,1,0]
]

B[7] = [
    [0,-1,-1,0],
    [1,0,0,1],
    [-1,0,0,0],
    [0,-1,0,0]
]

B[8] = [
    [0,0,-1,0],
    [0,0,0,-1],
    [0,0,0,0],
    [0,0,0,0]
]

B[9] = [
    [0,1,0,0],
    [-1,0,0,0],
    [1,0,0,1],
    [0,1,1,0]
]

B[10] = [
    [0,0,0,0],
    [0,0,0,0],
    [0,-1,0,0],
    [1,0,0,0]
]

B[11] = [
    [0,1,-1,0],
    [1,0,0,-1],
    [0,0,0,1],
    [0,0,-1,0]
]

B[12] = [
    [2,0,0,1],
    [0,0,1,0],
    [0,-1,-1,0],
    [1,0,0,1]
]

B[13] = [
    [1,0,0,0],
    [0,1,0,0],
    [0,-1,-1,0],
    [1,0,0,1]
]

B[14] = [
    [0,0,0,-1],
    [0,0,1,0],
    [0,1,-1,0],
    [1,0,0,-1]
]

B[15] = [
    [0,1,1,0],
    [-1,0,0,-1],
    [2,0,0,1],
    [0,0,1,0]
]

# =============================================================================
# 3. CONVERSÃO PARA SYMPY
# =============================================================================

B_sym = [
    sp.Matrix(B[k])
    for k in range(16)
]

B_matrix = sp.Matrix.hstack(
    *[
        M.reshape(16,1)
        for M in B_sym
    ]
)

print("="*78)
print("1. BASE ALPHA")
print("="*78)

rank_B = B_matrix.rank()

print("Rank da base Alpha =", rank_B)

if rank_B == 16:
    print("[OK] Os 16 geradores são linearmente independentes.")
else:
    print("[ERRO] Rank diferente de 16.")

# =============================================================================
# 4. CLIFFORD Cl(3,1)
# =============================================================================

G0 = mgamma(0)
G1 = mgamma(1)
G2 = mgamma(2)
G3 = mgamma(3)
G5 = mgamma(5)

I4 = sp.eye(4)

Gamma = [
    G0,
    G1,
    G2,
    G3
]

eta = sp.diag(
    1,-1,-1,-1
)

# =============================================================================
# 5. VERIFICAÇÃO CLIFFORD
# =============================================================================

print("\n" + "="*78)
print("2. VERIFICAÇÃO Cl(3,1)")
print("="*78)

clifford_ok = True

for a in range(4):

    for b in range(4):

        E = sp.simplify(
            Gamma[a]*Gamma[b]
            + Gamma[b]*Gamma[a]
            - 2*eta[a,b]*I4
        )

        if E != sp.zeros(4):

            clifford_ok = False

if clifford_ok:

    print("[OK] {Gamma_a,Gamma_b} = 2 eta_ab I")

else:

    print("[ERRO] Relação Clifford não satisfeita.")

# =============================================================================
# 6. GERADORES SPIN
# =============================================================================

Sigma = {}

for a in range(4):

    for b in range(a+1,4):

        Sigma[(a,b)] = sp.simplify(
            (
                Gamma[a]*Gamma[b]
                -
                Gamma[b]*Gamma[a]
            ) / 2
        )

Gamma5 = sp.I * G0*G1*G2*G3

Sigma13 = Sigma[(1,3)]

G5G0 = sp.simplify(
    Gamma5*G0
)

# =============================================================================
# 7. OPERADOR ANGULAR A(theta)
# =============================================================================

A = (
    I4
    + sp.I*G2
    - sp.cot(theta)*Sigma13
    + sp.tan(theta)*G5G0
)

print("\n" + "="*78)
print("3. OPERADOR ANGULAR A(theta)")
print("="*78)

print("""
A(theta) =
    I
  + i Gamma_2
  - cot(theta) Sigma_13
  + tan(theta) Gamma_5 Gamma_0
""")

# =============================================================================
# 8. H_mu DINÂMICO
# =============================================================================

H_mu = (
    G0*(G1*px + G2*py + G3*pz)
    + mu*G0
)

print("\n" + "="*78)
print("4. OPERADOR DINÂMICO H_mu")
print("="*78)

sp.pprint(H_mu)

# =============================================================================
# 9. OPERADOR COMPLETO
# =============================================================================

M = sp.simplify(
    A*H_mu
)

print("\n" + "="*78)
print("5. OPERADOR COMPLETO M(theta)")
print("="*78)

print("M(theta) = A(theta) H_mu")

# =============================================================================
# 10. BASE CLIFFORD COMPLETA
# =============================================================================

basis = [I4]
names = ["I"]

for a in range(4):

    basis.append(Gamma[a])
    names.append(f"Gamma_{a}")

for a in range(4):

    for b in range(a+1,4):

        basis.append(
            Sigma[(a,b)]
        )

        names.append(
            f"Sigma_{a}{b}"
        )

for a in range(4):

    basis.append(
        Gamma5*Gamma[a]
    )

    names.append(
        f"Gamma5Gamma_{a}"
    )

basis.append(Gamma5)
names.append("Gamma5")

Clifford_matrix = sp.Matrix.hstack(
    *[
        E.reshape(16,1)
        for E in basis
    ]
)

# =============================================================================
# 11. FUNÇÃO: DECOMPOSIÇÃO CLIFFORD
# =============================================================================

def clifford_decompose(M,label="M"):

    coeff = sp.simplify(
        Clifford_matrix.inv()
        * M.reshape(16,1)
    )

    reconstruction = sp.zeros(4)

    for c,E in zip(coeff,basis):

        reconstruction += (
            sp.simplify(c)*E
        )

    residual = sp.simplify(
        M-reconstruction
    )

    print("\n" + "-"*78)
    print(f"DECOMPOSIÇÃO CLIFFORD: {label}")
    print("-"*78)

    for name,c in zip(names,coeff):

        c = sp.simplify(c)

        if c != 0:

            print(
                f"{name:20s} = {c}"
            )

    if residual == sp.zeros(4):

        print("\n[OK] Resíduo Clifford = ZERO")

    else:

        print("\n[ERRO] Resíduo Clifford != ZERO")

        sp.pprint(residual)

    return coeff,residual

# =============================================================================
# 12. FUNÇÃO: DECOMPOSIÇÃO ALPHA
# =============================================================================

def alpha_decompose(M,label="M"):

    solution = sp.linsolve(
        (
            B_matrix,
            M.reshape(16,1)
        )
    )

    print("\n" + "-"*78)
    print(f"DECOMPOSIÇÃO ALPHA: {label}")
    print("-"*78)

    if solution == sp.EmptySet:

        print(
            "[NÃO] O operador não pertence ao span Alpha."
        )

        return None,None

    coeffs = list(solution)[0]

    reconstruction = sp.zeros(4)

    for c,Bi in zip(
        coeffs,
        B_sym
    ):

        reconstruction += (
            sp.simplify(c)*Bi
        )

    residual = sp.simplify(
        M-reconstruction
    )

    for k,c in enumerate(coeffs):

        c = sp.simplify(c)

        if c != 0:

            print(
                f"B{k+1:02d} = {c}"
            )

    if residual == sp.zeros(4):

        print("\n[OK] Resíduo Alpha = ZERO")

    else:

        print("\n[ERRO] Resíduo Alpha != ZERO")

        sp.pprint(residual)

    return sp.Matrix(coeffs),residual

# =============================================================================
# 13. DECOMPOSIÇÃO DE A
# =============================================================================

print("\n" + "="*78)
print("6. A(theta): CLIFFORD E ALPHA")
print("="*78)

coeff_A_Cl,res_A_Cl = clifford_decompose(
    A,
    "A(theta)"
)

coeff_A_Alpha,res_A_Alpha = alpha_decompose(
    A,
    "A(theta)"
)

# =============================================================================
# 14. DECOMPOSIÇÃO DE H_mu
# =============================================================================

print("\n" + "="*78)
print("7. H_mu: CLIFFORD E ALPHA")
print("="*78)

coeff_H_Cl,res_H_Cl = clifford_decompose(
    H_mu,
    "H_mu"
)

coeff_H_Alpha,res_H_Alpha = alpha_decompose(
    H_mu,
    "H_mu"
)

# =============================================================================
# 15. DECOMPOSIÇÃO DE M(theta)
# =============================================================================

print("\n" + "="*78)
print("8. M(theta)=A(theta)H_mu")
print("="*78)

coeff_M_Cl,res_M_Cl = clifford_decompose(
    M,
    "M(theta)"
)

coeff_M_Alpha,res_M_Alpha = alpha_decompose(
    M,
    "M(theta)"
)

# =============================================================================
# 16. TESTE DE PERTENÇA
# =============================================================================

print("\n" + "="*78)
print("9. TESTE DE PERTENÇA")
print("="*78)

print(
    "A(theta) em Alpha  :",
    res_A_Alpha == sp.zeros(4)
)

print(
    "H_mu em Alpha      :",
    res_H_Alpha == sp.zeros(4)
)

print(
    "M(theta) em Alpha  :",
    res_M_Alpha == sp.zeros(4)
)

print(
    "A(theta) em Clifford:",
    res_A_Cl == sp.zeros(4)
)

print(
    "H_mu em Clifford    :",
    res_H_Cl == sp.zeros(4)
)

print(
    "M(theta) em Clifford:",
    res_M_Cl == sp.zeros(4)
)

# =============================================================================
# 17. SUBESPAÇOS GERADOS
# =============================================================================

print("\n" + "="*78)
print("10. DIMENSÕES DOS SUBESPAÇOS")
print("="*78)

# Quatro componentes Clifford fundamentais
C1 = I4
C2 = sp.I*G2
C3 = Sigma13
C4 = G5G0

C_matrix = sp.Matrix.hstack(
    *[
        C.reshape(16,1)
        for C in [C1,C2,C3,C4]
    ]
)

print(
    "Rank do setor Clifford 4D =",
    C_matrix.rank()
)

# A, H e M como operadores individuais
AHM_matrix = sp.Matrix.hstack(
    A.reshape(16,1),
    H_mu.reshape(16,1),
    M.reshape(16,1)
)

print(
    "Rank{A,H_mu,M} =",
    AHM_matrix.rank()
)

# =============================================================================
# 18. DIFERENÇA PRODUZIDA POR H_mu
# =============================================================================

Delta = sp.simplify(
    M-A
)

print("\n" + "="*78)
print("11. CONTRIBUIÇÃO DE H_mu")
print("="*78)

print("Delta = M - A")

alpha_decompose(
    Delta,
    "Delta = M - A"
)

clifford_decompose(
    Delta,
    "Delta = M - A"
)

# =============================================================================
# 19. CONCLUSÃO AUTOMÁTICA
# =============================================================================

print("\n" + "="*78)
print("12. DIAGNÓSTICO FINAL")
print("="*78)

print("""
Perguntas fundamentais:

1. O setor Clifford 4D já pertence ao Alpha?
2. H_mu pertence ao espaço dos 16 geradores?
3. M(theta)=A(theta)H_mu pertence ao Alpha?
4. Quais novos B_i aparecem exclusivamente após H_mu?
5. Quais graus de liberdade são introduzidos por H_mu?
6. A dinâmica permanece confinada à álgebra Alpha?

IMPORTANTE:

O script NÃO assume que mu seja o idempotente matricial do Alpha.
Aqui mu é um parâmetro escalar na representação Clifford.

A identificação algébrica:

        mu^2 = mu
        i mu = - mu i

deve ser testada separadamente na álgebra Alpha.
""")
