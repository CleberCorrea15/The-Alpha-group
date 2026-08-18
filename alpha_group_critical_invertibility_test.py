# =============================================================================
# ALPHA GROUP — CRITICAL-POINT INVERTIBILITY TEST
# theta = pi/2
#
# Purpose:
# Test whether multiplication <-> division remains mutually invertible
# on the critical nodal modes induced by the projective critical class.
#
# IMPORTANT:
# - The critical matrix M_proj is allowed to be singular.
# - No pseudoinverse is used for division or inversion.
# - np.linalg.inv() is used for the actual inversions.
# - The pseudoinverse is used only to obtain coordinates in the 16-generator
#   basis and to reconstruct matrices.
# =============================================================================

import numpy as np
import time

TOL_RANK = 1e-10
TOL_INV = 1e-12
TOL_TEST = 1e-8

np.set_printoptions(precision=8, suppress=True)

inicio_global = time.time()

print("=" * 80)
print("ALPHA GROUP — CRITICAL-POINT INVERTIBILITY TEST")
print("theta = pi/2")
print("=" * 80)

# =============================================================================
# 1. ALPHA GENERATORS
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

GENERATORS = [I4, G_C, G_T, G_mu, G_const]


# =============================================================================
# 2. BUILD 16-DIMENSIONAL BASIS
# =============================================================================

def rank_of(lista):
    if not lista:
        return 0

    A = np.column_stack([M.reshape(16) for M in lista])
    return np.linalg.matrix_rank(A, tol=TOL_RANK)


def build_basis_fast(generators):
    basis = [G.copy() for G in generators]

    A = np.column_stack([M.reshape(16) for M in basis])
    rank = np.linalg.matrix_rank(A, tol=TOL_RANK)

    while rank < 16:
        curr_len = len(basis)
        adicionou = False

        for i in range(curr_len):
            for j in range(curr_len):

                prod = basis[i] @ basis[j]
                v = prod.reshape(16)

                test_A = np.column_stack([A, v])
                new_rank = np.linalg.matrix_rank(
                    test_A,
                    tol=TOL_RANK
                )

                if new_rank > rank:
                    basis.append(prod)
                    A = test_A
                    rank = new_rank
                    adicionou = True

                    if rank == 16:
                        break

            if rank == 16:
                break

        if not adicionou:
            break

    Base = []
    rank_final = 0

    for M in basis:
        teste = Base + [M]
        novo_rank = rank_of(teste)

        if novo_rank > rank_final:
            Base.append(M)
            rank_final = novo_rank

        if rank_final == 16:
            break

    return Base


Base = build_basis_fast(GENERATORS)

print("\nDimensão da base:", len(Base))

if len(Base) != 16:
    raise RuntimeError("A base não atingiu dimensão 16.")

print("✓ 16 geradores reconstruídos.")


# =============================================================================
# 3. COORDENADAS NA BASE
# =============================================================================

BASIS_NUMPY = np.column_stack([
    B.reshape(16) for B in Base
])

# SOMENTE para coordenadas/reconstrução.
BASIS_PINV = np.linalg.pinv(BASIS_NUMPY)


def alpha_coefficients(M):
    return BASIS_PINV @ M.reshape(16)


def reconstruct_numpy(coeffs):
    return (
        BASIS_NUMPY @ coeffs
    ).reshape(4, 4)


def reconstruction_error(M, coeffs):
    R = reconstruct_numpy(coeffs)
    return np.linalg.norm(R - M, ord="fro")


# =============================================================================
# 4. CLASSE PROJETIVA CRÍTICA
#
# M_proj(theta) = M(theta) / tan(theta)
#
# No limite theta -> pi/2:
#
# 1/tan(theta) -> 0
# cot(theta)/tan(theta) -> 0
# tan(theta)/tan(theta) -> 1
#
# O objeto crítico é A_PROJ.
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

A_PROJ = np.array([
    [0, 0,-1, 0],
    [0, 0, 0,-1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=float)


def build_projective_matrix():
    rows = []

    for i in range(4):
        blocks = []

        for j in range(4):
            blocks.append(
                A_PROJ[i, j] * INTERNAL[i][j]
            )

        rows.append(np.hstack(blocks))

    return np.vstack(rows)


M_CRIT = build_projective_matrix()


# =============================================================================
# 5. DIAGNÓSTICO DA ESTRUTURA CRÍTICA
# =============================================================================

print("\n" + "=" * 80)
print("ESTRUTURA CRÍTICA")
print("=" * 80)

rank_crit = np.linalg.matrix_rank(M_CRIT, tol=TOL_RANK)
det_crit = np.linalg.det(M_CRIT)
singular_crit = np.linalg.svd(M_CRIT, compute_uv=False)
eigvals, eigvecs = np.linalg.eig(M_CRIT)

print("\nrank(M_crit) =", rank_crit)
print("det(M_crit)  =", det_crit)

print("\nAutovalores:")
for i, lam in enumerate(eigvals, start=1):
    print(f"lambda {i:2d} = {lam}")

print("\nValores singulares:")
print(singular_crit)

print("\nObservação:")
print("A singularidade de M_crit NÃO é usada para a divisão.")
print("O teste de invertibilidade será feito nos modos X e Y.")


# =============================================================================
# 6. CONVERSÃO DE MODO NODAL PARA ELEMENTO ALPHA
# =============================================================================

def mode_to_matrix(v):
    v = np.asarray(v, dtype=complex)

    norma = np.linalg.norm(v)

    if norma < 1e-14:
        raise ValueError("Modo nodal nulo.")

    v = v / norma

    # Fixação de fase global
    idx = np.argmax(np.abs(v))
    fase = np.angle(v[idx])
    v = v * np.exp(-1j * fase)

    # Parte real usada na representação Alpha
    vr = np.real(v)

    return reconstruct_numpy(vr)


# =============================================================================
# 7. TESTE CENTRAL:
#    MULTIPLICAÇÃO <-> DIVISÃO NO PONTO CRÍTICO
# =============================================================================

def critical_invertibility_test(X, Y, nome):
    print("\n" + "-" * 80)
    print(f"TESTE CRÍTICO — {nome}")
    print("-" * 80)

    det_X = np.linalg.det(X)
    det_Y = np.linalg.det(Y)

    print(f"\ndet(X) = {det_X:.12e}")
    print(f"det(Y) = {det_Y:.12e}")

    if abs(det_X) < TOL_INV:
        print("\n⚠ X SINGULAR")
        return {
            "status": "X_SINGULAR",
            "passed": False
        }

    if abs(det_Y) < TOL_INV:
        print("\n⚠ Y SINGULAR")
        return {
            "status": "Y_SINGULAR",
            "passed": False
        }

    # -------------------------------------------------------------------------
    # INVERSÃO REAL DE Y
    # -------------------------------------------------------------------------

    try:
        Y_inv = np.linalg.inv(Y)
    except np.linalg.LinAlgError:
        print("\n⚠ Y^-1 NÃO EXISTE")
        return {
            "status": "Y_INVERSAO_FALHOU",
            "passed": False
        }

    # -------------------------------------------------------------------------
    # DIVISÃO
    # -------------------------------------------------------------------------

    Q = X @ Y_inv
    det_Q = np.linalg.det(Q)

    print(f"det(Q) = {det_Q:.12e}")

    if abs(det_Q) < TOL_INV:
        print("\n⚠ Q SINGULAR")
        return {
            "status": "Q_SINGULAR",
            "passed": False
        }

    # -------------------------------------------------------------------------
    # INVERSÃO REAL DE Q
    # -------------------------------------------------------------------------

    try:
        Q_inv = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        print("\n⚠ Q^-1 NÃO EXISTE")
        return {
            "status": "Q_INVERSAO_FALHOU",
            "passed": False
        }

    # -------------------------------------------------------------------------
    # MULTIPLICAÇÃO -> DIVISÃO
    # -------------------------------------------------------------------------

    erro_QY = np.linalg.norm(Q @ Y - X, ord="fro")

    # -------------------------------------------------------------------------
    # DIVISÃO -> MULTIPLICAÇÃO
    # -------------------------------------------------------------------------

    erro_QinvX = np.linalg.norm(Q_inv @ X - Y, ord="fro")

    # -------------------------------------------------------------------------
    # DUPLA INVERSÃO
    # -------------------------------------------------------------------------

    Q_double = np.linalg.inv(Q_inv)
    erro_double = np.linalg.norm(Q_double - Q, ord="fro")

    # -------------------------------------------------------------------------
    # RECONSTRUÇÃO DE Q NOS 16 GERADORES
    # -------------------------------------------------------------------------

    coef_Q = alpha_coefficients(Q)
    erro_recon = reconstruction_error(Q, coef_Q)

    print(f"Erro QY-X          = {erro_QY:.12e}")
    print(f"Erro Q^-1X-Y       = {erro_QinvX:.12e}")
    print(f"Erro dupla inversão = {erro_double:.12e}")
    print(f"Erro reconstrução Q = {erro_recon:.12e}")

    passed = (
        erro_QY < TOL_TEST
        and erro_QinvX < TOL_TEST
        and erro_double < TOL_TEST
        and erro_recon < TOL_TEST
    )

    if passed:
        print("\n✓ MULTIPLICAÇÃO ↔ DIVISÃO: APROVADA")
    else:
        print("\n⚠ MULTIPLICAÇÃO ↔ DIVISÃO: FALHOU")

    return {
        "status": "OK" if passed else "FAIL",
        "passed": passed,
        "det_X": det_X,
        "det_Y": det_Y,
        "det_Q": det_Q,
        "erro_QY": erro_QY,
        "erro_QinvX": erro_QinvX,
        "erro_double": erro_double,
        "erro_recon": erro_recon,
        "Q": Q,
        "Q_inv": Q_inv
    }


# =============================================================================
# 8. SELEÇÃO DOS DOIS MODOS CRÍTICOS
# =============================================================================

# Ordenação pelos módulos dos autovalores.
ordem = np.argsort(np.abs(eigvals))

v1 = eigvecs[:, ordem[0]]
v2 = eigvecs[:, ordem[1]]

try:
    X = mode_to_matrix(v1)
    Y = mode_to_matrix(v2)
except Exception as erro:
    raise RuntimeError(
        f"Erro na construção dos modos críticos: {erro}"
    )

resultado = critical_invertibility_test(
    X,
    Y,
    "MODOS CRÍTICOS"
)


# =============================================================================
# 9. TESTE INDEPENDENTE DE MU
# =============================================================================

print("\n" + "=" * 80)
print("TESTE INDEPENDENTE — MU")
print("=" * 80)

coef_mu = alpha_coefficients(G_mu)
mu_recon = reconstruct_numpy(coef_mu)
erro_mu = np.linalg.norm(mu_recon - G_mu, ord="fro")

print(f"\nErro de reconstrução de mu = {erro_mu:.12e}")

if erro_mu < TOL_TEST:
    print("✓ mu permanece no espaço dos 16 geradores.")
    mu_ok = True
else:
    print("⚠ Falha na reconstrução de mu.")
    mu_ok = False


# =============================================================================
# 10. RESULTADO FINAL
# =============================================================================

tempo_total = time.time() - inicio_global

print("\n" + "=" * 80)
print("RESULTADO FINAL — CRITICAL-POINT INVERTIBILITY TEST")
print("=" * 80)

print("\nPonto testado:")
print("  theta = pi/2")

print("\nEstrutura crítica:")
print(f"  rank(M_crit) = {rank_crit}")
print(f"  det(M_crit)  = {det_crit}")

print("\nInvertibilidade das operações:")
print(f"  Y^-1 existe              : {resultado['status'] == 'OK'}")
print(f"  Q = X Y^-1               : {resultado['status'] == 'OK'}")
print(f"  Q^-1 existe              : {resultado['status'] == 'OK'}")
print(f"  QY = X                   : {resultado['erro_QY'] < TOL_TEST if resultado['status'] == 'OK' else False}")
print(f"  Q^-1 X = Y               : {resultado['erro_QinvX'] < TOL_TEST if resultado['status'] == 'OK' else False}")
print(f"  (Q^-1)^-1 = Q            : {resultado['erro_double'] < TOL_TEST if resultado['status'] == 'OK' else False}")
print(f"  Q reconstruído nos 16 B_i: {resultado['erro_recon'] < TOL_TEST if resultado['status'] == 'OK' else False}")

print("\nMu:")
print(f"  mu reconstruído          : {mu_ok}")

print("\nResultado crítico:")
if resultado["passed"]:
    print("  ✓ MULTIPLICAÇÃO ↔ DIVISÃO INVERTÍVEIS NOS MODOS CRÍTICOS")
else:
    print("  ⚠ INVERTIBILIDADE NÃO CONFIRMADA")

print(f"\nTempo total: {tempo_total:.3f} s")

print("\n" + "=" * 80)
print("CONCLUSÃO")
print("=" * 80)

print("""
O teste foi formulado especificamente para theta = pi/2.

A matriz crítica M_crit pode ser singular.
Isso não é o objeto cuja inversibilidade está sendo
testada.

A pergunta é se os modos críticos X e Y geram uma
operação de divisão

        Q = X Y^-1

que permanece reversível por multiplicação e divisão.

Quando aprovado, o experimento verifica:

        QY = X
        Q^-1 X = Y
        (Q^-1)^-1 = Q

e a reconstrução de Q no espaço dos 16 geradores.

Este resultado é uma verificação computacional da
invertibilidade da multiplicação <-> divisão nos modos
críticos da representação Alpha.

Ele não constitui, por si só, uma prova de que a divisão
seja um novo colchete de Lie.
""")

print("=" * 80)
print("FIM")
print("=" * 80)
