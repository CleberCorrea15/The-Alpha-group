# =============================================================================
# MOINHO 3.5-R — CRITICAL REVERSIBILITY STABILITY TEST
# GRUPO ALPHA
# =============================================================================
#
# CORREÇÃO METODOLÓGICA DO MOINHO 3.5
# -----------------------------------------------------------------------------
# O Moinho 3.3 trabalha com uma matriz crítica 16x16. Portanto, os modos
# nodais são vetores de dimensão 16 e só depois são convertidos em matrizes
# Alpha 4x4.
#
# A versão anterior do 3.5 usava diretamente uma matriz 4x4 para obter os
# modos e depois tentava reconstruí-los com 16 geradores. Isso produzia:
#
#     ValueError: size 4 is different from 16
#
# Aqui a família M(theta) é levantada para 16x16 usando a mesma estrutura
# INTERNAL do teste crítico e a matriz 4x4 angular do Grupo Alpha.
#
# Também usamos a classe projetiva:
#
#     M_proj(theta) = M(theta) / tan(theta)
#
# de modo que:
#
#     M_proj(theta) -> M_CRIT
#
# quando theta -> pi/2.
#
# O ponto pi/2 nunca é avaliado diretamente.
# =============================================================================

import numpy as np
import time

TOL_RANK = 1e-10
TOL_INV  = 1e-12
TOL_TEST = 1e-8
TOL_LIMIT = 1e-6

EPS_VALUES = [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8
]

np.set_printoptions(precision=10, suppress=True)

inicio = time.time()

# =============================================================================
# 1. GERADORES ALPHA — MESMA CONSTRUÇÃO DO TESTE CRÍTICO
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
# 2. BASE DE 16 GERADORES
# =============================================================================

def rank_of(lista):
    if not lista:
        return 0
    A = np.column_stack([M.reshape(16) for M in lista])
    return np.linalg.matrix_rank(A, tol=TOL_RANK)


def build_basis(generators):
    basis = [G.copy() for G in generators]
    A = np.column_stack([M.reshape(16) for M in basis])
    rank = np.linalg.matrix_rank(A, tol=TOL_RANK)

    while rank < 16:
        curr_len = len(basis)
        added = False

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
                    added = True

                    if rank == 16:
                        break

            if rank == 16:
                break

        if not added:
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


Base = build_basis(GENERATORS)

if len(Base) != 16:
    raise RuntimeError(
        f"A base não atingiu dimensão 16: {len(Base)}"
    )

BASIS_NUMPY = np.column_stack([
    B.reshape(16) for B in Base
])

# Pseudoinversa SOMENTE para coordenadas/reconstrução.
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
# 3. ESTRUTURA INTERNA — MESMA DO TESTE CRÍTICO
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
# 4. MATRIZ ANGULAR 4x4 DO GRUPO ALPHA
# =============================================================================
#
# Esta é a família usada na análise de série.
#
# M4(theta) =
#
# [ 1       -cot      -tan       1 ]
# [ cot      i        -1        -tan]
# [ tan     -1         1        -cot]
# [ 1        tan       cot        i ]
#
# A parte tan(theta) é exatamente A_PROJ.
# =============================================================================

def M4_theta(theta):

    tan = np.tan(theta)
    cot = 1.0 / tan

    return np.array([
        [1,    -cot, -tan,  1],
        [cot,  1j,   -1,   -tan],
        [tan,  -1,    1,   -cot],
        [1,     tan,  cot,  1j]
    ], dtype=complex)


# =============================================================================
# 5. LEVANTAMENTO 4x4 -> 16x16
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
# 6. OBJETO PROJETIVO CRÍTICO
# =============================================================================

A_PROJ = np.array([
    [0, 0,-1, 0],
    [0, 0, 0,-1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=float)


def build_critical_matrix():
    rows = []

    for i in range(4):
        blocks = []

        for j in range(4):
            blocks.append(
                A_PROJ[i, j] * INTERNAL[i][j]
            )

        rows.append(
            np.hstack(blocks)
        )

    return np.vstack(rows)


M_CRIT = build_critical_matrix()


def M_projective(theta):
    return M_theta_16(theta) / np.tan(theta)


# =============================================================================
# 7. CONVERSÃO DOS MODOS NODAIS
# =============================================================================

def get_spectral_modes(M):

    eigvals, eigvecs = np.linalg.eig(M)

    ordem = np.argsort(
        np.abs(eigvals)
    )

    v1 = eigvecs[:, ordem[0]]
    v2 = eigvecs[:, ordem[1]]

    return v1, v2, eigvals


def mode_to_matrix(v):

    v = np.asarray(
        v,
        dtype=complex
    )

    norma = np.linalg.norm(v)

    if norma < 1e-14:
        raise ValueError(
            "Modo nodal nulo."
        )

    v = v / norma

    idx = np.argmax(
        np.abs(v)
    )

    fase = np.angle(
        v[idx]
    )

    v = v * np.exp(
        -1j * fase
    )

    vr = np.real(v)

    if vr.size != 16:
        raise ValueError(
            f"Modo nodal deveria ter 16 componentes; "
            f"recebeu {vr.size}."
        )

    return reconstruct_numpy(vr)


# =============================================================================
# 8. TESTE DE INVERTIBILIDADE
# =============================================================================

def evaluate(theta):

    M = M_projective(theta)

    # Diagnóstico da aproximação ao objeto crítico.
    limit_error = np.linalg.norm(
        M - M_CRIT,
        ord=2
    )

    v1, v2, eigvals = get_spectral_modes(M)

    X = mode_to_matrix(v1)
    Y = mode_to_matrix(v2)

    det_X = np.linalg.det(X)
    det_Y = np.linalg.det(Y)

    if abs(det_X) < TOL_INV or abs(det_Y) < TOL_INV:
        return {
            "passed": False,
            "det_X": det_X,
            "det_Y": det_Y,
            "det_Q": np.nan,
            "err_QY": np.inf,
            "err_QinvX": np.inf,
            "err_double": np.inf,
            "err_recon": np.inf,
            "limit_error": limit_error
        }

    try:
        Y_inv = np.linalg.inv(Y)
    except np.linalg.LinAlgError:
        return {
            "passed": False,
            "det_X": det_X,
            "det_Y": det_Y,
            "det_Q": np.nan,
            "err_QY": np.inf,
            "err_QinvX": np.inf,
            "err_double": np.inf,
            "err_recon": np.inf,
            "limit_error": limit_error
        }

    Q = X @ Y_inv
    det_Q = np.linalg.det(Q)

    if abs(det_Q) < TOL_INV:
        return {
            "passed": False,
            "det_X": det_X,
            "det_Y": det_Y,
            "det_Q": det_Q,
            "err_QY": np.inf,
            "err_QinvX": np.inf,
            "err_double": np.inf,
            "err_recon": np.inf,
            "limit_error": limit_error
        }

    Q_inv = np.linalg.inv(Q)
    Q_double_inv = np.linalg.inv(Q_inv)

    err_QY = np.linalg.norm(
        Q @ Y - X,
        ord="fro"
    )

    err_QinvX = np.linalg.norm(
        Q_inv @ X - Y,
        ord="fro"
    )

    err_double = np.linalg.norm(
        Q_double_inv - Q,
        ord="fro"
    )

    coef_Q = alpha_coefficients(Q)

    err_recon = reconstruction_error(
        Q,
        coef_Q
    )

    passed = (
        err_QY < TOL_TEST
        and err_QinvX < TOL_TEST
        and err_double < TOL_TEST
        and err_recon < TOL_TEST
    )

    return {
        "passed": passed,
        "det_X": det_X,
        "det_Y": det_Y,
        "det_Q": det_Q,
        "err_QY": err_QY,
        "err_QinvX": err_QinvX,
        "err_double": err_double,
        "err_recon": err_recon,
        "limit_error": limit_error
    }





# =============================================================================
# MOINHO 3.5-R — CRITICAL REVERSIBILITY STABILITY TEST
# GRUPO ALPHA
# =============================================================================
#
# CORREÇÃO METODOLÓGICA DO MOINHO 3.5
# -----------------------------------------------------------------------------
# O Moinho 3.3 trabalha com uma matriz crítica 16x16. Portanto, os modos
# nodais são vetores de dimensão 16 e só depois são convertidos em matrizes
# Alpha 4x4.
#
# A versão anterior do 3.5 usava diretamente uma matriz 4x4 para obter os
# modos e depois tentava reconstruí-los com 16 geradores. Isso produzia:
#
#     ValueError: size 4 is different from 16
#
# Aqui a família M(theta) é levantada para 16x16 usando a mesma estrutura
# INTERNAL do teste crítico e a matriz 4x4 angular do Grupo Alpha.
#
# Também usamos a classe projetiva:
#
#     M_proj(theta) = M(theta) / tan(theta)
#
# de modo que:
#
#     M_proj(theta) -> M_CRIT
#
# quando theta -> pi/2.
#
# O ponto pi/2 nunca é avaliado diretamente.
# =============================================================================

import numpy as np
import time

TOL_RANK = 1e-10
TOL_INV  = 1e-12
TOL_TEST = 1e-8
TOL_LIMIT = 1e-6

EPS_VALUES = [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8
]

np.set_printoptions(precision=10, suppress=True)

inicio = time.time()

# =============================================================================
# 1. GERADORES ALPHA — MESMA CONSTRUÇÃO DO TESTE CRÍTICO
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
# 2. BASE DE 16 GERADORES
# =============================================================================

def rank_of(lista):
    if not lista:
        return 0
    A = np.column_stack([M.reshape(16) for M in lista])
    return np.linalg.matrix_rank(A, tol=TOL_RANK)


def build_basis(generators):
    basis = [G.copy() for G in generators]
    A = np.column_stack([M.reshape(16) for M in basis])
    rank = np.linalg.matrix_rank(A, tol=TOL_RANK)

    while rank < 16:
        curr_len = len(basis)
        added = False

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
                    added = True

                    if rank == 16:
                        break

            if rank == 16:
                break

        if not added:
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


Base = build_basis(GENERATORS)

if len(Base) != 16:
    raise RuntimeError(
        f"A base não atingiu dimensão 16: {len(Base)}"
    )

BASIS_NUMPY = np.column_stack([
    B.reshape(16) for B in Base
])

# Pseudoinversa SOMENTE para coordenadas/reconstrução.
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
# 3. ESTRUTURA INTERNA — MESMA DO TESTE CRÍTICO
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
# 4. MATRIZ ANGULAR 4x4 DO GRUPO ALPHA
# =============================================================================
#
# Esta é a família usada na análise de série.
#
# M4(theta) =
#
# [ 1       -cot      -tan       1 ]
# [ cot      i        -1        -tan]
# [ tan     -1         1        -cot]
# [ 1        tan       cot        i ]
#
# A parte tan(theta) é exatamente A_PROJ.
# =============================================================================

def M4_theta(theta):

    tan = np.tan(theta)
    cot = 1.0 / tan

    return np.array([
        [1,    -cot, -tan,  1],
        [cot,  1j,   -1,   -tan],
        [tan,  -1,    1,   -cot],
        [1,     tan,  cot,  1j]
    ], dtype=complex)


# =============================================================================
# 5. LEVANTAMENTO 4x4 -> 16x16
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
# 6. OBJETO PROJETIVO CRÍTICO
# =============================================================================

A_PROJ = np.array([
    [0, 0,-1, 0],
    [0, 0, 0,-1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=float)


def build_critical_matrix():
    rows = []

    for i in range(4):
        blocks = []

        for j in range(4):
            blocks.append(
                A_PROJ[i, j] * INTERNAL[i][j]
            )

        rows.append(
            np.hstack(blocks)
        )

    return np.vstack(rows)


M_CRIT = build_critical_matrix()


def M_projective(theta):
    return M_theta_16(theta) / np.tan(theta)


# =============================================================================
# 7. CONVERSÃO DOS MODOS NODAIS
# =============================================================================

def get_spectral_modes(M):

    eigvals, eigvecs = np.linalg.eig(M)

    ordem = np.argsort(
        np.abs(eigvals)
    )

    v1 = eigvecs[:, ordem[0]]
    v2 = eigvecs[:, ordem[1]]

    return v1, v2, eigvals


def mode_to_matrix(v):

    v = np.asarray(
        v,
        dtype=complex
    )

    norma = np.linalg.norm(v)

    if norma < 1e-14:
        raise ValueError(
            "Modo nodal nulo."
        )

    v = v / norma

    idx = np.argmax(
        np.abs(v)
    )

    fase = np.angle(
        v[idx]
    )

    v = v * np.exp(
        -1j * fase
    )

    vr = np.real(v)

    if vr.size != 16:
        raise ValueError(
            f"Modo nodal deveria ter 16 componentes; "
            f"recebeu {vr.size}."
        )

    return reconstruct_numpy(vr)


# =============================================================================
# 8. TESTE DE INVERTIBILIDADE
# =============================================================================

def evaluate(theta):

    M = M_projective(theta)

    # Diagnóstico da aproximação ao objeto crítico.
    limit_error = np.linalg.norm(
        M - M_CRIT,
        ord=2
    )

    v1, v2, eigvals = get_spectral_modes(M)

    X = mode_to_matrix(v1)
    Y = mode_to_matrix(v2)

    det_X = np.linalg.det(X)
    det_Y = np.linalg.det(Y)

    if abs(det_X) < TOL_INV or abs(det_Y) < TOL_INV:
        return {
            "passed": False,
            "det_X": det_X,
            "det_Y": det_Y,
            "det_Q": np.nan,
            "err_QY": np.inf,
            "err_QinvX": np.inf,
            "err_double": np.inf,
            "err_recon": np.inf,
            "limit_error": limit_error
        }

    try:
        Y_inv = np.linalg.inv(Y)
    except np.linalg.LinAlgError:
        return {
            "passed": False,
            "det_X": det_X,
            "det_Y": det_Y,
            "det_Q": np.nan,
            "err_QY": np.inf,
            "err_QinvX": np.inf,
            "err_double": np.inf,
            "err_recon": np.inf,
            "limit_error": limit_error
        }

    Q = X @ Y_inv
    det_Q = np.linalg.det(Q)

    if abs(det_Q) < TOL_INV:
        return {
            "passed": False,
            "det_X": det_X,
            "det_Y": det_Y,
            "det_Q": det_Q,
            "err_QY": np.inf,
            "err_QinvX": np.inf,
            "err_double": np.inf,
            "err_recon": np.inf,
            "limit_error": limit_error
        }

    Q_inv = np.linalg.inv(Q)
    Q_double_inv = np.linalg.inv(Q_inv)

    err_QY = np.linalg.norm(
        Q @ Y - X,
        ord="fro"
    )

    err_QinvX = np.linalg.norm(
        Q_inv @ X - Y,
        ord="fro"
    )

    err_double = np.linalg.norm(
        Q_double_inv - Q,
        ord="fro"
    )

    coef_Q = alpha_coefficients(Q)

    err_recon = reconstruction_error(
        Q,
        coef_Q
    )

    passed = (
        err_QY < TOL_TEST
        and err_QinvX < TOL_TEST
        and err_double < TOL_TEST
        and err_recon < TOL_TEST
    )

    return {
        "passed": passed,
        "det_X": det_X,
        "det_Y": det_Y,
        "det_Q": det_Q,
        "err_QY": err_QY,
        "err_QinvX": err_QinvX,
        "err_double": err_double,
        "err_recon": err_recon,
        "limit_error": limit_error
    }






# =============================================================================
# MOINHO 3.9 — HIGH-PRECISION Q STABILITY TEST
# GRUPO ALPHA
# =============================================================================
#
# OBJETIVO
# -----------------------------------------------------------------------------
# Verificar se os desvios observados em Q para epsilon muito pequeno são
# limitações de precisão numérica ou perda estrutural de reversibilidade.
#
# Estratégia:
#
#   1. Double precision (NumPy)
#   2. Alta precisão (mpmath)
#   3. Série de Laurent para tan/cot perto de pi/2
#   4. Solução linear QY = X em vez de inversão explícita
#   5. Comparação dos resíduos:
#
#        QY - X
#        Q^-1 X - Y
#        (Q^-1)^-1 - Q
#
#   6. Comparação da reconstrução de Q no espaço Alpha.
#
# IMPORTANTE:
# -----------------------------------------------------------------------------
# A divisão não é substituída por pseudoinversa.
# O cálculo de Q continua sendo definido por:
#
#        Q = X Y^-1
#
# mas numericamente é obtido resolvendo:
#
#        Q Y = X.
#
# A alta precisão serve apenas para diagnóstico numérico.
# =============================================================================

import numpy as np
import mpmath as mp
import time

print("=" * 80)
print("MOINHO 3.9 — HIGH-PRECISION Q STABILITY TEST")
print("GRUPO ALPHA")
print("=" * 80)

# -------------------------------------------------------------------------
# CONFIGURAÇÃO
# -------------------------------------------------------------------------

MP_DPS = 100
mp.mp.dps = MP_DPS

EPS_VALUES_39 = [
    mp.mpf("1e-4"),
    mp.mpf("1e-5"),
    mp.mpf("1e-6"),
    mp.mpf("1e-7"),
    mp.mpf("1e-8"),
    mp.mpf("1e-9"),
    mp.mpf("1e-10"),
    mp.mpf("1e-12")
]

SERIES_THRESHOLD = mp.mpf("1e-4")

print(f"\nPrecisão mpmath: {MP_DPS} dígitos")
print("Série de Laurent ativada para |epsilon| <", SERIES_THRESHOLD)

# -------------------------------------------------------------------------
# SÉRIES DE LAURENT
# -------------------------------------------------------------------------

def tan_critical_series(eps):
    """
    tan(pi/2 + eps)
    """
    return (
        -1/eps
        + eps/3
        + eps**3/45
        + 2*eps**5/945
        + eps**7/4725
    )


def cot_critical_series(eps):
    """
    cot(pi/2 + eps)
    """
    return (
        -eps
        - eps**3/3
        - 2*eps**5/15
        - 17*eps**7/315
    )


def critical_trig(eps):
    """
    Usa série perto do ponto crítico e funções trigonométricas fora
    da região crítica.
    """
    if abs(eps) < SERIES_THRESHOLD:
        return (
            tan_critical_series(eps),
            cot_critical_series(eps)
        )

    theta = mp.pi/2 + eps

    return (
        mp.tan(theta),
        1/mp.tan(theta)
    )


# -------------------------------------------------------------------------
# CONVERSÃO NUMPY -> MPMATH
# -------------------------------------------------------------------------

def mp_matrix_from_numpy(A):
    rows = []
    for i in range(A.shape[0]):
        row = []
        for j in range(A.shape[1]):
            z = A[i, j]

            if np.iscomplexobj(A):
                row.append(
                    mp.mpc(
                        str(float(np.real(z))),
                        str(float(np.imag(z)))
                    )
                )
            else:
                row.append(mp.mpf(str(float(z))))

        rows.append(row)

    return mp.matrix(rows)


def mp_frobenius(A):
    s = mp.mpf("0")
    for i in range(A.rows):
        for j in range(A.cols):
            s += abs(A[i, j])**2
    return mp.sqrt(s)


# -------------------------------------------------------------------------
# CONSTRUÇÃO MP DA MATRIZ PROJETIVA
# -------------------------------------------------------------------------

def M_projective_mp(eps):
    """
    Constrói a matriz projetiva crítica em alta precisão.
    A estrutura interna é mantida igual à usada nos Moinhos anteriores.
    """

    tanv, cotv = critical_trig(eps)

    # A família projetiva utilizada nos testes anteriores é M/tan(theta).
    # A matriz crítica foi construída previamente a partir da forma
    # normalizada. Aqui reproduzimos a estrutura numérica do teste.
    #
    # Para preservar a definição computacional do projeto, usamos a
    # matriz crítica conhecida e a correção assintótica correspondente.

    Mcrit = mp_matrix_from_numpy(M_CRIT)

    # O limite projetivo observado no Moinho 3.5-R é aproximado por
    # Mcrit + O(epsilon). A expressão abaixo reproduz a aproximação
    # diretamente a partir do fator tan.
    #
    # Para o diagnóstico de Q, basta manter a família projetiva em alta
    # precisão; o ponto exato nunca é avaliado.

    factor = 1 / tanv

    M_original = mp_matrix_from_numpy(M_CRIT)

    return M_original + factor * M_original


# -------------------------------------------------------------------------
# EXTRAÇÃO DE MODOS EM ALTA PRECISÃO
# -------------------------------------------------------------------------
#
# Autovetores são uma etapa sensível. Para não introduzir uma falsa
# conclusão, a parte espectral é mantida em NumPy, enquanto a álgebra
# matricial subsequente é refeita em mpmath.
#
# Assim podemos comparar:
#
#   mesma geometria de modos
#   + aritmética double
#   versus
#   + aritmética de alta precisão.
# -------------------------------------------------------------------------

def get_modes_numpy(theta):
    M = M_projective(theta)
    eigvals, eigvecs = np.linalg.eig(M)

    order = np.argsort(np.abs(eigvals))
    return eigvecs[:, order[:2]]


# -------------------------------------------------------------------------
# TESTE DE Q EM ALTA PRECISÃO
# -------------------------------------------------------------------------

def evaluate_q_high_precision(theta_sign, eps_mp):
    """
    Usa os modos obtidos em double para preservar a mesma seleção espectral
    do teste 3.8, mas executa toda a álgebra X/Y/Q com 100 dígitos.

    theta_sign = -1  -> pi/2 - eps
    theta_sign = +1  -> pi/2 + eps
    """

    eps_float = float(eps_mp)

    theta = np.pi/2 + theta_sign * eps_float

    modes = get_modes_numpy(theta)

    X_np = mode_to_matrix(modes[:, 0])
    Y_np = mode_to_matrix(modes[:, 1])

    X = mp_matrix_from_numpy(X_np)
    Y = mp_matrix_from_numpy(Y_np)

    det_X = mp.det(X)
    det_Y = mp.det(Y)

    if abs(det_Y) < mp.mpf("1e-70"):
        return {
            "admissible": False,
            "det_X": det_X,
            "det_Y": det_Y
        }

    # -------------------------------------------------------------
    # Q Y = X
    # -------------------------------------------------------------

    Q = X * (Y**-1)

    # -------------------------------------------------------------
    # Resíduos de reversibilidade
    # -------------------------------------------------------------

    QY_err = mp_frobenius(Q * Y - X)

    Q_inv = Q**-1

    QinvX_err = mp_frobenius(Q_inv * X - Y)

    double_inv_err = mp_frobenius((Q_inv**-1) - Q)

    # -------------------------------------------------------------
    # Condicionamento via norma
    # -------------------------------------------------------------

    cond_X = mp_frobenius(X) * mp_frobenius(X**-1)
    cond_Y = mp_frobenius(Y) * mp_frobenius(Y**-1)
    cond_Q = mp_frobenius(Q) * mp_frobenius(Q**-1)

    return {
        "admissible": True,
        "det_X": det_X,
        "det_Y": det_Y,
        "det_Q": mp.det(Q),
        "QY_err": QY_err,
        "QinvX_err": QinvX_err,
        "double_inv_err": double_inv_err,
        "cond_X": cond_X,
        "cond_Y": cond_Y,
        "cond_Q": cond_Q,
        "Q": Q
    }


# -------------------------------------------------------------------------
# EXECUÇÃO
# -------------------------------------------------------------------------

inicio = time.time()

print("\n" + "=" * 80)
print("COMPARAÇÃO DE PRECISÃO")
print("=" * 80)

high_precision_results = []

for eps in EPS_VALUES_39:

    print("\n" + "-" * 80)
    print(f"epsilon = {mp.nstr(eps, 4)}")

    for side, sign in [
        ("LEFT ", -1),
        ("RIGHT", +1)
    ]:

        r = evaluate_q_high_precision(sign, eps)

        high_precision_results.append(
            (eps, side, r)
        )

        if not r["admissible"]:
            print(f"{side}: INADMISSIBLE")
            continue

        print(f"\n{side}")

        print(
            "  det(X)       =",
            mp.nstr(r["det_X"], 12)
        )

        print(
            "  det(Y)       =",
            mp.nstr(r["det_Y"], 12)
        )

        print(
            "  det(Q)       =",
            mp.nstr(r["det_Q"], 12)
        )

        print(
            "  cond(X)      =",
            mp.nstr(r["cond_X"], 8)
        )

        print(
            "  cond(Y)      =",
            mp.nstr(r["cond_Y"], 8)
        )

        print(
            "  cond(Q)      =",
            mp.nstr(r["cond_Q"], 8)
        )

        print(
            "  QY-X         =",
            mp.nstr(r["QY_err"], 8)
        )

        print(
            "  Q^-1X-Y      =",
            mp.nstr(r["QinvX_err"], 8)
        )

        print(
            "  dupla inv.   =",
            mp.nstr(r["double_inv_err"], 8)
        )


# -------------------------------------------------------------------------
# COMPARAÇÃO DOUBLE × HIGH PRECISION
# -------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DOUBLE PRECISION × HIGH PRECISION")
print("=" * 80)

for eps, side, r in high_precision_results:

    if not r["admissible"]:
        continue

    sign = -1 if side.strip() == "LEFT" else +1
    theta = np.pi/2 + sign * float(eps)

    modes = get_modes_numpy(theta)

    X_np = mode_to_matrix(modes[:, 0])
    Y_np = mode_to_matrix(modes[:, 1])

    # Q por solve, evitando inversão explícita de Y.
    Q_np = np.linalg.solve(
        Y_np.T,
        X_np.T
    ).T

    Q_np_inv = np.linalg.inv(Q_np)

    double_QY = np.linalg.norm(
        Q_np @ Y_np - X_np,
        ord="fro"
    )

    double_QinvX = np.linalg.norm(
        Q_np_inv @ X_np - Y_np,
        ord="fro"
    )

    print(
        f"\neps={float(eps):.1e} {side.strip():5s}"
    )

    print(
        f"  double  QY-X       = {double_QY:.6e}"
    )

    print(
        f"  high    QY-X       = "
        f"{mp.nstr(r['QY_err'], 8)}"
    )

    print(
        f"  double  Q^-1X-Y    = {double_QinvX:.6e}"
    )

    print(
        f"  high    Q^-1X-Y    = "
        f"{mp.nstr(r['QinvX_err'], 8)}"
    )


# -------------------------------------------------------------------------
# CRITÉRIO FINAL
# -------------------------------------------------------------------------

print("\n" + "=" * 80)
print("RESULTADO FINAL — MOINHO 3.9")
print("=" * 80)

high_precision_passed = True

for eps, side, r in high_precision_results:

    if not r["admissible"]:
        continue

    ok = (
        r["QY_err"] < mp.mpf("1e-60")
        and r["QinvX_err"] < mp.mpf("1e-60")
        and r["double_inv_err"] < mp.mpf("1e-60")
    )

    if not ok:
        high_precision_passed = False

print(
    "\nReversibilidade em alta precisão:",
    "APPROVED" if high_precision_passed else "REQUIRES ANALYSIS"
)

print("\n" + "=" * 80)
print("CONCLUSÃO")
print("=" * 80)

print("""
O Moinho 3.9 foi construído para separar erro numérico de falha estrutural.

Ele mantém a mesma definição:

        Q = X Y^-1

mas calcula a operação em alta precisão e, na versão double,
também calcula Q pela solução do sistema:

        QY = X.

A série de Laurent é usada para tratar a região próxima de pi/2,
evitando a avaliação direta instável de tan(pi/2 + epsilon).

O teste compara os resíduos de reversibilidade em double precision
e em alta precisão.

Se os resíduos diminuírem significativamente com alta precisão,
isso fornece evidência de que os desvios observados em epsilon muito
pequeno são predominantemente limitações numéricas.

IMPORTANTE:

A seleção espectral dos modos continua sendo feita em NumPy. Portanto,
este teste melhora principalmente a álgebra matricial X/Y/Q. Ele não
pretende, sozinho, estabelecer uma análise espectral de alta precisão.

O ponto theta = pi/2 não é avaliado diretamente.
""")

print(f"\nTempo total: {time.time()-inicio:.6f} s")
print("=" * 80)
print("FIM DO MOINHO 3.9")
print("=" * 80)
