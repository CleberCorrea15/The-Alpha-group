
# ============================================================================
# MOINHO DE CERVANTES 2 — MULTIPLICAÇÃO ↔ DIVISÃO
# GRUPO ALPHA — TESTE ESTATÍSTICO SOBRE OS 16 GERADORES
# ============================================================================
#
# OBJETIVO
#
# Para elementos invertíveis X e Y:
#
#       Q = X Y^(-1)
#
# verificar exatamente:
#
#       Q Y = X
#       Q^(-1) X = Y
#       (Q^(-1))^(-1) = Q
#
# Além disso:
#
#       Q
#
# deve permanecer dentro do espaço gerado pelos 16 geradores.
#
# Aritmética EXATA com SymPy.
# ============================================================================

import sympy as sp
import random
import time


# ============================================================================
# 1. GERADORES INICIAIS DO GRUPO ALPHA
# ============================================================================

I4 = sp.eye(4)

G_C = sp.Matrix([
    [0, -1,  0,  0],
    [1,  0,  0,  0],
    [0,  0,  0, -1],
    [0,  0,  1,  0]
])

G_T = sp.Matrix([
    [0,  0, -1,  0],
    [0,  0,  0, -1],
    [1,  0,  0,  0],
    [0,  1,  0,  0]
])

# ============================================================================
# G_mu
#
# Mantida como na versão corrigida do script anterior.
# ============================================================================

G_mu = sp.Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

G_const = sp.Matrix([
    [1,  0, 0, 1],
    [0,  1, 1, 0],
    [0, -1, 0, 0],
    [1,  0, 0, 0]
])

GENERATORS = [
    I4,
    G_C,
    G_T,
    G_mu,
    G_const
]


# ============================================================================
# 2. CONVERTER MATRIZES 4x4 EM VETORES DE 16 COMPONENTES
# ============================================================================

def matrix_to_columns(lista):

    return sp.Matrix.hstack(
        *[
            M.reshape(16, 1)
            for M in lista
        ]
    )


# ============================================================================
# 3. CONSTRUIR A BASE DE DIMENSÃO 16
# ============================================================================

def build_basis(generators):

    conjunto = list(generators)

    matriz = matrix_to_columns(conjunto)
    rank = matriz.rank()

    while True:

        adicionou = False

        # Base linear atual
        base_atual = [
            vec.reshape(4, 4)
            for vec in matrix_to_columns(
                conjunto
            ).columnspace()
        ]

        # Produtos entre elementos da base
        for A in base_atual:

            for B in base_atual:

                produto = A * B

                novo_rank = matrix_to_columns(
                    conjunto + [produto]
                ).rank()

                if novo_rank > rank:

                    conjunto.append(produto)
                    rank = novo_rank
                    adicionou = True

        if not adicionou:
            break

    Base = [
        vec.reshape(4, 4)
        for vec in matrix_to_columns(
            conjunto
        ).columnspace()
    ]

    return Base


# ============================================================================
# 4. DECOMPOSIÇÃO NA BASE
# ============================================================================

def decompose(M, Base):

    A = matrix_to_columns(Base)

    b = M.reshape(16, 1)

    sol = sp.linsolve(
        (A, b)
    )

    if sol == sp.EmptySet:

        raise ValueError(
            "Matriz fora do espaço gerado pelos 16 geradores."
        )

    vetor = list(sol)[0]

    return [
        sp.simplify(c)
        for c in vetor
    ]


# ============================================================================
# 5. RECONSTRUÇÃO A PARTIR DOS 16 GERADORES
# ============================================================================

def reconstruct(coeffs, Base):

    M = sp.zeros(4)

    for c, B in zip(coeffs, Base):

        M += c * B

    return sp.simplify(M)


# ============================================================================
# 6. GERAR ELEMENTO ALEATÓRIO DO ESPAÇO ALPHA
# ============================================================================

def random_alpha_element(Base, amplitude=3):

    coeffs = [
        random.randint(
            -amplitude,
            amplitude
        )
        for _ in range(16)
    ]

    # Evitar todos os coeficientes nulos
    if all(c == 0 for c in coeffs):

        indice = random.randrange(16)

        coeffs[indice] = 1

    M = reconstruct(
        coeffs,
        Base
    )

    return M, coeffs


# ============================================================================
# 7. GERAR ELEMENTO INVERTÍVEL
#
# IMPORTANTE:
#
# Para testar Q^(-1), precisamos garantir:
#
#       det(X) != 0
#       det(Y) != 0
#
# Portanto X e Y são sorteados até serem invertíveis.
# ============================================================================

def random_invertible_alpha_element(
    Base,
    amplitude=3,
    max_tentativas=1000
):

    for tentativa in range(
        1,
        max_tentativas + 1
    ):

        M, coeffs = random_alpha_element(
            Base,
            amplitude
        )

        det_M = sp.factor(
            M.det()
        )

        if det_M != 0:

            return (
                M,
                coeffs,
                det_M,
                tentativa
            )

    raise RuntimeError(
        "Não foi possível encontrar "
        "um elemento invertível após "
        f"{max_tentativas} tentativas."
    )


# ============================================================================
# 8. ERRO EXATO
# ============================================================================

def is_zero_matrix(M):

    return M == sp.zeros(
        M.rows,
        M.cols
    )


# ============================================================================
# 9. INÍCIO
# ============================================================================

print("=" * 80)
print("MOINHO 2 — MULTIPLICAÇÃO ↔ DIVISÃO")
print("GRUPO ALPHA")
print("=" * 80)

print(
    "\nConstruindo a base..."
)

Base = build_basis(
    GENERATORS
)

N = len(Base)

print(
    f"\nDimensão obtida: {N}"
)

if N != 16:

    print(
        "\nERRO: a base não possui dimensão 16."
    )

    raise SystemExit

print(
    "✓ Os 16 geradores foram reconstruídos."
)


# ============================================================================
# 10. CONFIGURAÇÃO
# ============================================================================

NUM_TESTES = 1000
AMPLITUDE = 3

print("\n" + "=" * 80)
print("CONFIGURAÇÃO")
print("=" * 80)

print(
    f"Casos de teste : {NUM_TESTES}"
)

print(
    f"Coeficientes   : inteiros entre "
    f"-{AMPLITUDE} e {AMPLITUDE}"
)

print(
    "Aritmética     : EXATA (SymPy)"
)

print(
    "Domínio        : X,Y invertíveis"
)


# ============================================================================
# 11. CONTADORES
# ============================================================================

reconstrucao_x_ok = 0
reconstrucao_y_ok = 0
reconstrucao_yinv_ok = 0
reconstrucao_q_ok = 0
reconstrucao_qinv_ok = 0

divisao_ok = 0
inversao_ok = 0
dupla_inversao_ok = 0

identidade_ok = 0

falhas = []

tentativas_x_total = 0
tentativas_y_total = 0

inicio = time.time()


# ============================================================================
# 12. MOINHO
# ============================================================================

for caso in range(
    1,
    NUM_TESTES + 1
):

    try:

        # ====================================================================
        # X INVERTÍVEL
        # ====================================================================

        X, coef_X, det_X, tent_X = (
            random_invertible_alpha_element(
                Base,
                AMPLITUDE
            )
        )

        tentativas_x_total += tent_X


        # ====================================================================
        # Y INVERTÍVEL
        # ====================================================================

        Y, coef_Y, det_Y, tent_Y = (
            random_invertible_alpha_element(
                Base,
                AMPLITUDE
            )
        )

        tentativas_y_total += tent_Y


        # ====================================================================
        # RECONSTRUÇÃO DE X
        # ====================================================================

        X_rec = reconstruct(
            coef_X,
            Base
        )

        if is_zero_matrix(
            sp.simplify(X_rec - X)
        ):

            reconstrucao_x_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "RECONSTRUCAO_X"
                )
            )


        # ====================================================================
        # RECONSTRUÇÃO DE Y
        # ====================================================================

        Y_rec = reconstruct(
            coef_Y,
            Base
        )

        if is_zero_matrix(
            sp.simplify(Y_rec - Y)
        ):

            reconstrucao_y_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "RECONSTRUCAO_Y"
                )
            )


        # ====================================================================
        # INVERSÃO DE Y
        # ====================================================================

        Y_inv = sp.simplify(
            Y.inv()
        )


        # ====================================================================
        # RECONSTRUIR Y^-1 NOS 16 GERADORES
        # ====================================================================

        coef_Y_inv = decompose(
            Y_inv,
            Base
        )

        Y_inv_rec = reconstruct(
            coef_Y_inv,
            Base
        )

        erro_Y_inv = sp.simplify(
            Y_inv_rec - Y_inv
        )

        if is_zero_matrix(
            erro_Y_inv
        ):

            reconstrucao_yinv_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "RECONSTRUCAO_Y_INV"
                )
            )


        # ====================================================================
        # DIVISÃO
        #
        # Q = X Y^-1
        # ====================================================================

        Q = sp.simplify(
            X * Y_inv
        )


        # ====================================================================
        # RECONSTRUIR Q
        # ====================================================================

        coef_Q = decompose(
            Q,
            Base
        )

        Q_rec = reconstruct(
            coef_Q,
            Base
        )

        erro_Q = sp.simplify(
            Q_rec - Q
        )

        if is_zero_matrix(
            erro_Q
        ):

            reconstrucao_q_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "RECONSTRUCAO_Q"
                )
            )


        # ====================================================================
        # TESTE FUNDAMENTAL
        #
        # QY = X
        # ====================================================================

        erro_QY = sp.simplify(
            Q * Y - X
        )

        if is_zero_matrix(
            erro_QY
        ):

            divisao_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "QY_NAO_E_X"
                )
            )


        # ====================================================================
        # INVERSÃO DA DIVISÃO
        #
        # Q^-1 X = Y
        # ====================================================================

        Q_inv = sp.simplify(
            Q.inv()
        )

        erro_QinvX = sp.simplify(
            Q_inv * X - Y
        )

        if is_zero_matrix(
            erro_QinvX
        ):

            inversao_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "Q_INV_X_NAO_E_Y"
                )
            )


        # ====================================================================
        # RECONSTRUIR Q^-1
        # ====================================================================

        coef_Q_inv = decompose(
            Q_inv,
            Base
        )

        Q_inv_rec = reconstruct(
            coef_Q_inv,
            Base
        )

        erro_Q_inv = sp.simplify(
            Q_inv_rec - Q_inv
        )

        if is_zero_matrix(
            erro_Q_inv
        ):

            reconstrucao_qinv_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "RECONSTRUCAO_Q_INV"
                )
            )


        # ====================================================================
        # DUPLA INVERSÃO
        #
        # (Q^-1)^-1 = Q
        # ====================================================================

        Q_inv_inv = sp.simplify(
            Q_inv.inv()
        )

        erro_dupla = sp.simplify(
            Q_inv_inv - Q
        )

        if is_zero_matrix(
            erro_dupla
        ):

            dupla_inversao_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "DUPLA_INVERSAO"
                )
            )


        # ====================================================================
        # IDENTIDADE DA DIVISÃO
        #
        # Q Y X^-1 = I
        # ====================================================================

        X_inv = sp.simplify(
            X.inv()
        )

        identidade = sp.simplify(
            Q * Y * X_inv
        )

        erro_identidade = sp.simplify(
            identidade - sp.eye(4)
        )

        if is_zero_matrix(
            erro_identidade
        ):

            identidade_ok += 1

        else:

            falhas.append(
                (
                    caso,
                    "IDENTIDADE"
                )
            )


    except Exception as erro:

        falhas.append(
            (
                caso,
                "EXCECAO",
                str(erro)
            )
        )


# ============================================================================
# 13. RESULTADOS
# ============================================================================

tempo = time.time() - inicio

print("\n" + "=" * 80)
print("RESULTADO DO MOINHO 2")
print("=" * 80)

print(
    f"\nCasos solicitados          : "
    f"{NUM_TESTES}"
)

print(
    f"Reconstrução de X          : "
    f"{reconstrucao_x_ok}/{NUM_TESTES}"
)

print(
    f"Reconstrução de Y          : "
    f"{reconstrucao_y_ok}/{NUM_TESTES}"
)

print(
    f"Y^-1 reconstruído          : "
    f"{reconstrucao_yinv_ok}/{NUM_TESTES}"
)

print(
    f"Q reconstruído             : "
    f"{reconstrucao_q_ok}/{NUM_TESTES}"
)

print(
    f"Q^-1 reconstruído          : "
    f"{reconstrucao_qinv_ok}/{NUM_TESTES}"
)

print(
    f"\nQY = X                     : "
    f"{divisao_ok}/{NUM_TESTES}"
)

print(
    f"Q^-1 X = Y                 : "
    f"{inversao_ok}/{NUM_TESTES}"
)

print(
    f"(Q^-1)^-1 = Q              : "
    f"{dupla_inversao_ok}/{NUM_TESTES}"
)

print(
    f"Q Y X^-1 = I               : "
    f"{identidade_ok}/{NUM_TESTES}"
)

print(
    f"\nFalhas encontradas         : "
    f"{len(falhas)}"
)

print(
    f"Tentativas médias para X  : "
    f"{tentativas_x_total / NUM_TESTES:.2f}"
)

print(
    f"Tentativas médias para Y  : "
    f"{tentativas_y_total / NUM_TESTES:.2f}"
)

print(
    f"Tempo total                : "
    f"{tempo:.2f} s"
)


# ============================================================================
# 14. TAXAS
# ============================================================================

print("\n" + "=" * 80)
print("TAXAS DE SUCESSO")
print("=" * 80)

def taxa(valor):

    return (
        100.0 * valor / NUM_TESTES
    )


print(
    f"Reconstrução X : "
    f"{taxa(reconstrucao_x_ok):.2f}%"
)

print(
    f"Reconstrução Y : "
    f"{taxa(reconstrucao_y_ok):.2f}%"
)

print(
    f"Y^-1           : "
    f"{taxa(reconstrucao_yinv_ok):.2f}%"
)

print(
    f"Q              : "
    f"{taxa(reconstrucao_q_ok):.2f}%"
)

print(
    f"Q^-1            : "
    f"{taxa(reconstrucao_qinv_ok):.2f}%"
)

print(
    f"QY = X           : "
    f"{taxa(divisao_ok):.2f}%"
)

print(
    f"Q^-1 X = Y       : "
    f"{taxa(inversao_ok):.2f}%"
)

print(
    f"Dupla inversão    : "
    f"{taxa(dupla_inversao_ok):.2f}%"
)

print(
    f"Identidade        : "
    f"{taxa(identidade_ok):.2f}%"
)


# ============================================================================
# 15. FALHAS
# ============================================================================

if len(falhas) == 0:

    print("\n" + "=" * 80)
    print("✓✓✓ NENHUMA FALHA ENCONTRADA ✓✓✓")
    print("=" * 80)

else:

    print("\n" + "=" * 80)
    print("FALHAS ENCONTRADAS")
    print("=" * 80)

    for falha in falhas[:20]:

        print(
            "\n",
            falha
        )


# ============================================================================
# 16. CONCLUSÃO AUTOMÁTICA
# ============================================================================

print("\n" + "=" * 80)
print("CONCLUSÃO COMPUTACIONAL")
print("=" * 80)

if (
    len(falhas) == 0
    and reconstrucao_x_ok == NUM_TESTES
    and reconstrucao_y_ok == NUM_TESTES
    and reconstrucao_yinv_ok == NUM_TESTES
    and reconstrucao_q_ok == NUM_TESTES
    and reconstrucao_qinv_ok == NUM_TESTES
    and divisao_ok == NUM_TESTES
    and inversao_ok == NUM_TESTES
    and dupla_inversao_ok == NUM_TESTES
    and identidade_ok == NUM_TESTES
):

    print("""
✓ 100% DOS CASOS FORAM APROVADOS.

✓ X pertence ao espaço dos 16 geradores.

✓ Y pertence ao espaço dos 16 geradores.

✓ Y^-1 pertence ao espaço dos 16 geradores.

✓ Q = X Y^-1 pertence ao espaço dos 16 geradores.

✓ Q^-1 pertence ao espaço dos 16 geradores.

✓ QY = X.

✓ Q^-1 X = Y.

✓ (Q^-1)^-1 = Q.

✓ Q Y X^-1 = I.

CONCLUSÃO:

A multiplicação e a divisão foram mutuamente
recuperadas em todos os casos testados, dentro
da representação matricial de dimensão 16.

ATENÇÃO:

Este experimento demonstra reconstrução e
inversibilidade dentro da representação matricial.
Não demonstra, por si só, que a divisão seja
um novo colchete de Lie.
""")

else:

    print("""
⚠ O experimento apresentou casos não aprovados.

Os casos de falha devem ser analisados antes
de qualquer conclusão estrutural.
""")


# ============================================================================
# 17. FIM
# ============================================================================

print("=" * 80)
print("FIM DO MOINHO 2")
print("=" * 80)
