# =============================================================================
# ALPHA GROUP — ANÁLISE DOS 120 COLCHETES + RELAÇÕES ESPECTRAIS
# =============================================================================
#
# OBJETIVO:
#   1. Construir os 16 geradores B1,...,B16
#   2. Calcular os 120 colchetes [Bi,Bj]
#   3. Separar 24 comutativos e 96 não-comutativos
#   4. Calcular os autovalores dos 16 geradores
#   5. Calcular os autovalores dos 96 colchetes não-comutativos
#   6. Detectar estruturas espectrais complexas
#   7. Procurar uma possível decomposição:
#
#               96 = 64 + 32
#
#   IMPORTANTE:
#   O programa NÃO assume que existam 32 relações espectrais.
#   Ele testa essa hipótese.
# =============================================================================

import sympy as sp
import numpy as np
from collections import Counter, defaultdict

np.set_printoptions(precision=8, suppress=True)

print("="*80)
print("ALPHA GROUP — 120 COLCHETES E ESTRUTURA ESPECTRAL")
print("="*80)


# =============================================================================
# 1. DEFINIÇÃO DOS 16 GERADORES
# =============================================================================

B = {}

B[1]  = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B[2]  = sp.Matrix([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 0,-1],
    [0,  0, 1, 0]
])

B[3]  = sp.Matrix([
    [0, 0,-1, 0],
    [0, 0, 0,-1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

B[4]  = sp.Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B[5]  = sp.Matrix([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0,-1, 0, 0],
    [1, 0, 0, 0]
])

B[6]  = sp.Matrix([
    [0, 0, 0, 1],
    [0, 0,-1, 0],
    [0,-1, 0, 0],
    [1, 0, 0, 0]
])

B[7]  = sp.Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0,-1],
    [0, 0, 1, 0]
])

B[8]  = sp.Matrix([
    [0,-1,-1, 0],
    [1, 0, 0, 1],
    [-1,0, 0, 0],
    [0,-1, 0, 0]
])

B[9]  = sp.Matrix([
    [0, 0,-1, 0],
    [0, 0, 0,-1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

B[10] = sp.Matrix([
    [0, 1, 0, 0],
    [-1,0, 0, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])

B[11] = sp.Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0,-1, 0, 0],
    [1, 0, 0, 0]
])

B[12] = sp.Matrix([
    [0, 1,-1, 0],
    [1, 0, 0,-1],
    [0, 0, 0, 1],
    [0, 0,-1, 0]
])

B[13] = sp.Matrix([
    [2, 0, 0, 1],
    [0, 0, 1, 0],
    [0,-1,-1, 0],
    [1, 0, 0, 1]
])

B[14] = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0,-1,-1, 0],
    [1, 0, 0, 1]
])

B[15] = sp.Matrix([
    [0, 0, 0,-1],
    [0, 0, 1, 0],
    [0, 1,-1, 0],
    [1, 0, 0,-1]
])

B[16] = sp.Matrix([
    [0, 1, 1, 0],
    [-1,0, 0,-1],
    [2, 0, 0, 1],
    [0, 0, 1, 0]
])


# =============================================================================
# 2. MATRIZ DA BASE E VERIFICAÇÃO DE RANK
# =============================================================================

def vec(M):
    return sp.Matrix(list(M))

M_base = sp.Matrix.hstack(*[vec(B[i]) for i in range(1,17)])

rank = M_base.rank()

print("\n" + "="*80)
print("1. DIMENSÃO DA BASE")
print("="*80)
print(f"Rank dos 16 geradores = {rank}")

if rank == 16:
    print("✓ Os 16 geradores são linearmente independentes.")
else:
    print("⚠ ATENÇÃO: rank diferente de 16.")


# Inversa calculada uma única vez
M_inv = M_base.inv()


# =============================================================================
# 3. FUNÇÕES
# =============================================================================

def bracket(A, C):
    """Colchete de Lie [A,C] = AC - CA"""
    return A*C - C*A


def coordinates(M):
    """Coordenadas de M na base B1,...,B16."""
    return sp.simplify(M_inv * vec(M))


def numerical_eigenvalues(M):
    """Autovalores numéricos."""
    A = np.array(M.tolist(), dtype=float)
    return np.linalg.eigvals(A)


def is_complex_eigenvalue(z, tol=1e-8):
    return abs(np.imag(z)) > tol


def conjugate_pair_exists(eigs, tol=1e-7):
    """
    Verifica se existe pelo menos um par lambda, conjugate(lambda).
    Para matrizes reais isso é esperado quando há autovalores complexos.
    """
    complex_vals = [z for z in eigs if abs(np.imag(z)) > tol]

    for z in complex_vals:
        target = np.conjugate(z)

        if any(abs(w-target) < tol for w in complex_vals):
            return True

    return False


def spectral_signature(eigs, tol=1e-7):
    """
    Classificação simples do espectro.
    """
    real_count = 0
    complex_count = 0

    for z in eigs:
        if abs(np.imag(z)) < tol:
            real_count += 1
        else:
            complex_count += 1

    if complex_count == 0:
        return "REAL"

    if complex_count == 2:
        return "1 PAR COMPLEXO"

    if complex_count == 4:
        return "2 PARES COMPLEXOS"

    return f"{complex_count} AUTOVALORES COMPLEXOS"


# =============================================================================
# 4. GERAR OS 120 COLCHETES
# =============================================================================

print("\n" + "="*80)
print("2. GERANDO OS 120 COLCHETES")
print("="*80)

all_brackets = {}
commutative = {}
noncommutative = {}

for i in range(1,17):
    for j in range(i+1,17):

        C = bracket(B[i], B[j])

        key = (i,j)

        all_brackets[key] = C

        if C == sp.zeros(4):
            commutative[key] = C
        else:
            noncommutative[key] = C


print(f"Total de pares               = {len(all_brackets)}")
print(f"Comutativos                  = {len(commutative)}")
print(f"Não-comutativos              = {len(noncommutative)}")

print("\nVerificação:")
print(f"{len(commutative)} + {len(noncommutative)} = "
      f"{len(commutative)+len(noncommutative)}")

if len(commutative) == 24 and len(noncommutative) == 96:
    print("✓ CONFIRMADO: 24 + 96 = 120")
else:
    print("⚠ A distribuição encontrada é diferente.")


# =============================================================================
# 5. LISTAR OS 24 COMUTATIVOS
# =============================================================================

print("\n" + "="*80)
print("3. OS 24 COLCHETES COMUTATIVOS")
print("="*80)

for (i,j) in commutative:
    print(f"[B{i:02d}, B{j:02d}] = 0")


# =============================================================================
# 6. ESPECTRO DOS 16 GERADORES
# =============================================================================

print("\n" + "="*80)
print("4. ESPECTRO DOS 16 GERADORES")
print("="*80)

generator_spectra = {}

for i in range(1,17):

    eigs = numerical_eigenvalues(B[i])
    generator_spectra[i] = eigs

    signature = spectral_signature(eigs)
    conjugate = conjugate_pair_exists(eigs)

    print(f"\nB{i:02d}")
    print(f"   autovalores = {eigs}")
    print(f"   classe      = {signature}")
    print(f"   par conjugado complexo? {conjugate}")


# =============================================================================
# 7. CONTAGEM DOS AUTOVALORES COMPLEXOS DOS 16 GERADORES
# =============================================================================

complex_eigenvalue_count = 0
complex_pairs = 0

for i,eigs in generator_spectra.items():

    complex_vals = [
        z for z in eigs
        if abs(np.imag(z)) > 1e-8
    ]

    complex_eigenvalue_count += len(complex_vals)

    if len(complex_vals) >= 2:
        complex_pairs += len(complex_vals)//2


print("\n" + "="*80)
print("5. CONTAGEM ESPECTRAL DOS GERADORES")
print("="*80)

print(f"Autovalores complexos encontrados = {complex_eigenvalue_count}")
print(f"Pares complexos                 = {complex_pairs}")

print("\nIMPORTANTE:")
print("O número de autovalores complexos NÃO é automaticamente")
print("o número de relações espectrais entre os colchetes.")


# =============================================================================
# 8. ESPECTRO DOS 96 COLCHETES NÃO-COMUTATIVOS
# =============================================================================

print("\n" + "="*80)
print("6. ESPECTRO DOS 96 COLCHETES NÃO-COMUTATIVOS")
print("="*80)

bracket_spectra = {}
spectral_brackets = {}
real_brackets = {}

for (i,j), C in noncommutative.items():

    eigs = numerical_eigenvalues(C)

    bracket_spectra[(i,j)] = eigs

    signature = spectral_signature(eigs)
    conjugate = conjugate_pair_exists(eigs)

    if conjugate:
        spectral_brackets[(i,j)] = eigs
    else:
        real_brackets[(i,j)] = eigs

    print(f"[B{i:02d}, B{j:02d}]")
    print(f"    espectro = {eigs}")
    print(f"    classe   = {signature}")
    print(f"    espectral = {conjugate}")


# =============================================================================
# 9. CONTAGEM FUNDAMENTAL
# =============================================================================

print("\n" + "="*80)
print("7. RESULTADO FUNDAMENTAL DA ANÁLISE ESPECTRAL")
print("="*80)

N_spectral = len(spectral_brackets)
N_other = len(real_brackets)

print(f"Não-comutativos totais       = {len(noncommutative)}")
print(f"Com estrutura espectral      = {N_spectral}")
print(f"Sem estrutura espectral      = {N_other}")

print("\nTeste da hipótese:")
print(f"96 = {N_other} + {N_spectral}")


if N_spectral == 32:

    print("\n" + "★"*80)
    print("RESULTADO:")
    print("✓ FORAM ENCONTRADAS EXATAMENTE 32 RELAÇÕES ESPECTRAIS")
    print("✓ A HIPÓTESE 96 = 64 + 32 É CONFIRMADA NUMERICAMENTE")
    print("★"*80)

elif N_spectral < 32:

    print("\nRESULTADO:")
    print(f"⚠ Foram encontradas apenas {N_spectral} relações espectrais.")
    print("A hipótese de 32 não é confirmada por este critério.")

else:

    print("\nRESULTADO:")
    print(f"⚠ Foram encontradas {N_spectral} relações espectrais.")
    print("O critério utilizado identifica mais de 32.")


# =============================================================================
# 10. LISTA DAS RELAÇÕES ESPECTRAIS
# =============================================================================

print("\n" + "="*80)
print("8. RELAÇÕES CLASSIFICADAS COMO ESPECTRAIS")
print("="*80)

if len(spectral_brackets) == 0:

    print("Nenhuma encontrada.")

else:

    for (i,j), eigs in spectral_brackets.items():

        print(f"\n[B{i:02d}, B{j:02d}]")

        for k,z in enumerate(eigs,1):
            print(f"    λ{k} = {z}")


# =============================================================================
# 11. AGRUPAMENTO POR PADRÃO ESPECTRAL
# =============================================================================

print("\n" + "="*80)
print("9. DISTRIBUIÇÃO DOS PADRÕES ESPECTRAIS")
print("="*80)

pattern_counter = Counter()

for pair,eigs in bracket_spectra.items():

    pattern = spectral_signature(eigs)

    pattern_counter[pattern] += 1


for pattern,count in pattern_counter.items():

    print(f"{pattern:25s} : {count:3d}")


# =============================================================================
# 12. RELAÇÕES ESPECTRAIS POR GERADOR
# =============================================================================

print("\n" + "="*80)
print("10. DISTRIBUIÇÃO DAS RELAÇÕES ESPECTRAIS")
print("="*80)

spectral_by_generator = Counter()

for (i,j) in spectral_brackets:

    spectral_by_generator[i] += 1
    spectral_by_generator[j] += 1


for i in range(1,17):

    print(
        f"B{i:02d}: "
        f"{spectral_by_generator[i]:3d} ocorrências em relações espectrais"
    )


# =============================================================================
# 13. MATRIZ DE INCIDÊNCIA DOS 96 NÃO-COMUTATIVOS
# =============================================================================

print("\n" + "="*80)
print("11. MATRIZ DE INCIDÊNCIA ESPECTRAL")
print("="*80)

incidence = np.zeros((16,16), dtype=int)

for (i,j) in noncommutative:

    if (i,j) in spectral_brackets:

        incidence[i-1,j-1] = 1
        incidence[j-1,i-1] = 1


print("\n1 = relação espectral")
print("0 = relação não espectral\n")

print("      " + " ".join(f"B{i:02d}" for i in range(1,17)))

for i in range(16):

    row = " ".join(str(x) for x in incidence[i])

    print(f"B{i+1:02d}   {row}")


# =============================================================================
# 14. VERIFICAÇÃO DA ANTICOMUTATIVIDADE
# =============================================================================

print("\n" + "="*80)
print("12. TESTE DE ANTICOMUTATIVIDADE")
print("="*80)

fail_anti = 0
tests_anti = 0

for i in range(1,17):

    for j in range(1,17):

        lhs = bracket(B[i],B[j])
        rhs = -bracket(B[j],B[i])

        tests_anti += 1

        if lhs != rhs:
            fail_anti += 1


print(f"Testes realizados = {tests_anti}")
print(f"Falhas            = {fail_anti}")

if fail_anti == 0:
    print("✓ Anticomutatividade confirmada.")


# =============================================================================
# 15. RESUMO FINAL
# =============================================================================

print("\n" + "="*80)
print("13. RESUMO FINAL")
print("="*80)

print("""
ESTRUTURA DOS 120 PARES
------------------------
Total de pares              : 120
Comutativos                  : 24
Não-comutativos              : 96

ANÁLISE ESPECTRAL
-----------------
Relações espectrais          : {}
Relações restantes           : {}

TESTE DA DECOMPOSIÇÃO
---------------------
96 = {} + {}

HIPÓTESE ORIGINAL
-----------------
96 = 64 + 32
""".format(
    N_spectral,
    N_other,
    N_other,
    N_spectral
))

print("="*80)
print("FIM DA ANÁLISE")
print("="*80)
