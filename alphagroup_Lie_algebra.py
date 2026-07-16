"""
==========================================================================
EXTENDED ALPHA GROUP
Computational Framework
==========================================================================

This repository contains the computational implementation developed for
the mathematical study of the Extended Alpha Group.

The programs included here provide symbolic and numerical tools for the
construction and analysis of the algebraic, geometric, spectral and
topological structures associated with the operator M(θ).

Depending on the selected module, the software is able to

• construct the matrix basis of the Extended Alpha Group;
• perform algebraic closure;
• decompose arbitrary matrices in the basis;
• compute matrix products;
• compute Lie brackets;
• determine the structure constants;
• verify the Jacobi identity;
• determine the center of the Lie algebra;
• verify the Central Element Theorem (μ);
• identify closed subalgebras;
• analyze solvability and nilpotency;
• construct graph representations;
• compute graph invariants and Laplacian operators;
• support Monte Carlo simulations;
• perform spectral and topological analyses associated with M(θ).

The codes contained in this repository were developed to support the
research program on the Extended Alpha Group and accompany the scientific
articles published by the author.

Author
------
Cleber Souza Corrêa

ORCID
------
0000-0001-5799-1982

License
-------
This software is provided for academic and research purposes.

==========================================================================
"""

import sympy as sp

# ==========================================================
# 02_basis.py
#
# Constrói automaticamente a base da álgebra do Grupo Alpha
# por fechamento multiplicativo.
# ==========================================================

# ----------------------------------------------------------
# Função auxiliar:
# transforma uma lista de matrizes 4x4 em vetores coluna 16x1
# ----------------------------------------------------------

def _matrix_to_columns(lista):
    return sp.Matrix.hstack(*[M.reshape(16, 1) for M in lista])


# ----------------------------------------------------------
# Constrói a base por fechamento da álgebra
# ----------------------------------------------------------

def build_basis(generators, verbose=True):
    """
    Constrói automaticamente a base linear fechada da álgebra.

    Parâmetros
    ----------
    generators : list
        Lista contendo os geradores iniciais.

    verbose : bool
        Imprime a evolução do processo.

    Retorna
    -------
    Base : list
        Lista contendo a base final.
    """

    conjunto = generators.copy()

    matriz = _matrix_to_columns(conjunto)
    rank = matriz.rank()

    if verbose:
        print("="*60)
        print("CONSTRUÇÃO DA BASE DA ÁLGEBRA")
        print("="*60)
        print(f"Geradores iniciais : {len(conjunto)}")
        print(f"Rank inicial       : {rank}")
        print()

    iteracao = 0

    while True:

        iteracao += 1
        adicionou = False

        # Base linear atual
        base_atual = [
            M.reshape(4,4)
            for M in _matrix_to_columns(conjunto).columnspace()
        ]

        for A in base_atual:
            for B in base_atual:

                produto = A*B

                novo_rank = _matrix_to_columns(
                    conjunto + [produto]
                ).rank()

                if novo_rank > rank:

                    conjunto.append(produto)

                    rank = novo_rank

                    adicionou = True

                    if verbose:
                        print(f"Iteração {iteracao}")
                        print(f"Nova matriz encontrada")
                        print(f"Rank = {rank}")
                        print("-"*40)

        if not adicionou:
            break

    Base = [
        vec.reshape(4,4)
        for vec in _matrix_to_columns(conjunto).columnspace()
    ]

    if verbose:

        print()
        print("="*60)
        print("BASE FINAL")
        print("="*60)

        print(f"Dimensão = {len(Base)}")

        for i,B in enumerate(Base,1):

            print()
            print(f"B{i}")
            sp.pprint(B)

    return Base

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

GENERATORS = [
I4,
G_C,
G_T,
G_mu,
G_const
]

Base = build_basis(GENERATORS)

# ==========================================================
# 03_decomposition.py
#
# Decompõe qualquer matriz 4x4 na base do Grupo Alpha.
# ==========================================================

import sympy as sp


# ----------------------------------------------------------
# Converte uma lista de matrizes em colunas 16x1
# ----------------------------------------------------------

def _matrix_to_columns(lista):
    return sp.Matrix.hstack(*[M.reshape(16,1) for M in lista])


# ----------------------------------------------------------
# Constrói o sistema linear
# ----------------------------------------------------------

def decompose(M, Base):
    """
    Decompõe uma matriz M na base fornecida.

    Parâmetros
    ----------
    M : sympy.Matrix
        Matriz 4x4.

    Base : list
        Lista contendo as 16 matrizes da base.

    Retorna
    -------
    coeficientes : list
        Lista [c1,c2,...,c16]
    """

    n = len(Base)

    coef = sp.symbols(f'c0:{n}')

    expr = sp.zeros(4)

    for c,B in zip(coef,Base):
        expr += c*B

    equacoes = []

    for i in range(4):
        for j in range(4):
            equacoes.append(
                sp.Eq(expr[i,j],M[i,j])
            )

    sol = sp.solve(equacoes,coef,dict=True)

    if len(sol)==0:
        raise ValueError("A matriz não pertence ao espaço gerado pela base.")

    sol = sol[0]

    return [sp.simplify(sol[c]) for c in coef]


# ----------------------------------------------------------
# Impressão amigável
# ----------------------------------------------------------

def print_decomposition(M, Base):

    coef = decompose(M,Base)

    print("="*60)
    print("DECOMPOSIÇÃO DA MATRIZ")
    print("="*60)

    sp.pprint(M)

    print("\nNa base:\n")

    vazio=True

    for i,c in enumerate(coef):

        if c!=0:

            vazio=False

            print(f"{c}*B{i+1}")

    if vazio:
        print("0")

print_decomposition(G_C,Base)

print_decomposition(G_mu,Base)

print_decomposition(G_T, Base)

print_decomposition(G_const, Base)

# ==========================================================
# 04_products.py
#
# Calcula todos os produtos Bi*Bj da base do Grupo Alpha.
#
# Requer:
#     Base = build_basis(GENERATORS)
#     decompose(M, Base)
#
# ==========================================================

import sympy as sp

# ----------------------------------------------------------
# Formata uma decomposição
# ----------------------------------------------------------

def format_decomposition(coef):

    termos = []

    for i, c in enumerate(coef):

        if c != 0:
            termos.append(f"{sp.simplify(c)}*B{i+1}")

    if len(termos) == 0:
        return "0"

    return " + ".join(termos)


# ----------------------------------------------------------
# Calcula todos os produtos
# ----------------------------------------------------------

produtos = {}

N = len(Base)

print("="*70)
print("TABELA DE PRODUTOS DA BASE")
print("="*70)

for i in range(N):

    for j in range(N):

        P = Base[i] * Base[j]

        coef = decompose(P, Base)

        produtos[(i+1, j+1)] = coef

        print("-"*60)
        print(f"B{i+1} × B{j+1}")
        print(format_decomposition(coef))
        print()

print("="*70)
print("TOTAL DE PRODUTOS:", len(produtos))
print("="*70)


# ==========================================================
# 05_lie_brackets.py
#
# Calcula todos os colchetes de Lie
#
# Requer:
#     Base
#     decompose()
#
# ==========================================================

import sympy as sp

# ----------------------------------------------------------
# Formatação
# ----------------------------------------------------------

def format_decomposition(coef):

    termos = []

    for i, c in enumerate(coef):

        if c != 0:
            termos.append(f"{sp.simplify(c)}*B{i+1}")

    if len(termos) == 0:
        return "0"

    return " + ".join(termos)


# ----------------------------------------------------------
# Cálculo dos colchetes
# ----------------------------------------------------------

lie = {}

comutativos = []

nao_comutativos = []

N = len(Base)

print("="*70)
print("COLCHETES DE LIE")
print("="*70)

for i in range(N):

    for j in range(i+1, N):      # apenas pares independentes

        C = Base[i]*Base[j] - Base[j]*Base[i]

        coef = decompose(C, Base)

        lie[(i+1, j+1)] = coef

        if all(c == 0 for c in coef):

            comutativos.append((i+1, j+1))

            status = "COMUTATIVO"

        else:

            nao_comutativos.append((i+1, j+1, coef))

            status = "NÃO COMUTATIVO"

        print("-"*60)
        print(f"[B{i+1}, B{j+1}]")

        print(status)

        print(format_decomposition(coef))

        print()


# ----------------------------------------------------------
# Estatísticas
# ----------------------------------------------------------

print("="*70)
print("ESTATÍSTICAS")
print("="*70)

print(f"Pares independentes : {N*(N-1)//2}")

print(f"Comutativos         : {len(comutativos)}")

print(f"Não comutativos     : {len(nao_comutativos)}")

print()

print("="*70)
print("LISTA DOS PARES COMUTATIVOS")
print("="*70)

for par in comutativos:

    print(par)

print()

print("="*70)
print("LISTA DOS PARES NÃO COMUTATIVOS")
print("="*70)

for i, j, coef in nao_comutativos:

    print(f"[B{i},B{j}] = {format_decomposition(coef)}")


# ==========================================================
# 05_LIE_ANALYZER_PLUS.py
#
# ANÁLISE COMPLETA DA ÁLGEBRA DE LIE
#
# Requer:
#   Base
#   decompose()
#
# ==========================================================

import sympy as sp
from collections import Counter

# ----------------------------------------------------------
# Formatação
# ----------------------------------------------------------

def format_decomposition(coeffs):

    terms = []

    for i,c in enumerate(coeffs):

        if c != 0:
            terms.append(f"{sp.simplify(c)}*B{i+1}")

    return " + ".join(terms) if terms else "0"


# ==========================================================
# ESTRUTURAS
# ==========================================================

N = len(Base)

Lie = {}

commuting = []

non_commuting = []

frequency = Counter()

weight = Counter()

complexity = Counter()

largest_size = 0

largest_pair = None

largest_expression = ""

commutation_matrix = [[0]*N for _ in range(N)]

print("="*70)
print("LIE ALGEBRA ANALYZER")
print("="*70)
print()


# ==========================================================
# CONSTRUÇÃO DOS COLCHETES
# ==========================================================

for i in range(N):

    for j in range(i+1,N):

        comm = Base[i]*Base[j] - Base[j]*Base[i]

        coef = decompose(comm,Base)

        Lie[(i+1,j+1)] = coef

        expr = format_decomposition(coef)

        if all(c==0 for c in coef):

            commuting.append((i+1,j+1))

            commutation_matrix[i][j]=0
            commutation_matrix[j][i]=0

        else:

            non_commuting.append((i+1,j+1,coef))

            commutation_matrix[i][j]=1
            commutation_matrix[j][i]=1

            nterms = 0

            for k,c in enumerate(coef):

                if c!=0:

                    nterms += 1

                    frequency[k+1]+=1

                    weight[k+1]+=abs(int(c))

            complexity[nterms]+=1

            if nterms>largest_size:

                largest_size = nterms

                largest_pair = (i+1,j+1)

                largest_expression = expr


# ==========================================================
# ESTATÍSTICAS
# ==========================================================

print("="*70)
print("ESTATÍSTICAS")
print("="*70)

pairs = N*(N-1)//2

print(f"Dimensão                 : {N}")
print(f"Pares independentes      : {pairs}")
print(f"Comutativos              : {len(commuting)}")
print(f"Não comutativos          : {len(non_commuting)}")

print(f"Percentual comutativo    : {100*len(commuting)/pairs:.2f}%")
print(f"Percentual não comutativo: {100*len(non_commuting)/pairs:.2f}%")

print()


# ==========================================================
# PARES COMUTATIVOS
# ==========================================================

print("="*70)
print("PARES COMUTATIVOS")
print("="*70)

for p in commuting:

    print(p)

print()


# ==========================================================
# PARES NÃO COMUTATIVOS
# ==========================================================

print("="*70)
print("PARES NÃO COMUTATIVOS")
print("="*70)

for i,j,coef in non_commuting:

    print(f"[B{i},B{j}] = {format_decomposition(coef)}")

print()


# ==========================================================
# FREQUÊNCIA
# ==========================================================

print("="*70)
print("FREQUÊNCIA DOS OPERADORES")
print("="*70)

for i in range(1,N+1):

    print(f"B{i:2d} : {frequency[i]}")

print()


# ==========================================================
# RANKING
# ==========================================================

print("="*70)
print("RANKING DOS OPERADORES")
print("="*70)

ranking = sorted(frequency.items(),
                 key=lambda x:x[1],
                 reverse=True)

for pos,(b,v) in enumerate(ranking,start=1):

    print(f"{pos:2d}º  B{b:<2d}  {v}")

print()


# ==========================================================
# PESO ALGÉBRICO
# ==========================================================

print("="*70)
print("PESO ALGÉBRICO")
print("="*70)

ranking_weight = sorted(weight.items(),
                        key=lambda x:x[1],
                        reverse=True)

for b,v in ranking_weight:

    print(f"B{b:2d} : {v}")

print()


# ==========================================================
# COMPLEXIDADE
# ==========================================================

print("="*70)
print("COMPLEXIDADE DOS COLCHETES")
print("="*70)

for n in sorted(complexity):

    print(f"{n} termos : {complexity[n]}")

print()


# ==========================================================
# MAIOR COLCHETE
# ==========================================================

print("="*70)
print("COLCHETE MAIS COMPLEXO")
print("="*70)

print("Par :",largest_pair)

print("Número de termos :",largest_size)

print(largest_expression)

print()


# ==========================================================
# MATRIZ DE COMUTAÇÃO
# ==========================================================

print("="*70)
print("MATRIZ DE COMUTAÇÃO")
print("="*70)

print("0 = comuta")
print("1 = não comuta")
print()

for linha in commutation_matrix:

    print(linha)

print()


# ==========================================================
# RESUMO
# ==========================================================

print("="*70)
print("RESUMO FINAL")
print("="*70)

print(f"Dimensão                 : {N}")
print(f"Pares independentes      : {pairs}")
print(f"Comutativos              : {len(commuting)}")
print(f"Não comutativos          : {len(non_commuting)}")

print()

print(f"Maior colchete           : {largest_pair}")

print(f"Número máximo de termos  : {largest_size}")

if ranking:
    print(f"Operador mais frequente  : B{ranking[0][0]}")

print()


# ==========================================================
# EXPORTAÇÃO
# ==========================================================

with open("Lie_Report.txt","w") as f:

    f.write("LIE ALGEBRA REPORT\n")
    f.write("="*60+"\n\n")

    f.write(f"Dimension: {N}\n")
    f.write(f"Independent pairs: {pairs}\n")
    f.write(f"Commuting: {len(commuting)}\n")
    f.write(f"Non-commuting: {len(non_commuting)}\n\n")

    f.write("COMMUTING PAIRS\n")

    for p in commuting:

        f.write(str(p)+"\n")

    f.write("\nNON-COMMUTING PAIRS\n")

    for i,j,coef in non_commuting:

        f.write(f"[B{i},B{j}] = {format_decomposition(coef)}\n")

print("Relatório salvo em Lie_Report.txt")

# ==========================================================
# 05_2_STRUCTURE_ANALYSIS.py
#
# Análise estrutural da Álgebra de Lie
#
# Requer:
#     Base
#     decompose()
#
# ==========================================================

import sympy as sp

# ==========================================================
# Formatação
# ==========================================================

def format_decomposition(coeffs):

    terms = []

    for i,c in enumerate(coeffs):

        if c != 0:
            terms.append(f"({sp.simplify(c)})*B{i+1}")

    return " + ".join(terms) if terms else "0"


# ==========================================================
# Produto de Lie
# ==========================================================

def lie_bracket(A,B):

    return A*B-B*A


# ==========================================================
# Constantes de estrutura
# ==========================================================

print("="*70)
print("STRUCTURE CONSTANTS")
print("="*70)

structure = {}

N=len(Base)

for i in range(N):

    for j in range(N):

        comm = lie_bracket(Base[i],Base[j])

        coef = decompose(comm,Base)

        structure[(i+1,j+1)] = coef

        if any(c!=0 for c in coef):

            print(f"[B{i+1},B{j+1}] = {format_decomposition(coef)}")

print()


# ==========================================================
# Identidade de Jacobi
# ==========================================================

print("="*70)
print("JACOBI IDENTITY")
print("="*70)

total=0

passed=0

failed=[]

for i in range(N):

    for j in range(N):

        for k in range(N):

            total+=1

            J = (
                lie_bracket(Base[i], lie_bracket(Base[j],Base[k]))
              + lie_bracket(Base[j], lie_bracket(Base[k],Base[i]))
              + lie_bracket(Base[k], lie_bracket(Base[i],Base[j]))
            )

            coef = decompose(J,Base)

            if all(c==0 for c in coef):

                passed+=1

            else:

                failed.append((i+1,j+1,k+1,coef))


print()

print("Total de triplos :",total)

print("Aprovados        :",passed)

print("Falhas           :",len(failed))

if len(failed)==0:

    print()
    print("STATUS : PASSED")

else:

    print()
    print("STATUS : FAILED")

    print()

    for i,j,k,coef in failed:

        print(f"({i},{j},{k})")

        print(format_decomposition(coef))

        print()


# ==========================================================
# Centro da Álgebra
# ==========================================================

print("="*70)
print("CENTER OF THE LIE ALGEBRA")
print("="*70)

center=[]

for i in range(N):

    central=True

    for j in range(N):

        C=lie_bracket(Base[i],Base[j])

        coef=decompose(C,Base)

        if any(c!=0 for c in coef):

            central=False

            break

    if central:

        center.append(i+1)

print()

print("Dimensão do centro :",len(center))

print("Elementos centrais :")

for b in center:

    print(f"B{b}")

print()


# ==========================================================
# Exportação
# ==========================================================

with open("Structure_Report.txt","w") as f:

    f.write("STRUCTURAL ANALYSIS\n")
    f.write("="*60+"\n\n")

    f.write("STRUCTURE CONSTANTS\n\n")

    for (i,j),coef in structure.items():

        if any(c!=0 for c in coef):

            f.write(f"[B{i},B{j}] = {format_decomposition(coef)}\n")

    f.write("\n")

    f.write("JACOBI\n")

    f.write(f"Total : {total}\n")

    f.write(f"Passed : {passed}\n")

    f.write(f"Failed : {len(failed)}\n\n")

    f.write("CENTER\n")

    f.write(f"Dimension : {len(center)}\n")

    for b in center:

        f.write(f"B{b}\n")

print()
print("Relatório salvo em Structure_Report.txt")

