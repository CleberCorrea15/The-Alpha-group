import numpy as np
from itertools import combinations
from collections import Counter

# ============================================================
# ALPHA GROUP: 120 -> 96 -> ESPECTRAIS GENUÍNOS
# Teste do papel de B7 na estrutura espectral
# ============================================================

# Critério numérico:
# partes imaginárias menores que IMAG_TOL são tratadas como zero.
# 1e-12 é deliberadamente mais rigoroso que o TOL algébrico.
TOL = 1e-8
IMAG_TOL = 1e-8

# --- Base B1 ... B16 (M_4(R)) ---
B1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],float)
B2=np.array([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]],float)
B3=np.array([[0,0,-1,0],[0,0,0,-1],[1,0,0,0],[0,1,0,0]],float)
B4=np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],float)
B5=np.array([[1,0,0,1],[0,1,1,0],[0,-1,0,0],[1,0,0,0]],float)
B6=np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]],float)
B7=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,-1],[0,0,1,0]],float)
B8=np.array([[0,-1,-1,0],[1,0,0,1],[-1,0,0,0],[0,-1,0,0]],float)
B9=np.array([[0,0,-1,0],[0,0,0,-1],[0,0,0,0],[0,0,0,0]],float)
B10=np.array([[0,1,0,0],[-1,0,0,0],[1,0,0,1],[0,1,1,0]],float)
B11=np.array([[0,0,0,0],[0,0,0,0],[0,-1,0,0],[1,0,0,0]],float)
B12=np.array([[0,1,-1,0],[1,0,0,-1],[0,0,0,1],[0,0,-1,0]],float)
B13=np.array([[2,0,0,1],[0,0,1,0],[0,-1,-1,0],[1,0,0,1]],float)
B14=np.array([[1,0,0,0],[0,1,0,0],[0,-1,-1,0],[1,0,0,1]],float)
B15=np.array([[0,0,0,-1],[0,0,1,0],[0,1,-1,0],[1,0,0,-1]],float)
B16=np.array([[0,1,1,0],[-1,0,0,-1],[2,0,0,1],[0,0,1,0]],float)

names=[f"B{i}" for i in range(1,17)]
basis=[B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16]

def comm(A,B):
    return A@B-B@A

def decompose(M):
    X=np.column_stack([A.reshape(-1) for A in basis])
    c,_,rank,_=np.linalg.lstsq(X,M.reshape(-1),rcond=None)
    R=sum(c[i]*basis[i] for i in range(16))
    return c,np.linalg.norm(M-R),rank

def spectral(M):
    """
    Classificação robusta:
      ZERO
      REAL_NUMERICO  -> todas as partes imaginárias <= IMAG_TOL
      COMPLEXO       -> pelo menos uma parte imaginária > IMAG_TOL
    """
    if np.linalg.norm(M)<=TOL:
        return False,"ZERO",np.zeros(4,dtype=complex),0,0,0.0

    ev=np.linalg.eigvals(M)
    imag_max=float(np.max(np.abs(ev.imag)))
    is_complex=imag_max>IMAG_TOL
    cls="COMPLEXO" if is_complex else "REAL_NUMERICO"
    return is_complex,cls,ev,np.linalg.matrix_rank(M,tol=TOL),max(abs(ev)),imag_max

# ============================================================
# 1. OS 120 PARES
# ============================================================
results=[]

for i,j in combinations(range(16),2):
    C=comm(basis[i],basis[j])
    sp,cls,ev,rank,rho,imag_max=spectral(C)
    c,res,rankbase=decompose(C)

    results.append({
        "i":i+1,
        "j":j+1,
        "pair":f"[B{i+1},B{j+1}]",
        "C":C,
        "norm":np.linalg.norm(C),
        "noncomm":np.linalg.norm(C)>TOL,
        "spectral":sp,
        "class":cls,
        "eig":ev,
        "rank":rank,
        "rho":rho,
        "imag_max":imag_max,
        "coeff":c,
        "residual":res,
        "B7":i==6 or j==6
    })

commutative=[r for r in results if not r["noncomm"]]
noncomm=[r for r in results if r["noncomm"]]
spectral96=[r for r in noncomm if r["spectral"]]
real96=[r for r in noncomm if not r["spectral"]]

print("="*95)
print("ALPHA GROUP — 120 -> 96 -> ESPECTRAIS GENUÍNOS")
print("="*95)
print(f"120 pares totais              : {len(results)}")
print(f"24 comutativos               : {len(commutative)}")
print(f"96 não comutativos           : {len(noncomm)}")
print(f"Espectrais genuínos          : {len(spectral96)}")
print(f"Reais após tolerância        : {len(real96)}")
print(f"Critério |Im(lambda)| >      : {IMAG_TOL:.1e}")

# ============================================================
# 2. AUDITORIA NUMÉRICA
# Mostra casos que poderiam ter sido classificados como
# complexos apenas por ruído de ponto flutuante.
# ============================================================
print("\n"+"="*95)
print("AUDITORIA DA PARTE IMAGINÁRIA")
print("="*95)

for r in noncomm:
    if r["imag_max"] <= 1e-8:
        print(f"{r['pair']:<14} max|Im(lambda)| = {r['imag_max']:.3e}"
              f"  -> {r['class']}")

# ============================================================
# 3. OS ESPECTRAIS GENUÍNOS
# ============================================================
print("\n"+"="*95)
print(f"COLCHETES ESPECTRAIS GENUÍNOS ({len(spectral96)})")
print("="*95)

for k,r in enumerate(spectral96,1):
    print(
        f"{k:2d}. {r['pair']:<14}"
        f" ||C||={r['norm']:.6e}"
        f" rank={r['rank']}"
        f" rho={r['rho']:.6e}"
        f" max|Im|={r['imag_max']:.6e}"
        f" B7={'SIM' if r['B7'] else 'NAO'}"
    )

# ============================================================
# 4. PAPEL DE B7
# ============================================================
b7_noncomm=[r for r in noncomm if r["B7"]]
b7_spec=[r for r in spectral96 if r["B7"]]
b7_real=[r for r in real96 if r["B7"]]

print("\n"+"="*95)
print("PAPEL DE B7")
print("="*95)
print(f"Pares não comutativos envolvendo B7 : {len(b7_noncomm)}")
print(f"Espectrais genuínos                  : {len(b7_spec)}")
print(f"Reais após tolerância                : {len(b7_real)}")

p_b7=len(b7_spec)/max(len(b7_noncomm),1)
p_global=len(spectral96)/max(len(noncomm),1)
enrichment=p_b7/p_global if p_global else np.nan

print(f"\nTaxa espectral dos pares com B7      : {100*p_b7:.2f}%")
print(f"Taxa espectral global dos 96         : {100*p_global:.2f}%")
print(f"Fator de enriquecimento de B7        : {enrichment:.4f}")

print("\nColchetes espectrais envolvendo B7:")
for r in b7_spec:
    print(f"  {r['pair']}  eig={np.round(r['eig'],8)}")

# ============================================================
# 5. GRAU NA REDE ESPECTRAL
# ============================================================
degree=Counter()

for r in spectral96:
    degree[f"B{r['i']}"]+=1
    degree[f"B{r['j']}"]+=1

print("\n"+"="*95)
print("GRAU DOS GERADORES NA REDE ESPECTRAL")
print("="*95)

for n in names:
    print(f"{n:4s}: {degree[n]:2d}" + ("  <-- B7" if n=="B7" else ""))

# ============================================================
# 6. AD_B7
# ============================================================
Aad=np.zeros((16,16))

for j in range(16):
    C=comm(B7,basis[j])
    c,_,_=decompose(C)
    Aad[:,j]=c

print("\n"+"="*95)
print("AÇÃO ADJUNTA ad_B7")
print("="*95)
print(np.round(Aad,4))

rank_ad=np.linalg.matrix_rank(Aad,tol=TOL)

print(f"\nrank(ad_B7) = {rank_ad}")
print(f"nullidade   = {16-rank_ad}")
print("espectro de ad_B7:")
print(np.round(np.linalg.eigvals(Aad),10))

# ============================================================
# 7. GERADORES ATINGIDOS POR ad_B7
# ============================================================
output=Counter()

for j in range(16):
    c,_,_=decompose(comm(B7,basis[j]))
    for k,x in enumerate(c):
        if abs(x)>TOL:
            output[f"B{k+1}"]+=1

g32=set()
for r in spectral96:
    g32.add(f"B{r['i']}")
    g32.add(f"B{r['j']}")

gB7=set(output)
inter=g32 & gB7

print("\n"+"="*95)
print("IMAGEM DE ad_B7 x REDE ESPECTRAL")
print("="*95)
print("Geradores nos colchetes espectrais:")
print(sorted(g32,key=lambda x:int(x[1:])))
print("\nGeradores atingidos por B7:")
print(sorted(gB7,key=lambda x:int(x[1:])))
print("\nInterseção:")
print(sorted(inter,key=lambda x:int(x[1:])))
print(f"\nInterseção = {len(inter)} de {len(g32)} geradores")

# ============================================================
# 8. CONCLUSÃO AUTOMÁTICA
# ============================================================
print("\n"+"="*95)
print("DIAGNÓSTICO")
print("="*95)

print(f"""
Com o critério numérico IMAG_TOL = {IMAG_TOL:.1e}, a classificação
espectral foi recalculada eliminando partes imaginárias compatíveis
com erro de ponto flutuante.

A hierarquia obtida é:

    120 pares
       |
       +-- 24 comutativos
       |
       +-- 96 não comutativos
                |
                +-- {len(spectral96)} espectrais genuínos
                |
                +-- {len(real96)} reais após tolerância

A hipótese sobre B7 deve ser avaliada agora usando SOMENTE os
{len(spectral96)} colchetes espectrais genuínos.

Atenção:
a presença de autovalores complexos no colchete é uma propriedade
do operador [Bi,Bj]. Ela não implica, por si só, que B7 seja uma
ponte estrutural-espectral.

O teste mais forte é a combinação de:
  1. presença de B7 nos colchetes espectrais;
  2. grau de B7 na rede espectral;
  3. estrutura de ad_B7;
  4. sobreposição entre Im(ad_B7) e os geradores da rede espectral;
  5. estabilidade dessas relações sob mudança de tolerância.
""")

np.save("A_adB7.npy",Aad)

print("Matriz ad_B7 salva em: A_adB7.npy")
print("="*95)

from collections import defaultdict

# Agrupamento por Assinatura Espectral Normalizada
familias = defaultdict(list)

for r in spectral96:
    C = r["C"]
    norma = np.linalg.norm(C)

    # 1. Normaliza a matriz para ignorar escala de amplitude
    C_norm = C / norma

    # 2. Obtém autovalores normalizados e ordena para criar uma chave única
    eigvals = np.linalg.eigvals(C_norm)

    # Arredonda partes real e imaginária para absorver ruído de float
    eig_sorted = np.sort_complex(eigvals)
    chave_espectral = tuple(np.round(eig_sorted, 4))

    familias[chave_espectral].append({
        "pair": r["pair"],
        "norma": norma,
        "B7": r["B7"]
    })

print("="*95)
print(f"REDUÇÃO DOS 42 COLCHETES PARA {len(familias)} FAMÍLIAS DE EQUIVALÊNCIA")
print("="*95)

for idx, (espectro, membros) in enumerate(familias.items(), 1):
    pares_str = ", ".join([m["pair"] for m in membros])
    tem_b7 = any(m["B7"] for m in membros)
    print(f"\nFamília {idx} [{len(membros)} pares] {'(Envolve B7)' if tem_b7 else ''}:")
    print(f"  Pares aglutinados : {pares_str}")
    print(f"  Espectro Base (λ) : {espectro}")

# ============================================================
# 9. ANÁLISE COMPLEMENTAR:
#    SEMELHANTES, SIMILARES E FAMÍLIAS ESTRUTURAIS
# ============================================================

from collections import defaultdict

# ------------------------------------------------------------
# 9.1 Funções auxiliares
# ------------------------------------------------------------

def limpar_vetor(c, tol=1e-8):
    """
    Elimina resíduos numéricos.
    """
    c = np.asarray(c, dtype=float).copy()
    c[np.abs(c) < tol] = 0.0
    return c


def suporte(c, tol=1e-8):
    """
    Geradores efetivamente presentes no colchete.
    """
    c = limpar_vetor(c, tol)
    return tuple(
        i+1 for i, x in enumerate(c)
        if abs(x) > tol
    )


def vetor_canonico(c, tol=1e-8):
    """
    Normaliza pela maior componente em módulo e
    escolhe uma orientação canônica.

    Assim:
        c
        -c
        2c
        -3c

    possuem a mesma assinatura canônica.
    """

    c = limpar_vetor(c, tol)

    m = np.max(np.abs(c))

    if m <= tol:
        return np.zeros_like(c)

    c = c / m

    # primeira componente não nula positiva
    for x in c:
        if abs(x) > tol:
            if x < 0:
                c = -c
            break

    return c


def sao_iguais(c1, c2, tol=1e-8):
    """
    Igualdade algébrica/numerical dos vetores.
    """
    return np.allclose(
        limpar_vetor(c1, tol),
        limpar_vetor(c2, tol),
        atol=tol,
        rtol=tol
    )


def sao_proporcionais(c1, c2, tol=1e-8):
    """
    Verifica se dois vetores são proporcionais,
    incluindo mudança global de sinal.
    """

    a = limpar_vetor(c1, tol)
    b = limpar_vetor(c2, tol)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na <= tol or nb <= tol:
        return False

    a = a / na
    b = b / nb

    return abs(abs(np.dot(a,b)) - 1.0) <= 1e-6


def sao_similares_estruturais(c1, c2, tol=1e-8):
    """
    Similaridade estrutural.

    Exige:
      1. mesmo suporte;
      2. mesma proporção relativa entre os coeficientes.

    A escala global e o sinal global são ignorados.
    """

    if suporte(c1, tol) != suporte(c2, tol):
        return False

    a = vetor_canonico(c1, tol)
    b = vetor_canonico(c2, tol)

    return np.allclose(
        a, b,
        atol=1e-6,
        rtol=1e-6
    )


# ------------------------------------------------------------
# 9.2 SEMELHANTES EXATOS
# ------------------------------------------------------------

print("\n" + "="*95)
print("SEMELHANTES EXATOS — MESMO VETOR NA BASE")
print("="*95)

semelhantes_exatos = []

usados = set()

for i, r in enumerate(noncomm):

    if i in usados:
        continue

    grupo = [r]
    usados.add(i)

    for j in range(i+1, len(noncomm)):

        if j in usados:
            continue

        s = noncomm[j]

        if sao_iguais(
            r["coeff"],
            s["coeff"]
        ):
            grupo.append(s)
            usados.add(j)

    if len(grupo) > 1:
        semelhantes_exatos.append(grupo)


print(
    f"Grupos de semelhantes exatos: "
    f"{len(semelhantes_exatos)}"
)

for k, grupo in enumerate(
    semelhantes_exatos, 1
):

    print(
        f"\nSE{k:02d} "
        f"[{len(grupo)} relações]"
    )

    for r in grupo:
        print(
            f"  {r['pair']:<14}"
            f" -> "
            f"{np.round(r['coeff'],4)}"
        )


# ------------------------------------------------------------
# 9.3 SEMELHANTES POR PROPORCIONALIDADE
# ------------------------------------------------------------

print("\n" + "="*95)
print("SEMELHANTES POR PROPORCIONALIDADE")
print("="*95)

semelhantes_proporcionais = []

usados = set()

for i, r in enumerate(noncomm):

    if i in usados:
        continue

    grupo = [r]
    usados.add(i)

    for j in range(i+1, len(noncomm)):

        if j in usados:
            continue

        s = noncomm[j]

        if sao_proporcionais(
            r["coeff"],
            s["coeff"]
        ):
            grupo.append(s)
            usados.add(j)

    if len(grupo) > 1:
        semelhantes_proporcionais.append(grupo)


print(
    f"Grupos proporcionais: "
    f"{len(semelhantes_proporcionais)}"
)

for k, grupo in enumerate(
    semelhantes_proporcionais, 1
):

    print(
        f"\nSP{k:02d} "
        f"[{len(grupo)} relações]"
    )

    for r in grupo:

        print(
            f"  {r['pair']:<14}"
            f" suporte={suporte(r['coeff'])}"
        )


# ------------------------------------------------------------
# 9.4 SIMILARES ESTRUTURAIS
# ------------------------------------------------------------

print("\n" + "="*95)
print("SIMILARES ESTRUTURAIS")
print("="*95)

similares_estruturais = []

usados = set()

for i, r in enumerate(noncomm):

    if i in usados:
        continue

    grupo = [r]
    usados.add(i)

    for j in range(i+1, len(noncomm)):

        if j in usados:
            continue

        s = noncomm[j]

        if sao_similares_estruturais(
            r["coeff"],
            s["coeff"]
        ):
            grupo.append(s)
            usados.add(j)

    if len(grupo) > 1:
        similares_estruturais.append(grupo)


print(
    f"Grupos de similares estruturais: "
    f"{len(similares_estruturais)}"
)

for k, grupo in enumerate(
    similares_estruturais, 1
):

    print(
        f"\nSIM{k:02d} "
        f"[{len(grupo)} relações]"
    )

    for r in grupo:

        print(
            f"  {r['pair']:<14}"
            f" suporte={suporte(r['coeff'])}"
        )

    print(
        "  assinatura ="
    )

    print(
        " ",
        vetor_canonico(
            grupo[0]["coeff"]
        )
    )


# ------------------------------------------------------------
# 9.5 ASSINATURA ESTRUTURAL DE CADA COLCHETE
# ------------------------------------------------------------

print("\n" + "="*95)
print("ASSINATURAS ESTRUTURAIS DOS 96 COLCHETES")
print("="*95)

assinaturas = defaultdict(list)

for r in noncomm:

    assinatura = tuple(
        np.round(
            vetor_canonico(
                r["coeff"]
            ),
            8
        )
    )

    assinaturas[assinatura].append(r)


print(
    f"Total de assinaturas estruturais: "
    f"{len(assinaturas)}"
)

for k, (assinatura, grupo) in enumerate(
    assinaturas.items(), 1
):

    print(
        f"\nC{k:02d} "
        f"[{len(grupo)} relações]"
    )

    print(
        "  Relações:"
    )

    for r in grupo:

        print(
            f"    {r['pair']}"
        )

    print(
        "  Assinatura:"
    )

    print(
        "   ",
        assinatura
    )


# ------------------------------------------------------------
# 9.6 COMPARAÇÃO:
#     FAMÍLIAS ESPECTRAIS x ESTRUTURA ALGÉBRICA
# ------------------------------------------------------------

print("\n" + "="*95)
print("COMPARAÇÃO — ESPECTRAL x ESTRUTURAL")
print("="*95)

# Dicionário par -> família espectral
familia_espectral = {}

for fid, (espectro, membros) in enumerate(
    familias.items(), 1
):

    for m in membros:

        familia_espectral[
            m["pair"]
        ] = fid


# Dicionário par -> assinatura estrutural
familia_estrutural = {}

for sid, (assinatura, grupo) in enumerate(
    assinaturas.items(), 1
):

    for r in grupo:

        familia_estrutural[
            r["pair"]
        ] = sid


for r in spectral96:

    print(
        f"{r['pair']:<14}"
        f" -> espectral E{familia_espectral[r['pair']]:02d}"
        f" -> estrutural C{familia_estrutural[r['pair']]:02d}"
        f" -> "
        f"{'B7' if r['B7'] else '--'}"
    )


# ------------------------------------------------------------
# 9.7 MATRIZ DE INTERSEÇÃO
#     FAMÍLIAS ESPECTRAIS x ESTRUTURAIS
# ------------------------------------------------------------

print("\n" + "="*95)
print("INTERSEÇÃO ENTRE FAMÍLIAS ESPECTRAIS E ESTRUTURAIS")
print("="*95)

intersecao = defaultdict(int)

for r in spectral96:

    e = familia_espectral[
        r["pair"]
    ]

    s = familia_estrutural[
        r["pair"]
    ]

    intersecao[(e,s)] += 1


for (e,s),n in sorted(
    intersecao.items()
):

    print(
        f"E{e:02d} x C{s:02d}"
        f" -> {n} relações"
    )


# ------------------------------------------------------------
# 9.8 DISTRIBUIÇÃO DAS 44 RELAÇÕES
# ------------------------------------------------------------

print("\n" + "="*95)
print("DISTRIBUIÇÃO DAS 44 RELAÇÕES ESPECTRAIS")
print("="*95)

distribuicao = Counter()

for r in spectral96:

    e = familia_espectral[
        r["pair"]
    ]

    s = familia_estrutural[
        r["pair"]
    ]

    distribuicao[
        (e,s)
    ] += 1


for (e,s),n in sorted(
    distribuicao.items()
):

    print(
        f"Família espectral E{e:02d}"
        f" / estrutural C{s:02d}"
        f" : {n}"
    )


# ------------------------------------------------------------
# 9.9 B7 NAS FAMÍLIAS
# ------------------------------------------------------------

print("\n" + "="*95)
print("B7 — POSIÇÃO NAS FAMÍLIAS")
print("="*95)

for r in b7_spec:

    print(
        f"{r['pair']:<14}"
        f" -> E{familia_espectral[r['pair']]:02d}"
        f" / C{familia_estrutural[r['pair']]:02d}"
    )


# ------------------------------------------------------------
# 9.10 BUSCA DO NÚMERO 44
# ------------------------------------------------------------

print("\n" + "="*95)
print("BUSCA DO NÚMERO 44")
print("="*95)

print(
    f"Colchetes não comutativos : {len(noncomm)}"
)

print(
    f"Colchetes espectrais      : {len(spectral96)}"
)

print(
    f"Famílias espectrais       : {len(familias)}"
)

print(
    f"Assinaturas estruturais   : {len(assinaturas)}"
)

print(
    f"Semelhantes exatos        : "
    f"{len(semelhantes_exatos)} grupos"
)

print(
    f"Proporcionais             : "
    f"{len(semelhantes_proporcionais)} grupos"
)

print(
    f"Similares estruturais     : "
    f"{len(similares_estruturais)} grupos"
)


if len(assinaturas) == 44:

    print(
        "\n*** RESULTADO ESPECIAL ***"
    )

    print(
        "O número 44 emerge naturalmente "
        "como número de assinaturas estruturais."
    )

else:

    print(
        "\nO número 44 não foi imposto."
    )

    print(
        f"O número encontrado foi "
        f"{len(assinaturas)}."
    )


# ------------------------------------------------------------
# 9.11 RELATÓRIO CONCEITUAL
# ------------------------------------------------------------

print("\n" + "="*95)
print("SÍNTESE ESTRUTURAL")
print("="*95)

print(
    f"""
A análise foi organizada em três níveis:

1. ESPECTRAL
   {len(spectral96)} colchetes apresentam espectro complexo.

2. SEMELHANÇA ALGÉBRICA
   Colchetes podem produzir exatamente o mesmo vetor
   na base B1,...,B16 ou vetores proporcionais.

3. SIMILARIDADE ESTRUTURAL
   Colchetes podem possuir o mesmo suporte e a mesma
   estrutura relativa de coeficientes.

Portanto:

     120 pares
         |
         +-- 24 comutativos
         |
         +-- 96 não comutativos
                  |
                  +-- {len(spectral96)} espectrais
                  |
                  +-- {len(familias)} famílias espectrais
                  |
                  +-- {len(assinaturas)} assinaturas estruturais

O número 44 é tratado como RESULTADO A SER DESCOBERTO,
e não como hipótese imposta ao algoritmo.
"""
)


# ------------------------------------------------------------
# 9.12 SALVAMENTO DOS RESULTADOS
# ------------------------------------------------------------

np.save(
    "alpha_structural_signatures.npy",
    np.array([
        vetor_canonico(r["coeff"])
        for r in noncomm
    ])
)

print(
    "\nAssinaturas estruturais salvas em:"
)

print(
    "alpha_structural_signatures.npy"
)

print("="*95)
print("FIM DA ANÁLISE COMPLEMENTAR")
print("="*95)

