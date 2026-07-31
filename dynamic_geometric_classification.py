# Script Python para Classificação Geométrica Dinâmica dos Geradores do Grupo Alpha
# Este script utiliza regras de Aritmética Modular (mod 16) para classificar
# automaticamente qualquer gerador de B1 a B120 a partir da semente inicial.

import pandas as pd

def classificar_gerador(indice):
    """
    Classifica o gerador Bi com base em regras aritméticas mod 16.
    A semente inicial (B1 a B16) define a assinatura periódica da álgebra.
    """
    if indice == 1:
        return "Central (z)", "Simetria global, comuta com toda a álgebra."

    # Avalia o comportamento com base no ciclo de 16 elementos
    resto = indice % 16

    if resto in [2, 3, 7]:
        return "Compacto (k)", "Subgrupos compactos, rotações e órbitas periódicas estáveis."
    elif resto in [4, 6]:
        return "Não-Compacto", "Transformações hiperbólicas, fluxos abertos e expansões."
    elif resto in [9, 11]:
        return "Nilpotente (n)", "Cisalhamentos, translações e operadores de escala com aniquilação finita."
    else:
        # resto in [0, 5, 8, 10, 12, 13, 14, 15]
        return "Projetivo", "Transformações fracionárias lineares e dinâmica nas fronteiras do espaço."

# Gera a classificação calculada dinamicamente para os 120 elementos
dados_calculados = []
for i in range(1, 121):
    classe, dinamica = classificar_gerador(i)
    dados_calculados.append({
        "Gerador": f"B{i}",
        "Índice": i,
        "Classe": classe,
        "Dinâmica": dinamica
    })

# Transforma em DataFrame para agrupar e validar
df = pd.DataFrame(dados_calculados)

# Agrupamento e contagem automática baseada na classificação dinâmica
resumo = df.groupby("Classe").agg(
    Quantidade=("Gerador", "count"),
    Exemplos=("Gerador", lambda x: ", ".join(list(x)[:4]) + ("..." if len(x) > 4 else ""))
).reset_index()

# Associa a descrição correspondente ao resumo
resumo["Comportamento Dinâmico"] = resumo["Classe"].apply(
    lambda c: df[df["Classe"] == c]["Dinâmica"].iloc[0]
)

# Saída limpa no terminal
print("=" * 110)
print(f"{'CLASSIFICAÇÃO DINÂMICA DA BASE DO GRUPO ALPHA (REGRAS EM BUTIDAS MOD 16)':^110}")
print("=" * 110)
for index, row in resumo.iterrows():
    print(f"Classe: {row['Classe']:<18} | Qtd: {row['Quantidade']:<3} | Exemplos: {row['Exemplos']:<35}")
    print(f"Dinâmica: {row['Comportamento Dinâmico']}")
    print("-" * 110)

print(f"TOTAL DE ELEMENTOS CLASSIFICADOS: {resumo['Quantidade'].sum()}")
print("=" * 110)
