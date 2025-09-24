import matplotlib.pyplot as plt
import pandas as pd

# Dados resumidos do Grupo Alpha
data = {
    "S^n": ["S3","S4","S5","S6","S7","S15","S23","S31"],
    "n": [3,4,5,6,7,15,23,31],
    "num_blocks": [1,2,2,2,2,4,6,8],
    "H1_grafo": [3,1,2,4,6,12,18,24],
    "H2_grafo": [4,1,2,5,8,16,24,32]
}

df = pd.DataFrame(data)

# Gráfico de H1 e H2 vs num_blocks
plt.figure(figsize=(10,6))
plt.plot(df["num_blocks"], df["H1_grafo"], 'o-', label="H_1 (graph)")
plt.plot(df["num_blocks"], df["H2_grafo"], 's-', label="H_2 (graph)")

# Destacar transição de coerência
plt.axvspan(0,3.5, color='yellow', alpha=0.2, label="Partial coherence (S3–S7)")
plt.axvspan(3.5,9, color='green', alpha=0.2, label="Dominant coherence (S15–S31)")

plt.xticks(df["num_blocks"], df["S^n"])
plt.xlabel("Number of blocks")
plt.ylabel("Betti numbers")
plt.title("H_1 and H_2 progression vs. number blocks — Alpha Group fiber coherence")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
