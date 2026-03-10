import os, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy.spatial import distance_matrix
from scipy import stats
import networkx as nx
from tqdm import tqdm

# Configuração para análise topológica
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

OUT_DIR = "results_arvore_ciclica"
N_POINTS = 200
N_SIM_EACH = 8
MAXDIM = 2

def build_M(theta):
    epsilon = 1e-12
    theta_adj = theta + 0j
    tan_theta = np.tan(theta_adj)
    cot_theta = 1.0/np.tan(theta_adj) if np.abs(np.tan(theta_adj)) > 1e-12 else 1e12
    max_val = 1e3
    tan_theta = np.clip(tan_theta, -max_val, max_val)
    cot_theta = np.clip(cot_theta, -max_val, max_val)
    mu = 1.0
    i = 1j
    M = np.array([
        [1, -cot_theta, -tan_theta, 1],
        [cot_theta, i, -1, -tan_theta],
        [tan_theta, -1, mu, -cot_theta],
        [1, tan_theta, cot_theta, i*mu]
    ], dtype=complex)
    return M

def generate_point_cloud_from_M(n_points, theta=0.0, W=1.0, seed=None):
    rng = np.random.default_rng(seed)
    M = build_M(theta)
    z_real = rng.normal(size=(n_points, 4))
    z_imag = rng.normal(size=(n_points, 4))
    Z = z_real + 1j * z_imag
    V = (M @ Z.T).T
    V_real = np.hstack([V.real, V.imag])
    V_centered = V_real - V_real.mean(axis=0, keepdims=True)
    U, S, VT = svd(V_centered, full_matrices=False)
    coords3 = U[:, :3] * S[:3]
    scales = np.linalg.norm(coords3, axis=1)
    median_r = np.median(scales)
    if median_r == 0:
        median_r = 1.0
    coords3 = coords3 / (median_r + 1e-12)
    return coords3

def analyze_vr_structure(points, eps):
    """Análise detalhada da estrutura do complexo Vietoris-Rips"""
    n = points.shape[0]
    D = distance_matrix(points, points)

    # Matriz de adjacência
    adj = (D <= eps).astype(int)
    np.fill_diagonal(adj, 0)

    # Grafo da estrutura
    G = nx.from_numpy_array(adj)

    # Estatísticas do grafo
    num_edges = G.number_of_edges()
    num_triangles = sum(nx.triangles(G).values()) // 3

    # Componentes conectados
    components = list(nx.connected_components(G))
    num_components = len(components)

    # Tamanho do maior componente
    if components:
        largest_component_size = max(len(c) for c in components)
    else:
        largest_component_size = 0

    # Coeficiente de agrupamento
    if num_edges > 0:
        clustering_coeff = nx.average_clustering(G)
    else:
        clustering_coeff = 0

    # Grau médio
    if n > 0:
        avg_degree = 2 * num_edges / n
    else:
        avg_degree = 0

    # Densidade do grafo
    if n > 1:
        max_possible_edges = n * (n - 1) / 2
        density = num_edges / max_possible_edges
    else:
        density = 0

    # Calcular H2
    h0 = num_components
    h1 = num_edges - n + h0
    h2 = max(0, num_triangles - num_edges + n - h0)

    return {
        'num_points': n,
        'num_edges': num_edges,
        'num_triangles': num_triangles,
        'num_components': num_components,
        'largest_component': largest_component_size,
        'clustering_coeff': clustering_coeff,
        'avg_degree': avg_degree,
        'graph_density': density,
        'h0': h0,
        'h1': h1,
        'h2': h2,
        'euler_characteristic': h0 - h1 + h2
    }

def analyze_epsilon_sweep():
    """Varredura detalhada de epsilon para entender a transição"""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parâmetros focados
    theta_focus = 1.55
    epsilon_range = np.linspace(0.1, 1.2, 30)

    print(f"ANÁLISE DA ESTRUTURA DA ÁRVORE CÍCLICA")
    print(f"θ = {theta_focus:.3f}")
    print(f"ε = {epsilon_range[0]:.1f} a {epsilon_range[-1]:.1f} ({len(epsilon_range)} pontos)")

    results = []
    start = time.time()

    for sim in tqdm(range(N_SIM_EACH), desc="Simulações"):
        seed = int((theta_focus * 1e6) % 2**31) + sim
        X = generate_point_cloud_from_M(N_POINTS, theta=theta_focus, seed=seed)

        for eps in epsilon_range:
            try:
                structure = analyze_vr_structure(X, eps)
                structure.update({
                    'sim': sim,
                    'theta': theta_focus,
                    'epsilon': eps
                })
                results.append(structure)
            except:
                continue

    if not results:
        return None, None

    df = pd.DataFrame(results)
    ts = int(time.time())
    csv_path = os.path.join(OUT_DIR, f"arvore_ciclica_analise_{ts}.csv")
    df.to_csv(csv_path, index=False)

    return df, csv_path, ts

def create_separate_figures(df, timestamp):
    """Cria cada figura em um arquivo PNG separado"""

    figure_paths = []

    # Médias por epsilon
    stats_eps = df.groupby('epsilon').agg({
        'h2': 'mean',
        'num_edges': 'mean',
        'num_triangles': 'mean',
        'num_components': 'mean',
        'largest_component': 'mean',
        'clustering_coeff': 'mean',
        'avg_degree': 'mean',
        'graph_density': 'mean'
    }).reset_index()

    # Calcular razões críticas
    stats_eps['edges_per_point'] = stats_eps['num_edges'] / N_POINTS
    stats_eps['triangles_per_edge'] = stats_eps['num_triangles'] / (stats_eps['num_edges'] + 1e-12)
    stats_eps['h2_per_triangle'] = stats_eps['h2'] / (stats_eps['num_triangles'] + 1e-12)

    # FIGURA 1: Transição de Fase Topológica - H2 vs ε
    plt.figure(figsize=(14, 8))
    plt.plot(stats_eps['epsilon'], stats_eps['h2'], 'r-o', linewidth=4, markersize=6, label='H2')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima H2')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('H2', fontsize=14)
    plt.title('TRANSÇÃO DE FASE TOPOLÓGICA: H2 vs ε\n(Pico em ε = 0.6-0.7)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig1_path = os.path.join(OUT_DIR, f"01_transicao_fase_h2_{timestamp}.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig1_path)
    print(f"Figura 1 salva: {fig1_path}")

    # FIGURA 2: Boxplot da Distribuição de H2 na Região Ótima
    plt.figure(figsize=(14, 8))

    # Filtrar região ótima
    df_optimal = df[(df['epsilon'] >= 0.55) & (df['epsilon'] <= 0.75)]

    # Criar categorias para o boxplot
    df_optimal['epsilon_bin'] = pd.cut(df_optimal['epsilon'], bins=8)

    sns.boxplot(data=df_optimal, x='epsilon_bin', y='h2', palette='YlOrRd')
    plt.xticks(rotation=45)
    plt.xlabel('ε (radius) - Optimal Region', fontsize=14)
    plt.ylabel('H2', fontsize=14)
    plt.title('Distribution of H2 in the Optimal Region (ε = 0.55-0.75) Boxplot by  ε range', fontsize=16, fontweight='bold')

    # Adicionar estatísticas no gráfico
    optimal_stats = df_optimal['h2'].describe()
    stats_text = f'General Statistics:\nAverage: {optimal_stats["mean"]:.1f}\nStd: {optimal_stats["std"]:.1f}\nMax: {optimal_stats["max"]:.0f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    fig2_path = os.path.join(OUT_DIR, f"02_boxplot_regiao_otima_{timestamp}.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig2_path)
    print(f"Figura 2 salva: {fig2_path}")

    # FIGURA 3: Crescimento da Estrutura
    plt.figure(figsize=(14, 8))
    plt.plot(stats_eps['epsilon'], stats_eps['num_edges'], color='blue', label='Arestas', linewidth=3, marker='o')
    plt.plot(stats_eps['epsilon'], stats_eps['num_triangles'], color='green', label='Triângulos', linewidth=3, marker='s')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    plt.title('CRESCIMENTO DA ESTRUTURA DO COMPLEXO VIETORIS-RIPS', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()

    fig3_path = os.path.join(OUT_DIR, f"03_crescimento_estrutura_{timestamp}.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig3_path)
    print(f"Figura 3 salva: {fig3_path}")

    # FIGURA 4: Conectividade do Grafo
    plt.figure(figsize=(14, 8))
    plt.plot(stats_eps['epsilon'], stats_eps['num_components'], color='purple', label='Componentes Conectados', linewidth=3, marker='o')
    plt.plot(stats_eps['epsilon'], stats_eps['largest_component'], color='orange', label='Maior Componente', linewidth=3, marker='s')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('Tamanho/Contagem', fontsize=14)
    plt.title('CONECTIVIDADE DO GRAFO - COMPONENTES CONECTADOS', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig4_path = os.path.join(OUT_DIR, f"04_conectividade_grafo_{timestamp}.png")
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig4_path)
    print(f"Figura 4 salva: {fig4_path}")

    # FIGURA 5: Propriedades do Grafo
    plt.figure(figsize=(14, 8))
    plt.plot(stats_eps['epsilon'], stats_eps['clustering_coeff'], color='brown', label='Coeficiente de Agrupamento', linewidth=3, marker='o')
    plt.plot(stats_eps['epsilon'], stats_eps['graph_density'], color='pink', label='Densidade do Grafo', linewidth=3, marker='s')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('Coeficiente', fontsize=14)
    plt.title('PROPRIEDADES DO GRAFO - AGRUPAMENTO E DENSIDADE', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig5_path = os.path.join(OUT_DIR, f"05_propriedades_grafo_{timestamp}.png")
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig5_path)
    print(f"Figura 5 salva: {fig5_path}")

    # FIGURA 6: Eficiência de Conectividade
    plt.figure(figsize=(14, 8))
    plt.plot(stats_eps['epsilon'], stats_eps['edges_per_point'], 'g-o', linewidth=3, markersize=6, label='Arestas por Ponto')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('Arestas/Ponto', fontsize=14)
    plt.title('EFICIÊNCIA DE CONECTIVIDADE - ARESTAS POR PONTO', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig6_path = os.path.join(OUT_DIR, f"06_eficiencia_conectividade_{timestamp}.png")
    plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig6_path)
    print(f"Figura 6 salva: {fig6_path}")

    # FIGURA 7: Eficiência de Triangulação
    plt.figure(figsize=(14, 8))
    plt.plot(stats_eps['epsilon'], stats_eps['triangles_per_edge'], 'b-o', linewidth=3, markersize=6, label='Triângulos por Aresta')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('Triângulos/Aresta', fontsize=14)
    plt.title('EFICIÊNCIA DE TRIANGULAÇÃO - TRIÂNGULOS POR ARESTA', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig7_path = os.path.join(OUT_DIR, f"07_eficiencia_triangulacao_{timestamp}.png")
    plt.savefig(fig7_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig7_path)
    print(f"Figura 7 salva: {fig7_path}")

    # FIGURA 8: Eficiência de Cavidades
    plt.figure(figsize=(14, 8))
    plt.plot(stats_eps['epsilon'], stats_eps['h2_per_triangle'], 'r-o', linewidth=3, markersize=6, label='H2 por Triângulo')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('H2/Triângulo', fontsize=14)
    plt.title('EFICIÊNCIA DE FORMAÇÃO DE CAVIDADES - H2 POR TRIÂNGULO', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig8_path = os.path.join(OUT_DIR, f"08_eficiencia_cavidades_{timestamp}.png")
    plt.savefig(fig8_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig8_path)
    print(f"Figura 8 salva: {fig8_path}")

    # FIGURA 9: Característica de Euler
    plt.figure(figsize=(14, 8))
    euler = stats_eps['num_components'] - stats_eps['num_edges'] + stats_eps['num_triangles']
    plt.plot(stats_eps['epsilon'], euler, 'm-o', linewidth=3, markersize=6, label='Característica de Euler')
    plt.axvspan(0.6, 0.7, alpha=0.3, color='yellow', label='Região Ótima')
    plt.xlabel('ε (raio)', fontsize=14)
    plt.ylabel('χ = Componentes - Arestas + Triângulos', fontsize=14)
    plt.title('CARACTERÍSTICA DE EULER DO COMPLEXO', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig9_path = os.path.join(OUT_DIR, f"09_caracteristica_euler_{timestamp}.png")
    plt.savefig(fig9_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig9_path)
    print(f"Figura 9 salva: {fig9_path}")

    # FIGURA 10: Correlação H2 vs Densidade
    plt.figure(figsize=(14, 8))
    plt.scatter(stats_eps['graph_density'], stats_eps['h2'], c=stats_eps['epsilon'],
                cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='ε', shrink=0.8)
    plt.xlabel('Densidade do Grafo', fontsize=14)
    plt.ylabel('H2', fontsize=14)
    plt.title('CORRELAÇÃO: H2 vs DENSIDADE DO GRAFO', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig10_path = os.path.join(OUT_DIR, f"10_correlacao_densidade_{timestamp}.png")
    plt.savefig(fig10_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig10_path)
    print(f"Figura 10 salva: {fig10_path}")

    # FIGURA 11: Teoria da Árvore Cíclica Ótima
    plt.figure(figsize=(14, 10))

    # Encontrar ponto ótimo
    max_h2_idx = stats_eps['h2'].idxmax()
    max_eps = stats_eps.loc[max_h2_idx, 'epsilon']
    max_h2 = stats_eps.loc[max_h2_idx, 'h2']

    theory_text = f'TEORIA DA ÁRVORE CÍCLICA ÓTIMA\n\n'
    theory_text += f'PONTO ÓTIMO ENCONTRADO:\n'
    theory_text += f'ε ótimo = {max_eps:.3f}\n'
    theory_text += f'H2 máximo = {max_h2:.1f}\n\n'
    theory_text += f'ESTRUTURA NO PONTO ÓTIMO:\n'
    theory_text += f'• Arestas/Ponto: {stats_eps.loc[max_h2_idx, "edges_per_point"]:.1f}\n'
    theory_text += f'• Triângulos/Aresta: {stats_eps.loc[max_h2_idx, "triangles_per_edge"]:.3f}\n'
    theory_text += f'• Coef. Agrupamento: {stats_eps.loc[max_h2_idx, "clustering_coeff"]:.3f}\n'
    theory_text += f'• Densidade: {stats_eps.loc[max_h2_idx, "graph_density"]:.3f}\n\n'
    theory_text += f'MECANISMO IDENTIFICADO:\n'
    theory_text += f'ε = 0.6-0.7 cria o "balanço perfeito":\n'
    theory_text += f'• Conectividade suficiente para formar cavidades\n'
    theory_text += f'• Triangulação eficiente sem preencher cavidades\n'
    theory_text += f'• Estrutura complexa mas não super-conectada\n'
    theory_text += f'• Máxima formação de "bolhas" topológicas (H2)\n\n'
    theory_text += f'ANALOGIA:\n'
    theory_text += f'"Rede de bolhas ideais" onde cada\n'
    theory_text += f'bolha é uma cavidade 2D (H2)'

    plt.text(0.5, 0.5, theory_text, transform=plt.gca().transAxes,
            fontsize=14, ha='center', va='center', linespacing=1.8,
            bbox=dict(boxstyle="round,pad=1.0", facecolor="lightyellow",
                     edgecolor="gold", linewidth=2))
    plt.axis('off')
    plt.title('RESUMO TEÓRICO - ÁRVORE CÍCLICA ÓTIMA', fontsize=18, fontweight='bold', pad=20)

    fig11_path = os.path.join(OUT_DIR, f"11_teoria_arvore_ciclica_{timestamp}.png")
    plt.savefig(fig11_path, dpi=150, bbox_inches='tight')
    plt.show()
    figure_paths.append(fig11_path)
    print(f"Figura 11 salva: {fig11_path}")

    return figure_paths, stats_eps

# EXECUÇÃO PRINCIPAL
print("ANÁLISE DA RELAÇÃO: ÁRVORE CÍCLICA vs ε = 0.6-0.7")
print("="*65)

df, csv_path, timestamp = analyze_epsilon_sweep()

if df is not None:
    print(f"\nAnálise estrutural concluída!")
    print(f"Total de configurações analisadas: {len(df)}")
    print(f"Range de ε: {df['epsilon'].min():.2f} a {df['epsilon'].max():.2f}")
    print(f"Dados salvos em: {csv_path}")

    print(f"\nGERANDO FIGURAS SEPARADAS...")
    figure_paths, stats_df = create_separate_figures(df, timestamp)

    # Análise final
    max_h2_idx = stats_df['h2'].idxmax()
    optimal_eps = stats_df.loc[max_h2_idx, 'epsilon']

    print(f"\n" + "="*70)
    print("RESUMO DA ANÁLISE - ÁRVORE CÍCLICA ÓTIMA")
    print("="*70)

    print(f"\nPONTO ÓTIMO: ε = {optimal_eps:.3f}")
    print(f"FIGURAS GERADAS: {len(figure_paths)} arquivos PNG")

    print(f"\nLISTA DE FIGURAS:")
    for i, path in enumerate(figure_paths, 1):
        print(f"  {i:2d}. {os.path.basename(path)}")

    print(f"\nAnálise completa concluída com sucesso!")

else:
    print("Erro na execução!")

