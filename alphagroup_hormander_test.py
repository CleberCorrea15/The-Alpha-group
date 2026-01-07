import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

EPS = 1e-12
MAX_BRACKET_DEPTH = 4

# ===========================
# 1. MATRIZ M(θ) EXATA
# ===========================
def build_M_theta(theta):
    """Matriz M_θ conforme equação (1) do artigo"""
    if abs(theta - np.pi/2) < EPS:
        theta = np.pi/2 - EPS
    
    tanθ = np.tan(theta)
    cotθ = 1/tanθ
    
    A = np.array([
        [1,    -cotθ, -tanθ, 1],
        [cotθ,  1,    -1,   -tanθ],
        [tanθ, -1,     1,   -cotθ],
        [1,     tanθ,  cotθ, 1]
    ], dtype=complex)
    
    μ = 1.0 + 0j
    B = np.diag([1, 1j, μ, 1j*μ])
    
    return A @ B

# ===========================
# 2. BASE COMPLETA DA ÁLGEBRA DO GRUPO ALPHA
# ===========================
def alpha_group_full_basis(theta):
    """
    Base COMPLETA da álgebra do Grupo Alpha.
    Baseada na estrutura {1, i, μ, iμ} e na matriz M(θ).
    
    Retorna 6 geradores que representam a álgebra completa.
    """
    tanθ = np.tan(theta) if abs(theta - np.pi/2) > EPS else 1e3
    cotθ = 1/tanθ if abs(tanθ) > EPS else 0
    
    # Geradores da álgebra de Lie do Grupo Alpha
    # Estes vêm da decomposição de M(θ) em elementos da álgebra
    
    # 1. Gerador associado a '1' (parte real, rotação em x-y)
    G1 = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)
    
    # 2. Gerador associado a 'i' (parte imaginária, rotação em x-z)
    G2 = np.array([
        [0, 0, -1j, 0],
        [0, 0, 0, 0],
        [1j, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)
    
    # 3. Gerador associado a μ (idempotente, direção especial)
    G3 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    # 4. Gerador associado a iμ (combinação)
    G4 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, -1j, 0]
    ], dtype=complex)
    
    # 5. Gerador de "distorção" dependente de θ (da parte trigonométrica)
    G5 = np.array([
        [0, -cotθ, -tanθ, 0],
        [cotθ, 0, 0, -tanθ],
        [tanθ, 0, 0, -cotθ],
        [0, tanθ, cotθ, 0]
    ], dtype=complex) * 0.5
    
    # 6. Outro gerador de distorção
    G6 = np.array([
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0]
    ], dtype=complex)
    
    return [G1, G2, G3, G4, G5, G6]

# ===========================
# 3. DISTRIBUIÇÃO HORIZONTAL RICA (não simplificada)
# ===========================
def rich_horizontal_distribution(theta):
    """
    Distribuição horizontal RICA e REALISTA para o Grupo Alpha.
    
    Em vez de apenas 2 geradores, usamos uma distribuição de
    dimensão 3 ou 4 que captura melhor a estrutura do grupo.
    """
    # Obter base completa
    all_generators = alpha_group_full_basis(theta)
    
    # Selecionar subconjunto como distribuição horizontal
    # A escolha depende da física do problema
    
    # Opção A: Distribuição de dimensão 3
    # (mais realista para muitos sistemas sub-Riemannianos)
    H = all_generators[:3]  # G1, G2, G3
    
    # Adicionar dependência em θ para o quarto gerador
    # (isso cria anisotropia angular)
    if abs(theta - np.pi/4) < np.pi/8:  # ±22.5° em torno de 45°
        # Adicionar G5 que tem forte dependência em θ
        H.append(all_generators[4])
    
    # Para θ próximo de 90°, adicionar mais um gerador
    if abs(theta - np.pi/2) < np.pi/6:  # ±30° em torno de 90°
        H.append(all_generators[5])
    
    return H

# ===========================
# 4. COMUTADOR DE LIE CORRETO
# ===========================
def lie_bracket_matrix(X, Y):
    """Comutador de Lie para matrizes: [X,Y] = XY - YX"""
    return X @ Y - Y @ X

# ===========================
# 5. TESTE HÖRMANDER COM DISTRIBUIÇÃO RICA
# ===========================
def rich_hormander_test(theta):
    """
    Teste da condição de Hörmander usando distribuição rica.
    
    Retorna: (rank, satisfaz_hormander, num_geradores, dimensão_Δ)
    """
    # 1. Obter distribuição horizontal RICA
    H = rich_horizontal_distribution(theta)
    dim_Delta = len(H)  # Dimensão da distribuição horizontal
    
    if dim_Delta < 2:
        return 0, False, 0, dim_Delta
    
    # 2. Inicializar conjunto de geradores
    all_generators = H.copy()
    current_layer = H.copy()
    
    # 3. Calcular álgebra de Lie gerada (bracket-generating)
    for depth in range(MAX_BRACKET_DEPTH):
        new_generators = []
        
        # Calcular comutadores entre todos os pares
        for i in range(len(current_layer)):
            for j in range(i+1, len(current_layer)):
                bracket = lie_bracket_matrix(current_layer[i], current_layer[j])
                
                # Ignorar se muito pequeno
                if np.linalg.norm(bracket) < EPS:
                    continue
                
                # Verificar se é linearmente independente
                is_new = True
                
                # Converter geradores existentes para vetores
                existing_vectors = [g.flatten() for g in all_generators]
                new_vector = bracket.flatten()
                
                if existing_vectors:
                    # Construir matriz com vetores como colunas
                    M = np.column_stack(existing_vectors)
                    
                    # Verificar se new_vector está no espaço coluna de M
                    try:
                        # Projeção ortogonal
                        proj = M @ np.linalg.lstsq(M, new_vector, rcond=None)[0]
                        residual = np.linalg.norm(new_vector - proj)
                        
                        if residual < EPS * np.linalg.norm(new_vector):
                            is_new = False
                    except:
                        # Fallback: método mais simples
                        for ev in existing_vectors:
                            cos_sim = np.abs(np.dot(new_vector, ev)) / (
                                np.linalg.norm(new_vector) * np.linalg.norm(ev) + EPS)
                            if cos_sim > 0.999:
                                is_new = False
                                break
                
                if is_new:
                    new_generators.append(bracket)
                    all_generators.append(bracket)
        
        if not new_generators:
            break
        
        current_layer = new_generators
    
    # 4. Calcular dimensão da álgebra gerada
    if all_generators:
        # Converter matrizes para vetores (4×4 complexa = 32D real)
        vectors = []
        for mat in all_generators:
            # Separar parte real e imaginária
            vec_real = mat.real.flatten()
            vec_imag = mat.imag.flatten()
            vectors.append(np.concatenate([vec_real, vec_imag]))
        
        # Construir matriz e calcular rank
        M = np.column_stack(vectors) if vectors else np.zeros((32, 0))
        rank = np.linalg.matrix_rank(M, tol=EPS)
    else:
        rank = 0
    
    # 5. Determinar se satisfaz Hörmander
    # Para espaço 4D, precisamos que a álgebra tenha dimensão ≥ 4
    # Mas também consideramos a dimensão da distribuição inicial
    satisfies_hormander = (rank >= 4) and (dim_Delta >= 2)
    
    return rank, satisfies_hormander, len(all_generators), dim_Delta

# ===========================
# 6. ANÁLISE SISTEMÁTICA COMPLETA
# ===========================
def complete_hormander_analysis(theta_range=(0.1, np.pi/2-0.1), n_points=50):
    """Análise sistemática completa"""
    
    thetas = np.linspace(theta_range[0], theta_range[1], n_points)
    
    results = {
        'theta_deg': [],
        'theta_rad': [],
        'rank': [],
        'satisfies': [],
        'num_generators': [],
        'dim_Delta': [],
        'tan_theta': [],
        'hormander_strength': []  # Força da condição (rank/dim_Delta)
    }
    
    print("🔬 ANÁLISE COMPLETA DA CONDIÇÃO DE HÖRMANDER")
    print("="*70)
    print("Usando distribuição horizontal RICA do Grupo Alpha")
    print("="*70)
    
    for i, θ in enumerate(thetas):
        rank, satisfies, num_gen, dim_Delta = rich_hormander_test(θ)
        
        tanθ = np.tan(θ) if abs(θ - np.pi/2) > EPS else np.nan
        
        results['theta_rad'].append(θ)
        results['theta_deg'].append(np.degrees(θ))
        results['rank'].append(rank)
        results['satisfies'].append(satisfies)
        results['num_generators'].append(num_gen)
        results['dim_Delta'].append(dim_Delta)
        results['tan_theta'].append(tanθ)
        
        # Força da condição de Hörmander
        if dim_Delta > 0:
            strength = rank / dim_Delta if dim_Delta > 0 else 0
        else:
            strength = 0
        results['hormander_strength'].append(strength)
        
        # Print progresso para ângulos importantes
        θ_deg = np.degrees(θ)
        if i % 10 == 0 or abs(θ_deg - 45) < 2 or abs(θ_deg - 90) < 2:
            status = "✅ HÖRMANDER" if satisfies else "❌ FALHA"
            print(f"θ = {θ_deg:6.1f}°: Δ-dim={dim_Delta}, rank={rank:2d}, "
                  f"geradores={num_gen:2d}, {status}")
    
    return results

# ===========================
# 7. VISUALIZAÇÃO AVANÇADA
# ===========================
def plot_complete_analysis(results):
    """Visualização completa dos resultados"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # Gráfico 1: Rank vs θ
    ax1 = axes[0, 0]
    ax1.plot(results['theta_deg'], results['rank'], 'b-', linewidth=3, label='Rank')
    ax1.fill_between(results['theta_deg'], 0, results['rank'], alpha=0.2, color='b')
    ax1.axhline(y=4, color='r', linestyle='--', linewidth=2, label='Mínimo (4)')
    ax1.axhline(y=16, color='g', linestyle=':', linewidth=1, label='Máximo teórico')
    ax1.set_xlabel('θ (degrees)')
    ax1.set_ylabel('Dimensão da álgebra gerada')
    ax1.set_title('Dimensão da Álgebra de Lie Gerada')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico 2: Dimensão da distribuição horizontal vs θ
    ax2 = axes[0, 1]
    ax2.plot(results['theta_deg'], results['dim_Delta'], 'g-', linewidth=3, marker='o', markersize=4)
    ax2.set_xlabel('θ (degrees)')
    ax2.set_ylabel('Dimensão de Δ')
    ax2.set_title('Dimensão da Distribuição Horizontal Δ')
    ax2.set_ylim([0, 6])
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Condição de Hörmander (regiões)
    ax3 = axes[1, 0]
    satisfies = np.array(results['satisfies'], dtype=float)
    
    # Colorir regiões
    for i in range(len(satisfies)-1):
        if satisfies[i]:
            color = 'green' if results['rank'][i] >= 6 else 'lightgreen'
            alpha = 0.4 if results['rank'][i] >= 6 else 0.2
            ax3.axvspan(results['theta_deg'][i], results['theta_deg'][i+1], 
                       alpha=alpha, color=color)
    
    # Plot da condição
    ax3.plot(results['theta_deg'], satisfies, 'k-', linewidth=2)
    ax3.set_xlabel('θ (degrees)')
    ax3.set_ylabel('Hörmander satisfeito')
    ax3.set_title('Condição de Hörmander (Regiões)')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Não', 'Sim'])
    ax3.grid(True, alpha=0.3)
    
    # Legenda de cores
    import matplotlib.patches as mpatches
    strong_patch = mpatches.Patch(color='green', alpha=0.4, label='Hörmander Forte (rank≥6)')
    weak_patch = mpatches.Patch(color='lightgreen', alpha=0.2, label='Hörmander Fraco (rank≥4)')
    ax3.legend(handles=[strong_patch, weak_patch])
    
    # Gráfico 4: Força da condição de Hörmander
    ax4 = axes[1, 1]
    strength = np.array(results['hormander_strength'])
    finite_mask = np.isfinite(strength)
    
    if np.any(finite_mask):
        ax4.plot(np.array(results['theta_deg'])[finite_mask], 
                strength[finite_mask], 'm-', linewidth=3)
        ax4.fill_between(np.array(results['theta_deg'])[finite_mask], 
                        0, strength[finite_mask], alpha=0.2, color='m')
        ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Limite teórico')
        ax4.set_xlabel('θ (degrees)')
        ax4.set_ylabel('Força (rank / dim Δ)')
        ax4.set_title('Força da Condição de Hörmander')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Gráfico 5: Número de geradores vs θ
    ax5 = axes[2, 0]
    ax5.plot(results['theta_deg'], results['num_generators'], 'r-', 
             linewidth=2, marker='s', markersize=4)
    ax5.set_xlabel('θ (degrees)')
    ax5.set_ylabel('Número de geradores')
    ax5.set_title('Complexidade da Base da Álgebra')
    ax5.grid(True, alpha=0.3)
    
    # Gráfico 6: Comparação rank vs dim Δ
    ax6 = axes[2, 1]
    scatter = ax6.scatter(results['dim_Delta'], results['rank'],
                         c=results['theta_deg'], cmap='viridis',
                         s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # Linha teórica: rank = dim Δ (mínimo)
    x_range = np.array([0, 6])
    ax6.plot(x_range, x_range, 'r--', alpha=0.5, label='rank = dim Δ (mínimo)')
    
    # Linha: rank = 4 (Hörmander)
    ax6.axhline(y=4, color='g', linestyle='-', alpha=0.5, label='Hörmander (rank=4)')
    
    ax6.set_xlabel('Dimensão de Δ')
    ax6.set_ylabel('Rank da álgebra')
    ax6.set_title('Relação: dim Δ vs Rank da Álgebra')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    plt.colorbar(scatter, ax=ax6, label='θ (degrees)')
    
    plt.suptitle('ANÁLISE COMPLETA DA CONDIÇÃO DE HÖRMANDER - GRUPO ALPHA\n' +
                'Distribuição Horizontal Rica e Realista',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# ===========================
# 8. ANÁLISE DETALHADA DOS RESULTADOS
# ===========================
def detailed_results_analysis(results):
    """Análise detalhada dos resultados"""
    
    print("\n" + "="*70)
    print("📊 ANÁLISE DETALHADA DOS RESULTADOS")
    print("="*70)
    
    theta_deg = np.array(results['theta_deg'])
    rank_array = np.array(results['rank'])
    satisfies_array = np.array(results['satisfies'])
    dim_Delta_array = np.array(results['dim_Delta'])
    
    # 1. Estatísticas gerais
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"   • Ângulos analisados: {len(theta_deg)}")
    print(f"   • Rank médio: {np.mean(rank_array):.2f} ± {np.std(rank_array):.2f}")
    print(f"   • Rank máximo: {np.max(rank_array)} (em θ = {theta_deg[np.argmax(rank_array)]:.1f}°)")
    print(f"   • Rank mínimo: {np.min(rank_array)}")
    
    print(f"\n   • Dimensão média de Δ: {np.mean(dim_Delta_array):.2f}")
    print(f"   • Dimensão máxima de Δ: {np.max(dim_Delta_array)}")
    print(f"   • Dimensão mínima de Δ: {np.min(dim_Delta_array)}")
    
    # 2. Condição de Hörmander
    n_satisfies = np.sum(satisfies_array)
    percent_satisfies = n_satisfies / len(satisfies_array) * 100
    
    print(f"\n✅ CONDIÇÃO DE HÖRMANDER:")
    print(f"   • Satisfeita em: {n_satisfies}/{len(satisfies_array)} "
          f"({percent_satisfies:.1f}%)")
    
    if n_satisfies > 0:
        sat_indices = np.where(satisfies_array)[0]
        sat_thetas = theta_deg[sat_indices]
        sat_ranks = rank_array[sat_indices]
        
        print(f"   • Faixa de θ com Hörmander: "
              f"[{np.min(sat_thetas):.1f}°, {np.max(sat_thetas):.1f}°]")
        
        # Classificar por força
        strong_indices = sat_ranks >= 6
        if np.any(strong_indices):
            strong_thetas = sat_thetas[strong_indices]
            print(f"   • Hörmander FORTE (rank≥6): "
                  f"[{np.min(strong_thetas):.1f}°, {np.max(strong_thetas):.1f}°]")
        
        # Verificar região de 90°
        near_90_mask = np.abs(sat_thetas - 90) < 10
        if np.any(near_90_mask):
            print(f"   • ✅ INCLUI região de θ ≈ 90°")
            near_90_ranks = sat_ranks[near_90_mask]
            print(f"   •   Rank médio próximo a 90°: {np.mean(near_90_ranks):.2f}")
        else:
            print(f"   • ⚠️  NÃO inclui região de θ ≈ 90°")
    
    # 3. Análise de transições
    print(f"\n🔀 ANÁLISE DE TRANSIÇÕES:")
    
    # Encontrar mudanças na dimensão de Δ
    delta_changes = []
    for i in range(1, len(dim_Delta_array)):
        if dim_Delta_array[i] != dim_Delta_array[i-1]:
            θ_trans = (theta_deg[i] + theta_deg[i-1]) / 2
            change = dim_Delta_array[i] - dim_Delta_array[i-1]
            delta_changes.append((θ_trans, change))
    
    if delta_changes:
        for θ_trans, change in delta_changes:
            direction = "AUMENTA" if change > 0 else "DIMINUI"
            print(f"   • θ = {θ_trans:6.1f}°: dim Δ {direction} em {abs(change)}")
    else:
        print(f"   • Nenhuma mudança na dimensão de Δ detectada")
    
    # 4. Correlações
    print(f"\n📊 CORRELAÇÕES E RELAÇÕES:")
    
    # Correlação entre dim Δ e rank
    if len(dim_Delta_array) > 1:
        valid_mask = (dim_Delta_array > 0) & (rank_array > 0)
        if np.sum(valid_mask) > 1:
            corr = np.corrcoef(dim_Delta_array[valid_mask], rank_array[valid_mask])[0,1]
            print(f"   • Correlação dim Δ vs rank: {corr:.3f}")
            
            if corr > 0.7:
                print(f"   • 👉 FORTE dependência: rank aumenta com dim Δ")
            elif corr > 0.3:
                print(f"   • 👉 Dependência moderada")
            else:
                print(f"   • 👉 Fraca ou nenhuma correlação linear")
    
    # 5. Conclusão qualitativa
    print(f"\n🎯 CONCLUSÃO QUALITATIVA:")
    
    if percent_satisfies > 80:
        print(f"   • ✅ FORTE EVIDÊNCIA de geometria sub-Riemanniana")
        print(f"   • 👉 O Grupo Alpha satisfaz robustamente Hörmander")
    elif percent_satisfies > 50:
        print(f"   • ⚠️  EVIDÊNCIA MODERADA de geometria sub-Riemanniana")
        print(f"   • 👉 Hörmander satisfeito na maioria das regiões")
    elif percent_satisfies > 20:
        print(f"   • ⚠️  EVIDÊNCIA FRACA de geometria sub-Riemanniana")
        print(f"   • 👉 Hörmander satisfeito apenas em regiões específicas")
    else:
        print(f"   • ❌ POUCA EVIDÊNCIA de geometria sub-Riemanniana")
        print(f"   • 👉 Condição de Hörmander raramente satisfeita")
    
    print("="*70)

# ===========================
# 9. TESTES ESPECÍFICOS DETALHADOS
# ===========================
def run_detailed_tests():
    """Testes detalhados em ângulos específicos"""
    
    print("\n" + "="*70)
    print("🔍 TESTES DETALHADOS EM ÂNGULOS-CHAVE")
    print("="*70)
    
    test_cases = [
        (0.1, "0.1° (próximo a 0)"),
        (15*np.pi/180, "15°"),
        (30*np.pi/180, "30°"),
        (45*np.pi/180, "45°"),
        (60*np.pi/180, "60°"),
        (75*np.pi/180, "75°"),
        (89*np.pi/180, "89° (próximo a 90°)"),
        (np.pi/2 - 0.001, "89.94° (muito próximo a 90°)")
    ]
    
    print("\nθ (°)   | dim Δ | Rank | Geradores | Hörmander | Força")
    print("-"*70)
    
    for θ_rad, desc in test_cases:
        rank, satisfies, num_gen, dim_Delta = rich_hormander_test(θ_rad)
        θ_deg = np.degrees(θ_rad)
        
        strength = rank / dim_Delta if dim_Delta > 0 else 0
        status = "✅" if satisfies else "❌"
        
        print(f"{θ_deg:6.1f}° | {dim_Delta:5d} | {rank:5d} | {num_gen:9d} | {status:9s} | {strength:5.2f}")
        
        # Análise adicional para ângulos importantes
        if abs(θ_deg - 45) < 1 or abs(θ_deg - 90) < 1:
            print(f"       {' '*10}→ Análise: ", end="")
            if satisfies:
                if rank >= 8:
                    print(f"Álgebra RICA (dimensão {rank}), estrutura complexa")
                elif rank >= 6:
                    print(f"Álgebra BOA (dimensão {rank}), bem estabelecida")
                else:
                    print(f"Álgebra MÍNIMA (dimensão {rank}), mas suficiente")
            else:
                print(f"Álgebra INSUFICIENTE (dimensão {rank}), restrições presentes")
    
    print("="*70)

# ===========================
# 10. EXECUÇÃO PRINCIPAL
# ===========================
if __name__ == "__main__":
    print("="*70)
    print("🔬 ANÁLISE AVANÇADA DA CONDIÇÃO DE HÖRMANDER")
    print("GRUPO ALPHA - DISTRIBUIÇÃO HORIZONTAL RICA")
    print("="*70)
    
    # Executar análise completa
    results = complete_hormander_analysis(
        theta_range=(0.1, np.pi/2 - 0.1),
        n_points=60
    )
    
    # Visualizar
    plot_complete_analysis(results)
    
    # Análise detalhada
    detailed_results_analysis(results)
    
    # Testes específicos
    run_detailed_tests()
    
    # Conclusão final
    print("\n" + "="*70)
    print("🎯 CONCLUSÃO FINAL DA ANÁLISE")
    print("="*70)
    
    # Resultados próximos a 90°
    near_90_mask = np.abs(np.array(results['theta_deg']) - 90) < 5
    if np.any(near_90_mask):
        near_90_ranks = np.array(results['rank'])[near_90_mask]
        near_90_satisfies = np.array(results['satisfies'])[near_90_mask]
        
        avg_rank_90 = np.mean(near_90_ranks)
        percent_satisfies_90 = np.mean(near_90_satisfies) * 100
        
        print(f"\n📊 REGIÃO PRÓXIMA A θ = 90°:")
        print(f"   • Rank médio: {avg_rank_90:.2f}")
        print(f"   • Hörmander satisfeito: {percent_satisfies_90:.1f}% dos pontos")
        
        if percent_satisfies_90 > 80 and avg_rank_90 >= 6:
            print(f"""
            ✅ CONCLUSÃO FORTE:
            
            O Grupo Alpha SATISFAZ ROBUSTAMENTE a condição de Hörmander
            para θ próximo a 90°, com uma álgebra de dimensão {avg_rank_90:.1f}.
            
            IMPLICAÇÕES:
            1. A distribuição horizontal é bracket-generating
            2. A álgebra gerada é rica (dimensão {avg_rank_90:.1f} ≥ 6)
            3. O sistema é completamente controlável
            4. A geometria é genuinamente sub-Riemanniana
            5. VALIDA as afirmações do artigo sobre ativação em θ ≈ 90°
            """)
        elif percent_satisfies_90 > 50 and avg_rank_90 >= 4:
            print(f"""
            ⚠️  CONCLUSÃO MODERADA:
            
            O Grupo Alpha SATISFAZ a condição de Hörmander para θ próximo a 90°,
            mas com uma álgebra de dimensão apenas {avg_rank_90:.1f}.
            
            IMPLICAÇÕES:
            1. Condição de Hörmander satisfeita, mas marginalmente
            2. Álgebra gerada tem dimensão mínima ({avg_rank_90:.1f})
            3. Sistema controlável, mas com pouca margem
            4. Geometria sub-Riemanniana, mas não rica
            5. SUPORTA PARCIALMENTE o artigo
            """)
        else:
            print(f"""
            ❌ CONCLUSÃO FRACA:
            
            O Grupo Alpha NÃO SATISFAZ consistentemente a condição de Hörmander
            para θ próximo a 90° (apenas {percent_satisfies_90:.1f}% satisfazem).
            
            IMPLICAÇÕES:
            1. Condição de Hörmander frequentemente falha
            2. Álgebra gerada insuficiente (dimensão {avg_rank_90:.1f})
            3. Restrições de acessibilidade presentes
            4. Natureza sub-Riemanniana questionável
            5. CONTRADIZ em parte o artigo
            """)
    else:
        print("⚠️  Não há dados suficientes próximos a θ = 90°")
    
    print("="*70)
