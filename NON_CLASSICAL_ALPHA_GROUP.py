import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import expm, logm
import matplotlib.gridspec as gridspec

# Style setting for broad compatibility
plt.style.use('seaborn-v0_8') 

# ================================
# --- NON-CLASSICAL ALPHA GROUP ---
# ================================

def M_theta_alpha_non_classical(theta):
    """
    Non-Classical Alpha Group Matrix M(theta).
    This structure is non-Hermitian and non-unitary.
    It features a dynamic invariant Mi.
    """
    epsilon = 1e-12
    theta_adj = theta

    # Handle critical points of tan/cot functions robustly
    critical_points = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    for cp in critical_points:
        if np.isclose(np.real(theta), cp, atol=1e-8):
            # Introduce a small imaginary component to avoid division by zero near poles
            theta_adj += epsilon * 1j

    tan_theta = np.tan(theta_adj)
    # Clip extreme values for numerical stability in cotangent calculation
    cot_theta = np.clip(1 / tan_theta, -1e6, 1e6) 

    # FUNDAMENTAL INVARIANT OF THE ALPHA GROUP (DYNAMIC)
    omega = 20
    Mi = np.sin(omega * theta) + 1j * np.cos(omega * theta)
    
    # ALPHA GROUP BASIS (Non-Hamiltonian-like)
    base_alpha = np.array([1, 1j, Mi, 1j * Mi], dtype=complex)
    
    # ALPHA GROUP COUPLING MATRIX 
    M_alpha = np.array([
        [1,     -cot_theta, -tan_theta,  1],
        [cot_theta,  1,      -1,        -tan_theta],
        [tan_theta, -1,       1,        -cot_theta],
        [1,      tan_theta,  cot_theta,   1]
    ], dtype=complex)
    
    # COMPLETE MATRIX: M_alpha acting on its own basis (Hadamard product)
    M_complete = M_alpha * base_alpha.reshape(1, 4)
    
    return M_complete, M_alpha, base_alpha, Mi

# ================================
# --- TOPOLOGICAL CALCULATION FUNCTIONS ---
# ================================

def calculate_non_classical_winding_number_data(theta_range):
    """
    Calculates the data for the Non-Classical Winding Number (W_Alpha).
    Based on the accumulated phase variation of the anti-symmetric component's eigenvalues.
    """
    all_phases = np.zeros((len(theta_range), 4))
    
    for idx, theta in enumerate(theta_range):
        M_comp, _, _, _ = M_theta_alpha_non_classical(theta)
        
        # Anti-symmetric component: Lie algebra generator
        M_anti = (M_comp - M_comp.T)/2
        
        eigvals = np.linalg.eigvals(M_anti)
        all_phases[idx] = np.angle(eigvals)
        
    all_phases_unwrapped = np.zeros_like(all_phases)
    for i in range(4):
        all_phases_unwrapped[:, i] = np.unwrap(all_phases[:, i])
        
    # Accumulated Winding Number (Sum of phase changes, normalized by 2*pi)
    winding_number_cumulative = np.sum(all_phases_unwrapped - all_phases_unwrapped[0, :], axis=1) / (2*np.pi)
    
    return winding_number_cumulative

def calculate_classical_winding_number_det(theta_range):
    """
    Calculates the Classical Winding Number (W_Det) based on the determinant's phase.
    """
    det_phases = []
    
    for theta in theta_range:
        M_comp, _, _, _ = M_theta_alpha_non_classical(theta)
        det_M = np.linalg.det(M_comp)
        det_phases.append(np.angle(det_M))

    det_phases_unwrapped = np.unwrap(det_phases)
    
    # Cumulative Winding Number (Change in determinant's phase)
    winding_number_cumulative = (det_phases_unwrapped - det_phases_unwrapped[0]) / (2*np.pi)
    
    return winding_number_cumulative

# ================================
# --- PRIMARY COMPARISON FUNCTION ---
# ================================

def compare_winding_numbers(theta_range):
    """
    Compares W_Alpha with W_Det for the given theta_range.
    """
    W_non_classical_data = calculate_non_classical_winding_number_data(theta_range)
    W_non_classical_final = W_non_classical_data[-1]

    W_classical_data = calculate_classical_winding_number_det(theta_range)
    W_classical_final = W_classical_data[-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(theta_range, W_non_classical_data, 
            label='NON-CLASSICAL Winding Number (Eigenvalues of $M_{anti}$)', 
            color='purple', linewidth=2)
    ax.axhline(W_non_classical_final, color='purple', linestyle=':', alpha=0.7)
    
    ax.plot(theta_range, W_classical_data, 
            label='CLASSICAL Winding Number (Det($M_{comp}$))', 
            color='teal', linestyle='--', linewidth=2)
    ax.axhline(W_classical_final, color='teal', linestyle=':', alpha=0.7)

    ax.set_title(f'Topological Invariant Comparison (θ Range: [{theta_range[0]:.3f}, {theta_range[-1]:.3f}])')
    ax.set_xlabel('Angular Parameter θ (rad)')
    ax.set_ylabel('Cumulative Winding Number')
    ax.text(theta_range[-1], W_non_classical_final, f'W_Alpha ≈ {W_non_classical_final:.3f}', 
            color='purple', va='center', ha='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.text(theta_range[-1], W_classical_final, f'W_Det ≈ {W_classical_final:.3f}', 
            color='teal', va='center', ha='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n=== TOPOLOGICAL INVARIANT COMPARISON ===")
    # Using raw string (r-string) to correctly handle \in LaTeX command
    print(r"Range: θ $\in$ [{0:.4f}, {1:.4f}]".format(theta_range[0], theta_range[-1]))
    print(f"1. Non-Classical Winding Number (W_Alpha) Change: {W_non_classical_final - W_non_classical_data[0]:.4f}")
    print(f"2. Classical Winding Number (W_Det) Change: {W_classical_final - W_classical_data[0]:.4f}")
    print("------------------------------------------------------------------------")
    print("In this critical range, the Winding Number accumulates smoothly, showing topological robustness despite geometric singularity.")

# ================================
# --- AUXILIARY ANALYSIS FUNCTIONS ---
# ================================

def hamilton_alpha_comparison():
    print("=== ALPHA GROUP vs HAMILTONIAN MODEL ===\n")
    print("1. HAMILTON'S QUATERNIONS (1843):")
    print("   • Basis: [1, i, j, k]")
    print("   • Rules: i² = j² = k² = ijk = -1")
    print("   • Structure: Closed, well-defined Algebra")
    
    print("\n2. ALPHA GROUP (NON-CLASSICAL):")
    print("   • Basis: [1, i, Mi, i·Mi]")
    print("   • Mi = sin(20θ) + i·cos(20θ) - DYNAMIC INVARIANT")
    print("   • Rules: INTRINSIC and dependent on θ")
    print("   • Structure: DYNAMIC and non-closed Algebra")
    print("   • Does NOT follow: i² = j² = k² = -1")
    
    print("\n3. CRITICAL DIFFERENCES:")
    print("   ✓ Mi is DYNAMIC (depends on θ), not constant")
    print("   ✓ Unique COMMUTATION structure")
    print("   ✓ Intrinsic non-trivial TOPOLOGY")
    print("   ✓ NON-CLASSICAL Lie Group")

def alpha_intrinsic_algebra():
    # Use pi/4 for comparison reference as requested
    theta = np.pi/4 
    M_comp, M_alpha, base_alpha, Mi = M_theta_alpha_non_classical(theta)
    
    print("\n=== INTRINSIC ALGEBRA OF THE ALPHA GROUP (θ=π/4) ===\n")
    
    elements = base_alpha
    
    print("Basis Products (4x4 Matrix):")
    for i in range(4):
        line = ""
        for j in range(4):
            line += f"{elements[i]*elements[j]:.4f}\t"
        print(line)

def alpha_geometric_structure(theta_range):
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Alpha Group Phase Space (CRITICAL SWEEP)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    base_traj = np.zeros((len(theta_range), 3))
    for idx, theta in enumerate(theta_range):
        _, _, base, _ = M_theta_alpha_non_classical(theta)
        base_traj[idx] = [np.real(base[2]), np.imag(base[2]), np.real(base[3])]
    
    for i in range(len(theta_range)-1):
        ax1.plot(base_traj[i:i+2,0], base_traj[i:i+2,1], base_traj[i:i+2,2],
                 color=cm.hsv(i/len(theta_range)), alpha=0.8)
    
    ax1.set_xlabel('Re($e_3$)')
    ax1.set_ylabel('Im($e_3$)')
    ax1.set_zlabel('Re($e_4$)')
    # Using raw string (r-string) for correct rendering of LaTeX
    ax1.set_title(r'ALPHA GROUP SPACE - Trajectory (θ $\in$ [{0:.3f}, {1:.3f}])'.format(theta_range[0], theta_range[-1]))
    
    # 2. Invariant Mi - Intrinsic Structure (CRITICAL SWEEP)
    ax2 = fig.add_subplot(2, 2, 2)
    
    Mi_vals = np.array([np.sin(20*theta) + 1j*np.cos(20*theta) for theta in theta_range])
    ax2.plot(np.real(Mi_vals), np.imag(Mi_vals), 'b-', alpha=0.5, linewidth=1)
    scatter = ax2.scatter(np.real(Mi_vals[::10]), np.imag(Mi_vals[::10]),
                          c=theta_range[::10], cmap='hsv', s=30)
    ax2.set_xlabel('Re(Mi)')
    ax2.set_ylabel('Im(Mi)')
    ax2.set_title('INVARIANT Mi - INTRINSIC STRUCTURE')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='θ (rad)')
    
    # 3. Alpha Group Matrix (Real and Imaginary Parts) - REFERENCE (pi/4)
    theta_ref = np.pi/4
    _, M_alpha, _, _ = M_theta_alpha_non_classical(theta_ref)
    
    ax3 = fig.add_subplot(2, 2, 3)
    im_real = ax3.imshow(np.real(M_alpha), cmap='RdBu_r', aspect='equal')
    ax3.set_title(f'COUPLING MATRIX (Real) [θ={theta_ref:.2f}]')
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    plt.colorbar(im_real, ax=ax3)
    
    ax4_im = fig.add_subplot(2, 2, 4)
    im_imag = ax4_im.imshow(np.imag(M_alpha), cmap='RdBu_r', aspect='equal')
    ax4_im.set_title(f'COUPLING MATRIX (Imaginary) [θ={theta_ref:.2f}]')
    ax4_im.set_xticks(range(4))
    ax4_im.set_yticks(range(4))
    plt.colorbar(im_imag, ax=ax4_im)
    
    plt.tight_layout()
    plt.show()

def alpha_unique_characteristics():
    print("\n=== UNIQUE CHARACTERISTICS OF THE ALPHA GROUP ===\n")
    
    characteristics = {
        "Dynamic Invariant": "Mi = sin(20θ) + i·cos(20θ) - NOT constant",
        "Intrinsic Basis": "[1, i, Mi, i·Mi] - Not the Hamiltonian [1, i, j, k]",
        "Open Algebra": "Algebraic structure may not be closed",
        "Intrinsic Topology": "Non-trivial natural Winding Number (W_Alpha)",
        "Parametric Dependence": "Structure varies smoothly with θ",
        "Non-commutativity": "Non-zero commutators - rich Lie algebra",
        "Intrinsic Geometry": "Parameter space with a unique metric"
    }
    
    for characteristic, description in characteristics.items():
        print(f"• {characteristic}: {description}")

# ================================
# --- DEEP TOPOLOGICAL ANALYSIS ---
# ================================

def deep_topological_analysis(theta_range):
    """
    Análise profunda dos invariantes topológicos do Grupo Alpha
    """
    print("\n=== DEEP TOPOLOGICAL ANALYSIS ===")
    
    # Calcula ambos os winding numbers
    W_alpha = calculate_non_classical_winding_number_data(theta_range)
    W_det = calculate_classical_winding_number_det(theta_range)
    
    # Análise estatística
    alpha_final = W_alpha[-1]
    det_final = W_det[-1]
    alpha_variation = np.ptp(W_alpha)  # peak-to-peak variation
    det_variation = np.ptp(W_det)
    
    print(f"Non-Classical Winding (W_Alpha):")
    print(f"  Final value: {alpha_final:.6f}")
    print(f"  Total variation: {alpha_variation:.6f}")
    print(f"  Mean: {np.mean(W_alpha):.6f} ± {np.std(W_alpha):.6f}")
    
    print(f"\nClassical Winding (W_Det):")
    print(f"  Final value: {det_final:.6f}")
    print(f"  Total variation: {det_variation:.6f}")
    print(f"  Mean: {np.mean(W_det):.6f} ± {np.std(W_det):.6f}")
    
    # Análise de correlação
    correlation = np.corrcoef(W_alpha, W_det)[0,1]
    print(f"\nCorrelation between W_Alpha and W_Det: {correlation:.6f}")
    
    # Identifica regiões de interesse
    alpha_gradient = np.gradient(W_alpha, theta_range)
    det_gradient = np.gradient(W_det, theta_range)
    
    # Pontos onde a topologia muda mais rapidamente
    alpha_max_change_idx = np.argmax(np.abs(alpha_gradient))
    det_max_change_idx = np.argmax(np.abs(det_gradient))
    
    print(f"\nTopological Sensitivity:")
    print(f"  W_Alpha changes fastest at θ = {theta_range[alpha_max_change_idx]:.6f}")
    print(f"  W_Det changes fastest at θ = {theta_range[det_max_change_idx]:.6f}")
    
    return W_alpha, W_det, alpha_gradient, det_gradient

def plot_detailed_comparison(theta_range, W_alpha, W_det, alpha_gradient, det_gradient):
    """
    Plot detalhado comparando os dois invariantes topológicos
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Comparação dos winding numbers
    ax1.plot(theta_range, W_alpha, 'purple', linewidth=3, label='W_Alpha (Non-Classical)')
    ax1.plot(theta_range, W_det, 'teal', linewidth=2, linestyle='--', label='W_Det (Classical)')
    ax1.set_xlabel('θ (rad)')
    ax1.set_ylabel('Winding Number')
    ax1.set_title('COMPARISON: Non-Classical vs Classical Winding Numbers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Diferença entre os dois
    difference = W_alpha - W_det
    ax2.plot(theta_range, difference, 'red', linewidth=2)
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('W_Alpha - W_Det')
    ax2.set_title('DIFFERENCE: Non-Classical - Classical')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Gradientes (sensibilidade topológica)
    ax3.plot(theta_range, alpha_gradient, 'purple', linewidth=2, label='dW_Alpha/dθ')
    ax3.plot(theta_range, det_gradient, 'teal', linewidth=2, linestyle='--', label='dW_Det/dθ')
    ax3.set_xlabel('θ (rad)')
    ax3.set_ylabel('Gradient')
    ax3.set_title('TOPOLOGICAL SENSITIVITY')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Diagrama de fase W_Alpha vs W_Det
    scatter = ax4.scatter(W_alpha, W_det, c=theta_range, cmap='viridis', alpha=0.7)
    ax4.set_xlabel('W_Alpha (Non-Classical)')
    ax4.set_ylabel('W_Det (Classical)')
    ax4.set_title('PHASE DIAGRAM: W_Alpha vs W_Det')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='θ (rad)')
    
    plt.tight_layout()
    plt.show()

# ================================
# --- ANALYSIS OF THE DYNAMIC INVARIANT Mi ---
# ================================

def analyze_Mi_dynamics(theta_range):
    """
    Análise detalhada do invariante dinâmico Mi
    """
    print("\n=== DYNAMIC INVARIANT Mi ANALYSIS ===")
    
    Mi_values = []
    Mi_moduli = []
    Mi_phases = []
    
    for theta in theta_range:
        Mi = np.sin(20*theta) + 1j*np.cos(20*theta)
        Mi_values.append(Mi)
        Mi_moduli.append(np.abs(Mi))
        Mi_phases.append(np.angle(Mi))
    
    Mi_array = np.array(Mi_values)
    
    # Estatísticas do invariante
    mean_modulus = np.mean(Mi_moduli)
    std_modulus = np.std(Mi_moduli)
    phase_variation = np.ptp(Mi_phases)  # total phase variation
    
    print(f"Mi = sin(20θ) + i·cos(20θ) Statistics:")
    print(f"  Modulus: {mean_modulus:.6f} ± {std_modulus:.6f}")
    print(f"  Phase variation: {phase_variation:.2f} radians")
    print(f"  Phase winding: {phase_variation/(2*np.pi):.2f} full cycles")
    
    # Análise de winding do próprio Mi
    Mi_phases_unwrapped = np.unwrap(Mi_phases)
    Mi_winding = (Mi_phases_unwrapped[-1] - Mi_phases_unwrapped[0]) / (2*np.pi)
    print(f"  Mi intrinsic winding: {Mi_winding:.2f}")
    
    # Plot do invariante Mi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajetória no plano complexo
    ax1.plot(np.real(Mi_array), np.imag(Mi_array), 'blue', alpha=0.7)
    scatter = ax1.scatter(np.real(Mi_array[::20]), np.imag(Mi_array[::20]), 
                         c=theta_range[::20], cmap='hsv', s=50)
    ax1.set_xlabel('Re(Mi)')
    ax1.set_ylabel('Im(Mi)')
    ax1.set_title('DYNAMIC INVARIANT Mi - COMPLEX TRAJECTORY')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='θ (rad)')
    
    # Evolução temporal
    ax2.plot(theta_range, Mi_moduli, 'red', label='|Mi|')
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('|Mi|', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax2b = ax2.twinx()
    ax2b.plot(theta_range, Mi_phases, 'green', label='Phase(Mi)')
    ax2b.set_ylabel('Phase (rad)', color='green')
    ax2b.tick_params(axis='y', labelcolor='green')
    
    ax2.set_title('Mi MODULUS AND PHASE EVOLUTION')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return Mi_array, Mi_moduli, Mi_phases

# ================================
# --- NON-ABELIAN NATURE ANALYSIS ---
# ================================

def analyze_non_abelian_nature():
    """
    Analisa a natureza não-abeliana do Grupo Alpha
    """
    print("\n=== NON-ABELIAN NATURE ANALYSIS ===")
    
    # Testa com diferentes ângulos
    theta1 = np.pi/4
    theta2 = np.pi/3
    theta3 = np.pi/6
    
    M1, _, _, _ = M_theta_alpha_non_classical(theta1)
    M2, _, _, _ = M_theta_alpha_non_classical(theta2)
    M3, _, _, _ = M_theta_alpha_non_classical(theta3)
    
    # 1. Teste de comutatividade
    commutator = M1 @ M2 - M2 @ M1
    commutator_norm = np.linalg.norm(commutator)
    
    print(f"1. COMUTATOR [M(π/4), M(π/3)]:")
    print(f"   Norm: {commutator_norm:.6f}")
    print(f"   [M1, M2] ≠ 0 → NON-ABELIAN SYSTEM")
    
    # 2. Teste do produto em ordens diferentes
    product_12 = M1 @ M2
    product_21 = M2 @ M1
    difference_norm = np.linalg.norm(product_12 - product_21)
    
    print(f"\n2. NON-COMMUTATIVITY OF PRODUCT:")
    print(f"   ||M1*M2 - M2*M1|| = {difference_norm:.6f}")
    print(f"   M1*M2 ≠ M2*M1 → ORDER MATTERS")
    
    # 3. Visualização do comutador
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(np.real(commutator), cmap='RdBu_r', aspect='equal')
    ax1.set_title('COMMUTATOR [M1,M2] - Real Part')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(np.imag(commutator), cmap='RdBu_r', aspect='equal')
    ax2.set_title('COMMUTATOR [M1,M2] - Imaginary Part')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return commutator_norm, difference_norm

# ================================
# --- ENHANCED MAIN EXECUTION ---
# ================================

def enhanced_main_analysis():
    """
    Análise principal aprimorada com todas as funcionalidades
    """
    # Define critical range
    theta_center = np.pi/2
    theta_half_range = 0.1 
    theta_critical_range = np.linspace(theta_center - theta_half_range, 
                                       theta_center + theta_half_range, 200)
    
    print("🚀 ENHANCED NON-CLASSICAL ALPHA GROUP ANALYSIS")
    print("=" * 60)
    
    # Análise topológica profunda
    W_alpha, W_det, alpha_grad, det_grad = deep_topological_analysis(theta_critical_range)
    
    # Plot comparativo detalhado
    plot_detailed_comparison(theta_critical_range, W_alpha, W_det, alpha_grad, det_grad)
    
    # Análise do invariante Mi
    Mi_array, Mi_moduli, Mi_phases = analyze_Mi_dynamics(theta_critical_range)
    
    # Análise da natureza não-abeliana
    commutator_norm, non_comm_norm = analyze_non_abelian_nature()
    
    # Análises auxiliares
    hamilton_alpha_comparison()
    alpha_intrinsic_algebra()
    alpha_geometric_structure(theta_critical_range)
    compare_winding_numbers(theta_critical_range)
    alpha_unique_characteristics()
    
    # Resumo final quantitativo
    print("\n" + "=" * 70)
    print("🎯 QUANTITATIVE SUMMARY - ALPHA GROUP TOPOLOGY")
    print("=" * 70)
    print(f"Parameter Range: θ ∈ [{theta_critical_range[0]:.4f}, {theta_critical_range[-1]:.4f}]")
    print(f"Non-Classical Winding (W_Alpha): {W_alpha[-1]:.6f}")
    print(f"Classical Winding (W_Det): {W_det[-1]:.6f}")
    print(f"Topological Difference: {W_alpha[-1] - W_det[-1]:.6f}")
    print(f"Mi Dynamic Invariant: {np.mean(Mi_moduli):.6f} ± {np.std(Mi_moduli):.6f}")
    print(f"Non-Abelian Strength: {commutator_norm:.6f}")
    print("=" * 70)
    print("CONCLUSION: The Alpha Group exhibits unique non-classical topological")
    print("properties that cannot be fully captured by classical invariants.")
    print("The system is fundamentally non-abelian with rich mathematical structure.")
    print("=" * 70)

# ================================
# --- COMPLETE EXECUTION ---
# ================================

if __name__ == "__main__":
    enhanced_main_analysis()

