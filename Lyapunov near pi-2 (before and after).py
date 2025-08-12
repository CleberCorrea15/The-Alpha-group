!pip install ripser
import numpy as np
from ripser import Rips
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm.notebook import tqdm
import pandas as pd

# Configurações gerais
rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')
np.random.seed(42)

def build_M(theta, mu=1.0, clip=1e6):
    """Matriz dinâmica M(θ) com tratamento de singularidades."""
    s, c = np.sin(theta), np.cos(theta)
    # Evita divisão por zero, com clip
    tan = np.clip(s / (c if abs(c) > 1e-12 else np.sign(c)*1e-12), -clip, clip)
    cot = np.clip(c / (s if abs(s) > 1e-12 else np.sign(s)*1e-12), -clip, clip)

    return np.array([
        [1.0, -cot, -tan, 1.0],
        [cot, 0+1j, -1.0, -tan],
        [tan, -1.0, mu, -cot],
        [1.0, tan, cot, 0+1j*mu]
    ], dtype=np.complex128)

def lyapunov_exponent(M, steps=3000):
    """Calcula o maior expoente de Lyapunov para a matriz M."""
    v = np.random.rand(M.shape[0])
    v /= np.linalg.norm(v)
    le_sum = 0.0
    for _ in range(steps):
        v = M @ v
        norm = np.linalg.norm(v)
        v /= norm
        le_sum += np.log(norm + 1e-12)
    return le_sum / steps

# Add a function to plot trajectories
def plot_trajectory(traj, theta):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Assuming the trajectory is complex, plotting the real part
    ax.plot(np.real(traj[:, 0]), np.real(traj[:, 1]), np.real(traj[:, 2]), lw=0.5)
    ax.set_title(f'Trajectory for θ = {theta:.4f}')
    ax.set_xlabel('Real Part of Dim 1')
    ax.set_ylabel('Real Part of Dim 2')
    ax.set_zlabel('Real Part of Dim 3')
    plt.show()


def monte_carlo_analysis(theta_values, n_samples=3, steps=3000, capture_theta=None):
    """Simula o atrator e calcula β₁ e Lyapunov via TDA."""
    results = []
    rips = Rips(maxdim=1, thresh=0.5)
    captured_trajectories = [] # List to store trajectories for the captured theta

    for theta in tqdm(theta_values, desc="Simulando Monte Carlo"):
        M = build_M(theta)

        beta1_samples = []
        lyapunov_samples = []

        for _ in range(n_samples):
            traj = np.zeros((steps, 4), dtype=np.complex128)
            state = np.random.randn(4) + 1j*np.random.randn(4)
            for i in range(steps):
                noise = 0.01 * (np.random.randn(4) + 1j*np.random.randn(4))
                state = M @ state + noise
                state /= np.linalg.norm(state) + 1e-12
                traj[i] = state

            # Capture trajectory if theta is close to capture_theta
            if capture_theta is not None and np.isclose(theta, capture_theta):
                 captured_trajectories.append(traj)

            # PCA e persistência (TDA)
            data = PCA(n_components=3).fit_transform(np.real(traj[::10]))
            dgms = rips.fit_transform(data)
            beta1 = len(dgms[1]) if len(dgms) > 1 else 0
            beta1_samples.append(beta1)

            # Expoente de Lyapunov
            lyap = lyapunov_exponent(M, steps=3000)
            lyapunov_samples.append(lyap)

        results.append({
            'theta': float(theta),
            'beta1_mean': float(np.mean(beta1_samples)),
            'beta1_std': float(np.std(beta1_samples)),
            'lyapunov_mean': float(np.mean(lyapunov_samples)),
            'lyapunov_std': float(np.std(lyapunov_samples))
        })

    return pd.DataFrame(results), captured_trajectories


def plot_results(df):
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.errorbar(df['theta'], df['beta1_mean'], yerr=df['beta1_std'],
                 fmt='o-', color='orange', label=r'$\beta_1$ (Monte Carlo)')
    ax1.set_xlabel(r'$\theta$ (radians)')
    ax1.set_ylabel(r'$\beta_1$', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')

    ax2 = ax1.twinx()
    ax2.errorbar(df['theta'], df['lyapunov_mean'], yerr=df['lyapunov_std'],
                 fmt='s-', color='blue', label='Lyapunov exponent')
    ax2.set_ylabel('Lyapunov exponent', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.axvline(np.pi/2, color='r', linestyle=':', label=r'$\pi/2$')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title(r'Monte Carlo: $\beta_1$ vs Lyapunov passing through $\pi/2$')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('montecarlo_beta1_lyapunov_1_57.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 11 pontos uniformemente espaçados de 1.55 a 1.59, incluindo 1.57
    theta_range = np.linspace(1.54, 1.60, 19)

    print("Theta range:", theta_range)

    print("\n=== Análise por Monte Carlo passando por label=r'$\pi/2$' ===")
    mc_df, traj_pi_2_cases = monte_carlo_analysis(theta_range, n_samples=3, steps=3000, capture_theta=np.pi/2)

    print("\n=== Resultados ===")
    print(mc_df)

    plot_results(mc_df)
