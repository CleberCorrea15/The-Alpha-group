!pip install ripser
import numpy as np
from ripser import Rips
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm.notebook import tqdm
import pandas as pd

rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')
np.random.seed(42)

def build_M(theta, mu=1.0, clip=1e6):
    s, c = np.sin(theta), np.cos(theta)
    tan = np.clip(s / (c if abs(c) > 1e-12 else np.sign(c)*1e-12), -clip, clip)
    cot = np.clip(c / (s if abs(s) > 1e-12 else np.sign(s)*1e-12), -clip, clip)
    return np.array([
        [1.0, -cot, -tan, 1.0],
        [cot, 0+1j, -1.0, -tan],
        [tan, -1.0, mu, -cot],
        [1.0, tan, cot, 0+1j*mu]
    ], dtype=np.complex128)

def lyapunov_exponent(M, steps=3000):
    v = np.random.rand(M.shape[0])
    v /= np.linalg.norm(v)
    le_sum = 0.0
    for _ in range(steps):
        v = M @ v
        norm = np.linalg.norm(v)
        v /= norm
        le_sum += np.log(norm + 1e-12)
    return le_sum / steps

# Define simulate_trajectory function
def simulate_trajectory(M, steps=3000):
    traj = np.zeros((steps, 4), dtype=np.complex128)
    state = np.random.randn(4) + 1j*np.random.randn(4)
    for i in range(steps):
        noise = 0.01 * (np.random.randn(4) + 1j*np.random.randn(4))
        state = M @ state + noise
        state /= np.linalg.norm(state) + 1e-12
        traj[i] = state
    return traj

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


def monte_carlo_analysis(theta_values, n_samples=3, steps=3000):
    results = []
    rips = Rips(maxdim=1, thresh=0.5)
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
            data = PCA(n_components=3).fit_transform(np.real(traj[::10]))
            dgms = rips.fit_transform(data)
            beta1 = len(dgms[1]) if len(dgms) > 1 else 0
            beta1_samples.append(beta1)
            lyap = lyapunov_exponent(M, steps=3000)
            lyapunov_samples.append(lyap)
        results.append({
            'theta': float(theta),
            'beta1_mean': float(np.mean(beta1_samples)),
            'beta1_std': float(np.std(beta1_samples)),
            'lyapunov_mean': float(np.mean(lyapunov_samples)),
            'lyapunov_std': float(np.std(lyapunov_samples))
        })
    return pd.DataFrame(results)

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
    ax1.axvline(0, color='r', linestyle=':', label='0 (radians)')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title(r'Monte Carlo: $\beta_1$ vs Lyapunov near zero (before and after)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('montecarlo_beta1_lyapunov_zero_sym.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    theta_range = np.linspace(-0.05, 0.05, 11)
    print("Theta range:", theta_range)
    print("\n=== Análise por Monte Carlo próximo de zero (antes e depois) ===")
    mc_df = monte_carlo_analysis(theta_range, n_samples=3, steps=3000)
    print("\n=== Resultados ===")
    print(mc_df)
    plot_results(mc_df)

   # if __name__ == "__main__":
    # Definir theta próximos de 0 e π/2
    theta_values = [0.0] # Removed np.pi/2 as plot_trajectory_and_field is commented out

    traj_cases = {}
    for theta in theta_values:
        M = build_M(theta)
        traj = simulate_trajectory(M, steps=3000)
        traj_cases[theta] = traj

    # Plotar para θ=0
    plot_trajectory(traj_cases[0.0], theta=0.0)

    # Plotar para θ=π/2
    #plot_trajectory_and_field(traj_cases[np.pi/2], theta=np.pi/2)
