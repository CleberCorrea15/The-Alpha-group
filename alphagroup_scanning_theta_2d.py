import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# ===========================
# Configurações
# ===========================
RES_GRID = 100        # grade 2D de x, y
RES_THETA = 101       # passos de theta
EPS = 1e-12           # para evitar divisão por zero
MAX_VAL = 1e3         # limite para estabilidade

# ===========================
# Métrica 2D projetada
# ===========================
def project_to_2D_metric(theta, x, y):
    """Tensor métrico 2D projetado"""
    if abs(theta - np.pi/2) < EPS:
        tanθ = np.sign(np.cos(theta)) * MAX_VAL
    else:
        tanθ = np.tan(theta)

    g00 = 1 + x**2
    g11 = 1 + y**2
    g01 = 0.3 * x * y * tanθ
    g01 = np.clip(g01, -MAX_VAL, MAX_VAL)

    return np.array([[g00, g01], [g01, g11]])

# ===========================
# Classificação de regimes
# ===========================
def classify_regime(theta, x, y):
    """0=Isotropic, 1=Moderate, 2=Strong"""
    G = project_to_2D_metric(theta, x, y)
    G_sym = 0.5 * (G + G.T)
    eigvals = np.linalg.eigvalsh(G_sym)
    eigvals = np.abs(eigvals) + EPS
    ratio = np.max(eigvals) / np.min(eigvals)

    if ratio > 3.0:
        return 2  # Strong
    elif ratio > 1.5:
        return 1  # Moderate
    else:
        return 0  # Isotropic

# ===========================
# Scanner do Alpha Group
# ===========================
class AlphaGroupScanner:
    def __init__(self, res_grid=RES_GRID):
        self.res = res_grid
        x = np.linspace(-1, 1, res_grid)
        y = np.linspace(-1, 1, res_grid)
        self.X, self.Y = np.meshgrid(x, y)

    def scan_theta_range(self, theta_min=40, theta_max=140, n_points=RES_THETA):
        """Varredura angular completa"""
        thetas_deg = np.linspace(theta_min, theta_max, n_points)
        thetas_rad = np.deg2rad(thetas_deg)

        results = {'theta_deg': thetas_deg,
                   'theta_rad': thetas_rad,
                   'isotropic': [], 'moderate': [], 'strong': []}

        for θ_rad in thetas_rad:
            iso = mod = strong = 0
            for i in range(self.res):
                for j in range(self.res):
                    regime = classify_regime(θ_rad, self.X[i,j], self.Y[i,j])
                    if regime == 0:
                        iso += 1
                    elif regime == 1:
                        mod += 1
                    else:
                        strong += 1
            results['isotropic'].append(iso)
            results['moderate'].append(mod)
            results['strong'].append(strong)
        return results

    def visualize_regimes_3D(self, theta_deg=90):
        """Visualização 3D para um θ específico"""
        θ_rad = np.deg2rad(theta_deg)
        regimes_grid = np.zeros((self.res, self.res))
        anisotropy_grid = np.zeros((self.res, self.res))

        for i in range(self.res):
            for j in range(self.res):
                regime = classify_regime(θ_rad, self.X[i,j], self.Y[i,j])
                regimes_grid[i,j] = regime
                G = project_to_2D_metric(θ_rad, self.X[i,j], self.Y[i,j])
                G_sym = 0.5 * (G + G.T)
                eigvals = np.linalg.eigvalsh(G_sym)
                eigvals = np.abs(eigvals) + EPS
                anisotropy_grid[i,j] = np.max(eigvals) / np.min(eigvals)

        # === Figura 1: Mapa de regimes ===
        fig1, ax1 = plt.subplots(figsize=(6,5))
        cmap_regimes = matplotlib.colormaps['viridis'].resampled(3)
        im1 = ax1.imshow(regimes_grid, cmap=cmap_regimes,
                         extent=[-1,1,-1,1], origin='lower')
        ax1.set_title(f'Regimes para θ = {theta_deg}°')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0,1,2])
        cbar1.ax.set_yticklabels(['Isotropic', 'Moderate', 'Strong'])

        # === Figura 2: Mapa de anisotropia ===
        fig2, ax2 = plt.subplots(figsize=(6,5))
        im2 = ax2.imshow(np.log10(anisotropy_grid), cmap='hot',
                         extent=[-1,1,-1,1], origin='lower')
        ax2.set_title(f'Anisotropia (log10) para θ = {theta_deg}°')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label='log10(λ_max/λ_min)')

        # === Figura 3: Superfície 3D de anisotropia ===
        fig3 = plt.figure(figsize=(8,6))
        ax3 = fig3.add_subplot(111, projection='3d')
        surf = ax3.plot_surface(self.X, self.Y, anisotropy_grid,
                                cmap='plasma', alpha=0.8)
        ax3.set_title(f'Superfície de Anisotropia - θ = {theta_deg}°')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('λ_max/λ_min')

        plt.show()
        return regimes_grid, anisotropy_grid

# ===========================
# Execução principal
# ===========================
if __name__ == "__main__":
    scanner = AlphaGroupScanner(res_grid=40)
    print("🔍 Varredura angular 40°-140°...")
    results = scanner.scan_theta_range(theta_min=40, theta_max=140, n_points=101)

    # === Figura 4: Gráfico da varredura angular ===
    plt.figure(figsize=(14,6))
    plt.plot(results['theta_deg'], results['isotropic'], 'b-o', label='Isotropic')
    plt.plot(results['theta_deg'], results['moderate'], 'g-s', label='Moderate')
    plt.plot(results['theta_deg'], results['strong'], 'r-^', label='Strong')
    plt.xlabel('θ (degrees)')
    plt.ylabel('Number of points')
    plt.title('Alpha Group: Angular Scanning (40°-140°)')
    plt.axvline(x=90, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Determinar ângulo ideal
    strong_array = np.array(results['strong'])
    idx_max = np.argmax(strong_array)
    theta_max = results['theta_deg'][idx_max]
    print(f"\n🎯 Ângulo ideal para anisotropia máxima: {theta_max:.1f}°")
