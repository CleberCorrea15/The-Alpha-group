# ===========================
# Improved dynamic-topology (Colab-ready)
# ===========================
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree
from scipy.linalg import eigh
from tqdm import tqdm

plt.rcParams['agg.path.chunksize'] = 100000
plt.rcParams['path.simplify_threshold'] = 1.0
%matplotlib inline

# -----------------------
# PARAMETERS (edit here)
# -----------------------
OUTDIR = "results_dynamic_topology_colab"
os.makedirs(OUTDIR, exist_ok=True)

THETA = np.pi/2 - 0.001      # angle (we’ll do a sweep later)
STEPS = 120000               # total simulation steps
RANDOM_SEED = 42
WINDOW_SIZE = 2000
STEP_SIZE = 500
DIST_THRESHOLD = 0.1         # KDTree linking distance
MATRIX_THRESHOLD = 1e-3      # threshold to build graph from M
NORMALIZE_TRAJECTORY = True  # z-score each dimension before KDTree
BOOTSTRAP_N = 3000           # bootstrap iterations for CI (per-window aggregated metrics)
SAVE_FIGS = True
SAVE_TO_DRIVE = False        # set True if you mounted Drive and want to copy outputs there
DRIVE_DIR = "/content/drive/MyDrive/dynamic_topology_results"  # only if SAVE_TO_DRIVE True

# ------------------------
# build_M (user-provided)
# ------------------------
def build_M(theta):
    epsilon = 1e-12
    theta_adj = theta + 0j
    critical_points = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    for cp in critical_points:
        if np.abs((theta_adj.real - cp) % (2*np.pi)) < 1e-8:
            theta_adj += epsilon * 1j

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

# ------------------------
# Simulation
# ------------------------
def simulate_single_run(theta, steps=STEPS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    M = build_M(theta)
    state = rng.randn(4)
    trajectory = np.zeros((steps, 4))

    alpha_norm = 0.98
    noise_scale = 0.02

    for step in range(steps):
        noise = noise_scale * rng.randn(4) * np.linalg.norm(state)
        state_new = np.real(M @ state) + noise

        norm_state = np.linalg.norm(state_new)
        if norm_state < 1e-8 or np.isnan(norm_state) or np.isinf(norm_state):
            state = rng.randn(4)
            state /= np.linalg.norm(state) + 1e-10
        else:
            state = alpha_norm * state_new / norm_state + (1 - alpha_norm) * state

        trajectory[step] = state

    return trajectory, M

# ------------------------
# Graph + Betti helpers
# ------------------------
def create_proximity_graph(points, distance_threshold=DIST_THRESHOLD, normalize=NORMALIZE_TRAJECTORY):
    G = nx.Graph()
    n = len(points)
    G.add_nodes_from(range(n))
    if n < 2:
        return G
    X = np.array(points)
    # optional per-dimension normalization
    if normalize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    tree = KDTree(X)
    pairs = tree.query_pairs(distance_threshold)
    for i, j in pairs:
        G.add_edge(i, j)
    return G

def calculate_betti_numbers(G):
    V = G.number_of_nodes()
    E = G.number_of_edges()
    H0 = nx.number_connected_components(G)
    H1 = max(0, E - V + H0)  # cyclomatic number
    triangles = nx.triangles(G)
    H2 = int(sum(triangles.values()) // 3) if V >= 3 else 0
    cliques = list(nx.find_cliques(G))
    H3 = sum(1 for c in cliques if len(c) == 4)
    return H0, H1, H2, H3, V, E, cliques

# ------------------------
# Trajectory analysis (windowed)
# ------------------------
def analyze_trajectory_topology(trajectory, window_size=WINDOW_SIZE, step_size=STEP_SIZE, dist_threshold=DIST_THRESHOLD):
    n_points = len(trajectory)
    results = {'window_center': [], 'H0': [], 'H1': [], 'H2': [], 'H3': [], 'density': []}
    for start in range(0, n_points - window_size + 1, step_size):
        end = start + window_size
        window = trajectory[start:end]
        G = create_proximity_graph(window, distance_threshold=dist_threshold, normalize=NORMALIZE_TRAJECTORY)
        H0, H1, H2, H3, V, E, cliques = calculate_betti_numbers(G)
        results['window_center'].append((start + end)//2)
        results['H0'].append(H0)
        results['H1'].append(H1)
        results['H2'].append(H2)
        results['H3'].append(H3)
        results['density'].append(nx.density(G))
    # convert to arrays
    for k in list(results.keys()):
        results[k] = np.array(results[k])
    # smoothing (moving average)
    def smooth(x, N=5):
        if len(x) < N:
            return x
        return np.convolve(x, np.ones(N)/N, mode='same')
    results['H2_smooth'] = smooth(results['H2'])
    results['H3_smooth'] = smooth(results['H3'])
    return results

# ------------------------
# Matrix topology
# ------------------------
def analyze_matrix_topology(M, threshold=MATRIX_THRESHOLD):
    H = M.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(H))
    for i in range(H):
        for j in range(i+1, H):
            if np.abs(M[i, j]) > threshold:
                G.add_edge(i, j)
    V = G.number_of_nodes()
    E = G.number_of_edges()
    H0 = nx.number_connected_components(G)
    H1 = max(0, E - V + H0)
    triangles = nx.triangles(G)
    H2 = int(sum(triangles.values()) // 3)
    cliques = list(nx.find_cliques(G))
    H3 = sum(1 for c in cliques if len(c) == 4)
    return {'H0': H0, 'H1': H1, 'H2': H2, 'H3': H3, 'nodes': V, 'edges': E, 'cliques': cliques}

# ------------------------
# Statistics (per-window + global with bootstrap CI)
# ------------------------
def window_level_statistics(df_windows, bootstrap_n=BOOTSTRAP_N, seed=RANDOM_SEED):
    agg = {}
    metrics = ['H0','H1','H2','H3','density']
    for m in metrics:
        vals = df_windows[m].values
        agg[m + '_mean'] = float(np.mean(vals))
        agg[m + '_std'] = float(np.std(vals, ddof=1))
        agg[m + '_median'] = float(np.median(vals))
        agg[m + '_min'] = float(np.min(vals))
        agg[m + '_max'] = float(np.max(vals))
        # bootstrap for 95% CI of the mean
        rng = np.random.RandomState(seed)
        boots = []
        for _ in range(bootstrap_n):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boots.append(np.mean(sample))
        agg[m + '_ci_lower'] = float(np.percentile(boots, 2.5))
        agg[m + '_ci_upper'] = float(np.percentile(boots, 97.5))
    return agg

# ------------------------
# Plot helpers (save separate files)
# ------------------------
def save_plot(fig, name):
    fp = os.path.join(OUTDIR, name)
    fig.savefig(fp, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return fp

def plot_and_save_trajectory_3d(traj, theta):
    """
    Improved 3D trajectory plot with equal axis scaling and margins so the
    vertical scale (z) is visible and not flattened. Also avoids z-label clipping.
    """
    fig = plt.figure(figsize=(10,7), dpi=220)
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = traj[:,0], traj[:,1], traj[:,2]
    t = np.linspace(0, 1, len(x))

    # line and points
    ax.plot(x, y, z, lw=0.6, alpha=0.9)
    ax.scatter(x[0], y[0], z[0], s=80, edgecolors='k', linewidths=0.8, label='Start', zorder=5)
    ax.scatter(x[-1], y[-1], z[-1], s=80, edgecolors='k', linewidths=0.8, label='End', zorder=5)

    # titles and labels
    ax.set_title(f"3D Trajectory (θ={theta:.6f})", fontsize=14, pad=12)
    ax.set_xlabel('Dim 1', fontsize=11, labelpad=10)
    ax.set_ylabel('Dim 2', fontsize=11, labelpad=10)
    # increase labelpad for z so it doesn't get clipped
    ax.set_zlabel('Dim 3', fontsize=11, labelpad=18)

    # compute ranges and set equal aspect (so z-scale is not squashed)
    def set_axes_equal(ax, xs, ys, zs, pad_frac=0.10):
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        z_min, z_max = float(np.min(zs)), float(np.max(zs))

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range, 1e-12)

        # add small padding
        pad = max_range * pad_frac
        x_center = 0.5 * (x_max + x_min)
        y_center = 0.5 * (y_max + y_min)
        z_center = 0.5 * (z_max + z_min)

        ax.set_xlim(x_center - max_range/2 - pad, x_center + max_range/2 + pad)
        ax.set_ylim(y_center - max_range/2 - pad, y_center + max_range/2 + pad)
        ax.set_zlim(z_center - max_range/2 - pad, z_center + max_range/2 + pad)

    set_axes_equal(ax, x, y, z, pad_frac=0.08)

    # nicer view angle
    ax.view_init(elev=25, azim=-60)

    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', fontsize=10)

    # increase tick label size for readability
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(9)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(9)
    for tick in ax.zaxis.get_ticklabels():
        tick.set_fontsize(9)

    # Adjust subplot margins to avoid clipping of 3D labels
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Save using tight bbox and a small pad to ensure labels not cut
    fname = f"trajectory_3d_theta_{theta:.6f}.png"
    fp = os.path.join(OUTDIR, fname)
    fig.savefig(fp, dpi=220, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    return fp

def plot_and_save_H0_H1(windows_df, theta):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(windows_df['window_center'], windows_df['H0'], '-b', label='H0')
    ax.plot(windows_df['window_center'], windows_df['H1'], '-r', label='H1')
    ax.set_title("Evolution H0 / H1")
    ax.set_xlabel('Steps'); ax.set_ylabel('Value')
    ax.grid(alpha=0.3); ax.legend()
    return save_plot(fig, f"H0_H1_theta_{theta:.6f}.png")

def plot_and_save_H2_H3(windows_df, theta):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(windows_df['window_center'], windows_df['H2_smooth'], '-g', label='H2 (smoothed)')
    ax.plot(windows_df['window_center'], windows_df['H3_smooth'], '-m', label='H3 (smoothed)')
    ax.set_title("Evolution H2 / H3 (smoothed)")
    ax.set_xlabel('Steps'); ax.set_ylabel('Value')
    ax.grid(alpha=0.3); ax.legend()
    return save_plot(fig, f"H2_H3_theta_{theta:.6f}.png")

def plot_and_save_density(windows_df, theta):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(windows_df['window_center'], windows_df['density'], '-o', color='orange')
    ax.set_title("Graph density per window")
    ax.set_xlabel('Steps'); ax.set_ylabel('Density')
    ax.grid(alpha=0.3)
    return save_plot(fig, f"density_theta_{theta:.6f}.png")

def plot_and_save_scatter_H2_H3(windows_df, theta):
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(windows_df['H2_smooth'], windows_df['H3_smooth'], c=windows_df['window_center'], cmap='viridis', alpha=0.8)
    ax.set_title("H2 vs H3 (color = time)")
    ax.set_xlabel('H2 (smoothed)'); ax.set_ylabel('H3 (smoothed)')
    plt.colorbar(sc, ax=ax, label='Steps')
    return save_plot(fig, f"H2_vs_H3_theta_{theta:.6f}.png")

def plot_and_save_boxplot_H(windows_df, theta):
    fig, ax = plt.subplots(figsize=(6,5))
    data = [windows_df['H0'], windows_df['H1'], windows_df['H2_smooth'], windows_df['H3_smooth']]
    labels = ['H0','H1','H2','H3']
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_title("Boxplot H0..H3 (windows)")
    return save_plot(fig, f"boxplot_Hs_theta_{theta:.6f}.png")

# ------------------------
# MAIN RUN
# ------------------------
def main():
    print("Simulating...")
    trajectory, M = simulate_single_run(THETA, steps=STEPS, seed=RANDOM_SEED)
    print("Analyzing matrix M...")
    matrix_results = analyze_matrix_topology(M, threshold=MATRIX_THRESHOLD)
    print("Analyzing trajectory (windows)...")
    traj_results = analyze_trajectory_topology(trajectory, window_size=WINDOW_SIZE, step_size=STEP_SIZE, dist_threshold=DIST_THRESHOLD)

    # Save windows to DataFrame
    df_windows = pd.DataFrame({
        'window_center': traj_results['window_center'],
        'H0': traj_results['H0'],
        'H1': traj_results['H1'],
        'H2': traj_results['H2'],
        'H3': traj_results['H3'],
        'H2_smooth': traj_results['H2_smooth'],
        'H3_smooth': traj_results['H3_smooth'],
        'density': traj_results['density']
    })
    windows_csv = os.path.join(OUTDIR, f"windows_betti_theta_{THETA:.6f}.csv")
    df_windows.to_csv(windows_csv, index=False)

    # Aggregated stats with bootstrap CI
    agg = window_level_statistics(df_windows, bootstrap_n=BOOTSTRAP_N, seed=RANDOM_SEED)
    agg['theta'] = float(THETA)
    agg['n_windows'] = int(len(df_windows))
    agg['steps'] = int(STEPS)
    agg['window_size'] = int(WINDOW_SIZE)
    agg['step_size'] = int(STEP_SIZE)
    # save aggregated stats
    agg_json = os.path.join(OUTDIR, f"aggregated_stats_theta_{THETA:.6f}.json")
    with open(agg_json, 'w') as f:
        json.dump(agg, f, indent=2)

    # Save matrix topology
    matrix_json = os.path.join(OUTDIR, f"matrix_topology_theta_{THETA:.6f}.json")
    with open(matrix_json, 'w') as f:
        json.dump(matrix_results, f, indent=2)

    # Produce and save separate figures
    saved_files = {}
    saved_files['trajectory_3d'] = plot_and_save_trajectory_3d(trajectory, THETA)
    saved_files['H0_H1'] = plot_and_save_H0_H1(df_windows, THETA)
    saved_files['H2_H3'] = plot_and_save_H2_H3(df_windows, THETA)
    saved_files['density'] = plot_and_save_density(df_windows, THETA)
    saved_files['H2_vs_H3'] = plot_and_save_scatter_H2_H3(df_windows, THETA)
    saved_files['boxplot'] = plot_and_save_boxplot_H(df_windows, THETA)

    # Summary CSV combining matrix + aggregated stats
    summary = {'theta': float(THETA), 'steps': STEPS, 'window_size': WINDOW_SIZE, 'step_size': STEP_SIZE}
    summary.update({f"matrix_{k}": v for k, v in matrix_results.items()})
    summary.update(agg)
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(OUTDIR, f"summary_theta_{THETA:.6f}.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Optionally copy to Drive
    if SAVE_TO_DRIVE:
        os.makedirs(DRIVE_DIR, exist_ok=True)
        import shutil
        for fname in os.listdir(OUTDIR):
            shutil.copy(os.path.join(OUTDIR, fname), os.path.join(DRIVE_DIR, fname))
        print("Files copied to Drive:", DRIVE_DIR)

    # Display inline previews in Colab
    from IPython.display import Image, display
    print("\nShowing saved figures (preview):")
    for k,v in saved_files.items():
        print(f"{k}: {v}")
        display(Image(v, width=800))

    print("\nFiles saved in:", OUTDIR)
    print(" - windows CSV:", windows_csv)
    print(" - aggregated JSON:", agg_json)
    print(" - matrix JSON:", matrix_json)
    print(" - summary CSV:", summary_csv)

    return {
        'outdir': OUTDIR,
        'files': saved_files,
        'windows_csv': windows_csv,
        'agg_json': agg_json,
        'matrix_json': matrix_json,
        'summary_csv': summary_csv,
        'matrix_results': matrix_results,
        'df_windows_head': df_windows.head().to_dict(orient='list')
    }

# Run it
results = main()
results
