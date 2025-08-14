import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def build_M(theta, mu=1.5, clip=1e6):
    epsilon = 1e-4
    s = np.sin(theta)
    c = np.cos(theta)
    s = np.where(np.abs(s) < epsilon, epsilon * np.sign(s) if np.sign(s)!=0 else epsilon, s)
    c = np.where(np.abs(c) < epsilon, epsilon * np.sign(c) if np.sign(c)!=0 else epsilon, c)
    tan = np.clip(s / c, -clip, clip)
    cot = np.clip(c / s, -clip, clip)
    return np.array([
        [1.0, -cot, -tan, 1.0],
        [cot, 0+1j, -1.0, -tan],
        [tan, -1.0, mu, -cot],
        [1.0, tan, cot, 0+1j*mu]
    ], dtype=np.complex128)

def simulate_psi_trajectory(theta, steps=1500, dt=0.01):
    psi = np.array([1+0j, 0.1+0.1j, 0+0j, 0+0j], dtype=np.complex128)
    M = build_M(theta)
    trajectory = np.zeros((steps, 4), dtype=np.complex128)
    for i in range(steps):
        dpsi = M @ psi
        psi = psi + dt * dpsi
        psi = psi / np.linalg.norm(psi)
        trajectory[i] = psi
    return trajectory

def plot_trajectory(theta):
    traj = simulate_psi_trajectory(theta)
    imag_traj = traj[:, :3].imag

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')

    points = imag_traj.reshape(-1,1,3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, len(segments))
    lc = Line3DCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(2)

    ax.add_collection(lc)

    ax.set_xlim(imag_traj[:,0].min(), imag_traj[:,0].max())
    ax.set_ylim(imag_traj[:,1].min(), imag_traj[:,1].max())
    ax.set_zlim(imag_traj[:,2].min(), imag_traj[:,2].max())

    ax.set_xlabel(r'Im$(\psi_1)$')
    ax.set_ylabel(r'Im$(\psi_2)$')
    ax.set_zlabel(r'Im$(\psi_3)$')
    ax.set_title(f'Órbita vibracional próxima de π/2 — θ = {theta:.4f} rad')
    plt.show()

interact(plot_trajectory,
         theta=FloatSlider(value=1.57, min=1.55, max=1.59, step=0.0005, description=r'$\theta$'))

