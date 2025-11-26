import numpy as np
import plotly.graph_objects as go
from functools import lru_cache

# -------------------------------
# MATRIZ ALPHA COM CACHE E LIMITES
# -------------------------------
@lru_cache(maxsize=1)
def build_M_alpha_cached(theta):
    theta = float(theta)
    tan_val = np.tan(theta)
    
    MAX_VAL = 1e2
    HUGE = 1e8
    if not np.isfinite(tan_val) or abs(tan_val) > HUGE:
        tan_val = MAX_VAL * np.sign(tan_val)
    
    eps = 1e-12
    cot_val = 1/tan_val if abs(tan_val) > eps else 1/eps
    
    i = 1j
    mu = 1.0

    M = np.array([
        [1,        -cot_val, -tan_val,  1],
        [cot_val,   i,       -1,       -tan_val],
        [tan_val,  -1,        mu,      -cot_val],
        [1,         tan_val,  cot_val,  i*mu]
    ], dtype=complex)

    M = np.clip(M.real, -MAX_VAL, MAX_VAL) + 1j*np.clip(M.imag, -MAX_VAL, MAX_VAL)
    return M

# -------------------------------
# PARÂMETROS
# -------------------------------
theta = 1.5602
n_steps = 1000
H0 = np.array([1+0j, 0.5+0j, 0.3+0j, 0.7+0j])

M_alpha = build_M_alpha_cached(theta)

# -------------------------------
# SIMULAÇÃO DA DINÂMICA (PARTES IMAGINÁRIAS)
# -------------------------------
H_list = []
H = H0.copy()
for t in range(n_steps):
    H = M_alpha @ H
    H = H / np.linalg.norm(H)  # normalização a cada passo
    # Projeção 3D das partes imaginárias
    X = H[0].imag
    Y = H[1].imag
    Z = np.sqrt(H[2].imag**2 + H[3].imag**2)
    intensity = np.abs(H[3].imag)  # cor baseada na 4ª dimensão imaginária
    H_list.append([X, Y, Z, intensity])

H_array = np.array(H_list)

# -------------------------------
# ANIMAÇÃO 3D INTERATIVA
# -------------------------------
fig = go.Figure(
    data=[go.Scatter3d(
        x=[H_array[0,0]],
        y=[H_array[0,1]],
        z=[H_array[0,2]],
        mode='markers',
        marker=dict(size=5, color=H_array[0,3], colorscale='Viridis',
                    cmin=0, cmax=np.max(H_array[:,3])),
        name='H(t)'
    )]
)

frames = []
for i in range(n_steps):
    frame = go.Frame(
        data=[go.Scatter3d(
            x=H_array[:i+1,0],
            y=H_array[:i+1,1],
            z=H_array[:i+1,2],
            mode='lines+markers',
            marker=dict(size=4, color=H_array[:i+1,3], colorscale='Viridis',
                        cmin=0, cmax=np.max(H_array[:,3]))
        )],
        name=f'frame{i}'
    )
    frames.append(frame)

fig.frames = frames

fig.update_layout(
    title="Dinâmica 3D das partes imaginárias de H0..H3",
    scene=dict(
        xaxis_title='Im(H0)',
        yaxis_title='Im(H1)',
        zaxis_title='sqrt(Im(H2)^2 + Im(H3)^2)',
        aspectmode='cube'
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration":50, "redraw": True},
                                   "fromcurrent": True}]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate"}])]
    )]
)

fig.show()

