import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(14,6))

# --- Geometria de Riemann ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Riemann geometry\n(local metric, dynamic bounding))')
# Criar esfera
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x, y, z, color='lightblue', alpha=0.5)

# Geodésicas locais (linhas curvas na esfera)
for phi in np.linspace(0, np.pi, 6):
    ax1.plot(np.sin(phi)*np.cos(u), np.sin(phi)*np.sin(u), np.cos(phi)*np.ones_like(u), color='blue', linewidth=1.5)

ax1.text(0,0,1.3,'Limited local paths', color='red')

# --- Alpha Group ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Alpha Group\n(superior structure, global dynamics)')
# Criar “camada superior” (plano acima)
X,Y = np.meshgrid(np.linspace(-1.5,1.5,10), np.linspace(-1.5,1.5,10))
Z = 1.2 + 0*X
ax2.plot_surface(X,Y,Z, color='orange', alpha=0.3)

# Grafo de nós e caminhos (projeção 3D simplificada)
nodes = np.array([[0,0,1],[1,0,1.1],[0.5,0.8,1.2],[0.5,0.3,1.4]])
paths = [[0,1,3],[0,2,3],[1,2,3]]
colors = ['red','green','blue']
for path, col in zip(paths, colors):
    for i in range(len(path)-1):
        n1,n2 = path[i], path[i+1]
        ax2.plot([nodes[n1,0],nodes[n2,0]],
                 [nodes[n1,1],nodes[n2,1]],
                 [nodes[n1,2],nodes[n2,2]], color=col, linewidth=2)
# Nós
ax2.scatter(nodes[:,0], nodes[:,1], nodes[:,2], c='black', s=60)
ax2.text(0,0,1.5,'Global exploration possible', color='red')

plt.show()


