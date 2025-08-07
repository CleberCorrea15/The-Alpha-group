import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4th_order(f, x0, t0, tf, h):
    num_steps = int((tf - t0) / h) + 1
    t_values = np.linspace(t0, tf, num_steps)
    x_values = np.zeros((num_steps, len(x0)))
    x_values[0] = x0

    for i in range(1, num_steps):
        k1 = h * f(t_values[i-1], x_values[i-1])
        k2 = h * f(t_values[i-1] + 0.5 * h, x_values[i-1] + 0.5 * k1)
        k3 = h * f(t_values[i-1] + 0.5 * h, x_values[i-1] + 0.5 * k2)
        k4 = h * f(t_values[i-1] + h, x_values[i-1] + k3)

        x_values[i] = x_values[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, x_values

# Sistema de equações diferenciais com a matriz A contendo tangente e cotangente
def system(t, x):
    theta = np.pi / 2.0048  # Pode ajustar conforme necessário
    A = np.array([
        [1, -1/np.tan(theta), -1*np.tan(theta), 1],
        [1/np.tan(theta), 1, -1, -1*np.tan(theta)],
        [1*np.tan(theta), -1, 1, -1/np.tan(theta)],
        [1, 1*np.tan(theta), 1/np.tan(theta), 1]
    ])
    return np.dot(A, x)

# Função de Lyapunov
def lyapunov(t_values, x_values):
    num_steps = len(t_values)
    V = np.identity(x_values.shape[1])  # Matriz de identidade inicial

    trace_values = []

    for i in range(1, num_steps):
        x_dot = system(t_values[i-1], x_values[i-1])
        V_dot = np.outer(x_dot, x_values[i-1]) + np.outer(x_values[i-1], x_dot)
        V += V_dot * h
        trace_values.append(np.trace(V))

    return t_values[1:], trace_values

# Condições iniciais e parâmetros
x0 = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)  # exemplo com quatro variáveis
t0 = 0
tf = 50.0
h = 0.001

# Executa o mapa de Poincaré no estilo de Hénon
t_values, x_values = runge_kutta_4th_order(system, x0, t0, tf, h)

# Calcula a função de Lyapunov
t_lyapunov, lyapunov_values = lyapunov(t_values, x_values)

# Plota os resultados
plt.figure(figsize=(12, 6))

# Plotagem do diagrama de fase
plt.subplot(1, 2, 1)
plt.plot(x_values[:, 0], x_values[:, 1])
plt.xlabel('$x_4$')
plt.ylabel('$x_2$')
plt.title('Phase Diagram')

# Plotagem da função de Lyapunov
plt.subplot(1, 2, 2)
plt.plot(t_lyapunov, lyapunov_values)
plt.xlabel('Time')
plt.ylabel('Lyapunov Trace')
plt.title('Lyapunov Function')

plt.tight_layout()
plt.show()

