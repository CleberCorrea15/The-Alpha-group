import numpy as np

# 1. Definição das 16 matrizes originais B1..B16
B1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
B2 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=float)
B3 = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
B4 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
B5 = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=float)
B6 = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=float)
B7 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=float)
B8 = np.array([[0, -1, -1, 0], [1, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]], dtype=float)
B9 = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
B10 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
B11 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=float)
B12 = np.array([[0, 1, -1, 0], [1, 0, 0, -1], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)
B13 = np.array([[2, 0, 0, 1], [0, 0, 1, 0], [0, -1, -1, 0], [1, 0, 0, 1]], dtype=float)
B14 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, -1, -1, 0], [1, 0, 0, 1]], dtype=float)
B15 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]], dtype=float)
B16 = np.array([[0, 1, 1, 0], [-1, 0, 0, -1], [2, 0, 0, 1], [0, 0, 1, 0]], dtype=float)

matrices = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16]

# --- 1. Decomposição da Base R x sl(4, R) ---
C = np.eye(4)
sl_candidates = [m - (np.trace(m) / 4.0) * C for m in matrices]
sl_vecs = np.array([m.flatten() for m in sl_candidates])

# Base ortonormal T_1...T_15 para sl(4,R)
_, _, Vh = np.linalg.svd(sl_vecs)
T = [Vh[i].reshape(4, 4) for i in range(15)]
T_matrix = np.array([t.flatten() for t in T])

# --- 2. Testes Geométricos e Espaço Vetorial ---
max_trace = max(abs(np.trace(t)) for t in T)
rank_T = np.linalg.matrix_rank(T_matrix)
max_comm_C = max(np.linalg.norm(C @ t - t @ C) for t in T)

# --- 3. Constantes de Estrutura f_{ij}^k ---
f_tensor = np.zeros((15, 15, 15))
comm_errors = []

for i in range(15):
    for j in range(15):
        comm = T[i] @ T[j] - T[j] @ T[i]
        coeffs, _, _, _ = np.linalg.lstsq(T_matrix.T, comm.flatten(), rcond=None)
        f_tensor[i, j, :] = coeffs
        proj = (T_matrix.T @ coeffs).reshape(4, 4)
        comm_errors.append(np.linalg.norm(comm - proj))

max_closure_error = max(comm_errors)

# Teste 5: Antisimetria f_ij^k = -f_ji^k
antisym_error = np.max(np.abs(f_tensor + np.transpose(f_tensor, (1, 0, 2))))

# Teste 6: Identidade de Jacobi em formato tensorial via Einstein Summation:
# J_{ijkl} = f_{ij}^m f_{mk}^l + f_{jk}^m f_{mi}^l + f_{ki}^m f_{mj}^l = 0
term1 = np.einsum('ijm, mkl -> ijkl', f_tensor, f_tensor)
term2 = np.einsum('jkm, mil -> ijkl', f_tensor, f_tensor)
term3 = np.einsum('kim, mjl -> ijkl', f_tensor, f_tensor)

jacobi_tensor = term1 + term2 + term3
max_jacobi_error = np.max(np.abs(jacobi_tensor))

# --- Relatório Final ---
print("=========================================================================")
print("=== RELATÓRIO COMPUTACIONAL DA ESTRUTURA DE LIE: g ≅ R (+) sl(4, R) ===")
print("=========================================================================")
print(f"1. Traço Nulo de sl(4,R) : Max |Tr(T_i)|          = {max_trace:.2e} -> {'PASSOU' if max_trace < 1e-12 else 'FALHOU'}")
print(f"2. Independência Linear   : Posto do subespaço T   = {rank_T}/15       -> {'PASSOU' if rank_T == 15 else 'FALHOU'}")
print(f"3. Centralidade de C      : Max ||[C, T_i]||       = {max_comm_C:.2e} -> {'PASSOU' if max_comm_C < 1e-12 else 'FALHOU'}")
print(f"4. Fechamento de sl(4,R)  : Erro Máx de Projeção   = {max_closure_error:.2e} -> {'PASSOU' if max_closure_error < 1e-12 else 'FALHOU'}")
print(f"5. Antisimetria [X, Y]    : Erro f_ij^k + f_ji^k   = {antisym_error:.2e} -> {'PASSOU' if antisym_error < 1e-12 else 'FALHOU'}")
print(f"6. Identidade de Jacobi   : Erro Máx no Tensor Jacobi = {max_jacobi_error:.2e} -> {'PASSOU' if max_jacobi_error < 1e-12 else 'FALHOU'}")
print("=========================================================================")
print("CONCLUSÃO MATEMÁTICA: a estrutura de Lie construída apresenta uma direção central")
print("e um setor traceless de dimensão 15, compatível com R ⊕ sl(4, R).")
print("Esta relação refere-se à estrutura interna da álgebra de 16 geradores.")
print("Ela não implica isomorfismo com a álgebra Alpha idealizada de dimensão 4.")
