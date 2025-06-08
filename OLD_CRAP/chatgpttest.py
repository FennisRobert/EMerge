import numpy as np

# Reference tetrahedron vertices
vertices = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([1.0, 0.0, 0.0]),
    3: np.array([0.0, 1.0, 0.0]),
    4: np.array([0.0, 0.0, 1.0]),
}

def compute_alpha_edge(ive1, ive2, C=1.0, num_pts=1001):
    # Physical edge length
    v1, v2 = vertices[ive1], vertices[ive2]
    length = np.linalg.norm(v2 - v1)

    # Quadratic shape polynomials on [0,1]
    def P_i(s): return 1 - 3*s + 3*s*s
    def P_j(s): return 3*s*s - 2*s
    def q0(s): return 1 - 2*s

    # Sample s in [0,1]
    s = np.linspace(0, 1, num_pts)
    integrand = P_i(s) * P_j(s) * q0(s)
    I = np.trapezoid(integrand, s)

    # Solve alpha * C * length * I = 1
    alpha = 1.0 / (C * length * I)
    return alpha, length, I

alpha, length, I = compute_alpha_edge(1, 2, C=1.0)
print(f"Edge (1,2): length={length:.6f}, I={I:.6f}, alpha (C=1)={alpha:.6f}")
