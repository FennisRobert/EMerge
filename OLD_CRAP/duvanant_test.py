import numpy as np
from emerge.solvers.fem.elements.dunavant import duvanant_points
# 1. Define triangle vertices on the XY plane
def define_triangle():
    # Example points A, B, C
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, 0.8])
    return np.vstack((A, B, C))

# 2. Define two example functions f1 and f2
def f1(x, y):
    """
    Example function: sine-cosine polynomial
    """
    return 0.5*x**2 + 0.5*x + 2*y + 3*y**2


def f2(x, y):
    """
    Example function: simple polynomial
    """
    return x**2 + y**2

# 3. Helper: check if a point p lies inside triangle defined by vertices[T x 2]
def point_in_triangle(p, T):
    A, B, C = T[0], T[1], T[2]
    # Compute barycentric coordinates
    v0 = C - A
    v1 = B - A
    v2 = p - A

    denom = v0[0]*v1[1] - v1[0]*v0[1]
    if np.isclose(denom, 0):
        return False

    a = (v2[0]*v1[1] - v1[0]*v2[1]) / denom
    b = (v0[0]*v2[1] - v2[0]*v0[1]) / denom

    # Point is inside if a >= 0, b >= 0 and a + b <= 1
    return (a >= 0) and (b >= 0) and (a + b <= 1)

# 4. Simple numerical integration over the triangle via pixel counting

def integrate_on_triangle(f, T, resolution=500):
    """
    Integrate function f(x, y) over triangle T by discretizing
    the bounding box into a grid of size resolution x resolution.

    Args:
        f: function of two variables (x, y)
        T: array of shape (3, 2) with triangle vertices
        resolution: number of subdivisions per axis

    Returns:
        Approximate integral value
    """
    # Bounding box
    xmin, ymin = T.min(axis=0)
    xmax, ymax = T.max(axis=0)

    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution

    total = 0.0
    # Loop over grid cells
    iis = np.arange(resolution)
    jjs = np.arange(resolution)

    I,J = np.meshgrid(iis, jjs, indexing='ij')

    x = xmin + (I.flatten() + 0.5) * dx
    y = ymin + (J.flatten() + 0.5) * dy


    p = np.array([x,y])

    A, B, C = T[0], T[1], T[2]
    # Compute barycentric coordinates
    v0 = C - A
    v1 = B - A
    v2 = p - A[:, np.newaxis]

    denom = v0[0]*v1[1] - v1[0]*v0[1]
    if np.isclose(denom, 0):
        return False

    a = (v2[0]*v1[1] - v1[0]*v2[1]) / denom
    b = (v0[0]*v2[1] - v2[0]*v0[1]) / denom

    mask = (a >= 0) & (b >= 0) & (a + b <= 1)

    total = np.sum(f(x[mask], y[mask])) * dx * dy
    return total

def duvanant_integral(f, order):
    pts = duvanant_points(order)
    A,B,C = define_triangle()
    x1, x2, x3 = A[0], B[0], C[0]
    y1, y2, y3 = A[1], B[1], C[1]

    Area = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    xs = x1*pts[1,:] + x2*pts[2,:] + x3*pts[3,:]
    ys = y1*pts[1,:] + y2*pts[2,:] + y3*pts[3,:]

    return Area*np.sum(pts[0,:] * f(xs, ys))


if __name__ == "__main__":
    # Build triangle
    triangle = define_triangle()

    # Choose resolution
    res = 5000  # increase for higher accuracy

    # Compute integrals
    integral_f1 = integrate_on_triangle(lambda x,y: f1(x,y)*f2(x,y), triangle, resolution=res)
    integral_f2 = duvanant_integral(lambda x,y: f1(x,y)*f2(x,y), 10)

    print(f"Approximate integral of f1 over triangle: {integral_f1}")
    print(f"Approximate integral of f2 over triangle: {integral_f2}")

    # TODO: Insert duvanant integration method here to compare results.
