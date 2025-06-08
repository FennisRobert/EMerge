import numpy as np

def fit_plane_basis(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    """
    Given N points (xs, ys, zs), compute an orthonormal basis (x̂, ŷ, n̂)
    and origin that best fits the points under the assumption they lie on a plane.
    
    Returns:
        xhat: np.ndarray (3,) - local x direction
        yhat: np.ndarray (3,) - local y direction
        nhat: np.ndarray (3,) - normal to the plane (local z direction)
        origin: np.ndarray (3,) - centroid of the points
    """
    # Stack the points into a (N, 3) matrix
    points = np.column_stack((xs, ys, zs))
    
    # Compute the centroid (origin)
    origin = points.mean(axis=0)
    
    # Center the points
    centered = points - origin

    # Perform SVD on the centered points
    # U, S, Vt = np.linalg.svd(centered)
    # The rows of Vt are the principal directions
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # The normal vector is the one corresponding to the smallest singular value
    nhat = Vt[2]
    
    # The first two vectors span the best-fit plane
    xhat = Vt[0]
    yhat = Vt[1]

    return xhat, yhat, nhat, origin

xs = np.linspace(0, 1, 10)
ys = np.linspace(0, 1, 10)

xs, ys = np.meshgrid(xs, ys)
xs = xs.flatten()
ys = ys.flatten()
zs = 0*xs

print(fit_plane_basis(xs, ys, zs))
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.cross(a,b)
print(np.array([a,b,c,c]).shape)