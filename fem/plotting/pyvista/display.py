from ...mesh3d import Mesh3D, SurfaceMesh
from ...geo3d import GMSHObject, GMSHSurface, GMSHVolume
from ...selection import FaceSelection, DomainSelection, EdgeSelection, Selection
from ...bc import PortBC
import numpy as np
import pyvista as pv
from typing import Iterable, Literal, Callable
from functools import wraps

def _logscale(dx, dy, dz):
    """
    Logarithmically scales vector magnitudes so that the largest remains unchanged
    and others are scaled down logarithmically.
    
    Parameters:
        dx, dy, dz (np.ndarray): Components of vectors.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Scaled dx, dy, dz arrays.
    """
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    dz = np.asarray(dz)

    # Compute original magnitudes
    mags = np.sqrt(dx**2 + dy**2 + dz**2)
    mags_nonzero = np.where(mags == 0, 1e-10, mags)  # avoid log(0)

    # Logarithmic scaling (scaled to max = original max)
    log_mags = np.log10(mags_nonzero)
    log_min = np.min(log_mags)
    log_max = np.max(log_mags)

    if log_max == log_min:
        # All vectors have the same length
        return dx, dy, dz

    # Normalize log magnitudes to [0, 1]
    log_scaled = (log_mags - log_min) / (log_max - log_min)

    # Scale back to original max magnitude
    max_mag = np.max(mags)
    new_mags = log_scaled * max_mag

    # Compute unit vectors
    unit_dx = dx / mags_nonzero
    unit_dy = dy / mags_nonzero
    unit_dz = dz / mags_nonzero

    # Apply scaled magnitudes
    scaled_dx = unit_dx * new_mags
    scaled_dy = unit_dy * new_mags
    scaled_dz = unit_dz * new_mags

    return scaled_dx, scaled_dy, scaled_dz

def _min_distance(xs, ys, zs):
    """
    Compute the minimum Euclidean distance between any two points
    defined by the 1D arrays xs, ys, zs.
    
    Parameters:
        xs (np.ndarray): x-coordinates of the points
        ys (np.ndarray): y-coordinates of the points
        zs (np.ndarray): z-coordinates of the points
    
    Returns:
        float: The minimum Euclidean distance between any two points
    """
    # Stack the coordinates into a (N, 3) array
    points = np.stack((xs, ys, zs), axis=-1)

    # Compute pairwise squared distances using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_squared = np.sum(diff ** 2, axis=-1)

    # Set diagonal to infinity to ignore zero distances to self
    np.fill_diagonal(dists_squared, np.inf)

    # Get the minimum distance
    min_dist = np.sqrt(np.min(dists_squared))
    return min_dist

def _norm(x, y, z):
    return np.sqrt(np.abs(x)**2 + np.abs(y)**2 + np.abs(z)**2)

def _select(obj: GMSHObject | Selection) -> Selection:
    if isinstance(obj, GMSHObject):
        return obj.select
    return obj

def _merge(lst: list[GMSHObject | Selection]) -> Selection:
    selections = [_select(item) for item in lst]
    dim = selections[0].dim
    all_tags = []
    for item in lst:
        all_tags.extend(_select(item).tags)
    
    if dim==1:
        return EdgeSelection(all_tags)
    elif dim==2:
        return FaceSelection(all_tags)
    elif dim==3:
        return DomainSelection(all_tags)

class Display:

    def __init__(self, mesh: Mesh3D, plotter: pv.Plotter = None):
        self._mesh: Mesh3D = mesh
        if plotter is None:
            plotter = pv.Plotter()
        self._plot: pv.Plotter = plotter

    def show(self):
        self._plot.show()
        
    def mesh_volume(self, volume: DomainSelection) -> pv.UnstructuredGrid:
        tets = self._mesh.get_tetrahedra(volume.tags)

        ntets = tets.shape[0]

        cells = np.zeros((ntets,5), dtype=np.int64)

        cells[:,1:] = self._mesh.tets[:,tets].T

        cells[:,0] = 4
        celltypes = np.full(ntets, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        points = self._mesh.nodes.T

        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh_surface(self, surface: FaceSelection) -> pv.UnstructuredGrid:
        tris = self._mesh.get_triangles(surface.tags)

        ntris = tris.shape[0]

        cells = np.zeros((ntris,4), dtype=np.int64)

        cells[:,1:] = self._mesh.tris[:,tris].T

        cells[:,0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = self._mesh.nodes.T

        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh(self, obj: GMSHObject | Selection | Iterable) -> pv.UnstructuredGrid:
        if isinstance(obj, Iterable):
            obj = _merge(obj)
        else:
            obj = _select(obj)
        
        if isinstance(obj, DomainSelection):
            return self.mesh_volume(obj)
        elif isinstance(obj, FaceSelection):
            return self.mesh_surface(obj)

    def add_mesh(self, obj: GMSHObject | Selection | Iterable,*args, **kwargs):
        self._plot.add_mesh(self.mesh(obj), *args, **kwargs)

    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        cloud = pv.PolyData(np.array([xs,ys,zs]).T)
        self._plot.add_points(cloud)

    def plot_portmode(self, port: PortBC, k0: float, Npoints: int = 10, dv=(0,0,0), XYZ=None,
                      field: Literal['E','H'] = 'E') -> pv.UnstructuredGrid:
        if XYZ:
            X,Y,Z = XYZ
        else:
            X,Y,Z = port.selection.sample(Npoints)
            for x,y,z in zip(X,Y,Z):
                self.plot_portmode(port, k0, Npoints, dv, (x,y,z), field)
            return
        X = X+dv[0]
        Y = Y+dv[1]
        Z = Z+dv[2]
        xf = X.flatten()
        yf = Y.flatten()
        zf = Z.flatten()

        d1 = np.sqrt((X[0,1]-X[0,0])**2 + (Y[0,1]-Y[0,0])**2 + (Z[0,1]-Z[0,0])**2)
        d2 = np.sqrt((X[1,0]-X[0,0])**2 + (Y[1,0]-Y[0,0])**2 + (Z[1,0]-Z[0,0])**2)
        d = min(d1, d2)

        F = port.port_mode_3d_global(xf,yf,zf, k0, which=field)

        Fx = F[0,:].reshape(X.shape).T
        Fy = F[1,:].reshape(X.shape).T
        Fz = F[2,:].reshape(X.shape).T

        if field=='H':
            F = np.imag(F.T)
            Fnorm = np.sqrt(Fx.imag**2 + Fy.imag**2 + Fz.imag**2)
        else:
            F = np.real(F.T)
            Fnorm = np.sqrt(Fx.real**2 + Fy.real**2 + Fz.real**2)

        grid = pv.StructuredGrid(X,Y,Z)
        self._plot.add_mesh(grid, scalars = Fnorm, opacity=0.8)

        Emag = F/np.max(Fnorm.flatten())*d*3
        self._plot.add_arrows(np.array([xf,yf,zf]).T, Emag)

    def plot_arrow(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              scale: float = 1,
              scalemode: Literal['lin','log'] = 'lin'):
        
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        dx = dx.flatten().real
        dy = dy.flatten().real
        dz = dz.flatten().real
        dmin = _min_distance(x,y,z)

        dmax = np.max(_norm(dx,dy,dz))
        
        Vec = scale * np.array([dx,dy,dz]).T / dmax * dmin 
        Coo = np.array([x,y,z]).T
        if scalemode=='log':
            dx, dy, dz = _logscale(Vec[:,0], Vec[:,1], Vec[:,2])
            Vec[:,0] = dx
            Vec[:,1] = dy
            Vec[:,2] = dz
        self._plot.add_arrows(Coo, Vec)


class PVPlotter:

    def __init__(self, mesh: Mesh3D):
        self.display: Display = Display(mesh)
    
