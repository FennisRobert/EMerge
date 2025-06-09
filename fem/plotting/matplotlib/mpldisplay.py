# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

from ...mesh3d import Mesh3D, SurfaceMesh
from ...geo3d import GMSHObject, GMSHSurface, GMSHVolume
from ...selection import FaceSelection, DomainSelection, EdgeSelection, Selection
from ...bc import PortBC
import numpy as np
from typing import Iterable, Literal, Callable
from functools import wraps
from ..display import BaseDisplay
from matplotlib.colors import Normalize
from matplotlib import cm
from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def align_triangle_normals(nodes: np.ndarray, triangles: np.ndarray):
    """
    Flip triangle winding to align with given normals.

    Parameters:
    - nodes: (3, N) array of 3D points.
    - triangles: (3, M) array of indices into nodes.
    - normals: (3, M) array of desired triangle normals.

    Returns:
    - triangles_aligned: (3, M) array with consistent winding.
    """
     # Get triangle vertices
    p0 = nodes[:, triangles[0,:]]
    p1 = nodes[:, triangles[1,:]]
    p2 = nodes[:, triangles[2,:]]

    # Compute current normals (not normalized)
    ns = np.cross(p1 - p0, p2 - p0, axis=0)
    tris_out = np.zeros_like(triangles, dtype=np.int64)
    for it in range(triangles.shape[1]):
        if (-ns[0,it]-ns[1,it]-ns[2,it]) < 0:
            tris_out[:,it] = triangles[:,it]
        else:
            tris_out[:,it] = triangles[(0,2,1),it]
    return tris_out

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

def _make_facecolors(C, cmap='viridis'):
    """
    Convert C (N, M) to facecolors (N-1, M-1, 4) using a colormap.
    """
    from matplotlib import cm
    C_avg = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])
    normed = (C_avg - np.min(C_avg)) / np.ptp(C_avg)
    return plt.get_cmap(cmap)(normed)

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

class MPLDisplay(BaseDisplay):
    COLORS: dict = {'green': (0,0.8,0), 'red': (0.8,0,0), 'blue': (0,0,0.8)}
    def __init__(self, mesh: Mesh3D):
        self._mesh: Mesh3D = mesh
        self._fig = None
        self._ax = None

    def init(self):
        if self._fig is None or self._ax is None:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._ax.axis('equal')
            self._ax.set_aspect('equal')
        
    def show(self):
        plt.show()
        self._fig = None
        self._ax = None

    ## OBLIGATORY METHODS
    def add_object(self, obj: GMSHObject | Selection | Iterable, opacity: float = 1, color: str = None, **kwargs):
        self.init()

        if color is not None:
            color = self.COLORS.get(color,(0,0.5,0))

        boundary_tris = self._mesh.boundary_triangles(obj.dimtags)
        tris = self._mesh.tris[:,boundary_tris]
        tris = align_triangle_normals(self._mesh.nodes, tris)
        x = self._mesh.nodes[0,:]
        y = self._mesh.nodes[1,:]
        z = self._mesh.nodes[2,:]
        surf = self._ax.plot_trisurf(
                x, y, z,triangles=tris.T,
                color=obj.color + (opacity,),
                linewidth=0.2,
                antialiased=True,
                shade=True
            )
          # Equal aspect ratio


    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        self.init()
        self._ax.scatter(xs,ys,zs)

    def add_portmode(self, port: PortBC, k0: float, Npoints: int = 5, dv=(0,0,0), XYZ=None,
                      field: Literal['E','H'] = 'E'):
        if XYZ:
            X,Y,Z = XYZ
        else:
            X,Y,Z = port.selection.sample(Npoints)
            for x,y,z in zip(X,Y,Z):
                self.add_portmode(port, k0, Npoints, dv, (x,y,z), field)
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

        Fx = F[0,:].reshape(X.shape)
        Fy = F[1,:].reshape(X.shape)
        Fz = F[2,:].reshape(X.shape)

        if field=='H':
            F = np.imag(F)
            Fnorm = np.sqrt(Fx.imag**2 + Fy.imag**2 + Fz.imag**2)
        else:
            F = np.real(F)
            Fnorm = np.sqrt(Fx.real**2 + Fy.real**2 + Fz.real**2)

        cmap = 'viridis'

        colors = _make_facecolors(Fnorm)
        surf = self._ax.plot_surface(X, Y, Z,facecolors=colors,
                           rstride=1, cstride=1, antialiased=True, linewidth=0)
       


        N = np.max(Fnorm.flatten())*d*3
        self.add_quiver(xf, yf, zf, F[0,:].real/N, F[1,:].real/N, F[2,:].real/N)

    def add_quiver(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              scale: float = 1,
              scalemode: Literal['lin','log'] = 'lin'):
        
        self.init()
        dlmax = np.sqrt(dx**2+dy**2+dz**2)
        dmin = _min_distance(x,y,z)
        scale = dmin/max(dlmax)
        self._ax.quiver(x,y,z,dx*scale,dy*scale,dz*scale, color='black')