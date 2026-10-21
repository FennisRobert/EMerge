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

# Last Cleanup: 2026-01-04

from __future__ import annotations
import numpy as np
from typing import Callable
from emsutil import Saveable
from emsutil import plot


def gauss3_composite(x: np.ndarray, y: np.ndarray) -> float:
    """
    Composite 3-point Gauss (order-2) integral
    on *equally spaced* 1-D data.

    Sections:
        (x0,x1,x2), (x2,x3,x4), (x4,x5,x6), ...

    The rule on one section [x_i, x_{i+2}] is

        ∫ f dx  ≈  (h/2) Σ_k w_k · f( x̄ + (h/2) ξ_k )

    where  h = x_{i+2}−x_i  and
           ξ = (−√3/5, 0, +√3/5),
           w = (8/9, 5/9, 8/9)

    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    if x.size % 2 == 0:
        raise ValueError("Number of samples must be odd (… 0,1,2; 2,3,4; …).")

    xi = np.sqrt(3 / 5)
    nodes = np.array([-xi, 0.0, +xi])
    weights = np.array([5 / 9, 8 / 9, 5 / 9])

    total = 0.0
    for i in range(0, x.size - 2, 2):
        y0, y1, y2 = y[i : i + 3]
        a = y0 * 0.5 - y1 + 0.5 * y2
        b = -y0 * 0.5 + 0.5 * y2
        c = y1
        poly_vals = a * nodes**2 + b * nodes + c
        total += np.dot(weights, poly_vals)
    return total


class Line(Saveable):
    """A Line class used for convenient definition of integration lines

    Args:
        xpts (np.ndarray): The x-coordinates of the line points
        ypts (np.ndarray): The y-coordinates of the line points
        zpts (np.ndarray): The z-coordinates of the line points
    """

    def __init__(self, xpts: np.ndarray, ypts: np.ndarray, zpts: np.ndarray):

        self.xs: np.ndarray = xpts
        self.ys: np.ndarray = ypts
        self.zs: np.ndarray = zpts
        self.dxs: np.ndarray = xpts[1:] - xpts[:-1]
        self.dys: np.ndarray = ypts[1:] - ypts[:-1]
        self.dzs: np.ndarray = zpts[1:] - zpts[:-1]
        self.dl = np.sqrt(self.dxs**2 + self.dys**2 + self.dzs**2)
        self.length: float = np.sum(np.sqrt(self.dxs**2 + self.dys**2 + self.dzs**2))
        self.l: np.ndarray = np.concatenate(
            (
                np.array(
                    [
                        0,
                    ]
                ),
                np.cumsum(self.dl),
            )
        )
        self.xmid: np.ndarray = 0.5 * (xpts[:-1] + xpts[1:])
        self.ymid: np.ndarray = 0.5 * (ypts[:-1] + ypts[1:])
        self.zmid: np.ndarray = 0.5 * (zpts[:-1] + zpts[1:])

        self.dx: float = self.dxs[0]
        self.dy: float = self.dys[0]
        self.dz: float = self.dzs[0]

    @property
    def cmid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The midpoints of the line segments.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The midpoint coordinates (xmid, ymid, zmid).
        """
        return self.xmid, self.ymid, self.zmid

    @property
    def cpoint(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The center points of the line segments."""
        return self.xs, self.ys, self.zs

    @staticmethod
    def from_points(start: np.ndarray, end: np.ndarray, Npts: int) -> Line:
        """Create a Line object from start to end with Npts points.

        Args:
            start (np.ndarray): start point coordinates (x,y,z).
            end (np.ndarray): end point coordinates (x,y,z).
            Npts (int): Number of points along the line.

        Returns:
            Line: The created Line object.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end
        xs = np.linspace(x1, x2, Npts)
        ys = np.linspace(y1, y2, Npts)
        zs = np.linspace(z1, z2, Npts)
        return Line(xs, ys, zs)

    @staticmethod
    def from_path(*points: tuple[float, float, float] | np.ndarray, ds: float) -> Line:
        """Creates a Line object from a series of points with approximate spacing ds.

        Args:
            ds (float): Approximate spacing between points.

        Returns:
            Line: The created Line object.
        """
        xpts = []
        ypts = []
        zpts = []
        points = [np.array(p) for p in points]
        xl = None
        yl = None
        zl = None
        for p1, p2 in zip(points[:-1], points[1:]):
            N = max(3, int(np.ceil(np.linalg.norm(p2 - p1) / ds)))
            *xs, xl = list(np.linspace(p1[0], p2[0], N))
            *ys, yl = list(np.linspace(p1[1], p2[1], N))
            *zs, zl = list(np.linspace(p1[2], p2[2], N))
            xpts.extend(xs)
            ypts.extend(ys)
            zpts.extend(zs)
        xpts.append(xl)
        ypts.append(yl)
        zpts.append(zl)
        return Line(np.array(xpts), np.array(ypts), np.array(zpts))

    def subsample(self, n: int) -> Line:
        """Split each segment into n sub-segments, returning a new Line.

        Args:
            n (int): Number of sub-segments per original segment.

        Returns:
            Line: A new Line with n*len(segments) segments.
        """
        xs, ys, zs = [], [], []
        for i in range(len(self.dxs)):
            t = np.linspace(0, 1, n + 1, endpoint=(i == len(self.dxs) - 1))
            if i < len(self.dxs) - 1:
                t = t[:-1]  # avoid duplicating shared vertices
            xs.append(self.xs[i] + t * self.dxs[i])
            ys.append(self.ys[i] + t * self.dys[i])
            zs.append(self.zs[i] + t * self.dzs[i])

        return Line(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))

    def smooth(self, nsteps: int = 1) -> "Line":
        """Smooth the path by Laplacian smoothing while preserving total arc length.

        Each step replaces every interior point with the average of its two
        neighbors, then rescales all segment lengths uniformly so the total
        arc length is unchanged. For closed loops (where the first and last
        point coincide), all points are treated as interior.

        Args:
            nsteps (int, optional): Number of smoothing iterations. Defaults to 1.

        Returns:
            Line: A new Line with smoothed coordinates.
        """
        xs = self.xs.copy()
        ys = self.ys.copy()
        zs = self.zs.copy()
        original_length = self.length

        # Detect closed loop: first point == last point
        closed = (
            np.abs(xs[0] - xs[-1]) < 1e-14
            and np.abs(ys[0] - ys[-1]) < 1e-14
            and np.abs(zs[0] - zs[-1]) < 1e-14
        )

        for _ in range(nsteps):
            if closed:
                # All points except the duplicated last one are smoothed
                # using periodic neighbors
                n = len(xs) - 1  # number of unique points
                new_xs = np.empty_like(xs)
                new_ys = np.empty_like(ys)
                new_zs = np.empty_like(zs)
                for i in range(n):
                    ip = (i + 1) % n
                    im = (i - 1) % n
                    new_xs[i] = 0.5 * (xs[ip] + xs[im])
                    new_ys[i] = 0.5 * (ys[ip] + ys[im])
                    new_zs[i] = 0.5 * (zs[ip] + zs[im])
                new_xs[-1] = new_xs[0]
                new_ys[-1] = new_ys[0]
                new_zs[-1] = new_zs[0]
            else:
                new_xs = xs.copy()
                new_ys = ys.copy()
                new_zs = zs.copy()
                new_xs[1:-1] = 0.5 * (xs[:-2] + xs[2:])
                new_ys[1:-1] = 0.5 * (ys[:-2] + ys[2:])
                new_zs[1:-1] = 0.5 * (zs[:-2] + zs[2:])

            # Rescale to preserve total arc length
            dxs = new_xs[1:] - new_xs[:-1]
            dys = new_ys[1:] - new_ys[:-1]
            dzs = new_zs[1:] - new_zs[:-1]
            new_length = np.sum(np.sqrt(dxs**2 + dys**2 + dzs**2))

            if new_length > 1e-30:
                scale = original_length / new_length
                center_x = new_xs.mean()
                center_y = new_ys.mean()
                center_z = new_zs.mean()
                new_xs = center_x + scale * (new_xs - center_x)
                new_ys = center_y + scale * (new_ys - center_y)
                new_zs = center_z + scale * (new_zs - center_z)

                if closed:
                    new_xs[-1] = new_xs[0]
                    new_ys[-1] = new_ys[0]
                    new_zs[-1] = new_zs[0]

            xs, ys, zs = new_xs, new_ys, new_zs

        return Line(xs, ys, zs)

    def line_integral(self, evalfunc: Callable) -> complex:
        """Computes a line integral of some vector field function.

        The function will be passed the arguments (x,y,z)
        The variables x,y,z will be np.ndarray's of shape (N,)
        The function must return a vector field of shape (3,N), (1,N) or (N,)

        For E-field integration you can typically use:
        >>> lambda x,y,z: field.interpolate(x,y,z).E

        Args:
            evalfunc (Callable): The vector field function

        Returns:
            complex: The line integral value
        """
        Fout = evalfunc(*self.cmid)

        N = self.dxs.shape[0]
        if isinstance(Fout, tuple):
            Fx, Fy, Fz = Fout
        elif isinstance(Fout, np.ndarray):
            if Fout.shape == (3, self.dxs.shape[0]):
                Fx, Fy, Fz = Fout
            elif Fout.shape == (1, N) or Fout.shape == (N,):
                ds = (self.dxs**2 + self.dys**2 + self.dzs**2) ** 0.5
                Fx = Fout * self.dxs / ds
                Fy = Fout * self.dys / ds
                Fz = Fout * self.dzs / ds
        else:
            raise ValueError(
                "Output of function must be either a (3,N) array, (1,N) array, or an (N,) array."
            )
        EdotL = Fx * self.dxs + Fy * self.dys + Fz * self.dzs
        # plot(np.cumsum(self.dl), EdotL)
        return np.sum(EdotL)

    def line_integral_precalc(
        self, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray
    ) -> complex:
        """Compute the line integral of a vector field sampled at the line points.

        Args:
            Ex (np.ndarray): The field x-component at the line points
            Ey (np.ndarray): The field y-component at the line points
            Ez (np.ndarray): The field z-component at the line points

        Returns:
            complex: The line integral
        """

        EdotL = Ex * self.dx + Ey * self.dy + Ez * self.dz
        return gauss3_composite(self.l, EdotL)

    def _integrate(self, quantity: np.ndarray) -> complex:
        """Integrates a quantity of values defined along the line.

        Args:
            quantity (np.ndarray): The quantity values along the line.

        Returns:
            complex: The integrated value.
        """
        return gauss3_composite(self.l, quantity)
