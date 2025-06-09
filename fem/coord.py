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

from __future__ import annotations
import numpy as np

class Line:
    """ A deprecated line class. Not used at the moment."""
    def __init__(self, xpts: np.ndarray,
                 ypts: np.ndarray,
                 zpts: np.ndarray):
        self.xs: np.ndarray = xpts
        self.ys: np.ndarray = ypts
        self.zs: np.ndarray = zpts
        self.dxs: np.ndarray = xpts[1:] - xpts[:-1]
        self.dys: np.ndarray = ypts[1:] - ypts[:-1]
        self.dzs: np.ndarray = zpts[1:] - zpts[:-1]
        self.dl = np.sqrt(self.dxs**2 + self.dys**2 + self.dzs**2)
        self.length: float = np.sum(np.sqrt(self.dxs**2 + self.dys**2 + self.dzs**2))

        self.xmid: np.ndarray = 0.5*(xpts[:-1] + xpts[1:])
        self.ymid: np.ndarray = 0.5*(ypts[:-1] + ypts[1:])
        self.zmid: np.ndarray = 0.5*(zpts[:-1] + zpts[1:])
    
    @property
    def cmid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.xmid, self.ymid, self.zmid

    @staticmethod
    def from_points(start: np.ndarray, end: np.ndarray, Npts: int) -> Line:
        x1, y1, z1 = start
        x2, y2, z2 = end
        xs = np.linspace(x1, x2, Npts)
        ys = np.linspace(y1, y2, Npts)
        zs = np.linspace(z1, z2, Npts)
        return Line(xs, ys, zs)
    
