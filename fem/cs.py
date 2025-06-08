from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal



@dataclass
class Point:
    x: float
    y: float
    z: float

    
@dataclass
class Axis:
    vector: np.ndarray

    def __repr__(self) -> str:
        return f"Axis({self.vector})"
    
    def __post_init__(self):

        self.vector = self.vector/np.linalg.norm(self.vector)
        self.np: np.ndarray = self.vector
    
    @property
    def neg(self) -> Axis:
        return Axis(-self.vector)
    
    def cross(self, other: Axis) -> Axis:
        return Axis(np.cross(self.vector, other.vector))

    def dot(self, other: Axis) -> float:
        return np.dot(self.vector, other.vector)
    
    def pair(self, other: Axis) -> Plane:
        return Plane(self, other)
    
    def construct_cs(self) -> CoordinateSystem:
        ax = Axis(np.array([1, 0, 0]))
        if np.abs(self.dot(ax)) > 0.999:
            
            ax = Axis(np.array([0, 1, 0]))
        ax1 = self.cross(ax)
        ax2 = self.cross(ax1).neg
        return CoordinateSystem(ax2, ax1, self, np.zeros(3))
    
    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, np.ndarray):
            return np.dot(self.vector, other)
        elif isinstance(other, (int, float)):
            return self.vector * other
        else:
            raise ValueError("Multiplication not supported for this type")
        
    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)
    

XAX: Axis = Axis(np.array([1, 0, 0]))
YAX: Axis = Axis(np.array([0, 1, 0]))
ZAX: Axis = Axis(np.array([0, 0, 1]))

def _parse_vector(vec: np.ndarray | tuple | list | Axis) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        return vec
    elif isinstance(vec, (list,tuple)):
        return np.array(vec)
    elif isinstance(vec, Axis):
        return vec.vector
    return None

def _parse_axis(vec: np.ndarray | tuple | list | Axis) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        return Axis(vec)
    elif isinstance(vec, (list,tuple)):
        return Axis(np.array(vec))
    elif isinstance(vec, Axis):
        return vec
    return None

@dataclass
class Plane:
    uax: Axis
    vax: Axis

    def __post_init__(self):
        # Check if axes are orthogonal
        if not np.isclose(np.dot(self.uax.vector, self.vax.vector), 0):
            raise ValueError("Axes are not orthogonal")

    def __repr__(self) -> str:
        return f"Plane({self.uax}, {self.vax})"
    
    @property
    def normal(self) -> Axis:
        return self.uax.cross(self.vax)
    
    def flip(self) -> Plane:
        return Plane(self.vax, self.uax)
    
    def cs(self, origin: np.ndarray = None) -> CoordinateSystem:
        if origin is None:
            origin = np.zeros(3)
        return CoordinateSystem(self.uax, self.vax, self.normal, origin)
    
    def grid(self, 
             uax: np.ndarray | tuple[float, float, int], 
             vax: np.ndarray | tuple[float, float, int], 
             origin: np.ndarray | tuple[float, float, float],
             indexing: Literal['xy','ij'] = 'xy') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if isinstance(uax, tuple):
            uax = np.linspace(*uax)
        if isinstance(vax, tuple):
            vax = np.linspace(*vax)
        if isinstance(origin, tuple):
            origin = np.array(origin)

        U, V = np.meshgrid(uax, vax, indexing=indexing)
        uax = U.flatten()
        vax = V.flatten()
        shp = U.shape
        xs = self.uax.np[0]*uax + self.vax.np[0]*vax + origin[0]
        ys = self.uax.np[1]*uax + self.vax.np[1]*vax + origin[1]
        zs = self.uax.np[2]*uax + self.vax.np[2]*vax + origin[2]

        return xs.reshape(shp), ys.reshape(shp), zs.reshape(shp)
    
    def span(self, u: float, v: float, N: int, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Create a grid of XYZ coordinates in the plane reaching from 0 to u and 0 to v in N steps at the given origin'''
        uax = np.linspace(0, u, N)
        vax = np.linspace(0, v, N)
        U, V = np.meshgrid(uax, vax, indexing='ij')
        uax = U.flatten()
        vax = V.flatten()
        shp = U.shape
        xs = self.uax.np[0]*uax + self.vax.np[0]*vax + origin[0]
        ys = self.uax.np[1]*uax + self.vax.np[1]*vax + origin[1]
        zs = self.uax.np[2]*uax + self.vax.np[2]*vax + origin[2]

        return xs.reshape(shp), ys.reshape(shp), zs.reshape(shp)

XYPLANE = Plane(XAX, YAX)
XZPLANE = Plane(XAX, ZAX)
YZPLANE = Plane(YAX, ZAX)
YXPLANE = Plane(YAX, XAX)
ZXPLANE = Plane(ZAX, XAX)
ZYPLANE = Plane(ZAX, YAX)


@dataclass
class CoordinateSystem:
    xax: Axis
    yax: Axis
    zax: Axis
    origin: np.ndarray
    _is_global: bool = False

    def __post_init__(self):
        self.xax = _parse_axis(self.xax)
        self.yax = _parse_axis(self.yax)
        self.zax = _parse_axis(self.zax)

        self._basis = np.array([self.xax.np, self.yax.np, self.zax.np]).T

        self._basis_inv = np.linalg.pinv(self._basis)

    def __repr__(self) -> str:
        return f"CoordinateSystem({self.xax}, {self.yax}, {self.zax}, {self.origin})"
    
    def copy(self) -> CoordinateSystem:
        return CoordinateSystem(self.xax, self.yax, self.zax, self.origin)
    
    def displace(self, dx: float, dy: float, dz: float) -> CoordinateSystem:
        csnew = CoordinateSystem(self.xax, self.yax, self.zax, self.origin + np.array([dx, dy, dz]))
        return csnew
    
    def rotate(self, axis: tuple | list | np.ndarray | Axis, 
               angle: float, 
               degrees: bool = True) -> CoordinateSystem:
        """
        Return a new CoordinateSystem rotated about the given axis (through the global origin)
        by `angle`. If `degrees` is True, `angle` is interpreted in degrees.
        """

        # Convert to radians if needed
        if degrees:
            theta = angle * np.pi/180

        # Normalize the rotation axis
        u = _parse_vector(axis)
        u = u / np.linalg.norm(u)

        # Build the skew-symmetric cross-product matrix K for u
        K = np.array([
            [   0,   -u[2],  u[1]],
            [ u[2],     0,  -u[0]],
            [-u[1],  u[0],     0]
        ], dtype=float)

        # Rodrigues' rotation formula: R = I + sinθ·K + (1−cosθ)·K²
        I = np.eye(3, dtype=float)
        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # Rotate each axis and the origin
        new_x = R @ self.xax.vector
        new_y = R @ self.yax.vector
        new_z = R @ self.zax.vector
        #new_o = R @ self.origin

        return CoordinateSystem(
            xax=new_x,
            yax=new_y,
            zax=new_z,
            origin=self.origin,
            _is_global=self._is_global
        )
    
    def swapxy(self) -> None:
        self.xax, self.yax = self.yax, self.xax
        self.__post_init__()
    
    def affine_from_global(self) -> np.ndarray:
        """
        xhat, yhat, zhat, origin  : each shape (3,) or (3,1)
        returns a 4×4 affine matrix T such that
            [R | origin]
            [0 |   1   ]
        where R = [xhat yhat zhat].
        """
        # ensure they’re 1-D
        x = self.xax.vector
        y = self.yax.vector
        z = self.zax.vector
        o = self.origin

        T = np.eye(4, dtype=float)
        T[0:3, 0] = x
        T[0:3, 1] = y
        T[0:3, 2] = z
        T[0:3, 3] = o
        return T
    
    def affine_to_global(self) -> np.ndarray:
        T = self.affine_from_global()
        R = T[0:3, 0:3]
        o = T[0:3, 3]
        R_inv = np.linalg.inv(R)         # general inverse of R
        o_new = - R_inv @ o              # new translation
        T_inv = np.eye(4, dtype=float)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3]   = o_new
        return T_inv
    
    def in_global_cs(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xg = self.xax.np[0]*x + self.yax.np[0]*y + self.zax.np[0]*z + self.origin[0]
        yg = self.xax.np[1]*x + self.yax.np[1]*y + self.zax.np[1]*z + self.origin[1]
        zg = self.xax.np[2]*x + self.yax.np[2]*y + self.zax.np[2]*z + self.origin[2]
        return xg, yg, zg
    
    def in_local_cs(self, x: np.ndarray,
                    y: np.ndarray,
                    z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = self._basis_inv
        xg = x - self.origin[0]
        yg = y - self.origin[1]
        zg = z - self.origin[2]
        x = B[0,0]*xg + B[0,1]*yg + B[0,2]*zg
        y = B[1,0]*xg + B[1,1]*yg + B[1,2]*zg
        z = B[2,0]*xg + B[2,1]*yg + B[2,2]*zg
        return x, y, z
    
    def in_global_basis(self, x: np.ndarray,
                        y: np.ndarray,
                        z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xg = self.xax.np[0]*x + self.yax.np[0]*y + self.zax.np[0]*z
        yg = self.xax.np[1]*x + self.yax.np[1]*y + self.zax.np[1]*z
        zg = self.xax.np[2]*x + self.yax.np[2]*y + self.zax.np[2]*z
        return xg, yg, zg
    
    def in_local_basis(self, x: np.ndarray,
                          y: np.ndarray,
                          z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
          B = self._basis_inv
          xg = x
          yg = y
          zg = z
          x = B[0,0]*xg + B[0,1]*yg + B[0,2]*zg
          y = B[1,0]*xg + B[1,1]*yg + B[1,2]*zg
          z = B[2,0]*xg + B[2,1]*yg + B[2,2]*zg
          return x, y, z
    
    @property
    def gx(self) -> float:
        return self.origin[0]
    
    @property
    def gy(self) -> float:
        return self.origin[1]
    
    @property
    def gz(self) -> float:
        return self.origin[2]
    
    @property
    def gxhat(self) -> np.ndarray:
        return self.xax.np
    
    @property
    def gyhat(self) -> np.ndarray:
        return self.yax.np
    
    @property
    def gzhat(self) -> np.ndarray:
        return self.zax.np
    
GCS = CoordinateSystem(XAX, YAX, ZAX, np.zeros(3), _is_global=True)
