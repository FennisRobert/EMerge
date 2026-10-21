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
from dataclasses import dataclass, field
from typing import Literal
from emsutil import Saveable


@dataclass
class Axis(Saveable):
    """A representation of an axis.
    An Axis object always has length 1 and points in some 3D direction.
    By default XAX, YAX, and ZAX are constructed and defined in the global namespace of the
    FEM module.

    Args:
        vector (np.ndarray): The axis direction (normalized automatically)
    """

    vector: np.ndarray

    def __repr__(self) -> str:
        return f"Axis({self.vector})"

    def __post_init__(self):
        self.vector = np.array(self.vector)
        self.vector = self.vector / np.linalg.norm(self.vector)
        self.np: np.ndarray = self.vector
        self.x: float = self.vector[0]
        self.y: float = self.vector[1]
        self.z: float = self.vector[2]

    def __neg__(self):
        return Axis(-self.vector)

    @property
    def tuple(self) -> tuple[float, float, float]:
        """The axis vector as an (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    @property
    def xy(self) -> tuple[float, float]:
        """The (x, y) components of the axis vector."""
        return (self.x, self.y)

    @property
    def neg(self) -> Axis:
        """The axis pointing in the opposite direction."""
        return Axis(-self.vector)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def cross(self, other: Axis) -> Axis:
        """Take the cross produt with another vector

        Args:
            other (Axis): Vector B in AxB

        Returns:
            Axis: The resultant Axis.
        """
        return Axis(np.cross(self.vector, other.vector))

    def dot(self, other: Axis) -> float:
        """Take the dot product of two vectors A·B

        Args:
            other (Axis): Vector B

        Returns:
            float: The resultant vector (A·B)
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def pair(self, other: Axis) -> Plane:
        """Pair this vector with another to span the plane A⨂B

        Args:
            other (Axis): Vector B

        Returns:
            Plane: The plane spanned by A and B.
        """
        return Plane(self, other)

    def __mul__(self, other: Axis) -> Plane:
        """The multiply binary operator

        Args:
            other (Axis): The multiplied Axis object

        Returns:
            Plane: The output plane
        """
        return self.pair(other)

    def construct_cs(
        self, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> CoordinateSystem:
        """Constructs a coordinate system where this vector is the Z-axis
        and the X and Y axis are normal to this axis but with an arbitrary rotation.

        Args:
            origin (tuple[float, float, float], optional): The origin of the coordinate system. Defaults to (0.0, 0.0, 0.0).

        Returns:
            CoordinateSystem: The resultant coordinate system
        """
        ax = Axis(np.array([1, 0, 0]))
        if np.abs(self.dot(ax)) > 0.999:
            ax = Axis(np.array([0, 1, 0]))
        ax1 = self.cross(ax)
        ax2 = self.cross(ax1).neg
        return CoordinateSystem(ax2, ax1, self, np.array(origin))

    def tensor(self, axis_value: complex, plane_value: complex) -> np.ndarray:
        """Generates a uniaxial (3x3) tensor with an axis value and orthogonal value

        Args:
            axis_value (complex): _description_
            plane_value (complex): _description_

        Returns:
            np.ndarray: _description_

        Yields:
            Iterator[np.ndarray]: _description_
        """
        n = self.np
        return plane_value*np.eye(3) + (axis_value-plane_value)*np.outer(n,n)

############################################################
#                         CONSTANTS                        #
############################################################

XAX: Axis = Axis(np.array([1, 0, 0]))
YAX: Axis = Axis(np.array([0, 1, 0]))
ZAX: Axis = Axis(np.array([0, 0, 1]))

############################################################
#                         FUNCTIONS                        #
############################################################


def _parse_vector(
    vec: np.ndarray | tuple[float, float, float] | list[float] | Axis | Anchor,
) -> np.ndarray:
    """Takes an array, tuple, list or Axis and alwasys returns an array."""
    if isinstance(vec, np.ndarray):
        return vec
    elif isinstance(vec, (list, tuple)):
        return np.array(vec)
    elif isinstance(vec, Axis):
        return vec.vector
    elif isinstance(vec, Anchor):
        return vec.c0
    return np.array(vec)


def _parse_axis(
    vec: np.ndarray | tuple[float, float, float] | list[float] | Axis,
) -> Axis:
    """Takes an array, tuple, list or Axis and always returns an Axis.

    Args:
        vec (np.ndarray | tuple | list | Axis): The Axis data

    Returns:
        Axis: The Axis object.
    """
    if isinstance(vec, np.ndarray):
        return Axis(vec)
    elif isinstance(vec, (list, tuple)):
        return Axis(np.array(vec))
    elif isinstance(vec, Axis):
        return vec
    return Axis(np.array(vec))


def argparse_xyz(
    x: float | np.ndarray | tuple[float, float, float] | list[float] | Axis | Anchor,
    y: float | None = None,
    z: float | None = None,
) -> tuple[float, float, float]:
    """A helper function to parse xyz arguments that can be given
    as single arrays, tuples, lists, Axis or Point objects or as separate
    x,y,z float values.
    Args:
        x (float | np.ndarray | tuple[float, float, float] | list[float] | Axis | Anchor): The x value or entire coordinate
        y (float | None, optional): The y value. Defaults to None.
        z (float | None, optional): The z value. Defaults to None.

    Returns:
        tuple[float, float, float]: The (x, y, z) coordinate
    """
    if y is None and z is None:
        if isinstance(x, (float, int, np.floating, np.integer)):
            raise ValueError(
                "If x is a single float value, y and z must also be provided."
            )
        return tuple(_parse_vector(x))
    else:
        if y is None:
            y = 0.0
        if z is None:
            z = 0.0
        return (x, y, z)


############################################################
#                        PLANE CLASS                       #
############################################################


@dataclass
class Plane(Saveable):
    """A generalization of any plane of inifinite size spanned by two Axis objects.

    Args:
        uax (Axis): The first in-plane axis
        vax (Axis): The second in-plane axis (orthogonal to uax)
    """

    uax: Axis
    vax: Axis

    def __post_init__(self):
        # Check if axes are orthogonal
        if not np.isclose(np.dot(self.uax.vector, self.vax.vector), 0):
            raise ValueError("Axes are not orthogonal")

    def __repr__(self) -> str:
        return f"Plane({self.uax}, {self.vax})"

    def __mul__(self, other: Axis) -> CoordinateSystem:
        return CoordinateSystem(self.uax, self.vax, other)

    @property
    def normal(self) -> Axis:
        """Returns the normal of the plane as u ⨉ v.

        Returns:
            Axis: The axis object normal to the plane.
        """
        return self.uax.cross(self.vax)

    def flip(self) -> Plane:
        """Flips the planes U and V axes.

        Returns:
            Plane: A new plane object.
        """
        return Plane(self.vax, self.uax)

    def cs(self, x0: float = 0, y0: float = 0, z0: float = 0) -> CoordinateSystem:
        """Returns a CoordinateSystem object for the plane where the XY axes are aligned
        with the plane UV axis and Z is normal.

        Args:
            x0 (float): The x coordinate of the origin
            y0 (float): The y coordinate of the origin
            z0 (float): The z coordinate of the origin

        Returns:
            CoordinateSystem: The coordinate system object
        """
        origin = np.array([x0, y0, z0])
        return CoordinateSystem(self.uax, self.vax, self.normal, origin)

    def grid(
        self,
        uax: np.ndarray | tuple[float, float, int],
        vax: np.ndarray | tuple[float, float, int],
        origin: np.ndarray | tuple[float, float, float],
        indexing: Literal["xy", "ij"] = "xy",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Spans a grid of points in the plane based on a np.linspace like argument type.
        The first uax argument should be a start, finish, Npoints type tuple of a float, float and integer.
        Item for the vax. The origin defines the coordinate at which u,v = 0 will be placed.
        The return type is an N,M np.meshgrid defined by the indexing 'xy' or 'ij'.

        Args:
            uax (np.ndarray | tuple[float, float, int]): The uax linspace argument values
            vax (np.ndarray | tuple[float, float, int]): The vax linspace argument values
            origin (np.ndarray | tuple[float, float, float]): The origin for u,v = 0
            indexing (Literal[&#39;xy&#39;,&#39;ij&#39;], optional): The indexing type. Defaults to 'xy'.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The X, Y, Z (N,M) meshgrid of coordinates.
        """
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
        xs = self.uax.np[0] * uax + self.vax.np[0] * vax + origin[0]
        ys = self.uax.np[1] * uax + self.vax.np[1] * vax + origin[1]
        zs = self.uax.np[2] * uax + self.vax.np[2] * vax + origin[2]

        return xs.reshape(shp), ys.reshape(shp), zs.reshape(shp)

    def span(
        self, u: float, v: float, N: int, origin: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a grid of XYZ coordinates in the plane reaching from 0 to u and 0 to v in N steps at the given origin

        Args:
            u (float): The distance along the first axis
            v (float): The distance along the second axis
            N (int): The number of sample points per axis
            origin (np.ndarray): The origin of the planar grid.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The set of X,Y,Z coordinates.
        """
        uax = np.linspace(0, u, N)
        vax = np.linspace(0, v, N)
        U, V = np.meshgrid(uax, vax, indexing="ij")
        uax = U.flatten()
        vax = V.flatten()
        shp = U.shape
        xs = self.uax.np[0] * uax + self.vax.np[0] * vax + origin[0]
        ys = self.uax.np[1] * uax + self.vax.np[1] * vax + origin[1]
        zs = self.uax.np[2] * uax + self.vax.np[2] * vax + origin[2]

        return xs.reshape(shp), ys.reshape(shp), zs.reshape(shp)


############################################################
#                         CONSTANTS                        #
############################################################

XYPLANE = Plane(XAX, YAX)
XZPLANE = Plane(XAX, ZAX)
YZPLANE = Plane(YAX, ZAX)
YXPLANE = Plane(YAX, XAX)
ZXPLANE = Plane(ZAX, XAX)
ZYPLANE = Plane(ZAX, YAX)

############################################################
#                  COORDINATE SYSTEM CLASS                 #
############################################################


@dataclass
class CoordinateSystem(Saveable):
    """A class representing CoordinateSystem information.

    This class is widely used throughout the FEM solver to embed objects in space properly.
    The x,y and z unit vectors are defined by Axis objects. The origin by a np.ndarray.

    The property _is_global is should only be set for any CoordinateSystem class that is wished to
    be considered as global. This is reserved for the GCS intance create automatically with:
        xax = (1,0,0)
        yax = (0,1,0)
        zax = (0,0,1)
        origin = (0., 0., 0.) meters

    Args:
        xax (Axis): The x-axis
        yax (Axis): The y-axis
        zax (Axis): The z-axis
        origin (np.ndarray, optional): The origin. Defaults to (0, 0, 0).
    """

    xax: Axis
    yax: Axis
    zax: Axis
    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    _basis: np.ndarray = field(default=None)
    _basis_inv: np.ndarray = field(default=None)
    _is_global: bool = False

    def __post_init__(self):
        self.xax = _parse_axis(self.xax)
        self.yax = _parse_axis(self.yax)
        self.zax = _parse_axis(self.zax)
        self._basis = np.array([self.xax.np, self.yax.np, self.zax.np]).T
        self._basis_inv = np.linalg.pinv(self._basis)

    def __repr__(self) -> str:
        return f"CS({self.xax}, {self.yax}, {self.zax}, {self.origin})"

    def copy(self) -> CoordinateSystem:
        """Creates a copy of this coordinate system.

        Returns:
            CoordinateSystem: A copy of this coordinate system
        """
        return CoordinateSystem(self.xax, self.yax, self.zax, self.origin)

    def flip_z(self) -> None:
        """Inverts the coordinate system along the Z-axis"""
        self.yax = -self.yax
        self.zax = -self.zax
        self.__post_init__()

    def displace(self, dx: float, dy: float, dz: float) -> CoordinateSystem:
        """Creates a displaced version of this coordinate system. The basis is kept the same.

        Args:
            dx (float): The X-displacement (meters)
            dy (float): The Y-displacement (meters)
            dz (float): The Z-displacement (meters)

        Returns:
            CoordinateSystem: The new CoordinateSystem object.
        """
        csnew = CoordinateSystem(
            self.xax, self.yax, self.zax, self.origin + np.array([dx, dy, dz])
        )
        return csnew

    def rotate(
        self,
        axis: tuple | list | np.ndarray | Axis,
        angle: float,
        degrees: bool = True,
        origin: bool | np.ndarray = False,
    ) -> CoordinateSystem:
        """Return a new CoordinateSystem rotated about the given axis (through the global origin)
        by `angle`. If `degrees` is True, `angle` is interpreted in degrees.

        Args:
            axis (tuple | list | np.ndarray | Axis): The rotation axis
            angle (float): The rotation angle (in degrees if degrees = True)
            degrees (bool, optional): Whether to use degrees. Defaults to True.
            origin (bool | np.ndarray, optional): Whether to rotate the origin as well. Defaults to False.

        Returns:
            CoordinateSystem: The new rotated coordinate system
        """

        # Convert to radians if needed
        if degrees:
            theta = angle * np.pi / 180

        # Normalize the rotation axis
        u = _parse_vector(axis)
        u = u / np.linalg.norm(u)

        # Build the skew-symmetric cross-product matrix K for u
        K = np.array(
            [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=np.float64
        )

        # Rodrigues' rotation formula: R = I + sinθ·K + (1−cosθ)·K²
        Imat = np.eye(3, dtype=np.float64)
        R = Imat + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # Rotate each axis and the origin
        new_x = R @ self.xax.vector
        new_y = R @ self.yax.vector
        new_z = R @ self.zax.vector

        if origin is not False:
            if isinstance(origin, bool):
                new_o = R @ self.origin
            else:
                new_o = (R @ (self.origin - np.array(origin))) + np.array(origin)
        else:
            new_o = self.origin.copy()

        return CoordinateSystem(
            xax=new_x, yax=new_y, zax=new_z, origin=new_o, _is_global=False
        )

    def swapxy(self) -> None:
        """Swaps the XY axes of the CoordinateSystem."""
        self.xax, self.yax = self.yax, self.xax
        self.__post_init__()

    def affine_from_global(self) -> np.ndarray:
        """Returns an Affine transformation matrix in order to transform coordinates from
        the global coordinate system to this coordinate system.

        Returns:
            np.ndarray: The affine transformation matrix.
        """
        x = self.xax.vector
        y = self.yax.vector
        z = self.zax.vector
        o = self.origin
        T = np.eye(4, dtype=np.float64)
        T[0:3, 0] = x
        T[0:3, 1] = y
        T[0:3, 2] = z
        T[0:3, 3] = o
        return T

    def affine_to_global(self) -> np.ndarray:
        """Returns an Affine transformation matrix in order to transform coordinates from
        this local coordinate system to the coordinate system.

        Returns:
            np.ndarray: The affine transformation matrix.
        """
        T = self.affine_from_global()
        R = T[0:3, 0:3]
        o = T[0:3, 3]
        R_inv = np.linalg.inv(R)
        o_new = -R_inv @ o
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3] = o_new
        return T_inv

    def in_global_cs(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts x,y,z coordinates into the global coordinate system.

        Args:
            x (np.ndarray): The x-coordinates (meter)
            y (np.ndarray): The y-coordinates (meter)
            z (np.ndarray): The z-coordinates (meter)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The resultant x, y and z coordinates.
        """
        xg = (
            self.xax.np[0] * x
            + self.yax.np[0] * y
            + self.zax.np[0] * z
            + self.origin[0]
        )
        yg = (
            self.xax.np[1] * x
            + self.yax.np[1] * y
            + self.zax.np[1] * z
            + self.origin[1]
        )
        zg = (
            self.xax.np[2] * x
            + self.yax.np[2] * y
            + self.zax.np[2] * z
            + self.origin[2]
        )
        return xg, yg, zg

    def in_local_cs(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts x,y,z coordinates into the local coordinate system.

        Args:
            x (np.ndarray): The x-coordinates (meter)
            y (np.ndarray): The y-coordinates (meter)
            z (np.ndarray): The z-coordinates (meter)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The resultant x, y and z coordinates.
        """
        B = self._basis_inv
        xg = x - self.origin[0]
        yg = y - self.origin[1]
        zg = z - self.origin[2]
        x = B[0, 0] * xg + B[0, 1] * yg + B[0, 2] * zg
        y = B[1, 0] * xg + B[1, 1] * yg + B[1, 2] * zg
        z = B[2, 0] * xg + B[2, 1] * yg + B[2, 2] * zg
        return x, y, z

    def in_global_basis(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts x,y,z vector components into the global coordinate basis.

        Args:
            x (np.ndarray): The x-vector components (meter)
            y (np.ndarray): The y-vector components (meter)
            z (np.ndarray): The z-vector components (meter)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The resultant x, y and z vectors.
        """
        xg = self.xax.np[0] * x + self.yax.np[0] * y + self.zax.np[0] * z
        yg = self.xax.np[1] * x + self.yax.np[1] * y + self.zax.np[1] * z
        zg = self.xax.np[2] * x + self.yax.np[2] * y + self.zax.np[2] * z
        return xg, yg, zg

    def in_local_basis(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts x,y,z vector components into the local coordinate basis.

        Args:
            x (np.ndarray): The x-vector components (meter)
            y (np.ndarray): The y-vector components (meter)
            z (np.ndarray): The z-vector components (meter)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The resultant x, y and z vectors.
        """
        B = self._basis_inv
        xg = x
        yg = y
        zg = z
        x = B[0, 0] * xg + B[0, 1] * yg + B[0, 2] * zg
        y = B[1, 0] * xg + B[1, 1] * yg + B[1, 2] * zg
        z = B[2, 0] * xg + B[2, 1] * yg + B[2, 2] * zg
        return x, y, z

    @property
    def gx(self) -> float:
        """The origin x-coordinate in global coordinates."""
        return self.origin[0]

    @property
    def gy(self) -> float:
        """The origin y-coordinate in global coordinates."""
        return self.origin[1]

    @property
    def gz(self) -> float:
        """The origin z-coordinate in global coordinates."""
        return self.origin[2]

    @property
    def gxhat(self) -> np.ndarray:
        """The x-axis unit vector in global coordinates."""
        return self.xax.np

    @property
    def gyhat(self) -> np.ndarray:
        """The y-axis unit vector in global coordinates."""
        return self.yax.np

    @property
    def gzhat(self) -> np.ndarray:
        """The z-axis unit vector in global coordinates."""
        return self.zax.np


############################################################
#                         CONSTANTS                        #
############################################################

CS = CoordinateSystem
GCS = CoordinateSystem(XAX, YAX, ZAX, np.zeros(3), _is_global=True)


############################################################
#                         FUNCTIONS                        #
############################################################


def cs(
    axes: str = "xyz",
    origin: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> CoordinateSystem:
    """Generate a coordinate system based on a simple string
    The string must contain the letters x, X, y, Y, z and/or Z.
    Small letters refer to positive axes and capitals to negative axes.

    The string must be 3 characters long.

    The first position indices which global axis gets assigned to the new local X-axis
    The second position indicates the Y-axis
    The third position indicates the Z-axis

    Thus, rotating the global XYZ coordinate system 90 degrees around the Z axis would yield: yXz

    Args:
        axes (str): The axis description
        origin (tuple[float, float, float], optional): The origin of the coordinate system. Defaults to (0.,0.,0.,).

    Returns:
        CoordinateSystem: The resultant coordinate system
    """
    if len(axes) != 3:
        raise ValueError("Axis object must be of length 3")

    axlib = {
        "x": Axis(np.array([1.0, 0.0, 0.0])),
        "X": Axis(np.array([-1.0, 0.0, 0.0])),
        "y": Axis(np.array([0.0, 1.0, 0.0])),
        "Y": Axis(np.array([0.0, -1.0, 0.0])),
        "z": Axis(np.array([0.0, 0.0, 1.0])),
        "Z": Axis(np.array([0.0, 0.0, -1.0])),
    }
    ax_obj = [axlib[ax] for ax in axes]

    return (ax_obj[0] * ax_obj[1] * ax_obj[2]).displace(*origin)


@dataclass
class Anchor(Saveable):
    """An Anchor is a generalization of a coordinate plus a local 3D axis system

    Anchors behave like coordinates when passed as arguments to functions
    that require coordinates. Additionally, they can be used in the .stick()
    method to move anchors of objects aligned with other ancors.

    Args:
        c0 (np.ndarray): The anchor coordinate
    """

    c0: np.ndarray
    _x: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    _y: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    _z: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self):
        self.c0 = np.array(self.c0)
        self._x = np.array(self._x)
        self._y = np.array(self._y)
        self._z = np.array(self._z)

    def __getattr__(self, name: str) -> Anchor:
        """An anchor with the coordinate systems xyz axis oriented

        in the provided order.
        Lower case means negative and upper-case positive.
        Example:
         >>> anchor.yZx is the axis system (-y, +z, -x)"""
        if len(name) != 3:
            raise AttributeError(f"There is no attribute named: {name}")
        axes = {
            "x": -self._x,
            "X": self._x,
            "y": -self._y,
            "Y": self._y,
            "z": -self._z,
            "Z": self._z,
        }
        return Anchor(self.c0, *[axes[c] for c in name])

    def __add__(self, other: Anchor) -> Anchor:
        """Adds two anchors by adding their origins and keeping the axis of self.

        Args:
            other (Anchor): The other anchor to add
        """
        return Anchor(self.c0 + other.c0, self._x, self._y, self._z)

    def __str__(self) -> str:
        return f"Anchor(c0={self.c0}, x={self._x}, y={self._y}, z={self._z})"

    @property
    def tx(self) -> Anchor:
        """Same anchor rotated around its positive X-axis 180 degrees"""
        return Anchor(self.c0, self._x, -self._y, -self._z)

    @property
    def ty(self) -> Anchor:
        """Same anchor rotated around its positive Y-axis 180 degrees"""
        return Anchor(self.c0, -self._x, self._y, -self._z)

    @property
    def tz(self) -> Anchor:
        """Same anchor rotated around its positive Z-axis 180 degrees"""
        return Anchor(self.c0, -self._x, -self._y, self._z)

    @property
    def mx(self) -> Anchor:
        """Same achor mirrored in its X-axis"""
        return Anchor(self.c0, -self._x, self._y, self._z)

    @property
    def my(self) -> Anchor:
        """Same anchor mirrored in its Y-axis"""
        return Anchor(self.c0, self._x, -self._y, self._z)

    @property
    def mz(self) -> Anchor:
        """Same anchor mirrored in its Z-axis"""
        return Anchor(self.c0, self._x, self._y, -self._z)

    def __iter__(self):
        yield self.c0[0]
        yield self.c0[1]
        yield self.c0[2]

    def as_homogeneous(self) -> np.ndarray:
        """The anchor as a 4x4 homogeneous transformation matrix.

        Returns:
            np.ndarray: The matrix with the anchor axes as rotation and its position as translation
        """
        R = np.column_stack((self._x, self._y, self._z))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.c0
        return T

    def cs(self) -> CoordinateSystem:
        """Creates a CoordinateSystem object at this anchor.

        Returns:
            CoordinateSystem: The resultant CoordinateSystem object.
        """
        return CoordinateSystem(self._x, self._y, self._z, self.c0)

    def compute_affine(self, other: Anchor) -> np.ndarray:
        """Compute affine transform mapping this anchor to `other`.

        Args:
            other (Anchor): The target anchor

        Returns:
            np.ndarray: The 4x4 affine transformation matrix that maps this anchor onto `other`
        """
        T_self = self.as_homogeneous()
        T_other = other.as_homogeneous()
        return T_other @ np.linalg.inv(T_self)

    def rotate(self, c0, ax, angle):
        """Rotate self.point and self.axis about axis `ax`, centered at `c0`, by `angle` radians.

        Args:
            c0 (np.ndarray): The center of rotation, shape (3,).
            ax (np.ndarray): The axis to rotate around, shape (3,). Need not be unit length.
            angle (float): Rotation angle in radians.
        """

        angle = -angle
        # Ensure axis is a unit vector
        k = ax / np.linalg.norm(ax)

        # Precompute trig values
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        def rodrigues(v: np.ndarray) -> np.ndarray:
            """
            Rotate vector v around axis k by angle using Rodrigues' formula.
            """
            # term1 = v * cosθ
            term1 = v * cos_theta
            # term2 = (k × v) * sinθ
            term2 = np.cross(k, v) * sin_theta
            # term3 = k * (k ⋅ v) * (1 - cosθ)
            term3 = k * (np.dot(k, v)) * (1 - cos_theta)
            return term1 + term2 + term3

        # Rotate the origin point about c0:
        rel_o = self.c0 - c0  # move to rotation-centre coordinates
        rot_o = rodrigues(rel_o)  # rotate
        self.c0 = rot_o + c0  # move back

        # Rotate the normal vector (pure direction, no translation)
        self._x = rodrigues(self._x)
        self._y = rodrigues(self._y)
        self._z = rodrigues(self._z)

    def translate(self, dx: float, dy: float, dz: float):
        """_summary_

        Args:
            dx (float): The translation x - displacement
            dy (float): The translation y - displacement
            dz (float): The translation z - displacement
        """
        self.c0 = self.c0 + np.array([dx, dy, dz])

    def mirror(self, c0: np.ndarray, pln: np.ndarray) -> None:
        """Reflect self.o and self.n across the plane passing through c0 with normal pln.

        Args:
            c0 (np.ndarray): A point on the mirror plane, shape (3,).
            pln (np.ndarray): The normal of the mirror plane, shape (3,). Need not be unit length.
        """
        # Normalize the plane normal
        k = pln / np.linalg.norm(pln)

        # Reflect the origin point:
        # compute vector from plane point to self.o
        v_o = self.c0 - c0
        # signed distance along normal
        dist_o = np.dot(v_o, k)
        # reflection
        self.c0 = self.c0 - 2 * dist_o * k

        # Reflect the normal/direction vector:
        dist_x = np.dot(self._x, k)
        dist_y = np.dot(self._y, k)
        dist_z = np.dot(self._z, k)
        self._x = self._x - 2 * dist_x * k
        self._y = self._y - 2 * dist_y * k
        self._z = self._z - 2 * dist_z * k

    def affine_transform(self, M: np.ndarray):
        """Apply a 4×4 affine transformation matrix to both self.o and self.n.

        Args:
            M (np.ndarray): The 4×4 affine transformation matrix. Points are transformed with homogeneous w=1,
                directions with w=0.
        """
        # Validate shape
        if M.shape != (4, 4):
            raise ValueError(f"Expected M to be 4×4, got shape {M.shape}")

        # Transform origin point (homogeneous w=1)
        homo_o = np.empty(4)
        homo_o[:3] = self.c0
        homo_o[3] = 1.0
        transformed_o = M @ homo_o

        self.c0 = transformed_o[:3]

        # Transform normal/direction vector (homogeneous w=0)
        homo_x = np.empty(4)
        homo_x[:3] = self._x
        homo_x[3] = 0.0

        homo_y = np.empty(4)
        homo_y[:3] = self._y
        homo_y[3] = 0.0

        homo_z = np.empty(4)
        homo_z[:3] = self._z
        homo_z[3] = 0.0

        self._x = (M @ homo_x)[:3]
        self._y = (M @ homo_y)[:3]
        self._z = (M @ homo_z)[:3]
