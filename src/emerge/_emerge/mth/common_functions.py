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

# Last Cleanup: 2025-01-01
import numpy as np
from ..cs import _parse_axis, Axis, _parse_vector
from emsutil import Material, FreqDependent
from emsutil.lib import EPS0

def norm(Field: np.ndarray) -> np.ndarray:
    """ Computes the complex norm of a field (3,N)

    Args:
        Field (np.ndarray): The input field, shape (3,N)

    Returns:
        np.ndarray: The complex norm in shape (N,)
    """
    return np.sqrt(np.abs(Field[0,:])**2 + np.abs(Field[1,:])**2 + np.abs(Field[2,:])**2)

def coax_rout(rin: float,
              eps_r: float = 1,
              Z0: float = 50) -> float:
    """Computes the outer radius given a dielectric constant, inner radius and characteristic impedance

    Args:
        rin (float): The inner radius
        eps_r (float, optional): The dielectric permittivity. Defaults to 1.
        Z0 (float, optional): The impedance. Defaults to 50.

    Returns:
        float: The outer radius
    """
    return rin*10**(Z0*np.sqrt(eps_r)/138)

def coax_rin(rout: float,
              eps_r: float = 1,
              Z0: float = 50) -> float:
    """Computes the inner radius given a dielectric constant, outer radius and characteristic impedance

    Args:
        rout (float): The outer radius
        eps_r (float, optional): The dielectric permittivity. Defaults to 1.
        Z0 (float, optional): The impedance. Defaults to 50.

    Returns:
        float: The inner radius
    """
    return rout/10**(Z0*np.sqrt(eps_r)/138)

def _move_3_to_last(x: np.ndarray) -> np.ndarray:
    """
    Ensure the axis of length 3 is the last axis.
    Accepts shapes (3,), (3, N), (N, 3).
    """
    x = np.asarray(x)

    if x.ndim == 1:
        if x.shape[0] != 3:
            raise ValueError(f"1D input must have shape (3,), got {x.shape}")
        return x  # shape (3,)

    if x.ndim != 2:
        raise ValueError(f"Input must be 1D or 2D, got {x.ndim}D with shape {x.shape}")

    axes_3 = [i for i, s in enumerate(x.shape) if s == 3]
    if len(axes_3) != 1:
        raise ValueError(f"Input must have exactly one axis of length 3, got shape {x.shape}")

    axis_3 = axes_3[0]
    if axis_3 == x.ndim - 1:
        return x  # already (..., 3)
    else:
        # swap the 3-axis to the last position: (3, N) -> (N, 3)
        return np.swapaxes(x, axis_3, x.ndim - 1)


def dot(A, B):
    """
    Dot product along the axis of length 3.

    A, B can be:
      - (3,)        : single 3-vector
      - (3, N)      : 3-by-N, treated as N vectors of length 3
      - (N, 3)      : N-by-3, treated as N vectors of length 3

    Args:
        A (np.ndarray): The first vector(s), shape (3,), (3, N) or (N, 3)
        B (np.ndarray): The second vector(s), same shape as A

    Returns:
      - scalar for (3,)·(3,)
      - (N,) for pairwise dot of N vectors.
    """
    A = _move_3_to_last(np.asarray(A))
    B = _move_3_to_last(np.asarray(B))

    # Broadcast over non-3 axes (e.g. (N,3) with (3,) -> (N,3))
    A, B = np.broadcast_arrays(A, B)

    return np.sum(A * B, axis=-1)


def cross(A, B):
    """
    Cross product along the axis of length 3.

    A, B can be:
      - (3,)
      - (3, N)
      - (N, 3)

    Args:
        A (np.ndarray): The first vector(s), shape (3,), (3, N) or (N, 3)
        B (np.ndarray): The second vector(s), same shape as A

    Returns:
      - (3,) for single cross product
      - (3, N) for pairwise cross of N vectors.
    """
    A = _move_3_to_last(np.asarray(A))
    B = _move_3_to_last(np.asarray(B))

    # Broadcast over non-3 axes
    A, B = np.broadcast_arrays(A, B)

    a1, a2, a3 = A[..., 0], A[..., 1], A[..., 2]
    b1, b2, b3 = B[..., 0], B[..., 1], B[..., 2]

    c1 = a2 * b3 - a3 * b2
    c2 = a3 * b1 - a1 * b3
    c3 = a1 * b2 - a2 * b1

    return np.array((c1, c2, c3))

def lumped_element_material(material_name: str,
                  direction: tuple[float, float, float] | Axis,
                  length: float, 
                  Area: float, 
                  C: float | None = None, 
                  L: float | None = None, 
                  R: float | None = None,
                  color: str = "#999999",
                  opacity: float = 0.2) -> Material:
    """Generates a material that effectively turns some volume into a lumped element.
    The volume must be approximately that of an extruded surface like a box or cylinder.
    This extruded volume must be extruded in the current direction and the start and end surfaces
    must be a conductor. Then the volume in between can be assigned this material which will turn the
    volume into an effective lumped element with the provided capacitance, inductance, resistance or mix of all.

    The materials must be assigned a unique name.

    Args:
        material_name (str): The material name.
        direction (tuple[float, float, float] | Axis): The direction in which the lumped element is oriented. This is the current direction.
        length (float): The length of the lumped element component.
        Area (float): The cross sectional area of the lumped element component.
        C (float | None, optional): The capacitance in F. Defaults to None.
        L (float | None, optional): The inductance in H. Defaults to None.
        R (float | None, optional): The resistance in Ω. Defaults to None.
        color (str, optional): The rendering color of the material
        opacity (float, optional): Trendering opacity of the material.

    Returns:
        Material: The effective lumped element material.
    """
    # Ensure direction is a unit vector
    dirvec = _parse_vector(direction)
    norm = np.linalg.norm(dirvec)
    if norm > 0:
        dirvec /= norm
        
    dvout = np.outer(dirvec, dirvec)
    
    # Presence flags
    incl_R = 1.0 if R is not None else 0.0
    incl_L = 1.0 if L is not None else 0.0
    
    # Set default dummy values for inactive elements to avoid division by zero
    C_val = C if C is not None else 0.0
    R_val = R if R is not None else 1.0
    L_val = L if L is not None else 1.0
    


    def _fer(f):
        w = 2 * np.pi * f
        
        # Real terms: Capacitance (positive) and Inductance (negative, dispersive)
        real_part = (C_val * length) / (EPS0 * Area) - (incl_L * length) / (w**2 * L_val * EPS0 * Area)
        
        # Imaginary term: Resistance (conductive loss)
        imag_part = - (incl_R * length) / (w * R_val * EPS0 * Area)
        
        # Scalar complex relative permittivity along the element axis
        er_scalar = real_part + 1j * imag_part
        
        # Dyadic projection tensor: er_scalar along u, 1.0 (free space) orthogonal to u
        return dvout * (er_scalar - 1.0) + np.eye(3)
    
    erf = FreqDependent(matrix=_fer)
    return Material(er=erf, color=color, opacity=opacity, name=material_name)