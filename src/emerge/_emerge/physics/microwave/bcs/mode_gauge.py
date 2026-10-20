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

"""Canonical sign/phase ("gauge") of port eigenmodes.

An eigenmode solver returns a mode up to an arbitrary complex factor. After normalizing the phase at the
largest DOF, the sign still depends on the mesh (DOF orientations, which DOF is largest), so the same
physical mode can come out flipped after remeshing, e.g. between adaptive mesh refinement passes.

The gauge fixes this with a mesh-independent rule: the overlap

    L(E) = integral over the port face of E_t . w dA

with a fixed, deliberately asymmetric test field w is made real and positive. w is defined in global
coordinates relative to the port centroid and size, so it does not depend on the mesh or on the port's
derived local coordinate system. Because w has no symmetry, the overlap is non-zero for practically every
mode, including symmetric ones such as the coaxial TEM mode.

Degenerate modes (that can rotate into each other) are not made unique by this; use explicit alignment
vectors for those.

No fixed rule can be perfect: any continuous functional that distinguishes E from -E must vanish for some
mode (Borsuk-Ulam theorem). The test field is tuned so that this does not happen for common port modes,
and canonical_mode_factor reports a quality value so near-zero cases can be detected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....mth.optimized import gaus_quad_tri
from emr.mth import generate_int_data_tri

# Fixed test field: per component i, w_i(u) = sin(a_i . u + phi_i) + sin(b_i . u + psi_i), with
# u = (x - centroid) / size. The wave vectors are "irrational" so w has no symmetry plane; the phases add
# an even part. The a_i (a few radians per port size) cover low order modes, the b_i (about twice that)
# keep the overlap clear for higher order modes such as TE20/TE30.
# Constants tuned by a random search that maximized the worst-case overlap quality over a suite of
# rectangular TEmn (m, n <= 3, aspect ratios 1 and 2), coaxial TEM and circular TM01 modes, with the
# port face in each of the three global planes (worst case 0.018, see the module docstring for why a
# universal non-zero lower bound cannot exist).
_GAUGE_K = np.array([
    [2.6499, 2.7637, 0.6228],
    [1.6060, -2.5534, -0.5864],
    [-2.7537, -2.7456, -0.0749],
])
_GAUGE_PHI = np.array([5.2368, 0.5469, 0.2128])
_GAUGE_K2 = np.array([
    [4.7254, -0.7236, -5.2791],
    [5.5632, 0.6883, 3.5246],
    [5.8182, -5.1402, -4.8331],
])
_GAUGE_PHI2 = np.array([5.8894, 2.0271, 3.7049])

# Overlap quality below which the gauge is considered unreliable (normalized: 0 <= quality <= 1).
GAUGE_QUALITY_WARNING = 1e-2


@dataclass
class PortFaceQuadrature:
    """Quadrature points and weights on a port face (global coordinates)."""
    points: np.ndarray      # (3, N)
    weights: np.ndarray     # (N,), the area integral of f is sum(f * weights)
    center: np.ndarray      # (3,) area weighted centroid
    size: float             # characteristic size, sqrt(area)
    normal: np.ndarray      # (3,) unit normal of the (planar) port face


def port_face_quadrature(nodes: np.ndarray, tris: np.ndarray, normal: np.ndarray, order: int = 4) -> PortFaceQuadrature:
    """Builds the quadrature of a port face from its triangles.

    Args:
        nodes: (3, Nnodes) mesh nodes.
        tris: (3, Ntris) node indices of the port face triangles.
        normal: (3,) normal of the port face (sign irrelevant).
        order: Gauss quadrature order on the triangles.
    """
    x, y, z, w, a, _ = generate_int_data_tri(nodes, tris, gaus_quad_tri(order))
    points = np.vstack([x, y, z])
    weights = w * a
    area = float(np.sum(weights))
    center = points @ weights / area
    n = np.asarray(normal, dtype=float)
    return PortFaceQuadrature(points, weights, center, np.sqrt(area), n / np.linalg.norm(n))


def gauge_test_field(points: np.ndarray, center: np.ndarray, size: float) -> np.ndarray:
    """The fixed gauge test field w at the given global points, shape (3, N)."""
    u = (points - center[:, None]) / size
    return np.sin(_GAUGE_K @ u + _GAUGE_PHI[:, None]) + np.sin(_GAUGE_K2 @ u + _GAUGE_PHI2[:, None])


def canonical_mode_factor(E: np.ndarray, quad: PortFaceQuadrature) -> tuple[complex, float]:
    """Returns the factor that puts a port mode in the canonical gauge, and the gauge quality.

    Multiplying the mode by the factor makes the overlap of its transverse field with the gauge test field
    real and positive. For a lossless mode (real transverse field) the factor is +1 or -1.

    Args:
        E: (3, N) electric field of the mode at quad.points (global components).
        quad: The port face quadrature.

    Returns:
        (factor, quality): factor is a unit complex number. quality = |L| / (||E_t|| ||w||) lies in [0, 1];
        values close to 0 mean the overlap is too small for a reliable decision.
    """
    n = quad.normal
    Et = E - n[:, None] * (n @ E)[None, :]          # transverse field only
    w = gauge_test_field(quad.points, quad.center, quad.size)

    overlap = np.sum(np.sum(Et * w, axis=0) * quad.weights)
    norm_E = np.sqrt(np.sum(np.sum(np.abs(Et) ** 2, axis=0) * quad.weights))
    norm_w = np.sqrt(np.sum(np.sum(w ** 2, axis=0) * quad.weights))
    if norm_E == 0 or abs(overlap) == 0:
        return 1.0 + 0.0j, 0.0
    quality = float(abs(overlap) / (norm_E * norm_w))
    return complex(np.conj(overlap) / abs(overlap)), quality
