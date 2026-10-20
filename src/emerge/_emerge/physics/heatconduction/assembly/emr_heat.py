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

# Last Cleanup: 2026-10-17
"""Python interface of the EMR (Rust) heat conduction assembly kernels."""
from typing import Callable
import numpy as np
from emr import heat_grad, heat_heatflux, heat_convection, heat_thermalcontact, heat_thinconductor
from ....elements.leg2 import Legrange2
from ....mth.csc_cast import CSCMapping
from ....mth.optimized import gaus_quad_tet

# Volume quadrature points of the heat source (must match keast(4) in emr/src/heat/heatflux.rs)
DPTS = gaus_quad_tet(4)


############################################################
#            CONDUCTION STIFFNESS AND CAPACITY             #
############################################################


def tet_stiffness_matrix(
    field: Legrange2, cond_termal: np.ndarray
) -> tuple[np.ndarray, CSCMapping]:
    """Main assembly function of the mass matrix for stationary heat transfer

    Args:
        field (Legrange2): The Legrange2 Field object
        cond_termal (np.ndarray): Array with 3x3 conductivity tensors for each tet

    Returns:
        tuple[np.ndarray, CSCMapping]: The A-matrix entry and CSCMapping object
    """
    tets = field.mesh.tets
    nodes = field.mesh.nodes

    tet_to_field = field.tet_to_field

    Amatrix, rows, cols = heat_grad._stiff_matrix_builder(nodes, tets, tet_to_field, cond_termal)

    cscmap = CSCMapping.from_rowcol(rows, cols, field.n_field)
    return Amatrix, cscmap


def tet_mass_matrix(
    field: Legrange2, rho_cp: np.ndarray
) -> tuple[np.ndarray, CSCMapping]:
    """Assemble the global mass (capacity) matrix for transient heat transfer.

    M_ij = integral rho * cp * N_i * N_j dV

    Args:
        field: Legrange2 field
        rho_cp: (n_tets,) array of density * specific heat per element [J/(m³·K)]

    Returns:
        M_coo: COO values
        cscmap: CSCMapping for conversion to CSC
    """
    tets = field.mesh.tets
    nodes = field.mesh.nodes
    tet_to_field = field.tet_to_field

    Mmatrix, rows, cols = heat_grad._mass_matrix_builder(nodes, tets, tet_to_field, rho_cp)

    cscmap = CSCMapping.from_rowcol(rows, cols, field.n_field)
    return Mmatrix, cscmap

############################################################
#               HEAT FLUX AND VOLUME SOURCES               #
############################################################


def assemble_surface_flux(
    field: Legrange2, face_tags: list[int], q_flux: float
) -> np.ndarray:
    """Assemble surface Neumann heat flux into global load vector.

    Args:
        field (Legrange2): Field object
        face_tags (list[int]): list/array of GMSH face tags for this BC
        q_flux (float): float, constant heat flux [W/m^2], positive = into domain

    Returns:
        f_global: (n_field,) load vector contribution
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return heat_heatflux._surface_flux_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        float(q_flux),
        field.n_field,
    )


def assemble_volume_source(
    field: Legrange2, tet_ids: list[int], q_func: Callable
) -> np.ndarray:
    """Assemble volumetric heat source into global load vector.

    Args:
        field: Legrange2 field
        tet_ids: array of tet indices where source is active
        q_func: callable(xs, ys, zs) -> array of q_V values [W/m³]

    Returns:
        f_global: (n_field,) load vector contribution
    """
    mesh = field.mesh
    tet_ids = np.asarray(tet_ids, dtype=np.int64)
    n_sel = len(tet_ids)
    nq = DPTS.shape[1]

    L1 = DPTS[1, :]
    L2 = DPTS[2, :]
    L3 = DPTS[3, :]
    L4 = DPTS[4, :]

    verts = mesh.tets[:, tet_ids]  # (4, n_sel)

    x0, x1, x2, x3 = mesh.nodes[0, verts]
    y0, y1, y2, y3 = mesh.nodes[1, verts]
    z0, z1, z2, z3 = mesh.nodes[2, verts]

    # Construct gauss quadrature integration points
    all_xs = x0[:, None] * L1 + x1[:, None] * L2 + x2[:, None] * L3 + x3[:, None] * L4
    all_ys = y0[:, None] * L1 + y1[:, None] * L2 + y2[:, None] * L3 + y3[:, None] * L4
    all_zs = z0[:, None] * L1 + z1[:, None] * L2 + z2[:, None] * L3 + z3[:, None] * L4

    # Call the volumetric heat flux function q_func
    q_at_quad = q_func(all_xs.ravel(), all_ys.ravel(), all_zs.ravel()).reshape(
        n_sel, nq
    )

    tet_to_field_sel = field.tet_to_field[:, tet_ids].copy()

    return heat_heatflux._volume_source_builder(
        mesh.nodes,
        tet_to_field_sel,
        q_at_quad,
        field.n_field,
    )

############################################################
#                 CONVECTION AND RADIATION                 #
############################################################


def assemble_radiation_bc(
    field: Legrange2,
    face_tags: list[int],
    emissivity: float,
    T_amb: float,
    T_solution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble blackbody radiation boundary condition.

    Linearized as Robin BC with temperature-dependent coefficient:
        h_rad = eps * sigma * (T^2 + T_amb^2) * (T + T_amb)

    Args:
        field: Legrange2 field
        face_tags: GMSH face tags for radiation boundary
        emissivity: surface emissivity (0 to 1)
        T_amb: ambient radiation temperature [K]
        T_solution: current temperature solution vector (n_field,)
                    used to evaluate h_rad at quadrature points

    Returns:
        K_values, K_rows, K_cols: COO stiffness contribution
        f_robin: (n_field,) load vector contribution
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return heat_convection._radiation_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        T_solution,
        float(emissivity),
        float(T_amb),
        field.n_field,
    )


def assemble_robin_bc(
    field: Legrange2, face_tags: list[int], h_coeff: float, T_amb: float
):
    """Assembles the Robin boundary condition which is the Convection boundary condition to an ambient temperature.

    Args:
        field (Legrange2): The problems Legrange basis function field object
        face_tags (list[int]): A list of face tag integers
        h_coeff (float): The heat flux coefficient in W/m^2
        T_amb (float): The ambient temperature
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    K_values, K_rows, K_cols = heat_convection._robin_stiffness_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        float(h_coeff),
    )

    f_robin = assemble_surface_flux(field, face_tags, float(h_coeff * T_amb))

    return K_values, K_rows, K_cols, f_robin

############################################################
#                     THERMAL CONTACT                      #
############################################################


def assemble_thermal_contact(field: Legrange2, face_tags: np.ndarray, h_c: float):
    """Assemble thermal contact coupling between two volume regions.

    Args:
        field: Legrange2 field (with _dof_mapping populated)
        face_tags: GMSH face tags of the contact interface
        h_c: thermal contact conductance [W/(m^2K)]

    Returns:
        K_values: COO values for stiffness matrix addition
        K_rows: COO row indices
        K_cols: COO column indices
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)
    n_triangles = len(tri_ids)

    # Build the DOF mapping arrays from dict
    dof_map = field._dof_mapping

    # Build tri_to_field for A side (original) and B side (remapped)
    tri_to_field_a = np.empty((6, n_triangles), dtype=np.int64)
    tri_to_field_b = np.empty((6, n_triangles), dtype=np.int64)

    for i in range(n_triangles):
        itri = tri_ids[i]

        fids_a = field.tri_to_field[:, itri].copy()
        tri_to_field_a[:, i] = fids_a

        fids_b = np.empty(6, dtype=np.int64)
        for j in range(6):
            fids_b[j] = dof_map[int(fids_a[j])]
        tri_to_field_b[:, i] = fids_b

    return heat_thermalcontact._thermal_contact_builder(
        mesh.nodes,
        mesh.tris,
        tri_to_field_a,
        tri_to_field_b,
        tri_ids,
        float(h_c),
    )

############################################################
#                  THIN CONDUCTIVE SHEETS                  #
############################################################


def assemble_conductive_sheet(field, face_tags, kappa_t):
    """Assemble thin conductive sheet stiffness matrix contribution.

    Models lateral heat conduction through a thin metallic layer
    (e.g. copper foil, ground plane) without volume meshing.

    Args:
        field: Legrange2 field
        face_tags: GMSH face tags for the sheet surface
        kappa_t: thermal conductivity of the sheet material times thickness [W/(K)]

    Returns:
        K_values: COO values for stiffness matrix addition
        K_rows: COO row indices
        K_cols: COO column indices
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return heat_thinconductor.leg2_tri_stiff(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        kappa_t,
    )
