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
from ..heatconduction_bc import (
    BoundaryCondition,
    ThermalContact,
    Convection,
    FixedTemperatureBoundary,
    FixedTemperatureVolume,
    HeatFluxBoundary,
    HeatFluxVolume,
    ThinConductor,
    BlackBodyRadiation,
    InitialTemperatureVolume,
    InitialTemperatureBoundary,
)
from ...material_assignment import MaterialAssignment
from ....elements.leg2 import Legrange2
from emsutil import Material
from ....settings import Settings
from scipy.sparse import csc_matrix
from loguru import logger
from ..simjob import SimJob
from typing import TypeVar


############################################################
#                    THE ASSEMBLER CLASS                   #
############################################################

TBC = TypeVar("T")


def filter_bc(listin: BoundaryCondition, bctype: type[TBC]) -> list[TBC]:
    return [x for x in listin if isinstance(x, bctype)]


class Assembler:
    """The assembler class is responsible for FEM EM problem assembly.

    It stores some cached properties to accellerate preformance.
    """

    def __init__(self, settings: Settings):

        self.cached_matrices = None
        self.settings: Settings = settings
        self.SELECT_INDEX: int = None
        self._partitioned: bool = False

    def assemble_hc_matrix(
        self,
        field: Legrange2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        T_initial_K: float | np.ndarray,
        transient: bool = False,
    ) -> tuple[SimJob, tuple[np.ndarray,]]:
        """Main ssembly routine of the heat conduction matrices


        It assembles both the stiffness matrix for stationary problems
        and the mass matrix for transient (optional if transiet = True)
        Args:
            field (Legrange2): The field object
            materials (list[Material]): List of all the assigned materials
            bcs (list[BoundaryCondition]): List of boundary conditions
            T_initial_K (float, np.ndarray): The initial temperature as scalar or array (previous solution)
            transient (bool): If a transient analysis is performed.

        """
        from .grad import tet_stiffness_matrix, tet_mass_matrix
        from .heatflux import assemble_surface_flux, assemble_volume_source
        from .convection import assemble_robin_bc, assemble_radiation_bc
        from .thinconductor import assemble_conductive_sheet
        from .thermalcontact import assemble_thermal_contact

        density = np.zeros((3, 3, field.mesh.n_tets), dtype=np.float64)
        cond_thermal = np.zeros((3, 3, field.mesh.n_tets), dtype=np.float64)
        specific_heat = np.zeros((3, 3, field.mesh.n_tets), dtype=np.float64)

        for mat, coords, ids in mat_assy.iter_materials():
            density = mat.density(1.0, density, coords, ids)
            cond_thermal = mat.cond_thermal(1.0, cond_thermal, coords, ids)
            specific_heat = mat.specific_heat(1.0, specific_heat, coords, ids)

        density = density[0, 0, :].squeeze()
        specific_heat = specific_heat[0, 0, :].squeeze()

        # --- DOF partitioning for ThermalContact

        thermal_contact = filter_bc(bcs, ThermalContact)

        if len(thermal_contact) > 0 and not self._partitioned:
            logger.info("Partitioning degrees of freedom on ThermalContact boundaries.")
            field.partition_dof([x.tags for x in thermal_contact])
            self._partitioned = True

        NF = field.n_field

        ############################################################
        #                     MATRIX ASSEMBLY                      #
        ############################################################

        # --- Stiffness Matrix (kept as flat COO parts; BC contributions
        # below are appended to the same lists and converted to CSC once,
        # at the end, instead of building/adding a CSC term per BC).
        Amat_data, Amat_rows, Amat_cols = tet_stiffness_matrix(field, cond_thermal)
        data_parts = [Amat_data]
        rows_parts = [Amat_rows]
        cols_parts = [Amat_cols]

        # --- Mass Matrix
        Bmat = None
        if transient:
            Bmat_data, Bmat_rows, Bmat_cols = tet_mass_matrix(field, density * specific_heat)
            Bmat = csc_matrix((Bmat_data, (Bmat_rows, Bmat_cols)), shape=(NF, NF))

        # --- Forcing vector
        bvec = np.zeros((NF,), dtype=np.float64)

        # --- Initial Temperature
        if isinstance(T_initial_K, (float, int)):
            Tvec = np.ones((NF,), dtype=np.float64) * T_initial_K
        else:
            Tvec = T_initial_K

        # --- Boundary condition pre-filtering
        fixed_temp_bound = filter_bc(bcs, FixedTemperatureBoundary)
        fixed_temp_volume = filter_bc(bcs, FixedTemperatureVolume)
        heat_flux_bound = filter_bc(bcs, HeatFluxBoundary)
        heat_flux_volume = filter_bc(bcs, HeatFluxVolume)
        convection = filter_bc(bcs, Convection)
        thinconductor = filter_bc(bcs, ThinConductor)
        blackbody = filter_bc(bcs, BlackBodyRadiation)
        init_volume = filter_bc(bcs, InitialTemperatureVolume)
        init_boundary = filter_bc(bcs, InitialTemperatureBoundary)
        prescribed = []

        # --- Extract the mesh
        mesh = field.mesh

        # --- Boundary condition parsing

        # --- Initial Temperature
        for bc in init_volume:
            tri_ids = mesh.get_tetrahedra(bc.tags)
            edge_ids = list(mesh.tet_to_edge[:, tri_ids].flatten())
            vertex_ids = list(mesh.tets[:, tri_ids].flatten())
            ids = [(field.edge_to_field[ii], bc.T) for ii in edge_ids] + [
                (field.node_to_field[ii], bc.T) for ii in vertex_ids
            ]
            Tvec[list(set(ids))] = bc.T

        for bc in init_boundary:
            tri_ids = mesh.get_triangles(bc.tags)
            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())
            vertex_ids = list(mesh.tris[:, tri_ids].flatten())
            ids = [(field.edge_to_field[ii], bc.T) for ii in edge_ids] + [
                (field.node_to_field[ii], bc.T) for ii in vertex_ids
            ]
            Tvec[list(set(ids))] = bc.T

        # --- Fixted Temperature
        for bc in fixed_temp_bound:
            logger.debug(f"Implementing: {bc}")
            face_tags = bc.tags
            tri_ids = mesh.get_triangles(face_tags)

            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())
            vertex_ids = list(mesh.tris[:, tri_ids].flatten())

            prescribed.extend([(field.edge_to_field[ii], bc.T) for ii in edge_ids])
            prescribed.extend([(field.node_to_field[ii], bc.T) for ii in vertex_ids])

        for bc in fixed_temp_volume:
            logger.debug(f"Implementing: {bc}")
            face_tags = bc.tags
            tet_ids = mesh.get_tetrahedra(face_tags)

            edge_ids = list(mesh.tet_to_edge[:, tet_ids].flatten())
            vertex_ids = list(mesh.tets[:, tet_ids].flatten())

            prescribed.extend([(field.edge_to_field[ii], bc.T) for ii in edge_ids])
            prescribed.extend([(field.node_to_field[ii], bc.T) for ii in vertex_ids])

        # --- Surface flux
        for bc in heat_flux_bound:
            logger.debug(f"Implementing: {bc}")
            face_tags = bc.tags
            tri_ids = mesh.get_triangles(face_tags)

            out = assemble_surface_flux(field, face_tags, bc.qm)
            bvec += out

        # --- Volume Flux
        for bc in heat_flux_volume:
            logger.debug(f"Implementing: {bc}")
            tet_ids = mesh.get_tetrahedra(bc.tags)
            out = assemble_volume_source(field, tet_ids, bc.fqm)
            bvec += out

        # ---convection
        for bc in convection:
            logger.debug(f"Implementing: {bc}")

            Kval, Krows, Kcols, out = assemble_robin_bc(field, bc.tags, bc.h, bc.Tamb)
            data_parts.append(Kval)
            rows_parts.append(Krows)
            cols_parts.append(Kcols)
            bvec += out

        # --- Black body radiation
        for bc in blackbody:
            logger.debug(f"Implementing: {bc}")
            Kval, Krows, Kcols, out = assemble_radiation_bc(
                field, bc.tags, bc.emissivity, bc.Tamb, Tvec
            )
            data_parts.append(Kval)
            rows_parts.append(Krows)
            cols_parts.append(Kcols)
            bvec += out

        # --- Thermal Contact
        for bc in thermal_contact:
            logger.debug(f"Implementing: {bc}")
            Kval, Krows, Kcols = assemble_thermal_contact(field, bc.tags, bc.h)
            data_parts.append(Kval)
            rows_parts.append(Krows)
            cols_parts.append(Kcols)

        # --- Thin Conductor
        for bc in thinconductor:
            logger.debug(f"Implementing: {bc}")

            kappa_t = bc.get_kappa()

            Kval, Krows, Kcols = assemble_conductive_sheet(field, bc.tags, kappa_t)
            data_parts.append(Kval)
            rows_parts.append(Krows)
            cols_parts.append(Kcols)

        # --- Finalize A matrix: one COO->CSC conversion covering the base
        # stiffness matrix plus every boundary condition's contribution
        # (scipy sums duplicate (row, col) entries automatically).
        Amat = csc_matrix(
            (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
            shape=(NF, NF),
        )

        # --- Finalize
        if len(prescribed) > 0:
            prescribed = sorted(prescribed, key=lambda x: x[0])
            arr = np.array(prescribed, dtype=np.float64)  # (N, 2)
            ids_raw = arr[:, 0].astype(np.int64)
            ts_raw = arr[:, 1]

            # Keep first occurrence of each DOF
            _, first_idx = np.unique(ids_raw, return_index=True)
            first_idx.sort()  # preserve original order, then sorted by unique

            ids = ids_raw[first_idx]
            ts = ts_raw[first_idx]

            # Sort by DOF index
            order = np.argsort(ids)
            ids = ids[order].tolist()
            ts = ts[order].tolist()

        else:
            ids = []
            ts = []

        simjob = SimJob(Amat, Bmat, bvec, NF, ids, ts)

        return simjob, [cond_thermal], Tvec
