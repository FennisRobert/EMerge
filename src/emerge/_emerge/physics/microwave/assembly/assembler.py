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

# Last Cleanup: 2026-08-19 (deduplication pass)
import numpy as np
from ..bcs import (
    PEC,
    BoundaryCondition,
    ScatteredField,
    RobinBC,
    PortBC,
    MWBoundaryConditionSet,
    SurfaceImpedance,
    WavePortIH,
    Void,
)
from ...material_assignment import MaterialAssignment
from ....periodic import Periodic
from ....elements.nedelec2 import Nedelec2
from ....elements.nedleg2 import NedelecLegrange2
from ....elements.dofsets import DoFSet
from ....mesh3d import Mesh3D
from ....settings import Settings
from scipy.sparse import csc_matrix
from .system_pattern import (SystemPattern, product_pattern, copy_onto_pattern,
                             bc_has_matrix_term, bc_has_abc2_term)
from loguru import logger
from ..simjob import SimJob
from ....const import EPS0, C0
import time
import hashlib
from typing import Generator
_PBC_DSMAX = 1e-15


############################################################
#                         FUNCTIONS                        #
############################################################

def _format_freq(freq: float) -> str:
    units = ["Hz", "kHz", "MHz", "GHz", "THz"]

    if freq == 0:
        return "0.00 Hz"

    i = int(np.floor(np.log10(abs(freq)) / 3))
    i = max(0, min(i, len(units) - 1))

    scaled_freq = freq / (1000.0 ** i)
    return f"{scaled_freq:.2f} {units[i]}"

def _pattern_hash(M: csc_matrix) -> str:
    hasher = hashlib.sha1()
    hasher.update(M.indptr.tobytes())
    hasher.update(M.indices.tobytes())
    return hasher.hexdigest()

def select_bc(bcs: list[BoundaryCondition], bctype: type[BoundaryCondition]) -> Generator[BoundaryCondition,None,None]:
    for bc in bcs:
        if isinstance(bc, bctype):
            yield bc

def diagnose_matrix(mat: csc_matrix, basis: "Nedelec2", solve_ids: np.ndarray) -> None:
    """
    Performs high-fidelity diagnostics on the REDUCED FEM system matrix,
    i.e. K[solve_ids,:][:,solve_ids] -- PEC/excluded DoFs already sliced
    out. Crashes with a detailed report if the matrix is numerically or
    structurally unfit.

    IMPORTANT: `mat` must already be the REDUCED matrix; `solve_ids` is
    only used to translate its local indices back to global DoF numbers
    for reporting. Passing the full, unreduced matrix here will raise an
    IndexError, since solve_ids is shorter than the full DoF count.
    """
    print("--- Starting FEM Matrix Diagnostics ---")

    n_dofs = mat.shape[0]
    report = []
    failed = False

    if mat.shape[0] != mat.shape[1]:
        report.append(f"CRITICAL: Non-square matrix detected ({mat.shape})")
        failed = True

    if n_dofs != len(solve_ids):
        report.append(
            f"CRITICAL: DoF mismatch! Matrix size {n_dofs} != len(solve_ids) {len(solve_ids)}"
        )
        failed = True

    col_counts = np.diff(mat.indptr)
    empty_cols_local = np.where(col_counts == 0)[0]

    row_present = np.zeros(n_dofs, dtype=bool)
    row_present[mat.indices] = True
    empty_rows_local = np.where(~row_present)[0]

    empty_cols = solve_ids[empty_cols_local]
    empty_rows = solve_ids[empty_rows_local]

    if len(empty_cols) > 0 or len(empty_rows) > 0:
        failed = True
        report.append(
            f"CRITICAL: Found {len(empty_cols)} empty columns and {len(empty_rows)} empty rows "
            f"among SOLVED (non-PEC-excluded) DoFs."
        )

    diag = mat.diagonal()
    zero_diag_local = np.where(np.isclose(diag, 0, atol=1e-15))[0]
    true_zero_diag_local = np.setdiff1d(zero_diag_local, empty_cols_local)
    if len(true_zero_diag_local) > 0:
        failed = True
        report.append(
            f"CRITICAL: {len(true_zero_diag_local)} non-empty (solved) columns have zero diagonal "
            f"(Numerical Singularity)."
        )

    if (mat - mat.T).nnz > 0:
        max_asym = np.max(np.abs((mat - mat.T).data)) if mat.nnz > 0 else 0
        if max_asym > 1e-12:
            report.append(f"WARNING: Matrix is asymmetric. Max diff: {max_asym}")

    if failed:
        print("\n" + "!" * 50)
        print("MATRIX DIAGNOSTICS FAILED")
        print("!" * 50)
        for line in report:
            print(line)

        print("\nHINT: PEC DoFs are intentionally excluded from `mat` via solve_ids")
        print("and are expected to be absent entirely. A problem DoF below IS in")
        print("solve_ids but has no matrix contribution or a zero diagonal -- THAT's")
        print("the real anomaly to investigate.")
        print("!" * 50)

        # Union every problem category into one set of GLOBAL dof ids, each
        # tagged with why it was flagged, then resolve to actual points.
        problem_dofs: dict[int, list[str]] = {}
        for gid in empty_cols:
            problem_dofs.setdefault(int(gid), []).append("empty column")
        for gid in empty_rows:
            problem_dofs.setdefault(int(gid), []).append("empty row")
        for gid in solve_ids[true_zero_diag_local]:
            problem_dofs.setdefault(int(gid), []).append("zero diagonal")

        points = list_problem_points(problem_dofs, basis)
        print(f"\n--- {len(points)} Problematic Point(s) ---")
        for p in points:
            tags_str = f", groups={p['groups']}" if p['groups'] else ""
            print(
                f"  dof={p['dof']:>8}  type={p['type']:<8}  "
                f"reasons={','.join(p['reasons']):<28}  "
                f"xyz=({p['x']:.6g}, {p['y']:.6g}, {p['z']:.6g}){tags_str}"
            )

        if points:
            coords_arr = np.array([[p['x'], p['y'], p['z']] for p in points]).T
            np.save("dead_coords.npy", coords_arr)
            print(f"\nSaved {len(points)} point coordinates to dead_coords.npy")

            # Copy-paste-ready for model.display.add_scatter(xs, ys, zs)
            xs_lit = ", ".join(f"{p['x']:.6g}" for p in points)
            ys_lit = ", ".join(f"{p['y']:.6g}" for p in points)
            zs_lit = ", ".join(f"{p['z']:.6g}" for p in points)
            print("\n--- Copy-paste for model.display.add_scatter(xs, ys, zs) ---")
            print(f"xs = [{xs_lit}]")
            print(f"ys = [{ys_lit}]")
            print(f"zs = [{zs_lit}]")

        raise MatrixDiagnosisError(
            "FEM Matrix is singular or improperly assembled.", points=points
        )

    print("Diagnostics Passed: Matrix is structurally sound.")

class MatrixDiagnosisError(RuntimeError):
    """Same as RuntimeError, but carries the resolved problem-point list so
    a caller can catch it and use the points directly (e.g. to visualize
    via add_solution_error / a scatter plot) instead of re-parsing stdout.
    """
    def __init__(self, message: str, points: list[dict]):
        super().__init__(message)
        self.points = points


def list_problem_points(problem_dofs: dict[int, list[str]], basis: "Nedelec2") -> list[dict]:
    """Resolves a {global_dof_id: [reasons]} dict into a list of dicts with
    actual coordinates and physical-group membership, for printing or for
    programmatic use (e.g. feeding a visualization).

    Returns a list of:
        {"dof": int, "type": "edge"|"tri", "x": float, "y": float, "z": float,
         "reasons": [str, ...], "groups": [str, ...]}
    """
    nedges = basis.nedges
    ntris = basis.ntris
    mesh = basis.mesh

    points = []
    for dof, reasons in sorted(problem_dofs.items()):
        if dof < nedges:
            entity_type = "edge"
            entity_id = dof
            coord = mesh.edge_centers[:, entity_id]
        elif nedges <= dof < (nedges + ntris):
            entity_type = "tri"
            entity_id = dof - nedges
            coord = mesh.tri_centers[:, entity_id]
        elif (nedges + ntris) <= dof < (2 * nedges + ntris):
            entity_type = "edge"
            entity_id = dof - (nedges + ntris)
            coord = mesh.edge_centers[:, entity_id]
        else:
            entity_type = "tri"
            entity_id = dof - (2 * nedges + ntris)
            coord = mesh.tri_centers[:, entity_id]

        groups = []
        if entity_type == "edge":
            for tag, edges in mesh.etag_to_edge.items():
                if entity_id in edges:
                    groups.append(f"Curve[{tag}]")
        else:
            for tag, tris in mesh.ftag_to_tri.items():
                if entity_id in tris:
                    groups.append(f"Surface[{tag}]")

        points.append({
            "dof": dof,
            "type": entity_type,
            "x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2]),
            "reasons": reasons,
            "groups": groups,
        })

    return points

def plane_basis_from_points(points: np.ndarray) -> np.ndarray:
    """
    Compute an orthonormal basis from a cloud of 3D points dominantly
    lying on one plane.
    """
    if points.shape[0] != 3:
        raise ValueError("Input must have shape (3, N)")

    centroid = points.mean(axis=1, keepdims=True)
    points_centered = points - centroid
    C = (points_centered @ points_centered.T) / points.shape[1]

    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    return eigvecs


############################################################
#                    THE ASSEMBLER CLASS                   #
############################################################

class Assembler:
    """The assembler class is responsible for FEM EM problem assembly.

    It stores some cached properties to accellerate preformance.
    """

    def __init__(self, settings: Settings):

        self._system: SystemPattern | None = None
        self._periodic_patterns: tuple[tuple, csc_matrix, csc_matrix] | None = None
        self.settings: Settings = settings
        self.mldata_filename: str | None = None

    def reset_cache(self) -> None:
        """Discards the cached sparsity pattern, boundary matrices and volume matrices."""
        self._system = None
        self._periodic_patterns = None

    # ------------------------------------------------------------------
    # Shared helpers (used by assemble_freq_matrix / assemble_scattering_matrix
    # / assemble_eig_matrix). assemble_bma_matrices is structurally different
    # (boundary-only, mixed-order field) and does not use these.
    # ------------------------------------------------------------------

    def _assemble_materials(
        self, mat_assy: MaterialAssignment, field: Nedelec2, frequency: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Evaluates er/ur/tand/cond over all tets and folds loss into er.

        Returns (er, ur, cond, is_frequency_dependent).
        """
        W0 = 2 * np.pi * frequency
        n = field.mesh.n_tets
        er = np.zeros((3, 3, n), dtype=np.complex128)
        ur = np.zeros((3, 3, n), dtype=np.complex128)
        tand = np.zeros((3, 3, n), dtype=np.complex128)
        cond = np.zeros((3, 3, n), dtype=np.complex128)

        for mat, centers, ids in mat_assy.iter_materials():
            er = mat.er(frequency, er, centers, ids)
            ur = mat.ur(frequency, ur, centers, ids)
            tand = mat.tand(frequency, tand, centers, ids)
            cond = mat.cond(frequency, cond, centers, ids)

        er = er * (1 - 1j * tand) - 1j * cond / (W0 * EPS0)

        is_freq_dep = mat_assy.frequency_dependent() or np.any(
            (cond > 0) & (cond < self.settings.mw_3d_peclim)
        )
        return er, ur, cond, is_freq_dep

    def _find_conductor_tets(self, field: Nedelec2, bcs: list[BoundaryCondition], cond: np.ndarray) -> np.ndarray:
        """Tets whose conductivity exceeds either the PEC or surf-impedance limit."""
        limit = min(self.settings.mw_3d_peclim, self.settings.mw_3d_surfimplim)
        mask = cond[0, 0, :] > limit

        for bc in bcs:
            if not isinstance(bc, Void):
                continue
            void_tet_ids = field.mesh.get_tetrahedra(bc.tags)
            mask[void_tet_ids] = True

        return np.flatnonzero(mask)

    def _collect_pec_dofs(
        self, field: Nedelec2, mesh, bcs: list[BoundaryCondition], conductor_tets: np.ndarray, cond: np.ndarray
    ) -> tuple[set[int], list[int]]:
        """Collects PEC degrees of freedom from volumetric conductors and
        explicit PEC boundary conditions. Returns (pec_dof_ids, pec_tri_ids).
        """
        pec_ids: list[int] = []
        pec_tris: list[int] = []

        # Step 1: Add all conductor tet DoF as PEC (0-field)
        for itet in conductor_tets:
            pec_ids.extend(field.tet_to_field[:, itet])
            pec_tris.extend(field.mesh.tet_to_tri[:, itet])
        
        if len(conductor_tets):
            logger.trace(
                f" - Extended PEC with {len(conductor_tets)} tets with a conductivity > {self.settings.mw_3d_peclim}."
            )
        pec_ids_set = set(pec_ids)

        # Step 2: Remove all Robin Bcs
        for bc in select_bc(bcs, RobinBC):
            tri_ids = field.mesh.get_triangles(bc.tags)
            dofs = set(field.tri_to_field[:, tri_ids].flatten())
            pec_ids_set = pec_ids_set.difference(dofs)

        # Step 3: Add back all actual PEC dofs
        for pec in select_bc(bcs, PEC):
            logger.trace(f" - Implementing: {pec}")
            if len(pec.tags) == 0:
                continue
            tri_ids = mesh.get_triangles(pec.tags)
            edge_ids = mesh.tri_to_edge[:, tri_ids].flatten()
            pec_ids_set.update(field.edge_to_field[:, edge_ids].flatten())
            pec_ids_set.update(field.tri_to_field[:, tri_ids].flatten())
            pec_tris.extend(tri_ids)

        # Step 4: add back pec tets
        pec_dofs = np.unique(field.tet_to_field[:, np.flatnonzero(cond[0,0,:] > self.settings.mw_2dbc_peclim)].flatten())
        pec_ids_set.update(pec_dofs)
        
        return pec_ids_set, pec_tris


    def _prepare_system(
        self,
        field: Nedelec2,
        er: np.ndarray,
        ur: np.ndarray,
        conductor_tets: np.ndarray,
        robin_bcs: list[RobinBC],
        use_cache: bool,
    ) -> tuple[SystemPattern, np.ndarray, np.ndarray]:
        """Returns the system pattern and the volume matrices E, B as data
        arrays on that pattern.

        The pattern (and the geometry-only boundary matrices it holds) is
        reused for as long as the field, the set of conductor tets and the
        Robin BC structure stay the same. With use_cache, E and B are stored
        on the pattern and reused as well.
        """
        from .curlcurl import tet_mass_stiffness_matrices

        key = SystemPattern.compute_key(field, conductor_tets, robin_bcs)
        system = self._system if (self._system is not None and self._system.key == key) else None

        if system is not None and use_cache and system.cached_E is not None:
            logger.debug(" - Using cached matrices.")
            return system, system.cached_E, system.cached_B

        logger.debug(" - Calling matrix assembler...")
        t0 = time.time()
        Evec, Bvec, volume_coo_to_csc = tet_mass_stiffness_matrices(
            field, er, ur, conductor_tets, None if system is None else system.volume_coo_to_csc
        )
        t1 = time.time()
        logger.debug(f' - Assembly speed: {(field.ntets - len(conductor_tets)) / (t1 - t0):.1f} tets/s')

        if system is None:
            if self._system is not None:
                logger.debug(" - Mesh, conductors or boundary conditions changed: rebuilding the sparsity pattern.")
            system = SystemPattern(key, field, volume_coo_to_csc, robin_bcs)
            self._system = system
            self._periodic_patterns = None

        E = system.volume_coo_to_system_data(Evec)
        del Evec
        B = system.volume_coo_to_system_data(Bvec)
        del Bvec

        if use_cache:
            system.cached_E, system.cached_B = E, B
        return system, E, B

    def _assemble_robin_terms(
        self,
        system: SystemPattern,
        out: np.ndarray,
        field: Nedelec2,
        mesh: Mesh3D,
        K0: float,
        er: np.ndarray,
        robin_bcs: list[RobinBC],
        scale: complex = 1.0,
        port_vectors: dict[int | float, np.ndarray] | None = None,
        background_fields: dict | None = None,
    ) -> None:
        """Adds scale * (all Robin BC matrix terms) into the system data array
        `out` in place, and accumulates the excitation vectors.

        Each Robin/ABC term is a cached geometry-only matrix on the system
        pattern times a coefficient (bc.get_gamma(K0), bc.get_abccorr(K0)).
        The dense Wave Port BC block is recomputed per frequency (the port
        mode changes) but always covers the same DoFs. PML BCs skip the
        matrix term only -- their excitation is still assembled.

        Excitation vectors are accumulated in place into whichever of
        port_vectors (driven ports) or background_fields (scattered field)
        is given; eigenmode assembly passes neither. For a WPBC bc the port
        excitation reuses assemble_wpbc's own vector (same mode overlap
        already computed for the matrix term).
        """
        from .wpbc import assemble_wpbc

        for bc in robin_bcs:
            logger.trace(f"   - Implementing {bc}")
            tri_ids = mesh.get_triangles(bc.tags)
            gamma_bc = bc.get_gamma(K0)

            if bc.material_correction:
                tet_ids = mesh.tri_to_tet[0, tri_ids]
                eravg = (er[0, 0, tet_ids] + er[1, 1, tet_ids] + er[2, 2, tet_ids]) / 3
                gamma = gamma_bc * np.sqrt(eravg)
            else:
                gamma = gamma_bc * np.ones_like(tri_ids, dtype=np.complex128)

            logger.trace(f"    - robin bc γ={np.mean(gamma):.3f}")

            wpbc_bvec = None
            if bc_has_matrix_term(bc):
                if isinstance(bc, WavePortIH):
                    logger.debug("    - Assembling dense Wave Port Boundary Condition.")
                    mprof, mode_xy, kappa_m = bc.get_modepf_kappa(K0, mesh.nodes, mesh.tris[:, tri_ids])
                    values, wpbc_bvec = assemble_wpbc(
                        field, tri_ids, system.wave_port_dofs(bc), mprof, mode_xy, kappa_m, gamma_bc, K0
                    )
                    system.add_wave_port_term(out, bc, values, scale)
                else:
                    system.add_robin_term(out, bc, gamma, scale)

            if port_vectors is not None:
                self._add_port_force(field, bc, tri_ids, K0, port_vectors, wpbc_bvec)
            if background_fields is not None:
                self._add_background_force(field, bc, tri_ids, K0, background_fields)

            if bc_has_abc2_term(bc):
                logger.debug("    - Implementing second order ABC correction.")
                system.add_abc2_term(out, bc, bc.get_abccorr(K0), scale)

    def _add_port_force(
        self,
        field: Nedelec2,
        bc: RobinBC,
        tri_ids: np.ndarray,
        K0: float,
        port_vectors: dict[int | float, np.ndarray],
        wpbc_bvec: np.ndarray | None = None,
    ) -> None:
        """Adds the driven-port excitation of bc into port_vectors (in place)."""
        from .robinbc import assemble_robin_bc_bvec

        if not (bc._include_force and bc.driven and not isinstance(bc, ScatteredField)):
            return
        if wpbc_bvec is not None:
            port_vectors[bc.port_number] += wpbc_bvec
            logger.trace(f"    - included WPBC force vector term with norm {np.linalg.norm(wpbc_bvec):.3f}")
            return
        for number, Ufunc in bc._iter_modes(K0):
            b_p = assemble_robin_bc_bvec(field, tri_ids, Ufunc)
            port_vectors[number] += b_p
            logger.trace(f"    - included force vector term with norm {np.linalg.norm(b_p):.3f}")

    def _add_background_force(
        self,
        field: Nedelec2,
        bc: RobinBC,
        tri_ids: np.ndarray,
        K0: float,
        background_fields: dict,
    ) -> None:
        """Adds the incident-field excitation of a ScatteredField bc into
        background_fields (in place). Assembled regardless of the PML flag,
        since the PML only suppresses the absorbing matrix term.
        """
        from .robinbc import assemble_robin_bc_bvec_scat

        if not isinstance(bc, ScatteredField):
            return
        normals = field.mesh.outward_normals(tri_ids)
        for bf in bc._iter_fields(K0):
            b_p = assemble_robin_bc_bvec_scat(field, tri_ids, bf.Uinc, bf.Uinc_curl, normals)
            if bf in background_fields:
                background_fields[bf] += b_p
            else:
                background_fields[bf] = b_p
            logger.debug(f".. Background field {bf} {np.linalg.norm(b_p):.3f}")

    def _assemble_periodic_terms(
        self, system: SystemPattern, field: Nedelec2, mesh: Mesh3D, K0: float, periodic_bcs: list[Periodic]
    ) -> tuple[csc_matrix | None, np.ndarray | None, bool]:
        """Builds the combined periodic reduction matrix P and the set of
        retained DOF indices. Returns (Pmat, keep_indices, has_periodic).

        P and the reduced system P^H K P are put on structural patterns that
        are computed once (from all-positive copies, so nothing can cancel),
        which keeps the reduced pattern identical at every frequency.
        """
        from ....mth.pairing import pair_coordinates
        from .periodicbc import gen_periodic_matrix

        if len(periodic_bcs) == 0:
            return None, None, False

        logger.debug(" - Implementing Periodic Boundary Conditions.")
        Pmats = []
        remove: set[int] = set()

        for pbc in periodic_bcs:
            logger.trace(f"    - Implementing {pbc}")
            tri_ids_1 = mesh.get_triangles(pbc.face1.tags)
            edge_ids_1 = mesh.get_edges(pbc.face1.tags)
            tri_ids_2 = mesh.get_triangles(pbc.face2.tags)
            edge_ids_2 = mesh.get_edges(pbc.face2.tags)
            dv = np.array(pbc.dv)
            logger.trace(f"    - displacement vector {dv}")
            linked_tris = pair_coordinates(mesh.tri_centers, tri_ids_1, tri_ids_2, dv, _PBC_DSMAX)
            linked_edges = pair_coordinates(mesh.edge_centers, edge_ids_1, edge_ids_2, dv, _PBC_DSMAX)
            phi = pbc.phi(K0)
            logger.trace(f"    - ϕ={phi} rad/m")
            Pmat, rows = gen_periodic_matrix(
                tri_ids_1,
                edge_ids_1,
                field.tri_to_field,
                field.edge_to_field,
                linked_tris,
                linked_edges,
                field.dofcodes2d,
                field.n_field,
                phi,
            )
            remove.update(rows)
            Pmats.append(Pmat)

        logger.trace(f"  - periodic bc removes {len(remove)} boundary DoF")
        keep_indices = np.setdiff1d(np.arange(field.n_field), np.sort(np.unique(list(remove))))

        key = (system.key, keep_indices.tobytes(), tuple(_pattern_hash(P) for P in Pmats))
        if self._periodic_patterns is None or self._periodic_patterns[0] != key:
            P_struct = product_pattern(*Pmats)[:, keep_indices].tocsc()
            K_struct = system.as_csc_matrix(np.ones(system.nnz))
            S_struct = product_pattern(P_struct.T, K_struct, P_struct)
            self._periodic_patterns = (key, P_struct, S_struct)

        Pmat = Pmats[0]
        for P2 in Pmats[1:]:
            Pmat = Pmat @ P2
        Pmat = copy_onto_pattern(Pmat[:, keep_indices], self._periodic_patterns[1])
        return Pmat, keep_indices, True

    def _reduce_periodic(self, M: csc_matrix, Pmat: csc_matrix) -> csc_matrix:
        """P^H M P on the cached reduced pattern."""
        return copy_onto_pattern(Pmat.getH() @ M @ Pmat, self._periodic_patterns[2])

    @staticmethod
    def _periodic_solve_ids(solve_ids: np.ndarray, keep_indices: np.ndarray, NF: int) -> np.ndarray:
        """Remaps solve_ids into the reduced periodic DOF numbering."""
        mask = np.zeros(NF, dtype=bool)
        mask[solve_ids] = True
        return np.flatnonzero(mask[keep_indices])

    @staticmethod
    def _solve_ids(NF: int, pec_ids: set[int]) -> np.ndarray:
        mask = np.ones(NF, dtype=bool)
        mask[list(pec_ids)] = False
        return np.flatnonzero(mask)

    # ------------------------------------------------------------------
    # Boundary mode analysis (unchanged -- different field type / shape,
    # does not share the Robin/periodic/PEC pattern above)
    # ------------------------------------------------------------------

    def assemble_bma_matrices(
        self,
        field: Nedelec2,
        er: np.ndarray,
        ur: np.ndarray,
        sig: np.ndarray,
        k0: float,
        port: PortBC,
        bc_set: MWBoundaryConditionSet,
        dofset: DoFSet

    ) -> tuple[csc_matrix, csc_matrix, np.ndarray, NedelecLegrange2]:
        """Computes the boundary mode analysis matrices

        Args:
            field (Nedelec2): The Nedelec2 field object
            er (np.ndarray): The relative permittivity tensor of shape (3,3,N)
            ur (np.ndarray): The relative permeability tensor of shape (3,3,N)
            sig (np.ndarray): The conductivity scalar of shape (N,)
            k0 (float): The simulation phase constant
            port (PortBC): The port boundary condition object
            bcs (MWBoundaryConditionSet): The other boundary conditions

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, NedelecLegrange2]: The E, B, solve ids and Mixed order field object.
        """
        from .generalized_eigen_hb import generelized_eigenvalue_matrix

        logger.debug("Assembling Boundary Mode Matrices")

        mesh = field.mesh
        tri_ids = mesh.get_triangles(port.tags)
        logger.trace(f".boundary face has {len(tri_ids)} triangles.")

        boundary_surface = mesh.boundary_surface(port.tags)
        nedlegfield = NedelecLegrange2(boundary_surface, port.cs, dofset)

        ermesh = er[:, :, tri_ids]
        urmesh = ur[:, :, tri_ids]
        sigmesh = sig[tri_ids]

        loss = -1j * sigmesh / (k0 * C0 * EPS0)
        ermesh[0, 0, :] = ermesh[0, 0, :] + loss
        ermesh[1, 1, :] = ermesh[1, 1, :] + loss
        ermesh[2, 2, :] = ermesh[2, 2, :] + loss

        logger.trace(f".assembling matrices for {nedlegfield} at k0={k0:.2f}")
        E, B = generelized_eigenvalue_matrix(
            nedlegfield, ermesh, urmesh, port.cs._basis, k0
        )

        # TODO: Simplified to all "conductors" loosely defined. Must change to implementing line robin boundary conditions.
        pecs: list[BoundaryCondition] = bc_set.get_conductors()

        if len(pecs) > 0:
            logger.debug(f".total of equiv. {len(pecs)} PEC BCs implemented for BMA")

        pec_ids = []

        for it in range(boundary_surface.n_tris):
            if (
                sigmesh[it] > self.settings.mw_3d_peclim
                or sigmesh[it] > self.settings.mw_3d_surfimplim
            ):
                pec_ids.extend(list(nedlegfield.tri_to_field[:, it]))

        for pec in pecs:
            logger.trace(f".implementing {pec}")
            if len(pec.tags) == 0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())
            for ii in edge_ids:
                i2 = nedlegfield.mesh.from_source_edge(ii)
                if i2 is None:
                    continue
                eids = nedlegfield.edge_to_field[:, i2]
                pec_ids.extend(list(eids))

        pec_ids_set: set[int] = set(pec_ids)

        logger.trace(f".total of {len(pec_ids_set)} pec DoF to remove.")
        solve_ids = [i for i in range(nedlegfield.n_field) if i not in pec_ids_set]

        return E, B, np.array(solve_ids), nedlegfield

    # ------------------------------------------------------------------
    # Frequency-domain (driven port) assembly
    # ------------------------------------------------------------------

    def assemble_freq_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
        cache_matrices: bool = False,
    ) -> tuple[SimJob, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Assembles the frequency domain FEM matrix

        Args:
            field (Nedelec2): The Nedelec2 object of the problems
            mat_assy (MaterialAssignment): Material assignment for the domain
            bcs (list[BoundaryCondition]): The boundary conditions
            frequency (float): The simulation frequency
            cache_matrices (bool, optional): Whether to use and cache matrices. Defaults to False.

        Returns:
            tuple[SimJob, tuple[np.ndarray, np.ndarray, np.ndarray]]: The SimJob and the (er, ur, cond) material tensors
        """
        logger.debug(f'Assembling frequency = {_format_freq(frequency)}')

        K0 = 2 * np.pi * frequency / C0
        mesh = field.mesh
        NF = field.n_field

        er, ur, cond, is_frequency_dependent = self._assemble_materials(mat_assy, field, frequency)
        conductor_tets = self._find_conductor_tets(field, bcs, cond)
        logger.debug(f' - Total of {len(conductor_tets)} PEC Tetrahedrons')

        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        port_bcs: list[PortBC] = [bc for bc in bcs if isinstance(bc, PortBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        system, E, B = self._prepare_system(
            field, er, ur, conductor_tets, robin_bcs, cache_matrices and not is_frequency_dependent
        )

        port_vectors: dict[int | float, np.ndarray] = {}
        for port in sorted(port_bcs, key=lambda x: x.port_number):
            for mat_index, mode_nr in port._iter_port_numbers():
                port_vectors[mat_index] = np.zeros((NF,), dtype=np.complex128)

        logger.debug(" - Implementing PEC Boundary Conditions.")
        pec_ids, pec_tris = self._collect_pec_dofs(field, mesh, bcs, conductor_tets, cond)

        K_data = system.a_plus_alpha_b(E, B, -K0 ** 2)
        if len(robin_bcs) > 0:
            logger.debug(" - Assembling Robin Boundary Conditions.")
        self._assemble_robin_terms(system, K_data, field, mesh, K0, er, robin_bcs, port_vectors=port_vectors)
        K = system.as_csc_matrix(K_data)

        solve_ids = self._solve_ids(NF, pec_ids)
        Pmat, keep_indices, has_periodic = self._assemble_periodic_terms(system, field, mesh, K0, periodic_bcs)
        if has_periodic:
            K = self._reduce_periodic(K, Pmat)
            solve_ids = self._periodic_solve_ids(solve_ids, keep_indices, NF)
            for key, b in port_vectors.items():
                port_vectors[key] = Pmat.getH() @ b

        logger.debug(f"  - Number of tets: {mesh.n_tets:,}")
        logger.debug(f"  - Number of DoF: {K.shape[0]:,}")
        logger.debug(f"  - Number of non-zero: {K.nnz:,}")

        if self.mldata_filename is not None:
            from ....mldata import MLPreconData
            Emat = system.as_csc_matrix(E)
            # K = E - k0^2 (B - R/k0^2): the boundary terms folded into the mass matrix.
            Bmat = system.as_csc_matrix((E - K_data) / K0 ** 2)

            # Discrete gradient G : Legrange2 -> Nedelec2. Exported alongside
            # E/B because it cannot be reconstructed from them, and an
            # auxiliary-space Maxwell preconditioner needs it to separate the
            # gradient (curl-free) modes from the rest.
            from .ams_export import (assemble_discrete_gradient,
                                     assemble_nedelec_interpolation)
            t_grad = time.time()
            Gmat = assemble_discrete_gradient(field)
            Pimat = assemble_nedelec_interpolation(field)
            logger.debug(f"  - AMS operators: G {Gmat.shape} nnz={Gmat.nnz:,}, "
                         f"Pi {Pimat.shape} nnz={Pimat.nnz:,} ({time.time() - t_grad:.1f}s)")

            mldataset = MLPreconData(self.mldata_filename, Emat, Bmat, K0, np.array(solve_ids), mesh.nodes, mesh.edges, mesh.tris, field.compute_global_dofcodes(), field.compute_global_dof_coords(), grad=Gmat, pi=Pimat)
        else:
            mldataset = None

        #diagnose_matrix(K[solve_ids,:][:,solve_ids], field, solve_ids)

        simjob = SimJob(
            K, port_vectors, K0 * 299792458 / (2 * np.pi), symmetric=not has_periodic, mldataset=mldataset
        )

        simjob.solve_ids = solve_ids
        simjob._pec_tris = pec_tris

        if has_periodic:
            simjob.P = Pmat
            simjob.has_periodic = has_periodic

        return simjob, (er, ur, cond)

    # ------------------------------------------------------------------
    # Scattered-field assembly
    # ------------------------------------------------------------------

    def assemble_scattering_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
        cache_matrices: bool = False,
    ) -> tuple[SimJob, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Assembles the scattered-field frequency domain FEM matrix

        Args:
            field (Nedelec2): The Nedelec2 object of the problems
            mat_assy (MaterialAssignment): Material assignment for the domain
            bcs (list[BoundaryCondition]): The boundary conditions
            frequency (float): The simulation frequency
            cache_matrices (bool, optional): Whether to use and cache matrices. Defaults to False.

        Returns:
            tuple[SimJob, tuple[np.ndarray, np.ndarray, np.ndarray]]: The SimJob and the (er, ur, cond) material tensors
        """
        K0 = 2 * np.pi * frequency / C0
        mesh = field.mesh
        NF = field.n_field

        er, ur, cond, is_frequency_dependent = self._assemble_materials(mat_assy, field, frequency)
        conductor_tets = self._find_conductor_tets(field, bcs, cond)

        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        system, E, B = self._prepare_system(
            field, er, ur, conductor_tets, robin_bcs, cache_matrices and not is_frequency_dependent
        )

        logger.debug("Implementing PEC Boundary Conditions.")
        pec_ids, pec_tris = self._collect_pec_dofs(field, mesh, bcs, conductor_tets, cond)

        background_fields: dict[tuple[float, float], np.ndarray] = {}

        K_data = system.a_plus_alpha_b(E, B, -K0 ** 2)
        self._assemble_robin_terms(
            system, K_data, field, mesh, K0, er, robin_bcs, background_fields=background_fields
        )
        matrix_fem = system.as_csc_matrix(K_data)

        solve_ids = self._solve_ids(NF, pec_ids)
        Pmat, keep_indices, has_periodic = self._assemble_periodic_terms(system, field, mesh, K0, periodic_bcs)
        if has_periodic:
            matrix_fem = self._reduce_periodic(matrix_fem, Pmat)
            solve_ids = self._periodic_solve_ids(solve_ids, keep_indices, NF)
            for key, b in background_fields.items():
                background_fields[key] = Pmat.getH() @ b

        logger.debug(f"Number of tets: {mesh.n_tets:,}")
        logger.debug(f"Number of DoF: {matrix_fem.shape[0]:,}")
        logger.debug(f"Number of non-zero: {matrix_fem.nnz:,}")

        simjob = SimJob(
            matrix_fem, background_fields, K0 * 299792458 / (2 * np.pi), symmetric=not has_periodic
        )

        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.has_periodic = has_periodic

        return simjob, (er, ur, cond)

    # ------------------------------------------------------------------
    # Eigenmode assembly
    # ------------------------------------------------------------------

    def assemble_eig_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
    ) -> tuple[SimJob, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Assembles the eigenmode analysis matrix

        The assembly process is frequency dependent because the frequency-dependent properties
        need a guess before solving. There is currently no adjustment after an eigenmode is found.
        The frequency-dependent properties are simply calculated once for the given frequency.

        Args:
            field (Nedelec2): The Nedelec2 field
            mat_assy (MaterialAssignment): Material assignment for the domain
            bcs (list[BoundaryCondition]): The list of boundary conditions
            frequency (float): The compilation frequency (for material properties only)

        Returns:
            tuple[SimJob, tuple[np.ndarray, np.ndarray, np.ndarray]]: The SimJob and the (er, ur, cond) material tensors
        """
        mesh = field.mesh
        k0 = 2 * np.pi * frequency / C0
        NF = field.n_field

        er, ur, cond, _ = self._assemble_materials(mat_assy, field, frequency)
        conductor_tets = self._find_conductor_tets(field, bcs, cond)

        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        logger.debug("Assembling matrices")
        system, E, B = self._prepare_system(field, er, ur, conductor_tets, robin_bcs, use_cache=False)

        logger.debug("Implementing PEC Boundary Conditions.")
        pec_ids, _ = self._collect_pec_dofs(field, mesh, bcs, conductor_tets, cond)

        # Eigenmode assembly has no excitation vectors.
        mass_data = B.copy()
        self._assemble_robin_terms(system, mass_data, field, mesh, k0, er, robin_bcs, scale=-1 / k0 ** 2)
        matrix_stiff = system.as_csc_matrix(E)
        matrix_mass = system.as_csc_matrix(mass_data)

        solve_ids = self._solve_ids(NF, pec_ids)
        Pmat, keep_indices, has_periodic = self._assemble_periodic_terms(system, field, mesh, k0, periodic_bcs)
        if has_periodic:
            matrix_stiff = self._reduce_periodic(matrix_stiff, Pmat)
            matrix_mass = self._reduce_periodic(matrix_mass, Pmat)
            solve_ids = self._periodic_solve_ids(solve_ids, keep_indices, NF)

        logger.debug(f"Number of tets: {mesh.n_tets}")
        logger.debug(f"Number of DoF: {matrix_stiff.shape[0]}")

        simjob = SimJob(matrix_stiff, None, frequency, B=matrix_mass)
        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.has_periodic = has_periodic

        return simjob, (er, ur, cond)
