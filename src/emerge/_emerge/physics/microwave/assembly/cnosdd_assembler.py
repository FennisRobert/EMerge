# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.
# ... (license header unchanged)

import numpy as np
from ..bcs import (
    PEC,
    BoundaryCondition,
    ScatteredField,
    RobinBC,
    PortBC,
    ThinConductor,
    SurfaceImpedance,
)
from ....geometry import GeoVolume
from ....selection import DomainSelection
from ...material_assignment import MaterialAssignment
from ....elements.nedelec2 import Nedelec2
from ....settings import Settings
from scipy.sparse import csc_matrix
from loguru import logger
from ....const import EPS0, C0
import time
from .dd_tools import generate_reduction_matrix

_PBC_DSMAX = 1e-15


class AssemblyError(Exception):
    pass


############################################################
#                    DIAGNOSTIC UTILITIES                  #
############################################################
# diagnose_matrix / identify_mesh_dead_zones / plane_basis_from_points /
# remove_from are unchanged from v1 -- unrelated to the assembly-path
# cleanup below, kept as-is for compatibility with any external callers.


def _format_freq(freq: float) -> str:
    units = ["Hz", "kHz", "MHz", "GHz", "THz"]
    if freq == 0:
        return "0.00 Hz"
    i = int(np.floor(np.log10(abs(freq)) / 3))
    i = max(0, min(i, len(units) - 1))
    return f"{freq / (1000.0 ** i):.2f} {units[i]}"


############################################################
#                          CNOSDD DATA                      #
############################################################

class CNOSDDData:

    def __init__(self,
                 n_doms: int,
                 n_intf: int,
                 domlink: dict[int, list[int]],
                 As: dict[int, csc_matrix],
                 Zs: dict[int, csc_matrix],
                 Rsd: dict[int, csc_matrix],
                 RsI: dict[int, csc_matrix],
                 RsId: dict[int, dict[int, csc_matrix]],
                 bs: dict[int | float, list[np.ndarray]],
                 gs: dict[tuple[int, int], np.ndarray],
                 dof_all: np.ndarray,
                 dof_domains: dict[int, np.ndarray],
                 dof_interfaces: dict[int, np.ndarray],
                 dof_pec: np.ndarray,
                 itfmap: dict[tuple[int, int] | int, tuple[int, int] | int]):
        self.n_doms = n_doms
        self.n_intf = n_intf
        self.domlink = domlink
        self.As = As
        self.Zs = Zs
        self.bs = bs
        self.Rsd = Rsd
        self.RsI = RsI
        self.RsId = RsId
        self.gs = gs
        
        # itfmap is stored keyed both by interface id (int) and by domain
        # pair (tuple) -- make sure both directions of every pair resolve.
        self.itfmap = dict(itfmap)
        extra = {}
        for key, val in itfmap.items():
            if isinstance(key, tuple):
                a, b = key
                extra[(b, a)] = val
        self.itfmap.update(extra)

        self.dof_all = dof_all
        self.dof_domains = dof_domains
        self.dof_interfaces = dof_interfaces
        self.dof_pec = dof_pec
        self.dof_solve = np.setdiff1d(dof_all, dof_pec, assume_unique=False)

        self.dof_solve_dom = {i: self._nopec_local(d, self.dof_solve) for i, d in dof_domains.items()}
        self.dof_solve_itf = {i: self._nopec_local(d, self.dof_solve) for i, d in dof_interfaces.items()}

    def get_matrix(self, index: int) -> csc_matrix:
        sl = self.dof_solve_dom[index]
        return self.As[index][sl, :][:, sl]

    def get_zmat(self, index: int) -> csc_matrix:
        sl = self.dof_solve_itf[index]
        return self.Zs[index][sl, :][:, sl]

    def get_b(self, modenr: int | float, index: int) -> np.ndarray:
        return self.bs[modenr][index][self.dof_solve_dom[index]]

    def get_Rm(self, idom: int, i_itf: int) -> np.ndarray:
        return self.RsId[idom][i_itf][self.dof_solve_itf[i_itf], :][:, self.dof_solve_dom[idom]]

    def get_g(self, key: tuple[int, int]) -> np.ndarray:
        return self.gs[key][self.dof_solve_itf[self.itfmap[key]]]

    @staticmethod
    def _nopec_local(sl_sub: np.ndarray, sl_nopec: np.ndarray) -> np.ndarray:
        """Local (within sl_sub) indices of the entries that are also in sl_nopec."""
        return np.flatnonzero(np.isin(sl_sub, sl_nopec, assume_unique=True))


############################################################
#                        ASSEMBLER                          #
############################################################

class CNOSDDAssembler:
    """FEM assembler for Conformal Non-Overlapping Schwarz Domain Decomposition."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.assy_field: Nedelec2 | None = None
        self.assy_freq: float = None
        self.assy_pecdof: set[int] = set()
        self.assy_port_vectors: dict[int | float, np.ndarray] = {}
        self.assy_port_vectors_split: dict[int | float, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Top level entry point
    # ------------------------------------------------------------------

    def assemble_freq_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
        domains: list[GeoVolume | DomainSelection],
    ) -> tuple[CNOSDDData, np.ndarray, np.ndarray, np.ndarray]:

        from .curlcurl import tet_mass_stiffness_matrices_subdom

        self.assy_field = field
        self.assy_freq = frequency
        logger.debug(f'Running DD Assembly for frequency = {_format_freq(frequency)}')

        mesh = field.mesh
        K0 = 2 * np.pi * frequency / C0

        er, ur, cond, conductor_tets = self._assemble_materials(mat_assy, field, K0)

        domains = {i: d for i, d in enumerate(domains)}
        domain_tets = {i: mesh.get_tetrahedra(d.tags) for i, d in domains.items()}
        tet_owner = self._build_tet_owner(mesh.n_tets, domain_tets)

        domain_links, intf_2_tris, itf_id, id_itf = self._find_interfaces(mesh, domains)
        n_domains, n_interfaces = len(domains), len(itf_id)

        dof_all = np.arange(field.n_field)
        Rsd, As, dof_per_subdomain = self._build_subdomain_systems(
            field, er, ur, K0, domain_tets, tet_mass_stiffness_matrices_subdom, conductor_tets, dof_all
        )

        dof_per_interface, RsId, RsI = self._build_interface_reductions(
            field, dof_all, dof_per_subdomain, intf_2_tris, id_itf, itf_id, domain_links, n_interfaces
        )

        gs: dict[tuple[int, int], np.ndarray] = {}
        for (a, b), i_itf in itf_id.items():
            shape = dof_per_interface[i_itf].shape
            gs[(a, b)] = np.zeros(shape, dtype=np.complex128)
            gs[(b, a)] = np.zeros(shape, dtype=np.complex128)

        self._get_pec_ids(bcs, conductor_tets)
        self._assemble_robin_bcs(bcs, mesh, tet_owner, Rsd, As, n_domains)

        Zs = self._assemble_interfaces(id_itf, intf_2_tris, domain_links, Rsd, RsI, As, n_domains)

        itfmap = dict(id_itf) | itf_id

        data = CNOSDDData(
            n_domains, n_interfaces, domain_links, As, Zs, Rsd, RsI, RsId,
            self.assy_port_vectors_split, gs, dof_all, dof_per_subdomain,
            dof_per_interface, np.sort(np.fromiter(self.assy_pecdof, dtype=np.int64)), itfmap,
        )
        return data, er, ur, cond

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def _assemble_materials(self, mat_assy: MaterialAssignment, field: Nedelec2, K0: float):
        n_tets = field.mesh.n_tets
        er = np.zeros((3, 3, n_tets), dtype=np.complex128)
        ur = np.zeros((3, 3, n_tets), dtype=np.complex128)
        tand = np.zeros((3, 3, n_tets), dtype=np.complex128)
        cond = np.zeros((3, 3, n_tets), dtype=np.complex128)

        for mat, centers, ids in mat_assy.iter_materials():
            er = mat.er(self.assy_freq, er, centers, ids)
            ur = mat.ur(self.assy_freq, ur, centers, ids)
            tand = mat.tand(self.assy_freq, tand, centers, ids)
            cond = mat.cond(self.assy_freq, cond, centers, ids)

        er = er * (1 - 1j * tand) - 1j * cond / (2 * np.pi * self.assy_freq * EPS0)

        limit = min(self.settings.mw_3d_peclim, self.settings.mw_3d_surfimplim)
        conductor_tets = np.flatnonzero(cond[0, 0, :field.n_tets] > limit)
        return er, ur, cond, conductor_tets

    # ------------------------------------------------------------------
    # Domain partitioning / interface discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tet_owner(n_tets: int, domain_tets: dict[int, np.ndarray]) -> np.ndarray:
        """tet index -> owning domain id (-1 if unowned)."""
        owner = np.full(n_tets, -1, dtype=np.int64)
        for idom, tets in domain_tets.items():
            owner[tets] = idom
        return owner

    @staticmethod
    def _find_interfaces(mesh, domains: dict[int, GeoVolume | DomainSelection]):
        """Detects shared-boundary interfaces between all domain pairs.

        Boundary triangle sets are computed once per domain instead of once
        per pair, since each domain participates in O(n_domains) comparisons.
        """
        n_domains = len(domains)
        boundary_tris = {i: set(mesh.get_triangles(d.boundary().tags)) for i, d in domains.items()}

        intf_2_tris: dict[tuple[int, int], list[int]] = {}
        domain_links: dict[int, list[int]] = {i: [] for i in range(n_domains)}

        for i1 in range(n_domains - 1):
            for i2 in range(i1 + 1, n_domains):
                shared = boundary_tris[i1] & boundary_tris[i2]
                if shared:
                    domain_links[i1].append(i2)
                    domain_links[i2].append(i1)
                    intf_2_tris[(i1, i2)] = sorted(shared)

        domain_links = {i: sorted(ls) for i, ls in domain_links.items()}
        itf_id = {key: i for i, key in enumerate(intf_2_tris)}
        id_itf = {i: key for key, i in itf_id.items()}

        for i, adj in domain_links.items():
            logger.debug(f' - Domain {i} linked to: {adj}')
        for i, key in id_itf.items():
            logger.debug(f' - Interface {i} = {key} ({len(intf_2_tris[key])} tris)')

        return domain_links, intf_2_tris, itf_id, id_itf

    # ------------------------------------------------------------------
    # Subdomain system matrices
    # ------------------------------------------------------------------

    def _build_subdomain_systems(self, field, er, ur, K0, domain_tets, mass_stiffness_fn, conductor_tets, dof_all):
        Rsd: dict[int, csc_matrix] = {}
        As: dict[int, csc_matrix] = {}
        dof_per_subdomain: dict[int, np.ndarray] = {}

        for i, tets in domain_tets.items():
            Evec, Bvec, rows, cols = mass_stiffness_fn(field, er, ur, tets, conductor_tets)
            D_s = np.sort(np.unique(field.tet_to_field[:, tets]))
            dof_per_subdomain[i] = D_s
            Rsd[i] = generate_reduction_matrix(dof_all, D_s)
            A_full = csc_matrix((Evec - Bvec * (K0 ** 2), (rows, cols)), shape=(field.n_field, field.n_field))
            As[i] = A_full[D_s, :][:, D_s]
            logger.debug(f' - Domain {i}: Rs.shape={Rsd[i].shape}, A.shape={As[i].shape}')

        return Rsd, As, dof_per_subdomain

    def _build_interface_reductions(self, field, dof_all, dof_per_subdomain, intf_2_tris, id_itf, itf_id, domain_links, n_interfaces):
        dof_per_interface = {
            i: np.sort(np.unique(field.tri_to_field[:, intf_2_tris[id_itf[i]]]))
            for i in range(n_interfaces)
        }

        RsId: dict[int, dict[int, csc_matrix]] = {i: {} for i in domain_links}
        for idom, adj in domain_links.items():
            for jdom in adj:
                i_itf = itf_id[(min(idom, jdom), max(idom, jdom))]
                RsId[idom][i_itf] = generate_reduction_matrix(dof_per_subdomain[idom], dof_per_interface[i_itf])

        RsI = {i: generate_reduction_matrix(dof_all, dof_per_interface[i]) for i in range(n_interfaces)}
        return dof_per_interface, RsId, RsI

    # ------------------------------------------------------------------
    # Robin / interface matrix construction
    # ------------------------------------------------------------------

    def _robin_bc_to_csc(self, tri_ids: np.ndarray, gamma: complex) -> csc_matrix:
        """Assembles a Robin-type surface integral over tri_ids into a global CSC matrix."""
        from .robinbc import assemble_robin_bc
        field = self.assy_field
        B = assemble_robin_bc(field, field.empty_tri_matrix(), tri_ids, gamma)
        return csc_matrix((B, (field._rows, field._cols)), shape=(field.n_field, field.n_field), dtype=np.complex128)

    def _get_robin_mat(self, bc: RobinBC) -> csc_matrix:
        from .robinbc import assemble_robin_bc_bvec
        from .robin_abc_order2 import abc_order_2_matrix

        field = self.assy_field
        K0 = 2 * np.pi * self.assy_freq / C0
        tri_ids = field.mesh.get_triangles(bc.tags)

        if isinstance(bc, (SurfaceImpedance, ThinConductor)):
            self.assy_pecdof -= set(field.tri_to_field[:, tri_ids].flatten())

        gamma = bc.get_gamma(K0)
        logger.trace(f"    - robin bc γ={gamma:.3f}")

        K = self._robin_bc_to_csc(tri_ids, gamma) if bc._assemble_matrix else csc_matrix(
            (field.n_field, field.n_field), dtype=np.complex128
        )

        if bc._include_force and bc.driven and not isinstance(bc, ScatteredField):
            for number, Ufunc in bc._iter_modes(K0):
                b_p = assemble_robin_bc_bvec(field, tri_ids, Ufunc)
                self.assy_port_vectors[number] += b_p
                logger.trace(f"    - force vector norm={np.linalg.norm(b_p):.3f}")

        if bc._isabc and bc.order == 2:
            logger.debug("    - Implementing second order ABC correction.")
            c2 = bc.get_abccorr(K0)
            mat = abc_order_2_matrix(field, tri_ids, c2)
            K = K + csc_matrix((mat, (field._rows, field._cols)), shape=K.shape, dtype=np.complex128)

        return K

    def _get_interface_matrix(self, tri_ids: list[int]) -> csc_matrix:
        """First-order (γ = j·k0) Robin transmission condition for DD interfaces."""
        K0 = 2 * np.pi * self.assy_freq / C0
        return self._robin_bc_to_csc(np.asarray(tri_ids), 1j * K0)

    # ------------------------------------------------------------------
    # BC application (each BC/interface touches only the domain(s) it belongs to)
    # ------------------------------------------------------------------

    def _owning_domain(self, mesh, tri_ids: np.ndarray, tet_owner: np.ndarray) -> int:
        tets = mesh.tri_to_tet[0, tri_ids]
        owners = np.unique(tet_owner[tets[tets >= 0]])
        owners = owners[owners >= 0]
        if owners.size == 0:
            raise AssemblyError("Boundary condition triangles are not owned by any domain.")
        if owners.size > 1:
            logger.warning(f"Boundary condition spans multiple domains {owners.tolist()}; "
                            f"assigning to {owners[0]}.")
        return int(owners[0])

    def _assemble_robin_bcs(self, bcs, mesh, tet_owner, Rsd, As, n_domains):
        robin_bcs = [bc for bc in bcs if isinstance(bc, RobinBC)]
        port_bcs = [bc for bc in bcs if isinstance(bc, PortBC)]

        self.assy_port_vectors = {}
        self.assy_port_vectors_split = {}
        for port in sorted(port_bcs, key=lambda x: x.port_number):
            for mat_index, _ in port._iter_port_numbers():
                self.assy_port_vectors[mat_index] = np.zeros(self.assy_field.n_field, dtype=np.complex128)
                self.assy_port_vectors_split[mat_index] = []

        for bc in robin_bcs:
            if len(bc.tags) == 0:
                continue
            K = self._get_robin_mat(bc)
            tri_ids = mesh.get_triangles(bc.tags)
            idom = self._owning_domain(mesh, tri_ids, tet_owner)
            R = Rsd[idom]
            As[idom] = As[idom] + R @ K @ R.T

        for mode, vector in self.assy_port_vectors.items():
            self.assy_port_vectors_split[mode] = [Rsd[i] @ vector for i in range(n_domains)]

    def _assemble_interfaces(self, id_itf, intf_2_tris, domain_links, Rsd, RsI, As, n_domains):
        Zs: dict[int, csc_matrix] = {}
        for i_itf, (i1, i2) in id_itf.items():
            K = self._get_interface_matrix(intf_2_tris[(i1, i2)])

            for idom in (i1, i2):
                R = Rsd[idom]
                As[idom] = As[idom] + R @ K @ R.T

            R = RsI[i_itf]
            Zs[i_itf] = R @ K @ R.T

        return Zs

    # ------------------------------------------------------------------
    # PEC handling (unchanged logic, tidied)
    # ------------------------------------------------------------------

    def _get_pec_ids(self, bcs: list[BoundaryCondition], conductor_tets: np.ndarray) -> None:
        logger.debug(" - Implementing PEC Boundary Conditions.")
        field = self.assy_field
        pec_ids: list[int] = []

        for itet in conductor_tets:
            pec_ids.extend(field.tet_to_field[:, itet])
        if len(conductor_tets):
            logger.trace(f" - Extended PEC with {len(conductor_tets)} tets above conductivity limit.")

        for pec in [bc for bc in bcs if isinstance(bc, PEC)]:
            if len(pec.tags) == 0:
                continue
            tri_ids = field.mesh.get_triangles(pec.tags)
            edge_ids = field.mesh.tri_to_edge[:, tri_ids].flatten()
            pec_ids.extend(field.edge_to_field[:, edge_ids].flatten())
            pec_ids.extend(field.tri_to_field[:, tri_ids].flatten())

        self.assy_pecdof = set(pec_ids)