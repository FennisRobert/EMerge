# [Written by Claude] All code in this module was written by Claude (AI).
"""Far fields of several port excitations on the same integration surface in one pass.

The Stratton-Chu phase factor exp(j k0 r.x') depends only on the direction r and the surface point x', not on
the excitation. `stratton_chu_ff` recomputes it for every excitation; `emr.sc_multi.stratton_chu_ff_multi`
computes it once per direction and surface point and applies it to the n x E and n x H of all excitations.
The integration surface and the point to tetrahedron mapping are also built once for all ports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from emr.sc_multi import stratton_chu_ff_multi
from emsutil.emdata import EHFieldFF

from ...mesh3d import SurfaceMesh
from .sc import _weighted_normals

if TYPE_CHECKING:
    from ...geometry import GeoSurface
    from ...selection import FaceSelection
    from .microwave_data import MWField


def stratton_chu_multi(
    Eins: list[np.ndarray],
    Hins: list[np.ndarray],
    mesh: SurfaceMesh,
    theta: np.ndarray,
    phi: np.ndarray,
    k0: float,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """The far field of several excitations on the same surface, as `sc.stratton_chu` gives it per excitation.

    Args:
        Eins (list[np.ndarray]): The surface E field (3, Ns) of every excitation at the edge centers of mesh
        Hins (list[np.ndarray]): The surface H field (3, Ns) of every excitation
        mesh (SurfaceMesh): The integration surface with outward normals
        theta (np.ndarray): The theta angles (N,)
        phi (np.ndarray): The phi angles (N,)
        k0 (float): The free space wave number

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], np.ndarray]: E (3, N) and H (3, N) of every excitation and the
            weighted normals (3, Ns)
    """
    wns = _weighted_normals(mesh)
    K = len(Eins)
    Ein = np.ascontiguousarray(np.concatenate([np.asarray(E) for E in Eins], axis=0), dtype=np.complex128)
    Hin = np.ascontiguousarray(np.concatenate([np.asarray(H) for H in Hins], axis=0), dtype=np.complex128)
    Eout, Hout = stratton_chu_ff_multi(
        Ein,
        Hin,
        np.ascontiguousarray(mesh.edge_centers, dtype=np.float64),
        np.ascontiguousarray(wns, dtype=np.float64),
        np.ascontiguousarray([theta, phi], dtype=np.float64),
        np.float64(k0),
    )
    Es = [Eout[3 * k : 3 * k + 3] for k in range(K)]
    Hs = [Hout[3 * k : 3 * k + 3] for k in range(K)]
    return Es, Hs, wns


def farfield_ports(
    field: MWField,
    faces: FaceSelection | GeoSurface,
    thetas: np.ndarray | None = None,
    phis: np.ndarray | None = None,
    origin: tuple[float, float, float] | None = None,
    syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
    ports: list[int | float] | None = None,
) -> dict[int | float, EHFieldFF]:
    """The 3D far field of every port excited on its own, as `MWField.farfield_3d` gives it per port.

    Args:
        field (MWField): The field solution of one frequency
        faces (FaceSelection | GeoSurface): The integration faces
        thetas (np.ndarray, optional): The 1D array of theta values. Defaults to 91 points from 0 to pi.
        phis (np.ndarray, optional): The 1D array of phi values. Defaults to 181 points from -pi to pi.
        origin (tuple[float, float, float], optional): The boundary normal alignment origin. Defaults to None.
        syms (list[Literal['Ex','Ey','Ez','Hx','Hy','Hz']], optional): E and H-plane symmetry planes. Defaults to None.
        ports (list[int | float], optional): The port numbers to excite. Defaults to all active ports.

    Returns:
        dict[int | float, EHFieldFF]: The far field of every port
    """
    from .microwave_data import _mirrored_sources, _symmetry_factor

    if syms is None:
        syms = []
    if thetas is None:
        thetas = np.linspace(0, np.pi, 91)
    if phis is None:
        phis = np.linspace(-np.pi, np.pi, 181)
    if ports is None:
        ports = list(dict.fromkeys(pm.port_number for pm in field.port_modes))
        if field._active_ports is not None:
            ports = [p for p in ports if p in field._active_ports]

    T, P = np.meshgrid(thetas, phis, indexing="ij")
    theta, phi = T.flatten(), P.flatten()

    surface = field._integration_surface(faces, origin)
    xs, ys, zs = surface.exyz
    mapping = field.basis.interpolate_index(xs.flatten(), ys.flatten(), zs.flatten(), usenan=True)

    excitation = field.excitation
    ehfields = []
    try:
        for p in ports:
            field.excite_port(p)
            ehfields.append(field.interpolate(xs, ys, zs, mapping=mapping))
    finally:
        field.excitation = excitation

    Effs, Hffs, wns = stratton_chu_multi(
        [eh.E for eh in ehfields], [eh.H for eh in ehfields], surface, theta, phi, field.k0
    )
    Ptots = [np.sum(eh.Smx * wns[0, :] + eh.Smy * wns[1, :] + eh.Smz * wns[2, :]).real for eh in ehfields]

    if len(syms) > 0:
        # every mirror image is one more surface with the same far field directions
        mirrors = [list(_mirrored_sources(surface, eh, syms)) for eh in ehfields]
        for m in range(len(mirrors[0])):
            surf = mirrors[0][m][0]
            E2s, H2s, _ = stratton_chu_multi(
                [mir[m][1] for mir in mirrors], [mir[m][2] for mir in mirrors], surf, theta, phi, field.k0
            )
            Effs = [E + E2 for E, E2 in zip(Effs, E2s)]
            Hffs = [H + H2 for H, H2 in zip(Hffs, H2s)]
        # same scaling as MWField._farfield
        factor = _symmetry_factor(syms)
        Effs = [E * factor for E in Effs]
        Hffs = [H * factor for H in Hffs]
        Ptots = [Pt * factor**2 for Pt in Ptots]

    out = dict()
    for p, E, H, Pt in zip(ports, Effs, Hffs, Ptots):
        out[p] = EHFieldFF(E.reshape((3,) + T.shape), H.reshape((3,) + T.shape), T, P, Pt, freq=field.freq)
    return out
