from ._emerge.physics.microwave.microwave_data import MWData
from ._emerge.selection import FaceSelection
from ._emerge.geometry import GeoSurface
from types import SimpleNamespace
import numpy as np
from typing import Literal

def surface_model(mwdata: MWData, farfield_boundary: FaceSelection | GeoSurface | None = None) -> dict:
    """Generate a dataset that Optycal can interpret to form a surface model for Near and Farfield calculations

    Args:
        dataset (MWData): _description_
        farfield_boundary (FaceSelection | GeoSurface | None, optional): _description_. Defaults to None.

    Returns:
        SimpleNamespace: _description_
    """
    dataset = dict(
        solutions=[],
        vertices=None,
        triangles=None,
        normals=None
    )
    for field in mwdata.field.iter():

        basis = field.basis
        mesh = basis.mesh

        data_out = dict()
        data_out['freq'] = field.freq
        data_out['k0'] = field.k0
        excitations = []
        for portmode in field.port_modes:
            excitations.append(dict(
                port_number=portmode.port_number,
                mode_number=portmode.mode_number,
                smat_index=portmode.smat_index,
                Z0=portmode.Z0,
                beta=portmode.beta
            ))
        n_excitations = len(excitations)
        data_out['excitations'] = excitations
        data_out['n_excitations'] = n_excitations

        if farfield_boundary is None:
            tags = mesh.exterior_face_tags
        else:
            tags = farfield_boundary.tags

        center = np.mean(mesh.nodes, axis=1).squeeze()
        surface = mesh.boundary_surface(tags, False, center)

        fields = []
        for i in range(n_excitations):
            excitation = dict()
            excitations = [0.0 + 0.0j for _ in range(n_excitations)]
            excitations[i] = 1.0 + 0.0j
            EHfield = field.interpolate(*surface.exyz)
            vertices = surface.nodes
            triangles = surface.tris
            normals = surface.normals
            E = EHfield.E
            H = EHfield.H
            excitation['E'] = E
            excitation['H'] = H
            fields.append(excitation)
        data_out['fields'] = fields
        dataset['solutions'].append(data_out)
    dataset['vertices'] = vertices
    dataset['triangles'] = triangles
    dataset['normals'] = normals
    return dataset

def farfield_model(mwdata: MWData, 
                   farfield_boundary: FaceSelection | GeoSurface | None = None, 
                   dangle_deg: float = 1.0,
                   syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,) -> dict:
    
    """Generate a dataset that Optycal can interpret to form a farfield model for Near and Farfield calculations

    Args:
        dataset (MWData): _description_
        farfield_boundary (FaceSelection | GeoSurface | None, optional): _description_. Defaults to None.

    Returns:
        SimpleNamespace: _description_
    """

    dangle_rad = dangle_deg * np.pi/180
    thetas = np.linspace(0, np.pi, int(np.pi/(dangle_rad))+1)
    phis = np.linspace(-np.pi, np.pi, int(2*np.pi/(dangle_rad))+1)
    dataset = dict(
        solutions=[],
        thetas=thetas,
        phis=phis
    )

    for field in mwdata.field.iter():
        data_out = dict()
        data_out['freq'] = field.freq
        data_out['k0'] = field.k0
        excitations = []
        for portmode in field.port_modes:
            excitations.append(dict(
                port_number=portmode.port_number,
                mode_number=portmode.mode_number,
                smat_index=portmode.smat_index,
                Z0=portmode.Z0,
                beta=portmode.beta
            ))
        n_excitations = len(excitations)
        data_out['excitations'] = excitations
        data_out['n_excitations'] = n_excitations

        fields = []
        for i in range(n_excitations):
            farfield = dict()
            excitations = [0.0 + 0.0j for _ in range(n_excitations)]
            excitations[i] = 1.0 + 0.0j
            field.set_excitations(*excitations)
            ffdata = field.farfield_3d(farfield_boundary, thetas, phis, syms=syms)
            
            farfield['E'] = ffdata.E.F
            farfield['H'] = ffdata.H.F
            fields.append(farfield)
        data_out['fields'] = fields
        dataset['solutions'].append(data_out)
    return dataset