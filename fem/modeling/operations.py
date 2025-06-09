from typing import TypeVar
from ..geo3d import GMSHSurface, GMSHVolume
from ..cs import CoordinateSystem, GCS
import gmsh
import numpy as np

T = TypeVar('T', GMSHSurface, GMSHVolume)

def add(main: T, tool: T, 
             remove_object: bool = True,
             remove_tool: bool = True) -> T:
    ''' Adds two GMSH objects together, returning a new object that is the union of the two.
    
    Parameters
    ----------
    main : GMSHSurface | GMSHVolume
    tool : GMSHSurface | GMSHVolume
    remove_object : bool, optional
        If True, the main object will be removed from the model after the operation. Default is True.
    remove_tool : bool, optional
        If True, the tool object will be removed from the model after the operation. Default is True.
    
    Returns
    -------
    GMSHSurface | GMSHVolume
        A new object that is the union of the main and tool objects.
    '''
    out_dim_tags, out_dim_tags_map = gmsh.model.occ.fuse(main.dimtags, tool.dimtags, removeObject=remove_object, removeTool=remove_tool)
    
    if out_dim_tags[0][0] == 3:
        return GMSHVolume([dt[1] for dt in out_dim_tags])
    elif out_dim_tags[0][0] == 2:
        return GMSHSurface([dt[1] for dt in out_dim_tags])

def remove(main: T, other: T, 
             remove_object: bool = True,
             remove_tool: bool = True) -> T:
    ''' Subtractes a tool object GMSH from the main object, returning a new object that is the difference of the two.
    
    Parameters
    ----------
    main : GMSHSurface | GMSHVolume
    tool : GMSHSurface | GMSHVolume
    remove_object : bool, optional
        If True, the main object will be removed from the model after the operation. Default is True.
    remove_tool : bool, optional
        If True, the tool object will be removed from the model after the operation. Default is True.
    
    Returns
    -------
    GMSHSurface | GMSHVolume
        A new object that is the difference of the main and tool objects.
    '''
    out_dim_tags, out_dim_tags_map = gmsh.model.occ.cut(main.dimtags, other.dimtags, removeObject=remove_object, removeTool=remove_tool)
    if out_dim_tags[0][0] == 3:
        return GMSHVolume([dt[1] for dt in out_dim_tags])
    elif out_dim_tags[0][0] == 2:
        return GMSHSurface([dt[1] for dt in out_dim_tags])

subtract = remove

def embed(main: GMSHVolume, other: GMSHSurface) -> None:
    ''' Embeds a surface into a volume in the GMSH model.
    Parameters
    ----------
    main : GMSHVolume
        The volume into which the surface will be embedded.
    other : GMSHSurface
        The surface to be embedded into the volume.
    
    Returns
    -------
    None
    '''
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(other.dim, [other.tag,], main.dim, main.tags)

def rotate(main: GMSHVolume, 
           c0: tuple[float, float, float],
           ax: tuple[float, float, float],
           angle: float,
           degree=True) -> GMSHVolume:
    """Rotates a GMSHVolume object around an axist defined at a coordinate.

    Args:
        main (GMSHVolume): The object to rotate
        c0 (tuple[float, float, float]): The point of origin for the rotation axis
        ax (tuple[float, float, float]): A vector defining the rotation axis
        angle (float): The angle in degrees (if degree is True)
        degree (bool, optional): Whether to interpret the angle in degrees. Defaults to True.

    Returns:
        GMSHVolume: The rotated GMSHVolume object.
    """
    if degree:
        angle = angle * np.pi/180
    gmsh.model.occ.rotate(main.dimtags, *c0, *ax, -angle)
    return main

def translate(main: GMSHVolume,
              dx: float = 0,
              dy: float = 0,
              dz: float = 0) -> GMSHVolume:
    """Translates the GMSHVolume object along a given displacement

    Args:
        main (GMSHVolume): The object to translate
        dx (float, optional): The X-displacement in meters. Defaults to 0.
        dy (float, optional): The Y-displacement in meters. Defaults to 0.
        dz (float, optional): The Z-displacement in meters. Defaults to 0.

    Returns:
        GMSHVolume: The translated object
    """
    gmsh.model.occ.translate(main.dimtags, dx, dy, dz)
    return main

def mirror(main: GMSHVolume,
           origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
           direction: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> GMSHVolume:
    """Mirrors a GMSHVolume object along a miror plane defined by a direction originating at a point

    Args:
        main (GMSHVolume): The object to mirror
        origin (tuple[float, float, float], optional): The point of origin in meters. Defaults to (0.0, 0.0, 0.0).
        direction (tuple[float, float, float], optional): The normal axis defining the plane of reflection. Defaults to (1.0, 0.0, 0.0).

    Returns:
        GMSHVolume: The mirrored GMSHVolume object
    """
    a, b, c = direction
    x0, y0, z0 = origin
    d = -(a*x0 + b*y0 + c*z0)
    if (a==0) and (b==0) and (c==0):
        return main
    gmsh.model.occ.mirror(main.dimtags, a,b,c,d)
    return main

def change_coordinate_system(main: GMSHVolume,
                             new_cs: CoordinateSystem = GCS,
                             old_cs: CoordinateSystem = GCS):
    """Moves the GMSHVolume object from a current coordinate system to a new one.

    The old and new coordinate system by default are the global coordinate system.
    Thus only one needs to be provided to transform to and from these coordinate systems.

    Args:
        main (GMSHVolume): The object to transform
        new_cs (CoordinateSystem): The new coordinate system. Defaults to GCS
        old_cs (CoordinateSystem, optional): The old coordinate system. Defaults to GCS.

    Returns:
        _type_: _description_
    """
    if new_cs._is_global and old_cs._is_global:
        return main
    # Transform to the global coordinate system.
    if not old_cs._is_global:
        gmsh.model.occ.affine_transform(main.dimtags, old_cs.affine_to_global().flatten()[:12])
    # Transform to a new coordinate system.
    if not new_cs._is_global:
        gmsh.model.occ.affineTransform(main.dimtags, new_cs.affine_from_global().flatten()[:12])
    return main