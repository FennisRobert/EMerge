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

from typing import TypeVar, overload
from ..geometry import GeoSurface, GeoVolume, GeoObject, GeoPoint, GeoEdge, GeoPolygon
from ..cs import CoordinateSystem, GCS
import gmsh
import numpy as np

T = TypeVar('T', GeoSurface, GeoVolume, GeoObject, GeoPoint, GeoEdge)

def _gen_mapping(obj_in, obj_out) -> dict:
    tag_mapping: dict[int, dict] = {0: dict(),
                                        1: dict(),
                                        2: dict(),
                                        3: dict()}
    for domain, mapping in zip(obj_in, obj_out):
        tag_mapping[domain[0]][domain[1]] = [o[1] for o in mapping]
    return tag_mapping

def add(main: T, tool: T, 
             remove_object: bool = True,
             remove_tool: bool = True) -> T:
    ''' Adds two GMSH objects together, returning a new object that is the union of the two.
    
    Parameters
    ----------
    main : GeoSurface | GeoVolume
    tool : GeoSurface | GeoVolume
    remove_object : bool, optional
        If True, the main object will be removed from the model after the operation. Default is True.
    remove_tool : bool, optional
        If True, the tool object will be removed from the model after the operation. Default is True.
    
    Returns
    -------
    GeoSurface | GeoVolume
        A new object that is the union of the main and tool objects.
    '''
    out_dim_tags, out_dim_tags_map = gmsh.model.occ.fuse(main.dimtags, tool.dimtags, removeObject=remove_object, removeTool=remove_tool)
    if out_dim_tags[0][0] == 3:
        output = GeoVolume([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    elif out_dim_tags[0][0] == 2:
        output = GeoSurface([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    if remove_object:
        main._exists = False
    if remove_tool:
        tool._exists = False
    return output.set_material(main.material) # type: ignore

def remove(main: T, tool: T, 
             remove_object: bool = True,
             remove_tool: bool = True) -> T:
    ''' Subtractes a tool object GMSH from the main object, returning a new object that is the difference of the two.
    
    Parameters
    ----------
    main : GeoSurface | GeoVolume
    tool : GeoSurface | GeoVolume
    remove_object : bool, optional
        If True, the main object will be removed from the model after the operation. Default is True.
    remove_tool : bool, optional
        If True, the tool object will be removed from the model after the operation. Default is True.
    
    Returns
    -------
    GeoSurface | GeoVolume
        A new object that is the difference of the main and tool objects.
    '''
    out_dim_tags, out_dim_tags_map = gmsh.model.occ.cut(main.dimtags, tool.dimtags, removeObject=remove_object, removeTool=remove_tool)
    if out_dim_tags[0][0] == 3:
        output = GeoVolume([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    elif out_dim_tags[0][0] == 2:
        output = GeoSurface([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    if remove_object:
        main._exists = False
    if remove_tool:
        tool._exists = False
    return output.set_material(main.material) # type: ignore

subtract = remove

def intersect(main: T, tool: T, 
             remove_object: bool = True,
             remove_tool: bool = True) -> T:
    ''' Intersection of a tool object GMSH with the main object, returning a new object that is the intersection of the two.
    
    Parameters
    ----------
    main : GeoSurface | GeoVolume
    tool : GeoSurface | GeoVolume
    remove_object : bool, optional
        If True, the main object will be removed from the model after the operation. Default is True.
    remove_tool : bool, optional
        If True, the tool object will be removed from the model after the operation. Default is True.
    
    Returns
    -------
    GeoSurface | GeoVolume
        A new object that is the difference of the main and tool objects.
    '''
    out_dim_tags, out_dim_tags_map = gmsh.model.occ.intersect(main.dimtags, tool.dimtags, removeObject=remove_object, removeTool=remove_tool)
    if out_dim_tags[0][0] == 3:
        output = GeoVolume([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    elif out_dim_tags[0][0] == 2:
        output = GeoSurface([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    if remove_object:
        main._exists = False
    if remove_tool:
        tool._exists = False
    return output.set_material(main.material) #type:ignore

def embed(main: GeoVolume, other: GeoSurface) -> None:
    ''' Embeds a surface into a volume in the GMSH model.
    Parameters
    ----------
    main : GeoVolume
        The volume into which the surface will be embedded.
    other : GeoSurface
        The surface to be embedded into the volume.
    
    Returns
    -------
    None
    '''
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(other.dim, other.tags, main.dim, main.tags)

def rotate(main: GeoVolume, 
           c0: tuple[float, float, float],
           ax: tuple[float, float, float],
           angle: float,
           make_copy: bool = False,
           degree=True) -> GeoObject:
    """Rotates a GeoVolume object around an axist defined at a coordinate.

    Args:
        main (GeoVolume): The object to rotate
        c0 (tuple[float, float, float]): The point of origin for the rotation axis
        ax (tuple[float, float, float]): A vector defining the rotation axis
        angle (float): The angle in degrees (if degree is True)
        degree (bool, optional): Whether to interpret the angle in degrees. Defaults to True.

    Returns:
        GeoVolume: The rotated GeoVolume object.
    """
    if degree:
        angle = angle * np.pi/180
    
    if make_copy:
        rotate_obj = main.make_copy()
    else:
        rotate_obj = main
     
    gmsh.model.occ.rotate(rotate_obj.dimtags, *c0, *ax, -angle)
    # Rotate the facepointers
    for fp in rotate_obj._all_pointers:
        fp.rotate(c0, ax, angle)
    return rotate_obj

def translate(main: GeoVolume,
              dx: float = 0,
              dy: float = 0,
              dz: float = 0,
              make_copy: bool = False) -> GeoObject:
    """Translates the GeoVolume object along a given displacement

    Args:
        main (GeoVolume): The object to translate
        dx (float, optional): The X-displacement in meters. Defaults to 0.
        dy (float, optional): The Y-displacement in meters. Defaults to 0.
        dz (float, optional): The Z-displacement in meters. Defaults to 0.
        make_copy (bool, optional): Whether to make a copy first before translating.

    Returns:
        GeoObject: The translated object
    """
    
    if make_copy:
        trans_obj = main.make_copy()
    else:
        trans_obj = main
    gmsh.model.occ.translate(trans_obj.dimtags, dx, dy, dz)
    
     # Rotate the facepointers
    for fp in trans_obj._all_pointers:
        fp.translate(dx, dy, dz)

    return trans_obj

def mirror(main: GeoObject,
           origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
           direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
           make_copy: bool = True) -> GeoObject:
    """Mirrors a GeoVolume object along a miror plane defined by a direction originating at a point

    Args:
        main (GeoVolume): The object to mirror
        origin (tuple[float, float, float], optional): The point of origin in meters. Defaults to (0.0, 0.0, 0.0).
        direction (tuple[float, float, float], optional): The normal axis defining the plane of reflection. Defaults to (1.0, 0.0, 0.0).
        make_copy (bool, optional): Whether to make a copy first before mirroring.

    Returns:
        GeoVolume: The mirrored GeoVolume object
    """
    origin = np.array(origin)
    direction = np.array(direction)

    a, b, c = direction
    x0, y0, z0 = origin
    d = -(a*x0 + b*y0 + c*z0)
    if (a==0) and (b==0) and (c==0):
        return main
    
    mirror_obj = main
    if make_copy:
        mirror_obj = main.make_copy()
    gmsh.model.occ.mirror(mirror_obj.dimtags, a,b,c,d)
    
    for fp in mirror_obj._all_pointers:
        fp.mirror(origin, direction)
    return mirror_obj

def change_coordinate_system(main: GeoObject,
                             new_cs: CoordinateSystem = GCS,
                             old_cs: CoordinateSystem = GCS) -> GeoObject:
    """Moves the GeoVolume object from a current coordinate system to a new one.

    The old and new coordinate system by default are the global coordinate system.
    Thus only one needs to be provided to transform to and from these coordinate systems.

    Args:
        main (GeoVolume): The object to transform
        new_cs (CoordinateSystem): The new coordinate system. Defaults to GCS
        old_cs (CoordinateSystem, optional): The old coordinate system. Defaults to GCS.

    Returns:
        GeoObject: The output object
    """
    if new_cs._is_global and old_cs._is_global:
        return main
    
    M1 = old_cs.affine_to_global()
    M2 = new_cs.affine_from_global()
    # Transform to the global coordinate system.
    if not old_cs._is_global:
        gmsh.model.occ.affine_transform(main.dimtags, M1.flatten()[:12])
    # Transform to a new coordinate system.
    if not new_cs._is_global:
        gmsh.model.occ.affineTransform(main.dimtags, M2.flatten()[:12])

    for fp in main._all_pointers:
        fp.affine_transform(M1)
        fp.affine_transform(M2)
    return main

@overload
def unite(*objects: GeoVolume) -> GeoVolume: ...

@overload
def unite(*objects: GeoSurface) -> GeoSurface: ...

@overload
def unite(*objects: GeoEdge) -> GeoEdge: ...

@overload
def unite(*objects: GeoPolygon) -> GeoSurface: ...

def unite(*objects: GeoObject) -> GeoObject:
    """Applies a fusion consisting of all geometries in the argument.

    Returns:
        GeoObject: The resultant object
    """
    main, *rest = objects
    if not rest:
        return main
    main._exists = False
    dts = []
    for other in rest:
        dts.extend(other.dimtags)
        other._exists = False
    
    new_dimtags, mapping = gmsh.model.occ.fuse(main.dimtags, dts)
    
    new_obj = GeoObject.from_dimtags(new_dimtags)._take_tools(*objects)
    new_obj.set_material(main.material)
    new_obj.prio_set(main._priority)
    
    return new_obj

def expand_surface(surface: GeoSurface, distance: float) -> GeoSurface:
    """EXPERIMENTAL: Expands an input surface. The surface must exist on a 2D plane.
    
    The output surface does not inherit material properties.
    
    If any problems occur, reach out through email.

    Args:
        surface (GeoSurface): The input surface to expand
        distance (float): The exapansion distance

    Returns:
        GeoSurface: The output surface
    """
    surfs = [] 
    for tag in surface.tags:
        looptags, _ = gmsh.model.occ.get_curve_loops(tag)
        new_curves = []
        for looptag in looptags:
            curve_tags = gmsh.model.occ.offset_curve(looptag, distance)
            loop_tag = gmsh.model.occ.addCurveLoop([t for d,t in curve_tags])
            new_curves.append(loop_tag)
        surftag = gmsh.model.occ.addPlaneSurface(new_curves)
        surfs.append(surftag)
    surf = GeoSurface(surfs)
    return surf