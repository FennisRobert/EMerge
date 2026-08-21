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

# Last Cleanup: 2025-03-12
from typing import TypeVar, overload, Literal
from ..geometry import GeoSurface, GeoVolume, GeoObject, GeoPoint, GeoEdge, GeoPolygon
from ..cs import CoordinateSystem, GCS, Anchor
import gmsh
import numpy as np
from emsutil import AIR

T = TypeVar('T', GeoSurface, GeoVolume, GeoObject, GeoPoint, GeoEdge)

def _tags(dimtags: list[tuple[int, int]]) -> list[int]:
    return [t for d,t in dimtags]

def _dimtags(dim: int, tags: list[int]) -> list[tuple[int, int]]:
    return [(dim,tag) for tag in tags]


def add(main: T, 
        tool: T, 
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
    
    gmsh.model.occ.synchronize()
    if out_dim_tags[0][0] == 3:
        output = GeoVolume([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    elif out_dim_tags[0][0] == 2:
        output = GeoSurface([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    
    if remove_object:
        main._exists = False
    if remove_tool:
        tool._exists = False

    if main.material is not None:
        output.set_material(main.material)
    return output # type: ignore

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
    gmsh.model.occ.synchronize()
    out_dim_tags, out_dim_tags_map = gmsh.model.occ.cut(main.dimtags, tool.dimtags, removeObject=remove_object, removeTool=remove_tool)
    
    gmsh.model.occ.synchronize()
    if out_dim_tags[0][0] == 3:
        output = GeoVolume([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    elif out_dim_tags[0][0] == 2:
        output = GeoSurface([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    if remove_object:
        main._exists = False
    if remove_tool:
        tool._exists = False

    if main.material is not None:
        output.set_material(main.material)
    return output # type: ignore

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
    
    gmsh.model.occ.synchronize()
    if out_dim_tags[0][0] == 3:
        output = GeoVolume([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    elif out_dim_tags[0][0] == 2:
        output = GeoSurface([dt[1] for dt in out_dim_tags])._take_tools(tool,main)
    if remove_object:
        main._exists = False
    if remove_tool:
        tool._exists = False

    if main.material is not None:
        output.set_material(main.material)
    return output # type: ignore

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

def rotate(main: GeoObject, 
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
    for fp in rotate_obj._all_transformable:
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
    for fp in trans_obj._all_transformable:
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
    
    for fp in mirror_obj._all_transformable:
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

    for fp in main._all_transformable:
        fp.affine_transform(M1)
        fp.affine_transform(M2)
        
    return main

def stretch(main: GeoObject, fx: float = 1, fy: float = 1, fz: float = 1, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> GeoObject:
    """Stretches a geometry with a factor fx, fy and fz along the x, y and Z axes respectively
    
    The stretch origin is centered at the provided origin.

    Returns:
        _type_: _description_
    """
    gmsh.model.occ.dilate(main.dimtags, *origin, fx, fy, fz)
    
    return main

def extrude(main: GeoSurface, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> GeoVolume:
    """Extrudes a surface entity by a displacement

    Args:
        main (GeoSurface): _description_
        dx (float): _description_
        dy (float): _description_
        dz (float): _description_

    Returns:
        GeoObject: _description_
    """
    dtout = gmsh.model.occ.extrude(main.dimtags, dx, dy, dz)
    out = [dt[1] for dt in dtout if dt[0]==3]
    obj_out = GeoVolume(out, name=f'Extrusion[{main.name}]')
    main.remove()
    gmsh.model.occ.synchronize()
    return obj_out
    

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
    gmsh.model.occ.synchronize()
    
    newname = 'Union[' + ','.join([obj.name for obj in objects]) + ']'
    new_obj = GeoObject.from_dimtags(new_dimtags)._take_tools(*objects)
    new_obj.name = newname
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

def stick(obj: GeoObject, p1: Anchor , p2: Anchor ) -> GeoObject:
    """Glues an objects face anchor to another face anchor point with the appropriate orientation.

    Args:
        object (GeoObject): The object to stick to something else
        p1 (Anchor): The main objects achor point to move
        p2 (Anchor): The anchor point to move it to

    Returns:
        GeoObject: The moved object
    """
    affine = p1.compute_affine(p2)
    gmsh.model.occ.affine_transform(obj.dimtags, affine.flatten()[:12])
    
    for fp in obj._all_transformable:
        fp.affine_transform(affine)
    
    return obj

def bounding_box() -> tuple[float, float, float, float, float, float]:
    """Compute the bounding box of the current geometry state

    Returns:
        tuple[float, float, float, float, float, float]: xmin, xmax, ymin, ymax, zmin, zmax
    """
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    zmins = []
    zmaxs = []
    gmsh.model.occ.synchronize()
    for dim in (0,1,2,3):
        for tag in gmsh.model.occ.get_entities(dim):

            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(*tag)
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            zmins.append(zmin)
            zmaxs.append(zmax)
    
    return min(xmins), max(xmaxs), min(ymins), max(ymaxs), min(zmins), max(zmaxs)


def connect_faces(face1: GeoSurface, face2: GeoSurface,
                  make_solid: bool = True,
                  make_ruled: bool = True,
                  parametrization: Literal["ChordLength", "Centripetal", "IsoParametric"] = "IsoParametric") -> GeoVolume:
    """Connect two surface objects together

    The underlaying operation requires that the points on the circumpherence of both surfaces need to be the same.

    Args:
        face1 (GeoSurface): The first surface to connect
        face2 (GeoSurface): The second surface to connect

    Returns:
        GeoVolume: _description_
    """
    wire1, _ = gmsh.model.occ.getCurveLoops(face1.tags[0])
    wire2, _ = gmsh.model.occ.getCurveLoops(face2.tags[0])
    dts = gmsh.model.occ.addThruSections([wire1[0], wire2[0]], make_solid, make_ruled, parametrization=parametrization)
    tags = [t for d,t in dts]
    return GeoVolume(tags)
    

def thick_wall(thickness: float, *volumes):
    """Add thickness to volumes and return a solid expanded geometry without internal holes."""
    gmsh.model.occ.synchronize()

    dimtags = []
    for vol in volumes:
        dimtags.extend(vol.dimtags)

    dimtags = gmsh.model.occ.copy(dimtags)

    if len(dimtags) > 1:
        union_dimtags, _ = gmsh.model.occ.fuse([dimtags[0]], dimtags[1:], removeObject=True, removeTool=True)
    else:
        union_dimtags = dimtags
    print('Entities before:', gmsh.model.occ.getEntities(3))
    print('2D Entities before:', gmsh.model.occ.getEntities(2))
    new_solids = []
    for dim, tag in union_dimtags:
        print(f'Dim = {dim}, tag = {tag}')
        # Hollow shell: outer offset surface + inner surface (the cavity)
        thick_dimtags = gmsh.model.occ.addThickSolid(tag, [], offset=thickness)
        gmsh.model.occ.synchronize()
        print(f'Thick dimtags: {thick_dimtags}')
        for d,t in thick_dimtags:
            print(f'Mass in = {gmsh.model.occ.getMass(d,t)}')
            surfloop, _ = gmsh.model.occ.getSurfaceLoops(t)
            gmsh.model.occ.remove([(d,t),])
            print(f'Surface loop = {surfloop}')
            new_volume = gmsh.model.occ.addVolume(surfloop)
            print(f'New Volume: {new_volume} with mass {gmsh.model.occ.getMass(3, new_volume)}')
            new_solids.append((3,new_volume))
    
    gmsh.model.occ.synchronize()
    
    print('Entities after:', gmsh.model.occ.getEntities(3))
    print('2D Entities before:', gmsh.model.occ.getEntities(2))
    print(new_solids, union_dimtags)
    outdts,_ = gmsh.model.occ.fuse(union_dimtags, new_solids)
    print(outdts)
    new_vol = GeoVolume.from_dimtags(outdts).set_material(AIR)
    gmsh.model.occ.synchronize()
    print('Entities final:', gmsh.model.occ.getEntities(3))
    print('2D Entities final:', gmsh.model.occ.getEntities(2))
    print(new_vol)
    return new_vol
