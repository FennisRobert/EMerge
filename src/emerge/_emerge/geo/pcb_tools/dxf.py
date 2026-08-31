
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
import re
from typing import Any
from ..pcb import PCBNew, RouteException
from ...selection import Selection
from ...geometry import GeoObject
from emsutil import Material, PEC
from ...cs import GCS, CoordinateSystem
from loguru import logger
from math import hypot
from ...simmodel import Simulation
from ...mth.loopsplit import Loop
try: 
    import ezdxf
    from ezdxf.recover import readfile as recover_readfile
    from ezdxf.path import make_path, from_hatch
except ImportError as e:
    logger.error('Cannot find the required ezdxf library. Install using: pip install ezdxf')
    raise e
import numpy as np

INSUNITS_TO_NAME = {
    0: "unitless",
    1: "inch",
    2: "foot",
    3: "mile",
    4: "millimeter",
    5: "centimeter",
    6: "meter",
    7: "kilometer",
    8: "microinch",
    9: "mil",          # thousandth of an inch
    10: "yard",
    11: "angstrom",
    12: "nanometer",
    13: "micrometer",
    14: "decimeter",
    15: "decameter",
    16: "hectometer",
    17: "gigameter",
    18: "astronomical unit",
    19: "light year",
    20: "parsec",
}

# scale factor: drawing units -> millimeters
INSUNITS_TO_MM = {
    0: 1.0,           # unitless; treat as mm by convention
    1: 25.4,
    2: 304.8,
    3: 1609344.0,
    4: 1.0,
    5: 10.0,
    6: 1000.0,
    7: 1_000_000.0,
    8: 25.4e-6,
    9: 0.0254,
    10: 914.4,
    11: 1e-7,
    12: 1e-6,
    13: 1e-3,
    14: 100.0,
    15: 10_000.0,
    16: 100_000.0,
    17: 1e12,
    18: 1.495978707e14,     # AU in mm
    19: 9.460730472e18,     # ly in mm
    20: 3.085677581e19,     # pc in mm
}

def cluster_values(values, tol):
    """Return [(center, count), ...] clustering sorted values by tolerance."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    out = []
    for c in clusters:
        out.append((sum(c)/len(c), len(c)))
    return out

STACKUP_TOKENS = [
    r"fr-?4", r"rogers ?\d+", r"core", r"prepreg", r"dielectric",
    r"cu(?:\W|$)|copper", r"\b1\.?6\s*mm\b", r"\b0\.?8\s*mm\b",
    r"\b[12]\s*oz\b", r"\b35\s*µ?m\b", r"\b70\s*µ?m\b",
    r"stack[- ]?up", r"layer\s*\d+", r"pcb", r"thickness",
]

def inspect_pcb_from_dxf(
    filename: str,
    flatten_tol: float = 0.25,    # curve flattening tolerance (drawing units)
    z_tol: float | None = None,   # cluster tolerance for Z (drawing units); default auto
    layer_filter: str | None = None,
):
    """
    Returns a dict with:
      {
        'units': {'code': int, 'name': str, 'to_mm': float},
        'z_levels': [{'z': float, 'count': int}],           # in drawing units
        'thickness_units': 'same as units',
        'thickness': float,                                  # in drawing units
        'thickness_mm': float,                               # convenience
        'notes': {'materials': [...], 'raw_text_hits': [...]}
      }
    """
    doc, auditor = recover_readfile(filename)
    msp = doc.modelspace()

    # Units
    code = int(doc.header.get("$INSUNITS", 0))
    unit_name = INSUNITS_TO_NAME.get(code, f"unknown({code})")
    to_mm = INSUNITS_TO_MM.get(code, 1.0)

    # Gather geometry Zs (WCS), also consider entity.dxf.elevation as fallback
    z_values = []

    def on_layer(e) -> bool:
        return layer_filter is None or e.dxf.layer == layer_filter

    for e in msp:
        if not on_layer(e):
            continue
        dxtype = e.dxftype()
        paths = []
        try:
            if dxtype in ("HATCH", "MPOLYGON"):
                paths = list(from_hatch(e))
            else:
                paths = [make_path(e)]
        except TypeError:
            # skip unsupported entity types
            pass
        except Exception:
            pass

        for p in paths:
            subs = p.sub_paths() if getattr(p, "has_sub_paths", False) and p.has_sub_paths else [p]
            for sp in subs:
                try:
                    verts = list(sp.flattening(distance=flatten_tol))
                except Exception:
                    continue
                for v in verts:
                    # v may be Vec2 or Vec3; handle both
                    z = getattr(v, "z", None)
                    if z is None:
                        # fallback to entity elevation if available
                        z = float(getattr(e.dxf, "elevation", 0.0))
                    z_values.append(float(z))

        # Entities with explicit THICKNESS (rare but possible)
        thick = getattr(e.dxf, "thickness", None)
        if thick not in (None, 0):
            # If an entity is extruded, its "top" would be at elevation+thickness along normal.
            # We can't reliably map OCS normal here; just record the magnitude as a hint.
            pass

    # Cluster Zs
    auto_tol = (1e-6 if to_mm >= 1.0 else 1e-6 / to_mm)  # ~1 nm in mm space → tiny in most units
    ztol = z_tol if z_tol is not None else auto_tol
    z_clusters = cluster_values(z_values, tol=ztol)
    z_levels = [{"z": z, "count": n} for z, n in z_clusters]

    # Thickness from geometry (only meaningful if multiple distinct Zs)
    if z_values:
        zmin, zmax = min(z_values), max(z_values)
        thickness = zmax - zmin
    else:
        zmin = zmax = thickness = 0.0

    # Parse material/thickness hints from TEXT/MTEXT
    material_hits = set()
    raw_text_hits = []
    token_re = re.compile("|".join(STACKUP_TOKENS), re.IGNORECASE)

    for e in msp.query("TEXT MTEXT"):
        try:
            text = e.dxf.text if e.dxftype() == "TEXT" else e.text
        except Exception:
            continue
        if not text:
            continue
        if token_re.search(text):
            raw_text_hits.append(text.strip())
            # simple keyword extraction
            for kw in ["FR4", "Rogers", "core", "prepreg", "copper", "stackup", "thickness", "oz", "µm", "um", "mm"]:
                if re.search(kw, text, re.IGNORECASE):
                    material_hits.add(kw.upper())

    return {
        "units": {"code": code, "name": unit_name, "to_mm": to_mm},
        "z_levels": z_levels,
        "thickness_units": unit_name,
        "thickness": thickness,
        "thickness_mm": thickness * to_mm,
        "notes": {
            "materials": sorted(material_hits),
            "raw_text_hits": raw_text_hits[:50],  # cap to keep output sane
            "comment": (
                "Single Z level detected; geometry alone cannot determine PCB thickness."
                if len(z_levels) <= 1 else
                "Thickness estimated from min/max Z across all geometry."
            )
        },
    }

def path_items_with_semantics(e, p, distance=0.25):
    """Yield dicts describing each sub-path with geometry + semantics we can infer."""
    subs = p.sub_paths() if getattr(p, "has_sub_paths", False) and p.has_sub_paths else [p]
    for sp in subs:
        verts = list(sp.flattening(distance=distance))
        if len(verts) < 2:
            continue

        # geometry in XY (keep Z if you like)
        pts = [(v.x, v.y) for v in verts]
        is_closed = pts[0] == pts[-1]

        item = {
            "layer": e.dxf.layer,
            "entity": e.dxftype(),
            "handle": e.dxf.handle,
            "is_closed": is_closed,
            "points": pts if is_closed else None,       # polygon ring for closed shapes
            "start": None,
            "end": None,
            "length": None,                             # centerline length for open paths
            "width": None,                              # LWPOLYLINE width if available
        }

        if not is_closed:
            item["start"] = pts[0]
            item["end"]   = pts[-1]
            # polyline centerline length
            item["length"] = sum(
                hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(pts, pts[1:])
            )

        # Preserve potential trace width info from LWPOLYLINE
        if e.dxftype() == "LWPOLYLINE":
            cw = getattr(e.dxf, "const_width", None)
            if cw not in (None, 0):
                item["width"] = float(cw)
            else:
                # per-vertex widths (rare but possible)
                try:
                    item["widths"] = [(v[0], v[1]) for v in zip(e.get_start_widths(), e.get_end_widths())]
                except Exception:
                    pass

        yield item
        
def extract_polygons_with_meta(
    filename: str,
    layer: str | None = None,
    distance: float = 0.25,
    skip_open: bool = True,
) -> dict[str, Any]:
    """
    Returns a list of dicts:
      {
        'ring': [(x, y), ...],         # closed ring
        'layer': 'LayerName',
        'z': 12.34,                    # representative height (median Z of ring)
        'entity': 'LWPOLYLINE' | ...,
        'handle': 'ABCD'
      }
    """
    doc, auditor = recover_readfile(filename)
    msp = doc.modelspace()

    def on_layer(e) -> bool:
        return layer is None or e.dxf.layer == layer

    items = []

    for e in msp:
        if not on_layer(e):
            continue

        dxtype = e.dxftype()
        paths = []
        try:
            if dxtype in ("HATCH", "MPOLYGON"):
                paths = list(from_hatch(e))
            else:
                paths = [make_path(e)]
        except TypeError:
            continue
        except Exception:
            continue

        for p in paths:
            subs = p.sub_paths() if p.has_sub_paths else [p]
            for sp in subs:
                
                verts = list(sp.flattening(distance=distance))
                if len(verts) < 2:
                    continue

                # ensure not closed
                if (verts[0].x, verts[0].y) == (verts[-1].x, verts[-1].y):
                    verts = verts[:-1]

                ring_xy = [(v.x, v.y) for v in verts]
                zs = sorted(v.z for v in verts)
                z = zs[len(zs)//2] if zs else float(getattr(e.dxf, "elevation", 0.0))  # median Z

                items.append({
                    "ring": ring_xy,
                    "layer": e.dxf.layer,
                    "z": float(z),
                    "entity": dxtype,
                    "handle": e.dxf.handle,
                })

    return items

def import_dxf(filename: str, 
               material: Material, 
               thickness: float | None = None,
               unit: float | None = None,
               cs: CoordinateSystem | None = GCS,
               trace_material: Material = PEC,
               split: bool = True) -> PCBNew:
    """Imports a DXF file as PCBNew object

    Args:
        filename (str): The DXF Filename
        material (Material): The material to use for the PCB
        thickness (float | None, optional): The thickness to use for the PCB. Defaults to None.
        unit (float | None, optional): The pcb unit. Defaults to None.
        cs (CoordinateSystem | None, optional): The coordinate system. Defaults to GCS.
        trace_material (Material, optional): The material for traces. Defaults to PEC.
        split (bool, optional): If a loop splitting algorithm is to be used. Defaults to True.

    Raises:
        RouteException: _description_
        RouteException: _description_

    Returns:
        PCB: _description_
    """
    polies = extract_polygons_with_meta(filename)
    prop = inspect_pcb_from_dxf(filename)
    
    if prop['units']['name'] == 'unitless':
        if unit is None:
            raise RouteException('Cannot generate PCB because the unit is not found in the DXF file or provided in the import_dxf function.')
        pcb_unit = unit
    else:
        pcb_unit = 0.001 * prop['units']['to_mm']
    
    if prop['thickness'] == 0.0:
        if thickness is None:
            raise RouteException('Cannot generate PCB because no thickness is found int he DXF file and none is provided in the import_dxf function.')
        pcb_thickness = thickness
    else:
        pcb_thickness = 0.001 * prop['thickness_mm'] / pcb_unit
    
    if cs is None:
        cs = GCS
    
    zs = sorted(list(set([pol['z'] for pol in polies])))
    pcb = PCBNew(pcb_thickness, pcb_unit, cs, material=material, trace_material=trace_material)
    
    for poly in polies:
        xs, ys = zip(*poly['ring'])
        z = poly['z']
        zs.append(z)
        xs = [x for x in xs]
        ys = [y for y in ys]
        loop = Loop(xs, ys)
        
        if split:
            to_add, to_remove = loop.split()
            for i, (xs, ys) in enumerate(to_add):
                pcb.add_poly(xs, ys, z=z, name=poly['handle']+f'_{i}')
            for i, (xs, ys) in enumerate(to_remove):
                pcb.add_hole(xs, ys, z=z, name=poly['handle']+f'_remove_{i}')
        else:
            xs = list(loop.x)
            ys = list(loop.y)
            pcb.add_poly(xs, ys, z, name=poly['handle'])
            
    return pcb

def _extract_polygons(nodes: np.ndarray, tris: np.ndarray, tri_ids: np.ndarray) -> list[list[tuple[float, float]]]:
    from collections import defaultdict
    edge_counter = defaultdict(int)
    
    nT = tri_ids.shape[1]
    for it in range(nT):
        i1, i2, i3 = [int(x) for x in tri_ids[:,it]]
        edge_counter[(min(i1,i2), max(i1,i2))] += 1
        edge_counter[(min(i1,i3), max(i1,i3))] += 1
        edge_counter[(min(i2,i3), max(i2,i3))] += 1

    edges: list[tuple[int, int]] = [edge for edge, counter in edge_counter.items() if counter == 1]
    
    node_sequence = defaultdict(list)
    
    for i1, i2 in edges:
        node_sequence[i1].append(i2)
        node_sequence[i2].append(i1)
    
    islands = []
    
    while True:
        first_node = node_sequence.keys()
        if not first_node:
            break
        first_node = list(first_node)[0]
        second_node = node_sequence[first_node][0]
        
        loop = [first_node, second_node]
        
        while True:
            next_nodes = node_sequence[second_node]
            next_node = next_nodes[0] if next_nodes[0] != first_node else next_nodes[1]
            if next_node == loop[0]:
                break
            loop.append(next_node)
            first_node = second_node
            second_node = next_node
        
        islands.append(loop)
        for node in loop:
            del node_sequence[node]
    
    polygons = []
    
    for loop in islands:
        poly = []
        for node in loop:
            x = nodes[0, node]
            y = nodes[1, node]
            poly.append((x,y))
        
        # remove last point if its within 1e-8 of the first
        if hypot(poly[0][0]-poly[-1][0], poly[0][1]-poly[-1][1]) < 1e-8:
            poly = poly[:-1]    
        
        # remove all collinear points
        simplified_poly = []
        nP = len(poly)
        for i in range(nP):
            p_prev = poly[i-1]
            p_curr = poly[i]
            p_next = poly[(i+1)%nP]
            
            # vector from prev to curr
            v1 = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1])
            # vector from curr to next
            v2 = (p_next[0]-p_curr[0], p_next[1]-p_curr[1])
            
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if abs(cross) > 1e-10:
                simplified_poly.append(p_curr)
        
        simplified_poly.append(simplified_poly[0])  # close the polygon
        polygons.append(simplified_poly)
    
    return polygons

def _export_dxf_single(simulation: Simulation, filename: str, z_height: float, selection: Selection | GeoObject | list[Selection | GeoObject] | None = None) -> None:
    
    if selection is None:
        selection = []
    elif isinstance(selection, (Selection, GeoObject)):
        selection = [selection]
    
    all_tags = set()
    
    mesh = simulation.mesh
    
    for key in mesh.ftag_to_tri.keys():
        all_tags.add(key)
    
    
    if len(selection) > 0:
        selected_tags = set()
        for sel in selection:
            selected_tags.update(sel.tags)
        # intersect with all_tags
        selected_tags = selected_tags.intersection(all_tags)
    else:
        selected_tags = all_tags
    
    nodes = mesh.nodes
    tris = mesh.tris
    
    poly_triset = []
    for geo in simulation.all_geos():
        if geo.dim != 2:
            continue
        
        tags = [tag for tag in geo.tags if tag in selected_tags]
        if not tags:
            continue
        tri_ids = mesh.get_triangles(tags)
        
        polynodes = nodes[:,np.unique(tris[:,tri_ids])]
        
        if not np.all(np.abs(polynodes[2,:]-z_height) < 1e-8):
            continue

        poly_triset.append(tris[:,tri_ids])
        
    
    # entire tri_set is on the correct z-height
    polies: list[list] = [_extract_polygons(nodes, tris, triset) for triset in poly_triset]
    
    # Reduce polies to a single list of polygons
    final_polies = []
    for polyset in polies:
        for poly in polyset:
            
            final_polies.append(poly)
    
    if not final_polies:
        logger.warning(f'No 2D geometry found at z={z_height} to export to DXF.')
        return
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    for poly in final_polies:
        # add as solid region
        
        msp.add_lwpolyline(poly, close=True, dxfattribs={'layer': 'F.Cu'})
    # add .dxf if it is not thetere
    if not filename.lower().endswith('.dxf'):
        filename += '.dxf'
        
    doc.saveas(filename)
    return

def pcb_to_dxf(pcb: PCBNew, filename: str) -> None:
    """Exports all PCBNew traces to a DXF file.

    Args:
        PCB (PCB): Your PCB object.
        filename (str): The filename to save the DXF as.
        
    Returns:
        None
    """
    
    from collections import defaultdict
    
    unit = pcb.unit
    layers: dict[float, list] = defaultdict(list)
    for pcbpoly in pcb._poly_out:
        xs = list(np.array(pcbpoly.xs)*unit)
        ys = list(np.array(pcbpoly.ys)*unit)
        poly = [(x,y) for x,y in zip(xs, ys)]
        # make sure the last point is the same as the first
        if poly[0] != poly[-1]:
            poly.append(poly[0])
        layers[pcbpoly.layer].append(poly)
    
    for z in layers:
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()
        
        for poly in layers[z]:
            msp.add_lwpolyline(poly, close=True, dxfattribs={'layer': f'Z{z:.3f}'})
        # add .dxf if it is not there
        fname_parts = filename.rsplit('.', maxsplit=1)
        if len(fname_parts) == 2:
            fname = f'{fname_parts[0]}_z{z:.3f}.{fname_parts[1]}'
        else:
            fname = f'{filename}_layer{z}.dxf'
            
        doc.saveas(fname)

    bxs = pcb.board_outline_xs
    bys = pcb.board_outline_ys

    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    msp.add_lwpolyline([(x,y) for x,y in zip(bxs,bys)], close=True)
    fname_parts = filename.rsplit('.', maxsplit=1)
    if len(fname_parts) == 2:
        fname = f'{fname_parts[0]}_outline.{fname_parts[1]}'
    else:
        fname = f'{filename}_outline.dxf'
    doc.saveas(fname)
    
def export_dxf(simulation: Simulation, filename: str, z_height: float | list[float], selection: list[Selection | GeoObject] = None) -> None:
    """Exports all 2D Geometries on a single or multiple  z-heights to a DXF file.

    Args:
        simulation (Simulation): Your simulation object.
        filename (str): The filename to save the DXF as.
        z_height (float): The z-height to export the 2D geometries at.
        
    Returns:
        None
    """
    
    if not simulation.mesh.defined:
        raise Exception('Cannot export DXF because the simulation mesh is not defined yet. Please run generate_mesh() first.')

    if isinstance(z_height, list):
        for zh in z_height:
            fname_parts = filename.rsplit('.', maxsplit=1)
            if len(fname_parts) == 2:
                fname = f'{fname_parts[0]}_z{zh:.3f}.{fname_parts[1]}'
            else:
                fname = f'{filename}_z{zh:.3f}.dxf'
            _export_dxf_single(simulation, fname, zh, selection)
    else:
        _export_dxf_single(simulation, filename, z_height, selection)
    return