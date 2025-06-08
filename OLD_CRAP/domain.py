import shapely as shp
import gmsh
import numpy as np
from ..fem.material import Material, AIR
from loguru import logger

def _unpack(poly) -> list[shp.Polygon]:
    if isinstance(poly, shp.MultiPolygon):
        poly = [poly for poly in poly.geoms]
    else:
        poly = [
            poly,
        ]
    return poly


def _flatten(basepoly: shp.Polygon, otherpoly: shp.Polygon) -> list[shp.Polygon]:
    polyA = _unpack(basepoly.difference(otherpoly))
    polyB = _unpack(otherpoly.difference(basepoly))
    polyC = _unpack(basepoly.intersection(otherpoly))
    output = [
        p
        for p in polyA + polyB + polyC
        if isinstance(p, (shp.Polygon,)) and not p.is_empty
    ]
    return output

def _disjoint(basepoly: shp.Polygon, otherpoly: shp.Polygon) -> list[shp.Polygon]:
    polyA = _unpack(basepoly.difference(otherpoly))
    polyB = _unpack(otherpoly.difference(basepoly))
    polyC = _unpack(basepoly.intersection(otherpoly))
    return polyA, polyB, polyC

def _introduce(basepolys: list[shp.Polygon], otherpoly: shp.Polygon) -> list[shp.Polygon]:
    new_base_polies = []
    intersections = []
    leftover = otherpoly
    for base in basepolys:
        new_base_polies += _unpack(base.difference(otherpoly))
        intersections += _unpack(base.intersection(otherpoly))
        leftover = leftover.difference(base)
    return [p for p in new_base_polies + intersections + _unpack(leftover) if not p.is_empty and isinstance(p, shp.Polygon)]


class Point:

    def __init__(self, x: float, y: float, precision: float):
        self.x = x
        self.y = y
        self._precision = precision
        self._ix: int = int(round(self.x*self._precision))
        self._iy: int = int(round(self.y*self._precision))
        self._hash: int = hash((self._ix, self._iy))
        self.tag = None
        self.key = self._hash
        self._ds: float = None
        self._embedded: bool = False
        self._embedded_entity: Edge | Domain = None

    def __repr__(self) -> str:
        return f'Point[{self.tag}]({self.x:.3f}, {self.y:.3f})'
    
    @property
    def dimtag(self) -> tuple[int, int]:
        return (0, self.tag)
    
    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def update_ds(self, ds: float) -> None:
        if self._ds is None:
            self._ds = ds
        else:
            self._ds = min(ds, self._ds)

    def __str__(self) -> str:
        return f'Point[{self.tag}]({self.x:.3f}, {self.y:.3f})'
    
    def __hash__(self) -> int:
        return self._hash
    
    def __eq__(self, other) -> bool:
        return (self._ix == other._ix) and (self._iy == other._iy)

class Edge:

    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2
        self.tag = None
        self._hash: int = hash((min(self.p1._hash, self.p2._hash), max(self.p1._hash, self.p2._hash)))
        self.key = self._hash
        self.point_tags = (self.p1.tag, self.p2.tag)

    @property
    def dimtag(self) -> tuple[int, int]:
        return (1, self.tag)
    
    @property
    def xy(self) -> np.ndarray:
        return 0.5*(self.p1.xy + self.p2.xy)
    
    def __str__(self) -> str:
        return f'Edge[{self.tag}]({self.p1},{self.p2})'
    
    def __repr__(self) -> str:
        return f'Edge[{self.tag}]({self.p1},{self.p2})'
    
    def __hash__(self) -> int:
        return self._hash
    
class Curve:

    def __init__(self, loop_tags: list[int]):
        self.tags = loop_tags
        self._hash = hash(tuple(loop_tags))
        self.tag = None
        self.key = self._hash

    @property
    def dimtag(self) -> tuple[int, int]:
        return (1, self.tag)
    
    def __repr__(self) -> str:
        return f'Curve[{self.tag}]({self.tags})'
    
    def __str__(self) -> str:
        return f'Curve[{self.tag}]({self.tags})'
    
    def __hash__(self) -> int:
        return self._hash

class EdgeSet:

    def __init__(self):
        self.edges: dict[int, Edge] = dict()
        self.tag_to_edge: dict[int, Edge] = dict()
        self.tag_counter = 1

    def __repr__(self) -> str:
        return f'EdgeSet({self.edges})'
    
    def __str__(self) -> str:
        return f'EdgeSet({self.edges})'
    
    def closest(self, x: float, y: float) -> Point:
        dist = lambda item: np.sqrt((x-item.xy[0])**2 + (y-item.xy[1])**2)
        edges = [(p, dist(p)) for p in self.edges.values()]
        return sorted(edges, key=lambda x: x[1])[0][0]
    
    def edges_on_string(self, linestring: shp.LineString) -> list[Edge]:
        edges = []
        domain = linestring.buffer(1e-6)
        for edge in self.edges.values():
            if domain.contains(shp.Point(*edge.xy)):
                edges.append(edge)
        return edges

    def new_tag(self):
        tag = self.tag_counter
        self.tag_counter += 1
        return tag
    
    def get_edge(self, tag: int) -> Edge:
        return self.tag_to_edge[abs(tag)]
    
    def get_edges(self, tags: list[int]) -> list[Edge]:
        return [self.get_edge(tag) for tag in tags]
    
    def get_signed_edge(self, edge: Edge):
        if edge.key in self.edges:
            if edge.p1 == self.edges[edge.key].p2:
                return -self.edges[edge.key].tag
            else:
                return self.edges[edge.key].tag
        else:
            return KeyError()
        
    def submit_edge(self, point1: Point, point2: Point) -> int:
        edge = Edge(point1, point2)
        if edge.key in self.edges:
            return self.get_signed_edge(edge)
        else:
            edge.tag = self.new_tag()
            self.edges[edge.key] = edge
            self.tag_to_edge[edge.tag] = edge
            return edge.tag

class CurveSet:

    def __init__(self):
        self.curves: dict[int, Curve] = dict()
        self.tag_to_curve: dict[int, Curve] = dict()
        self.tag_counter = 1

    def __repr__(self) -> str:
        return f'CurveSet({self.curves})'
    
    def new_tag(self):
        tag = self.tag_counter
        self.tag_counter += 1
        return tag
    
    def submit_curve(self, taglist: list[int]) -> int:
        curve = Curve(taglist)
        if curve.key not in self.curves:
            curve.tag = self.new_tag()
            self.curves[curve.key] = curve
            self.tag_to_curve[curve.tag] = curve
            return curve.tag
        else:
            return self.curves[curve.key].tag

    def get_curve(self, tag: int) -> Curve:
        return self.tag_to_curve[tag]
    
class PointSet:

    def __init__(self, precision: float):
        self.points: dict[int, Point] = dict()
        self.tag_counter = 1
        self._precision = precision
        self.tag_to_point: dict[int, Point] = dict()

    def __repr__(self) -> str:
        return f'PointSet({self.points})'
    
    def closest(self, x: float, y: float) -> Point:
        dist = lambda item: np.sqrt((x-item.xy[0])**2 + (y-item.xy[1])**2)
        points = [(p, dist(p)) for p in self.points.values()]
        return sorted(points, key=lambda x: x[1])[0][0]
    
    def new_tag(self):
        tag = self.tag_counter
        self.tag_counter += 1
        return tag
    
    def get_point(self, tag: int) -> Point:
        return self.tag_to_point[tag]
    
    def submit_coordinates(self, coords: list[tuple]) -> list[int]:
        tags = []
        for coord in coords:
            pnt = Point(coord[0], coord[1], self._precision)
            if pnt.key not in self.points:
                pnt.tag = self.new_tag()
                self.points[pnt.key] = pnt
                self.tag_to_point[pnt.tag] = pnt
                tags.append(pnt.tag)
            else:
                tags.append(self.points[pnt.key].tag)
        
        return tags

class Domain:

    def __init__(self, polygon: shp.Polygon):
        self.polygon = polygon
        self.point_tags = None
        self.edge_tags = None
        self.curve_tag = None
        self.hole_polygons = []
        self.tag = None
        self.material = AIR
    
    @property
    def dimtag(self) -> tuple[int, int]:
        return (2, self.tag)
    
    @property
    def xy(self) -> np.ndarray:
        return np.array(self.polygon.centroid.xy)
    
    @property
    def plane_tags(self) -> list[int]:
        return [self.curve_tag,] + [hole.curve_tag for hole in self.hole_polygons]

    def __repr__(self) -> str:
        return f'ProcessedPolygon({self.edge_tags})'
    
class SelectByXYZ:

    def __init__(self, point, edge, domain):
        self.point: Point = point
        self.edge: Edge = edge
        self.domain: Domain = domain

class Geometry:

    def __init__(self):
        self._input_polygons = []
        self._total_domain: shp.Polygon = None
        self._boundary_domain: shp.Polygon = None
        self.edges: EdgeSet = EdgeSet()
        self.points: PointSet = PointSet(1e6)
        self.curves: CurveSet = CurveSet()
        self.polygons: list[Domain] = []
        self.ds = 0.1

    def _merge_polygons(self, polygons: list[shp.Polygon]) -> None:
        
        points = [poly for poly in polygons if isinstance(poly,shp.Point)]
        polygons = [poly for poly in polygons if isinstance(poly,shp.Polygon)]
        self._input_polygons = polygons
        base_poly = [
            self._input_polygons[0],
        ]
        self._total_domain = shp.union_all(polygons)
        
        other_polys = polygons[1:]

        for poly in other_polys:
            base_poly = _introduce(base_poly, poly)

        self._boundary_domain = self._total_domain.exterior.buffer(1e-6)
        self._input_polygons = base_poly + points
        return base_poly

    def is_boundary(self, edge: Edge) -> bool:
        ''' Returns true if the given edge is an absolute boundary of the simulation domain ( not contained inside )'''
        return self._boundary_domain.contains(shp.Point(*edge.xy))
    
    def retreive(self, coordinates: np.ndarray, material_selector: callable):
        xs = coordinates[0,:]
        ys = coordinates[1,:]
        tups = [(x,y) for x,y in zip(xs,ys)]
        arry = np.zeros((xs.shape[0],))
        for domain in self.polygons:
            inside = np.array(shp.contains_xy(domain.polygon, tups), dtype=np.int32)
            value = material_selector(domain.material, xs, ys)
            arry[inside==1] = value[inside==1]
        return arry

    def closest(self, x: float, y: float) -> Point:
        dist = lambda item: np.sqrt((x-item.xy[0])**2 + (x-item.xy[1])**2)
        domains = [(p, dist(p)) for p in self.polygons]
        return sorted(domains, key=lambda x: x[1])[0][0]
    
    def on_boundary(self, boundary: shp.LineString) -> list[Edge]:
        edges = self.edges.edges_on_string(boundary)
        edges = [edge for edge in edges if self.is_boundary(edge)]
        #logger.debug(f'Selected edges on boundary {boundary}: {edges}')
        return edges
    
    def on_internal_boundary(self, boundary: shp.LineString) -> list[Edge]:
        edges = self.edges.edges_on_string(boundary)
        #edges = [edge for edge in edges if self.is_boundary(edge)]
        #logger.debug(f'Selected edges on boundary {boundary}: {edges}')
        return edges

    def select(self, x: float, y: float) -> SelectByXYZ:
        point = self.points.closest(x,y)
        edge = self.edges.closest(x,y)
        domain = self.closest(x,y)
        return SelectByXYZ(point, edge, domain)

    def compile(self) -> None:
        for polygon in self._input_polygons:
            if isinstance(polygon, shp.Point):
                self._process_point(polygon)
                continue

            # Only if polygons are left
            self._process_polygon(polygon)

    def get_domain(self, tag: int) -> Domain:
        for poly in self.polygons:
            if poly.curve_tag == tag:
                return poly
        raise KeyError(f'There is no Domain with tag {tag}')
    
    def _process_point(self, point: shp.Point) -> None:

        point_tag = self.points.submit_coordinates([(point.x, point.y)])

    def _process_polygon(self, polygon: shp.Polygon) -> None:
        
        if isinstance(polygon, shp.Polygon):
            coords = polygon.exterior.coords
        elif isinstance(polygon, shp.LinearRing):
            coords = polygon.coords
        else:
            raise TypeError(f'I cant process an object of the type {type(polygon)}')
        # Process exterior coordinates
        exterior_coords = list(coords)[:-1]
        exterior_point_tags = self.points.submit_coordinates(exterior_coords)
        exterior_point_tags.append(exterior_point_tags[0])
        exterior_edge_tags = []
        
        for tag1, tag2 in zip(exterior_point_tags[:-1], exterior_point_tags[1:]):
            p1 = self.points.get_point(tag1)
            p2 = self.points.get_point(tag2)
            edge_tag = self.edges.submit_edge(p1, p2)
            exterior_edge_tags.append(edge_tag)

        exterior_loop_tag = self.curves.submit_curve(exterior_edge_tags)
        processed_polygon = Domain(polygon)
        processed_polygon.point_tags = exterior_point_tags
        processed_polygon.edge_tags = exterior_edge_tags
        processed_polygon.curve_tag = exterior_loop_tag
        
        # Process interior rings (holes)
        if isinstance(polygon, shp.LinearRing):
            return processed_polygon
        
        for interior in polygon.interiors:
            processed_polygon.hole_polygons.append(self._process_polygon(interior))

        self.polygons.append(processed_polygon)
        return processed_polygon
        
    def update_point_ds(self, discretizer: callable, resolution: float) -> None:
        for domain in self.polygons:
            ds = discretizer(domain.material)*resolution
            for tag in domain.point_tags:
                self.points.get_point(tag).update_ds(ds)
        
        for point in self.points.points.values():
            for domain in self.polygons:
                if domain.polygon.contains(shp.Point(point.x, point.y)):
                    ds = discretizer(domain.material)*resolution
                    point.update_ds(ds)
                    point._embedded = True
                    point._embedded_entity = domain

    def _commit_gmsh(self):
        for tag, point in self.points.tag_to_point.items():
            gmsh.model.geo.addPoint(point.x, point.y, 0, point._ds)
        
        for tag, edge in self.edges.tag_to_edge.items():
            gmsh.model.geo.addLine(edge.p1.tag, edge.p2.tag, tag)

        for tag, curve in self.curves.tag_to_curve.items():
            gmsh.model.geo.addCurveLoop(curve.tags, tag)

        for polygon in self.polygons:
            tag = gmsh.model.geo.addPlaneSurface(polygon.plane_tags)
            polygon.tag = tag

        gmsh.model.geo.synchronize()

        for tag, point in self.points.tag_to_point.items():
            if point._embedded is False:
                continue
            gmsh.model.mesh.embed(0, [point.tag,], 2, point._embedded_entity.tag)

    def overview(self) -> None:
        """
        Generates a visual overview of all polygons and lines, labeling each with their Gmsh tags.
        """
        # Synchronize the Gmsh model to ensure all entities are defined
        gmsh.model.geo.synchronize()

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        # Plot the lines and annotate with line IDs
        for tag, edge in self.edges.tag_to_edge.items():
            
            # Get the coordinates of the start and end points
            start_coord = edge.p1.xy
            end_coord = edge.p2.xy
            # Plot the line
            ax.plot(
                [start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], "k-"
            )
            # Compute the midpoint of the line to place the label
            mid_x = (start_coord[0] + end_coord[0]) / 2
            mid_y = (start_coord[1] + end_coord[1]) / 2
            # Annotate with the line ID
            ax.text(
                mid_x,
                mid_y,
                str(tag),
                color="blue",
                fontsize=12,
                ha="center",
                va="center",
                bbox=bbox_props,
            )

        # Plot the polygons and annotate with plane surface IDs
        for polygon in self.polygons:

            print(f'Adding curve tag [{polygon.curve_tag}], {polygon}')
            x, y = polygon.polygon.exterior.xy
            ax.fill(x, y, alpha=0.3)
            # Compute the centroid
            centroid = polygon.polygon.centroid
            # Annotate with the plane surface ID
            
            centroid = polygon.polygon.point_on_surface()
            ax.text(
                centroid.x,
                centroid.y,
                str(polygon.curve_tag),
                color="red",
                fontsize=13,
                ha="center",
                va="center",
                bbox=bbox_props,
            )

        # Set aspect ratio and labels
        ax.set_aspect("equal", "box")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Overview of Gmsh Tags")
        plt.grid(True)
        plt.show()