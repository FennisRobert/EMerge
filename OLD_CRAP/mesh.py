import gmsh
import shapely as shp
import numpy as np
from collections import defaultdict

def smrz(arry: np.ndarray):
    print(f'Array | shape = {arry.shape}, min={np.min(arry.flatten())}, max={np.max(arry.flatten())}')

def _get_edge(tag) -> list:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, nodetags = gmsh.model.mesh.getElements(1, tag)
    nodetags = list(nodetags[0])
    tags = [
        nodetags[0],
    ] + nodetags[1::2]
    return tags

def _get_point(tag) -> int:
    node_tag, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, nodetags = gmsh.model.mesh.getElements(0,tag)
    nodetags = list(nodetags[0])
    tags = [
        nodetags[0],
    ] + nodetags[1::2]
    return tags[0]


class Mesh:

    def __init__(self):
        # COORDINATES
        self.nodes: np.ndarray = None

        # ELEMENTS
        self.edges: np.ndarray = None
        self.tris: np.ndarray = None
        self.tets: np.ndarray = None

        # QUANTITIES
        self.n_edges: int = None
        self.n_tris: int = None
        self.n_tets: int = None
        self.n_nodes: int = None

        # MEASURES
        self.edge_lengths: np.ndarray = None
        self.tri_areas: np.ndarray = None
        self.tet_volumes: np.ndarray = None

        # DERIVED COORDINATES
        self.edge_centers: np.ndarray = None
        self.tri_centers: np.ndarray = None
        self.tet_centers: np.ndarray = None

        # MAPPING
        self.tet_to_edge: np.ndarray = None
        self.tet_to_tri: np.ndarray = None

        self.tri_to_edge: np.ndarray = None
        self.tri_to_tet: np.ndarray = None

        self.edge_to_tri: np.ndarray = None
        self.edge_to_tet: np.ndarray = None
        self.node_to_edge: dict[int, list[int]] = None
        

    def find_edge_groups(self, edge_ids: np.ndarray) -> dict:
        """
        Find the groups of edges in the mesh.

        Split an edge list into sets (islands) whose vertices are mutually connected.

        Parameters
        ----------
        edges : np.ndarray, shape (2, N)
            edges[0, i] and edges[1, i] are the two vertex indices of edge *i*.
            The array may contain any (hashable) integer vertex labels, in any order.

        Returns
        -------
        List[Tuple[int, ...]]
            A list whose *k*‑th element is a `tuple` with the (zero‑based) **edge IDs**
            that belong to the *k*‑th connected component.  Ordering is:
            • components appear in the order in which their first edge is met,  
            • edge IDs inside each tuple are sorted increasingly.

        Notes
        -----
        * Only the connectivity of the supplied edges is considered.  
        In particular, vertices that never occur in `edges` do **not** create extra
        components.
        * Runtime is *O*(N + V), with N = number of edges, V = number of
        distinct vertices.  No external libraries are needed.
        """
        edges = self.edges[:,edge_ids]
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("`edges` must have shape (2, N)")

        n_edges: int = edges.shape[1]

        # --- build “vertex ⇒ incident edge IDs” map ------------------------------
        vert2edges = defaultdict(list)
        for eid in edge_ids:
            v1, v2 = self.edges[0, eid], self.edges[1, eid]
            vert2edges[v1].append(eid)
            vert2edges[v2].append(eid)
        
        groups = []

        ungrouped = set(edge_ids)

        group = [edge_ids[0],]
        ungrouped.remove(edge_ids[0])

        while True:
            new_edges = set()
            for edge in group:
                v1, v2 = self.edges[0, edge], self.edges[1, edge]
                new_edges.update(set(vert2edges[v1]))
                new_edges.update(set(vert2edges[v2]))

            new_edges = new_edges.intersection(ungrouped)
            if len(new_edges) == 0:
                groups.append(tuple(sorted(group)))
                if len(ungrouped) == 0:
                    break
                group = [ungrouped.pop(),]
            else:
                group += list(new_edges)
                ungrouped.difference_update(new_edges)

        return groups

class Mesh2D(Mesh):
    def __init__(self, name: str = "UnnamedMesh", element_order: int = 2):

        self.name: str = name

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        self.t2vertex: dict = dict()
        self.t2tri: dict = dict()
        self.vertices = None
        self.triangles = None
        self.boundary_normals = None
        self.extract_mesh_data()
        self.vertices = self.vertices[:2, :]
        self.nv = self.vertices.shape[1]
        self.nt = self.triangles.shape[1]
        self.nF = self.nv
        self.ne = None
        self.centroids = 1/3 * (self.vertices[:,self.triangles[0,:]] + 
                               self.vertices[:,self.triangles[1,:]] +
                               self.vertices[:,self.triangles[2,:]])
        
        self.quad_vertices = None
        self.quad_edges = None
        self.quad_elements = None
        self.quad_vertex_pair_edge = None
        self.quad_boundary_normals = None

        self.element_order = element_order

        if element_order == 2:
            self._generate_quad_data()
            self.nF = self.nv + self.ne

        self._generate_boundary_normals()

        

    @property
    def xs(self) -> np.ndarray:
        if self.element_order==1:
            return self.vertices[0,:]
        else:
            return self.quad_vertices[0,:]
    @property
    def ys(self) -> np.ndarray:
        if self.element_order==1:
            return self.vertices[1,:]
        else:
            return self.quad_vertices[1,:]
    
    @property
    def mvertices(self) -> np.ndarray:
        if self.element_order==1:
            return self.vertices
        else:
            return self.quad_vertices
        
    @property
    def mtriangles(self) -> np.ndarray:
        if self.element_order==1:
            return self.triangles
        else:
            long_tris = []
            for it in range(self.nt):
                i1, i2, i3, i4, i5, i6 = self.quad_elements[:,it]
                long_tris += [(i1, i4, i6),(i2, i4, i5),(i4, i6, i5),(i5, i3, i6)]
            out = np.array(long_tris).T
            return out

    def _generate_quad_data(self) -> None:
        ''' Generates quadratic interpolation data of the mesh'''
        edge_elements = []
        for it in range(self.nt):
            i1, i2, i3 = sorted(list(self.triangles[:,it]))
            edge_elements.append((i1, i2))
            edge_elements.append((i2, i3))
            edge_elements.append((i1, i3))

        edge_elements = list(set(edge_elements))
        
        edge_element_mapping = {edgepair: i for i,edgepair in enumerate(edge_elements)}

        self.ne = len(edge_element_mapping)

        self.quad_vertex_pair_edge = dict()
        self.quad_elements = np.zeros((6, self.nt))
        self.quad_vertices = np.zeros((2, self.nv+self.ne))

        self.quad_edges = np.array(edge_elements).T

        self.quad_elements[:3, :] = self.triangles

        nv = self.nv
        for it in range(self.nt):
            i1, i2, i3 = sorted(list(self.triangles[:,it]))
            self.triangles[:,it] = (i1, i2, i3)
            k1 = (i1, i2)
            k2 = (i2, i3)
            k3 = (i1, i3)
            ie1 = edge_element_mapping[k1]+nv
            ie2 = edge_element_mapping[k2]+nv
            ie3 = edge_element_mapping[k3]+nv
            self.quad_elements[:,it] = (i1, i2, i3, ie1, ie2, ie3)
            self.quad_vertex_pair_edge[k1] = ie1
            self.quad_vertex_pair_edge[k2] = ie2
            self.quad_vertex_pair_edge[k3] = ie3
            self.quad_vertex_pair_edge[(i2, i1)] = ie1
            self.quad_vertex_pair_edge[(i3, i2)] = ie2
            self.quad_vertex_pair_edge[(i3, i1)] = ie3

        self.quad_elements = self.quad_elements.astype(np.int32)
        ve1 = self.vertices[:,self.quad_edges[0,:]]
        ve2 = self.vertices[:,self.quad_edges[1,:]]
        self.quad_vertices[:,:self.nv] = self.vertices
        self.quad_vertices[:,self.nv:] = 0.5*(ve1 + ve2)
        #for it in range(self.nt):
        #    self.plot_mesh(highlight_vertices=self.quad_elements[:,it])

        print('Quadratic Mesh data generated successfully')

    def _generate_boundary_normals(self) -> None:
        self.boundary_normals = defaultdict(list)
        self.quad_boundary_normals = defaultdict(list)

        for it in range(self.nt):
            cx, cy = self.centroids[:,it]

            i1, i2, i3 = self.triangles[:,it]
            x1, x2, x3 = self.vertices[0,self.triangles[:,it]]
            y1, y2, y3 = self.vertices[1,self.triangles[:,it]]

            dx1, dy1 = x2-x1, y2-y1
            dx2, dy2 = x3-x2, y3-y2
            dx3, dy3 = x1-x3, y1-y3

            l1 = np.sqrt(dx1**2+dy1**2)
            l2 = np.sqrt(dx2**2+dy2**2)
            l3 = np.sqrt(dx3**2+dy3**2)

            mx1, my1 = 0.5*(x1+x2), 0.5*(y1+y2)
            mx2, my2 = 0.5*(x2+x3), 0.5*(y2+y3)
            mx3, my3 = 0.5*(x3+x1), 0.5*(y3+y1)

            nx1, ny1 = dy1/l1, dx1/l1
            nx2, ny2 = dy2/l2, dx2/l2
            nx3, ny3 = dy3/l3, dx3/l3

            dir1 = np.sign(nx1*(mx1-cx) +ny1*(my1-cy))
            dir2 = np.sign(nx2*(mx2-cx) +ny2*(my2-cy))
            dir3 = np.sign(nx3*(mx3-cx) +ny3*(my3-cy))

            nx1, ny1 = dir1*nx1, dir1*ny1
            nx2, ny2 = dir2*nx2, dir2*ny2
            nx3, ny3 = dir3*nx3, dir3*ny3

            self.boundary_normals[(i1, i2)].append((nx1, ny1))
            self.boundary_normals[(i2, i3)].append((nx2, ny2))
            self.boundary_normals[(i3, i1)].append((nx3, ny3))
            self.boundary_normals[(i3, i1)].append((nx1, ny1))
            self.boundary_normals[(i2, i2)].append((nx2, ny2))
            self.boundary_normals[(i1, i3)].append((nx3, ny3))
        #
        #         (3)
        #     5  / | 4       
        #     (6) (5)
        #  6 /     | 3
        # (1)-(4)-(2)
        #    1   2
        #
        if self.element_order == 2:
            for it in range(self.nt):
                i1, i2, i3, i4, i5, i6 = self.quad_elements[:,it]
                cx, cy = self.centroids[:,it]

                x1, x2, x3, x4, x5, x6 = self.quad_vertices[0,self.quad_elements[:,it]]
                y1, y2, y3, y4, y5, y6 = self.quad_vertices[1,self.quad_elements[:,it]]

                dx1, dy1 = x4-x1, y4-y1
                dx2, dy2 = x2-x4, y2-y4
                dx3, dy3 = x5-x2, y5-y2
                dx4, dy4 = x3-x5, y3-y5
                dx5, dy5 = x6-x3, y6-y3
                dx6, dy6 = x1-x6, y1-y6
                l1 = np.sqrt(dx1**2 + dy1**2)
                l2 = np.sqrt(dx2**2 + dy2**2)
                l3 = np.sqrt(dx3**2 + dy3**2)
                l4 = np.sqrt(dx4**2 + dy4**2)
                l5 = np.sqrt(dx5**2 + dy5**2)
                l6 = np.sqrt(dx6**2 + dy6**2)

                mx1, my1 = 0.5*(x4+x1), 0.5*(y4+y1)
                mx2, my2 = 0.5*(x2+x4), 0.5*(y2+y4)
                mx3, my3 = 0.5*(x5+x2), 0.5*(y5+y2)
                mx4, my4 = 0.5*(x3+x5), 0.5*(y3+y5)
                mx5, my5 = 0.5*(x6+x3), 0.5*(y6+y3)
                mx6, my6 = 0.5*(x1+x6), 0.5*(y1+y6)

                nx1, ny1 = dy1/l1, -dx1/l1
                nx2, ny2 = dy2/l2, -dx2/l2
                nx3, ny3 = dy3/l3, -dx3/l3
                nx4, ny4 = dy4/l4, -dx4/l4
                nx5, ny5 = dy5/l5, -dx5/l5
                nx6, ny6 = dy6/l6, -dx6/l6

                dir1 = np.sign(nx1*(mx1-cx) + ny1*(my1-cy))
                dir2 = np.sign(nx2*(mx2-cx) + ny2*(my2-cy))
                dir3 = np.sign(nx3*(mx3-cx) + ny3*(my3-cy))
                dir4 = np.sign(nx4*(mx4-cx) + ny4*(my4-cy))
                dir5 = np.sign(nx5*(mx5-cx) + ny5*(my5-cy))
                dir6 = np.sign(nx6*(mx6-cx) + ny6*(my6-cy))

                nx1, ny1 = dir1*nx1, dir1*ny1
                nx2, ny2 = dir2*nx2, dir2*ny2
                nx3, ny3 = dir3*nx3, dir3*ny3
                nx4, ny4 = dir4*nx4, dir4*ny4
                nx5, ny5 = dir5*nx5, dir5*ny5
                nx6, ny6 = dir6*nx6, dir6*ny6

                self.quad_boundary_normals[(i1, i4)].append((nx1, ny1))
                self.quad_boundary_normals[(i4, i2)].append((nx2, ny2))
                self.quad_boundary_normals[(i2, i5)].append((nx3, ny3))
                self.quad_boundary_normals[(i5, i3)].append((nx4, ny4))
                self.quad_boundary_normals[(i3, i6)].append((nx5, ny5))
                self.quad_boundary_normals[(i6, i1)].append((nx6, ny6))

                self.quad_boundary_normals[(i4, i1)].append((nx1, ny1))
                self.quad_boundary_normals[(i2, i4)].append((nx2, ny2))
                self.quad_boundary_normals[(i5, i2)].append((nx3, ny3))
                self.quad_boundary_normals[(i3, i5)].append((nx4, ny4))
                self.quad_boundary_normals[(i6, i3)].append((nx5, ny5))
                self.quad_boundary_normals[(i1, i6)].append((nx6, ny6))

        self.quad_boundary_normals = {index: value[0] for index, value in self.quad_boundary_normals.items() if len(value)==1}
        self.boundary_normals = {index: value[0] for index, value in self.boundary_normals.items() if len(value)==1}


    def ids(self, indices):
        return [self.t2vertex[i] for i in indices]

    def get_edge(self, edgetag: int) -> list[int]:
        """Returns the list of vertex indices associated with the edge corresponding to the provided tag"""
        ids = _get_edge(edgetag)
        ids = [self.t2vertex[i] for i in ids]
        if self.element_order==2:
            ids_out = [ids[0],]
            for i1, i2 in zip(ids[:-1],ids[1:]):
                ic = self.quad_vertex_pair_edge[(i1, i2)]
                ids_out += [ic,i2]
            return ids_out
        else:
            return ids

    def get_point(self, pointtag: int) -> int:
        ids = _get_point(pointtag)
        ids = self.t2vertex[ids]
        print(f'FOUND VERTEX INDEX FOR POINT {pointtag} = {ids}')
        return ids
    
    def get_domain(self, domaintag: int) -> list[int]:
        """Return the list of triangles indices"""
        element_types, element_tags, nodetags = gmsh.model.mesh.getElements(
            2, domaintag
        )
        return [self.t2tri[i] for i in element_tags[0]]

    def extract_mesh_data(self) -> None:
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        t2idx = {t: i for i, t in enumerate(node_tags)}
        self.t2vertex = t2idx
        points = np.array(node_coords).reshape(-1, 3).T

        element_types, element_tags, nodetags = gmsh.model.mesh.getElements(2, -1)

        self.t2tri = {t: i for i, t in enumerate(element_tags[0])}

        nodetags = [t2idx[t] for t in nodetags[0]]
        triangles = np.array(nodetags).reshape(-1, 3).T

        gmsh.write(f"meshes/{self.name}.msh")
        self.vertices = points
        self.triangles = triangles

    def plot_gmsh(self) -> None:
        gmsh.fltk.run()

    def plot_mesh(
        self,
        highlight_vertices: list[int] = None,
        highlight_triangles: list[int] = None,
    ) -> None:
        # Extract x and y coordinates of vertices
        if self.element_order==1:
            x = self.vertices[0, :]
            y = self.vertices[1, :]
        else:
            x = self.quad_vertices[0,:]
            y = self.quad_vertices[1,:]

        tvx = self.vertices[0,:]
        tvy = self.vertices[1,:]
        TRIS = self.mtriangles
        N = TRIS.shape[1]
        cx = np.zeros((N,))
        cy = np.zeros((N,))
        
        # print('TRIS SHAPE:', N)
        # for ii in range(N):
        #     (i1, i2, i3) = TRIS[:,ii]
        #     cx[ii] = (x[i1] + x[i2] + x[i3])/3
        #     cy[ii] = (y[i1] + y[i2] + y[i3])/3
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.triplot(
            tvx, tvy, self.triangles.T, color="black", lw=0.5
        )  # Use triplot for 2D triangular meshes
        plt.scatter(x, y, color="red", s=10)  # Plot vertices for reference
        #plt.scatter(cx,cy)
        # Set plot labels and title
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("2D Mesh Plot")
        plt.gca().set_aspect("equal", adjustable="box")  # Set equal aspect ratio

        if highlight_triangles is not None:
            for index in highlight_triangles:
                plt.fill(
                    x[self.triangles[:, index]],
                    y[self.triangles[:, index]],
                    alpha=0.3,
                    color="blue",
                )
        if highlight_vertices is not None:
            ids = np.array(highlight_vertices).astype(np.int32)
            x = self.xs[ids]
            y = self.ys[ids]
            plt.scatter(x, y)
        # Show the plot
        plt.show()