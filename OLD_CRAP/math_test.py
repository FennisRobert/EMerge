import numpy as np
import pyescher as pe
from emerge.solvers.fem.elements.nedelec2.assembly import interpolate_solution
from emerge.solvers.fem.mesh3d import Mesh3D


class CustomMesh:

    def __init__(self):
        txs = np.array([0, 1, 0, 0])
        tys = np.array([0, 0, 1, 0])
        tzs = np.array([0, 0, 0, 1])

        self.nodes = np.array([txs, tys, tzs])
        self.tris = np.array([[0, 1, 2],
                     [0, 1, 3],
                     [0, 2, 3],
                     [1, 2, 3]]).T
        self.edges = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]).T
        self.tet_to_mesh = dict()
        self.tet_to_edge = np.array([0,1,2,3,4,5]).reshape((6,1))
        self.tets = np.array([[0, 1, 2, 3]]).reshape((4,1))
        self.tet_to_tri = np.array([0,1,2,3]).reshape((4,1))

        nedges = 6
        ntris = 4
        self.nfield = 2*nedges + 2*ntris
        self.tet_to_field = np.zeros((20, self.tets.shape[1]), dtype=int)
        self.tet_to_field[:6,:] = self.tet_to_edge
        self.tet_to_field[6:10,:] = self.tet_to_tri + nedges
        self.tet_to_field[10:16,:] = self.tet_to_edge + (ntris+nedges)
        self.tet_to_field[16:20,:] = self.tet_to_tri + (ntris+2*nedges)

        self.edge_to_field = np.zeros((2,nedges), dtype=int)

        self.edge_to_field[0,:] = np.arange(nedges)
        self.edge_to_field[1,:] = np.arange(nedges) + ntris + nedges

        self.tri_to_field = np.zeros((2,ntris), dtype=int)

        self.tri_to_field[0,:] = np.arange(ntris) + nedges
        self.tri_to_field[1,:] = np.arange(ntris) + 2*nedges + ntris


mesh = CustomMesh()

view = pe.Viewer()

v0 = 0.0001
v1 = 0.9999
xs = np.linspace(v0, v1, 11)
ys = np.linspace(v0, v1, 11)
zs = np.linspace(v0, v1, 11)
xs, ys, zs = np.meshgrid(xs, ys, zs)

xs = xs.flatten()
ys = ys.flatten()
zs = zs.flatten()


sol = np.ones((20,))
sol[0] = 0
sol[10] = 0

coords = np.array([xs, ys, zs])
Ex, Ey, Ez = interpolate_solution(coords, sol, mesh.tets, mesh.tris, mesh.edges, mesh.nodes, mesh.tet_to_field)


txs = mesh.nodes[0,:]
tys = mesh.nodes[1,:]
tzs = mesh.nodes[2,:]

with view.new3d('Tetrahedron') as v:
    v.scatter(txs, tys, tzs, size=0.01)
    #v.scatter(xs, ys, zs, s=P1+P2)
    v.mesh(np.array([txs, tys, tzs]), mesh.tris)
    v.quiver3d(xs, ys, zs, Ex, Ey, Ez)