#import fem

from fem.elements.nedleg2 import NedelecLegrange2
from fem.mesh3d import SurfaceMesh
from fem.mth.tri import ned2_tri_interp
import matplotlib.pyplot as plt

import numpy as np

#import pyvista as pv

#from fem.plotting.pyvista import Display


####

def draw_triangle(ax, triangle_coords, **kwargs):
    """Draws a triangle from three (x, y) tuples."""
    triangle = np.array(triangle_coords + [triangle_coords[0]])  # Close the loop
    ax.plot(triangle[:, 0], triangle[:, 1], **kwargs)
    ax.fill(triangle[:, 0], triangle[:, 1], alpha=0.1, **kwargs)

def draw_vectors(ax, positions, vectors, **kwargs):
    """Draws vector arrows at specified positions."""
    positions = np.array(positions)
    vectors = np.array(vectors)
    ax.quiver(positions[:, 0], positions[:, 1], vectors[:, 0], vectors[:, 1],
              angles='xy', **kwargs)

def plot_triangle_with_vectors(triangle_coords, vector_positions, vector_arrows, title="Triangle with Vectors"):
    """Plots a triangle and vector field in the same figure."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    draw_triangle(ax, triangle_coords, color='blue', linewidth=2)
    draw_vectors(ax, vector_positions, vector_arrows, color='red')
    
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    plt.show()


##########
nodes = np.array([[0., 1., 0., 1.],[0., 0., 1., 1.0], [0., 0., 0., 0.]])
tris = np.array([[0, 1],[1, 2],[2, 3]])
edges = np.array([[0,1,0,1,2],[1,2,2,3,3]])

xs = np.linspace(0., 1., 21)
ys = np.linspace(0., 1., 21)

X, Y = np.meshgrid(xs, ys)
xf = X.flatten()
yf = Y.flatten()

for j in range(14):
    active_fields = [j,]

    ####
    field = np.zeros((14,))

    for i in active_fields:
        field[i] = 1

    #field = [0,1,2,3,4],[5,6],[7,8,9,10,11],[12,13]
    tri_to_field = np.array([[0,1,2,5,7,8, 9, 12],
                             [1,3,4,6,8,10,11,13]]).T
    print(tri_to_field.shape)
    Ex, Ey, Ez = ned2_tri_interp(np.array([xf, yf]), field, tris, edges, nodes, tri_to_field)

    plot_triangle_with_vectors(nodes.T, np.array([xf, yf]).T, np.array([Ex, Ey, Ez]).T, 'Test')