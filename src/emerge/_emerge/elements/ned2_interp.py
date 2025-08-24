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
from numba import njit, f8, c16, i8, types # type: ignore
import numpy as np
from ..mth.optimized import compute_distances

@njit(types.Tuple((f8[:], f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]), cache = True, nogil=True)
def tet_coefficients(xs, ys, zs):
    ## THIS FUNCTION WORKS
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(-x1*y2*z3/6 + x1*y2*z4/6 + x1*y3*z2/6 - x1*y3*z4/6 - x1*y4*z2/6 + x1*y4*z3/6 + x2*y1*z3/6 - x2*y1*z4/6 - x2*y3*z1/6 + x2*y3*z4/6 + x2*y4*z1/6 - x2*y4*z3/6 - x3*y1*z2/6 + x3*y1*z4/6 + x3*y2*z1/6 - x3*y2*z4/6 - x3*y4*z1/6 + x3*y4*z2/6 + x4*y1*z2/6 - x4*y1*z3/6 - x4*y2*z1/6 + x4*y2*z3/6 + x4*y3*z1/6 - x4*y3*z2/6)
    
    aas[0] = x2*y3*z4 - x2*y4*z3 - x3*y2*z4 + x3*y4*z2 + x4*y2*z3 - x4*y3*z2
    aas[1] = -x1*y3*z4 + x1*y4*z3 + x3*y1*z4 - x3*y4*z1 - x4*y1*z3 + x4*y3*z1
    aas[2] = x1*y2*z4 - x1*y4*z2 - x2*y1*z4 + x2*y4*z1 + x4*y1*z2 - x4*y2*z1
    aas[3] = -x1*y2*z3 + x1*y3*z2 + x2*y1*z3 - x2*y3*z1 - x3*y1*z2 + x3*y2*z1
    bbs[0] = -y2*z3 + y2*z4 + y3*z2 - y3*z4 - y4*z2 + y4*z3
    bbs[1] = y1*z3 - y1*z4 - y3*z1 + y3*z4 + y4*z1 - y4*z3
    bbs[2] = -y1*z2 + y1*z4 + y2*z1 - y2*z4 - y4*z1 + y4*z2
    bbs[3] = y1*z2 - y1*z3 - y2*z1 + y2*z3 + y3*z1 - y3*z2
    ccs[0] = x2*z3 - x2*z4 - x3*z2 + x3*z4 + x4*z2 - x4*z3
    ccs[1] = -x1*z3 + x1*z4 + x3*z1 - x3*z4 - x4*z1 + x4*z3
    ccs[2] = x1*z2 - x1*z4 - x2*z1 + x2*z4 + x4*z1 - x4*z2
    ccs[3] = -x1*z2 + x1*z3 + x2*z1 - x2*z3 - x3*z1 + x3*z2
    dds[0] = -x2*y3 + x2*y4 + x3*y2 - x3*y4 - x4*y2 + x4*y3
    dds[1] = x1*y3 - x1*y4 - x3*y1 + x3*y4 + x4*y1 - x4*y3
    dds[2] = -x1*y2 + x1*y4 + x2*y1 - x2*y4 - x4*y1 + x4*y2
    dds[3] = x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2

    return aas, bbs, ccs, dds, V

@njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:]), cache = True, nogil=True)
def tri_coefficients(vxs, vys):

    x1, x2, x3 = vxs
    y1, y2, y3 = vys

    a1 = x2*y3-y2*x3
    a2 = x3*y1-y3*x1
    a3 = x1*y2-y1*x2
    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    #A = 0.5*(b1*c2 - b2*c1)
    sA = 0.5*(((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)))
    sign = np.sign(sA)
    A = np.abs(sA)
    As = np.array([a1, a2, a3])*sign
    Bs = np.array([b1, b2, b3])*sign
    Cs = np.array([c1, c2, c3])*sign
    return As, Bs, Cs, A

@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex 0.1005964238ers of one tetrahedron, in *its* order
        (v0, v1, v2, v3).

    triangle_ids : 2-D int64 array (nTri × 3)
        Each row is a global-ID triple of one face that belongs to this tet.

    Returns
    -------
    local_tris   : 2-D int64 array (nTri × 3)
        Same triangles, but every entry replaced by the local index
        0,1,2,3 that the vertex has inside this tetrahedron.
        (Guaranteed to be ∈{0,1,2,3}; no -1 ever appears if the input
        really belongs to the tet.)
    """
    ndim = triangle_ids.shape[0]
    ntri = triangle_ids.shape[1]
    out  = np.zeros(triangle_ids.shape, dtype=np.int64)

    for t in range(ntri):                 # each triangle
        for j in range(ndim):                # each vertex in that triangle
            gid = triangle_ids[j, t]      # global ID to look up

            # linear search over the four tet vertices
            for k in range(4):
                if vertex_ids[k] == gid:
                    out[j, t] = k         # store local index 0-3
                    break                 # stop the k-loop

    return out

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], i8[:,:], i8[:,:], f8[:,:], i8[:,:], i8[:]), cache=True, nogil=True)
def ned2_tet_interp(coords: np.ndarray,
                         solutions: np.ndarray, 
                         tets: np.ndarray, 
                         tris: np.ndarray,
                         edges: np.ndarray,
                         nodes: np.ndarray,
                         tet_to_field: np.ndarray,
                         tetids: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation'''
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    nEdges = edges.shape[1]

    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]

    Ex = np.full((nNodes, ),np.nan, dtype=np.complex128)
    Ey = np.full((nNodes, ),np.nan, dtype=np.complex128)
    Ez = np.full((nNodes, ),np.nan, dtype=np.complex128)
    for i_iter in range(tetids.shape[0]):
        itet = tetids[i_iter]

        iv1, iv2, iv3, iv4 = tets[:, itet]

        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_field[:6, itet]]
        g_tri_ids = tris[:, tet_to_field[6:10, itet]-nEdges]

        l_edge_ids = local_mapping(g_node_ids, g_edge_ids)
        l_tri_ids = local_mapping(g_node_ids, g_tri_ids)

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]
        v4 = nodes[:,iv4]

        bv1 = v2 - v1
        bv2 = v3 - v1
        bv3 = v4 - v1

        blocal = np.zeros((3,3))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        blocal[:,2] = bv3
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tet_to_field[:, itet]
        Etet = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:] + coords_local[2,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6) & (coords_local[2,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TETRAHEDRON #########
        
        x = xs[inside==1]
        y = ys[inside==1]
        z = zs[inside==1]

        xvs = nodes[0, tets[:,itet]]
        yvs = nodes[1, tets[:,itet]]
        zvs = nodes[2, tets[:,itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
        
        Em1s = Etet[0:6]
        Ef1s = Etet[6:10]
        Em2s = Etet[10:16]
        Ef2s = Etet[16:20]
        
        Exl = np.zeros(x.shape, dtype=np.complex128)
        Eyl = np.zeros(x.shape, dtype=np.complex128)
        Ezl = np.zeros(x.shape, dtype=np.complex128)
        for ie in range(6):
            Em1, Em2 = Em1s[ie], Em2s[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            d1, d2 = d_s[edgeids]
            x1, x2 = xvs[edgeids]
            y1, y2 = yvs[edgeids]
            z1, z2 = zvs[edgeids]

            L = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            ex =  L*(Em1*(a1 + b1*x + c1*y + d1*z) + Em2*(a2 + b2*x + c2*y + d2*z))*(b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))/(216*V**3)
            ey =  L*(Em1*(a1 + b1*x + c1*y + d1*z) + Em2*(a2 + b2*x + c2*y + d2*z))*(c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))/(216*V**3)
            ez =  L*(Em1*(a1 + b1*x + c1*y + d1*z) + Em2*(a2 + b2*x + c2*y + d2*z))*(d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))/(216*V**3)

            Exl += ex
            Eyl += ey
            Ezl += ez
        
        for ie in range(4):
            Em1, Em2 = Ef1s[ie], Ef2s[ie]
            triids = l_tri_ids[:, ie]
            a1, a2, a3 = a_s[triids]
            b1, b2, b3 = b_s[triids]
            c1, c2, c3 = c_s[triids]
            d1, d2, d3 = d_s[triids]

            x1, x2, x3 = xvs[l_tri_ids[:, ie]]
            y1, y2, y3 = yvs[l_tri_ids[:, ie]]
            z1, z2, z3 = zvs[l_tri_ids[:, ie]]

            L1 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
            L2 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

            ex =  (-Em1*L1*(b1*(a3 + b3*x + c3*y + d3*z) - b3*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z) + Em2*L2*(b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))*(a3 + b3*x + c3*y + d3*z))/(216*V**3)
            ey =  (-Em1*L1*(c1*(a3 + b3*x + c3*y + d3*z) - c3*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z) + Em2*L2*(c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))*(a3 + b3*x + c3*y + d3*z))/(216*V**3)
            ez =  (-Em1*L1*(d1*(a3 + b3*x + c3*y + d3*z) - d3*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z) + Em2*L2*(d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))*(a3 + b3*x + c3*y + d3*z))/(216*V**3)
            
            Exl += ex
            Eyl += ey
            Ezl += ez

        Ex[inside] = Exl
        Ey[inside] = Eyl
        Ez[inside] = Ezl
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], i8[:,:], i8[:,:], f8[:,:], i8[:,:], c16[:], i8[:]), cache=True, nogil=True)
def ned2_tet_interp_curl(coords: np.ndarray,
                         solutions: np.ndarray, 
                         tets: np.ndarray, 
                         tris: np.ndarray,
                         edges: np.ndarray,
                         nodes: np.ndarray,
                         tet_to_field: np.ndarray,
                         c: np.ndarray,
                         tetids: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation of the analytic curl'''
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    nEdges = edges.shape[1]

    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]

    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)

    for i_iter in range(tetids.shape[0]):
        itet = tetids[i_iter]
        
        iv1, iv2, iv3, iv4 = tets[:, itet]

        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_field[:6, itet]]
        g_tri_ids = tris[:, tet_to_field[6:10, itet]-nEdges]

        l_edge_ids = local_mapping(g_node_ids, g_edge_ids)
        l_tri_ids = local_mapping(g_node_ids, g_tri_ids)

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]
        v4 = nodes[:,iv4]

        bv1 = v2 - v1
        bv2 = v3 - v1
        bv3 = v4 - v1

        blocal = np.zeros((3,3))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        blocal[:,2] = bv3
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tet_to_field[:, itet]
        Etet = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:] + coords_local[2,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6) & (coords_local[2,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        const = c[itet]
        ######### INSIDE THE TETRAHEDRON #########
        
        x = xs[inside==1]
        y = ys[inside==1]
        z = zs[inside==1]

        xvs = nodes[0, tets[:,itet]]
        yvs = nodes[1, tets[:,itet]]
        zvs = nodes[2, tets[:,itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
        
        Em1s = Etet[0:6]
        Ef1s = Etet[6:10]
        Em2s = Etet[10:16]
        Ef2s = Etet[16:20]
        
        Exl = np.zeros(x.shape, dtype=np.complex128)
        Eyl = np.zeros(x.shape, dtype=np.complex128)
        Ezl = np.zeros(x.shape, dtype=np.complex128)
        for ie in range(6):
            Em1, Em2 = Em1s[ie], Em2s[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            d1, d2 = d_s[edgeids]
            x1, x2 = xvs[edgeids]
            y1, y2 = yvs[edgeids]
            z1, z2 = zvs[edgeids]

            L = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            ex =  L*(-Em1*a1*c1*d2 + Em1*a1*c2*d1 - Em1*b1*c1*d2*x + Em1*b1*c2*d1*x - Em1*c1**2*d2*y + Em1*c1*c2*d1*y - Em1*c1*d1*d2*z + Em1*c2*d1**2*z - Em2*a2*c1*d2 + Em2*a2*c2*d1 - Em2*b2*c1*d2*x + Em2*b2*c2*d1*x - Em2*c1*c2*d2*y - Em2*c1*d2**2*z + Em2*c2**2*d1*y + Em2*c2*d1*d2*z)/(72*V**3)
            ey =  L*(Em1*a1*b1*d2 - Em1*a1*b2*d1 + Em1*b1**2*d2*x - Em1*b1*b2*d1*x + Em1*b1*c1*d2*y + Em1*b1*d1*d2*z - Em1*b2*c1*d1*y - Em1*b2*d1**2*z + Em2*a2*b1*d2 - Em2*a2*b2*d1 + Em2*b1*b2*d2*x + Em2*b1*c2*d2*y + Em2*b1*d2**2*z - Em2*b2**2*d1*x - Em2*b2*c2*d1*y - Em2*b2*d1*d2*z)/(72*V**3)
            ez =  L*(-Em1*a1*b1*c2 + Em1*a1*b2*c1 - Em1*b1**2*c2*x + Em1*b1*b2*c1*x - Em1*b1*c1*c2*y - Em1*b1*c2*d1*z + Em1*b2*c1**2*y + Em1*b2*c1*d1*z - Em2*a2*b1*c2 + Em2*a2*b2*c1 - Em2*b1*b2*c2*x - Em2*b1*c2**2*y - Em2*b1*c2*d2*z + Em2*b2**2*c1*x + Em2*b2*c1*c2*y + Em2*b2*c1*d2*z)/(72*V**3)
            Exl += ex
            Eyl += ey
            Ezl += ez
        
        for ie in range(4):
            Em1, Em2 = Ef1s[ie], Ef2s[ie]
            triids = l_tri_ids[:, ie]
            a1, a2, a3 = a_s[triids]
            b1, b2, b3 = b_s[triids]
            c1, c2, c3 = c_s[triids]
            d1, d2, d3 = d_s[triids]

            x1, x2, x3 = xvs[l_tri_ids[:, ie]]
            y1, y2, y3 = yvs[l_tri_ids[:, ie]]
            z1, z2, z3 = zvs[l_tri_ids[:, ie]]

            L1 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
            L2 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

            ex =  (Em1*L1*(-c2*(d1*(a3 + b3*x + c3*y + d3*z) - d3*(a1 + b1*x + c1*y + d1*z)) + d2*(c1*(a3 + b3*x + c3*y + d3*z) - c3*(a1 + b1*x + c1*y + d1*z)) + 2*(c1*d3 - c3*d1)*(a2 + b2*x + c2*y + d2*z)) - Em2*L2*(-c3*(d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z)) + d3*(c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z)) + 2*(c1*d2 - c2*d1)*(a3 + b3*x + c3*y + d3*z)))/(216*V**3)
            ey =  (-Em1*L1*(-b2*(d1*(a3 + b3*x + c3*y + d3*z) - d3*(a1 + b1*x + c1*y + d1*z)) + d2*(b1*(a3 + b3*x + c3*y + d3*z) - b3*(a1 + b1*x + c1*y + d1*z)) + 2*(b1*d3 - b3*d1)*(a2 + b2*x + c2*y + d2*z)) + Em2*L2*(-b3*(d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z)) + d3*(b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z)) + 2*(b1*d2 - b2*d1)*(a3 + b3*x + c3*y + d3*z)))/(216*V**3)
            ez =  (Em1*L1*(-b2*(c1*(a3 + b3*x + c3*y + d3*z) - c3*(a1 + b1*x + c1*y + d1*z)) + c2*(b1*(a3 + b3*x + c3*y + d3*z) - b3*(a1 + b1*x + c1*y + d1*z)) + 2*(b1*c3 - b3*c1)*(a2 + b2*x + c2*y + d2*z)) - Em2*L2*(-b3*(c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z)) + c3*(b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z)) + 2*(b1*c2 - b2*c1)*(a3 + b3*x + c3*y + d3*z)))/(216*V**3)
            
            Exl += ex
            Eyl += ey
            Ezl += ez

        Ex[inside] = Exl*const
        Ey[inside] = Eyl*const
        Ez[inside] = Ezl*const
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], f8[:,:], i8[:,:]), cache=True, nogil=True)
def ned2_tri_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation'''
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0,:]
    ys = coords[1,:]

    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)

    nodes = nodes[:2,:]

    l_edge_ids = np.array([[0,1,0],[1,2,2]])

    for itri in range(tris.shape[1]):

        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2,2))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.0001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TETRAHEDRON #########
        
        x = xs[inside==1]
        y = ys[inside==1]

        xvs = nodes[0, tris[:,itri]]
        yvs = nodes[1, tris[:,itri]]

        Ds = compute_distances(xvs, yvs, 0*xvs)

        L1 = Ds[0,1]
        L2 = Ds[1,2]
        L3 = Ds[0,2]

        mult = np.array([L1,L2,L3,L3,L1,L2,L3,L1])

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        
        Etri = Etri*mult
        
        Em1s = Etri[:3]
        Ef1s = Etri[3]
        Em2s = Etri[4:7]
        Ef2s = Etri[7]
        
        Exl = np.zeros(x.shape, dtype=np.complex128)
        Eyl = np.zeros(x.shape, dtype=np.complex128)


        for ie in range(3):
            Em1, Em2 = Em1s[ie], Em2s[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            
            ex =  (Em1*(a1 + b1*x + c1*y) + Em2*(a2 + b2*x + c2*y))*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))/(8*A**3)
            ey =  (Em1*(a1 + b1*x + c1*y) + Em2*(a2 + b2*x + c2*y))*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))/(8*A**3)
            
            Exl += ex
            Eyl += ey
        
    
        Em1, Em2 = Ef1s, Ef2s
        triids = np.array([0,1,2])

        a1, a2, a3 = a_s[triids]
        b1, b2, b3 = b_s[triids]
        c1, c2, c3 = c_s[triids]

        ex =  (-Em1*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + Em2*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        ey =  (-Em1*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + Em2*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        
        Exl += ex
        Eyl += ey

        Ex[inside] = Exl
        Ey[inside] = Eyl
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], f8[:,:], i8[:,:]), cache=True, nogil=True)
def ned2_tri_interp_full(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation'''
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0,:]
    ys = coords[1,:]

    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)

    nodes = nodes[:2,:]

    for itri in range(tris.shape[1]):

        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2,2))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.00001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TRIANGLE #########
        
        x = xs[inside==1]
        y = ys[inside==1]

        xvs = nodes[0, tris[:,itri]]
        yvs = nodes[1, tris[:,itri]]

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s

        # original Nedelec-1 order 2 formulation 
        # ex =  (e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        # ey =  (e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        # ez =  (a1 + b1*x + c1*y)*(e10*(-A + a1 + b1*x + c1*y) + e11*(-A + a1 + b1*x + c1*y) + e12*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y) + e14*(a2 + b2*x + c2*y) + e9*(-A + a1 + b1*x + c1*y))/(2*A**2)
        
        # New Nedelec-1 order 2 formulation
        ex = (-2*A*(e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))) - e4*((b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) - e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) - e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) - e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) + e8*((b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y)))/(8*A**3)
        ey = (-2*A*(e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))) - e4*((c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) - e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) - e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) - e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) + e8*((c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y)))/(8*A**3)
        ez = (-e10*(a2 + b2*x + c2*y)*(A - a2 - b2*x - c2*y)/2 - e11*(a3 + b3*x + c3*y)*(A - a3 - b3*x - c3*y)/2 + e12*(a1 + b1*x + c1*y)*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y)*(a3 + b3*x + c3*y) + e14*(a1 + b1*x + c1*y)*(a3 + b3*x + c3*y) - e9*(a1 + b1*x + c1*y)*(A - a1 - b1*x - c1*y)/2)/A**2
        Ex[inside] = ex
        Ey[inside] = ey
        Ez[inside] = ez
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], f8[:,:], i8[:,:], c16[:,:,:], c16), cache=True, nogil=True)
def ned2_tri_interp_curl(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray,
                    diadic: np.ndarray,
                    beta: float):
    ''' Nedelec 2 tetrahedral interpolation'''
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0,:]
    ys = coords[1,:]
    jB = 1j*beta
    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)

    nodes = nodes[:2,:]

    for itri in range(tris.shape[1]):
        
        dc = diadic[:,:,itri]

        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2,2))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.0001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TETRAHEDRON #########
        
        x = xs[inside==1]
        y = ys[inside==1]

        xvs = nodes[0, tris[:,itri]]
        yvs = nodes[1, tris[:,itri]]

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)

        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s
        
        # original Nedelec-1 order 2 formulation 
        #hx =  (4*A*(c1*(e10*(-A + a1 + b1*x + c1*y) + e11*(-A + a1 + b1*x + c1*y) + e12*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y) + e14*(a2 + b2*x + c2*y) + e9*(-A + a1 + b1*x + c1*y)) + (a1 + b1*x + c1*y)*(c1*e10 + c1*e11 + c1*e9 + c2*e12 + c2*e13 + c2*e14)) + jB*(e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y)))/(8*A**3)
        # hy =  (4*A*(b1*(e10*(-A + a1 + b1*x + c1*y) + e11*(-A + a1 + b1*x + c1*y) + e12*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y) + e14*(a2 + b2*x + c2*y) + e9*(-A + a1 + b1*x + c1*y)) + (a1 + b1*x + c1*y)*(b1*e10 + b1*e11 + b1*e9 + b2*e12 + b2*e13 + b2*e14)) - jB*(e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y)))/(8*A**3)
        # hz =  (-3*a1*b1*c2*e1 - 3*a1*b1*c3*e3 + 3*a1*b2*c1*e1 + a1*b2*c3*e4 + a1*b2*c3*e8 + 3*a1*b3*c1*e3 - a1*b3*c2*e4 - a1*b3*c2*e8 - 3*a2*b1*c2*e5 + 2*a2*b1*c3*e4 - a2*b1*c3*e8 + 3*a2*b2*c1*e5 - 3*a2*b2*c3*e2 - 2*a2*b3*c1*e4 + a2*b3*c1*e8 + 3*a2*b3*c2*e2 + a3*b1*c2*e4 - 2*a3*b1*c2*e8 - 3*a3*b1*c3*e7 - a3*b2*c1*e4 + 2*a3*b2*c1*e8 - 3*a3*b2*c3*e6 + 3*a3*b3*c1*e7 + 3*a3*b3*c2*e6 - 3*b1**2*c2*e1*x - 3*b1**2*c3*e3*x + 3*b1*b2*c1*e1*x - 3*b1*b2*c2*e5*x + 3*b1*b2*c3*e4*x + 3*b1*b3*c1*e3*x - 3*b1*b3*c2*e8*x - 3*b1*b3*c3*e7*x - 3*b1*c1*c2*e1*y - 3*b1*c1*c3*e3*y - 3*b1*c2**2*e5*y + 3*b1*c2*c3*e4*y - 3*b1*c2*c3*e8*y - 3*b1*c3**2*e7*y + 3*b2**2*c1*e5*x - 3*b2**2*c3*e2*x - 3*b2*b3*c1*e4*x + 3*b2*b3*c1*e8*x + 3*b2*b3*c2*e2*x - 3*b2*b3*c3*e6*x + 3*b2*c1**2*e1*y + 3*b2*c1*c2*e5*y + 3*b2*c1*c3*e8*y - 3*b2*c2*c3*e2*y - 3*b2*c3**2*e6*y + 3*b3**2*c1*e7*x + 3*b3**2*c2*e6*x + 3*b3*c1**2*e3*y - 3*b3*c1*c2*e4*y + 3*b3*c1*c3*e7*y + 3*b3*c2**2*e2*y + 3*b3*c2*c3*e6*y)/(8*A**3)
       
        # New Nedelec-1 order 2 formulation
        hx = (4*A*(2*c1*e12*(a2 + b2*x + c2*y) + 2*c1*e14*(a3 + b3*x + c3*y) + c1*e9*(a1 + b1*x + c1*y) - c1*e9*(A - a1 - b1*x - c1*y) + c2*e10*(a2 + b2*x + c2*y) - c2*e10*(A - a2 - b2*x - c2*y) + 2*c2*e12*(a1 + b1*x + c1*y) + 2*c2*e13*(a3 + b3*x + c3*y) + c3*e11*(a3 + b3*x + c3*y) - c3*e11*(A - a3 - b3*x - c3*y) + 2*c3*e13*(a2 + b2*x + c2*y) + 2*c3*e14*(a1 + b1*x + c1*y)) + jB*(2*A*(e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))) + e4*((c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) + e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) + e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) + e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) - e8*((c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y))))/(8*A**3)
        hy = (4*A*(-2*b1*e12*(a2 + b2*x + c2*y) - 2*b1*e14*(a3 + b3*x + c3*y) - b1*e9*(a1 + b1*x + c1*y) + b1*e9*(A - a1 - b1*x - c1*y) - b2*e10*(a2 + b2*x + c2*y) + b2*e10*(A - a2 - b2*x - c2*y) - 2*b2*e12*(a1 + b1*x + c1*y) - 2*b2*e13*(a3 + b3*x + c3*y) - b3*e11*(a3 + b3*x + c3*y) + b3*e11*(A - a3 - b3*x - c3*y) - 2*b3*e13*(a2 + b2*x + c2*y) - 2*b3*e14*(a1 + b1*x + c1*y)) - jB*(2*A*(e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))) + e4*((b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) + e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) + e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) + e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) - e8*((b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y))))/(8*A**3)
        hz = (4*A*(e1*(b1*c2 - b2*c1) + e2*(b2*c3 - b3*c2) + e3*(b1*c3 - b3*c1)) - e4*(b1*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + b2*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y)) - (b1*c3 - b3*c1)*(a2 + b2*x + c2*y) - (b2*c3 - b3*c2)*(a1 + b1*x + c1*y)) + e4*(c1*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + c2*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y)) + (b1*c3 - b3*c1)*(a2 + b2*x + c2*y) + (b2*c3 - b3*c2)*(a1 + b1*x + c1*y)) - e5*(b1 - b2)*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) + e5*(c1 - c2)*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + 2*e5*(b1*c2 - b2*c1)*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) - e6*(b2 - b3)*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + e6*(c2 - c3)*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + 2*e6*(b2*c3 - b3*c2)*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) - e7*(b1 - b3)*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y)) + e7*(c1 - c3)*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y)) + 2*e7*(b1*c3 - b3*c1)*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) + e8*(b2*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y)) + b3*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) - (b1*c2 - b2*c1)*(a3 + b3*x + c3*y) - (b1*c3 - b3*c1)*(a2 + b2*x + c2*y)) - e8*(c2*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y)) + c3*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + (b1*c2 - b2*c1)*(a3 + b3*x + c3*y) + (b1*c3 - b3*c1)*(a2 + b2*x + c2*y)))/(8*A**3)     
        
        Ex[inside] = hx*dc[0,0]
        Ey[inside] = hy*dc[1,1]
        Ez[inside] = hz*dc[2,2]
    return Ex, Ey, Ez
