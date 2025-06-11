
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

from numba import njit, f8, i8, types, c16, prange
import numpy as np

from .optimized import local_mapping, cross_c, dot_c, cross, dot, compute_distances, tet_coefficients, tet_coefficients_bcd,volume_coeff, area_coeff, tri_coefficients

NFILL = 5
VOLUME_COEFF_CACHE_BASE = np.zeros((NFILL,NFILL,NFILL,NFILL), dtype=np.float64)
for I in range(NFILL):
    for J in range(NFILL):
        for K in range(NFILL):
            for L in range(NFILL):
                VOLUME_COEFF_CACHE_BASE[I,J,K,L] = volume_coeff(I,J,K,L)

VOLUME_COEFF_CACHE = VOLUME_COEFF_CACHE_BASE

@njit(c16[:](c16[:,:], c16[:]), cache=True, nogil=True)
def matmul(Mat, Vec):
    ## Matrix multiplication of a 3D vector
    Vout = np.empty((3,), dtype=np.complex128)
    Vout[0] = Mat[0,0]*Vec[0] + Mat[0,1]*Vec[1] + Mat[0,2]*Vec[2]
    Vout[1] = Mat[1,0]*Vec[0] + Mat[1,1]*Vec[1] + Mat[1,2]*Vec[2]
    Vout[2] = Mat[2,0]*Vec[0] + Mat[2,1]*Vec[1] + Mat[2,2]*Vec[2]
    return Vout
    
#########################################
###### LEGRANGE 2 BASIS FUNCTIONS ########
#########################################

def leg2_tet_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tets: np.ndarray, 
                    tris: np.ndarray,
                    edges: np.ndarray,
                    nodes: np.ndarray,
                    tet_to_field: np.ndarray,
                    tet_to_edge: np.ndarray,
                    tet_to_tri: np.ndarray,
                    tet_ids: np.ndarray):

    # Solution has shape (nEdges, nsols)
    nPoints = coords.shape[1]
    nEdges = edges.shape[1]

    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]

    Potential = np.zeros((nPoints, ), dtype=np.complex128)

    for itet in tet_ids:

        iv1, iv2, iv3, iv4 = tets[:, itet]

        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_edge[:, itet]]
        g_tri_ids = tris[:, tet_to_tri[:,itet]]

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
        
        Ev = Etet[0:4]
        Ee = Etet[4:]
        
        
        pot = np.zeros(x.shape, dtype=np.complex128)
        

        for ie in range(4):
            e = Ev[ie]
            
            a1 = a_s[ie]
            b1 = b_s[ie]
            c1 = c_s[ie]
            d1 = d_s[ie]

            Vv = e*(a1 + b1*x + c1*y + d1*z)*(-3*V + a1 + b1*x + c1*y + d1*z)/(18*V**2)
            
            pot += Vv

        for ie in range(6):
            e = Ee[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            d1, d2 = d_s[edgeids]

            Ve = e*(a1 + b1*x + c1*y + d1*z)*(a2 + b2*x + c2*y + d2*z)/(9*V**2)

            pot += Ve
            
        
        
        Potential[inside] = pot
    return Potential

def leg2_tet_grad_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tets: np.ndarray, 
                    tris: np.ndarray,
                    edges: np.ndarray,
                    nodes: np.ndarray,
                    tet_to_field: np.ndarray,
                    tet_to_edge: np.ndarray,
                    tet_to_tri: np.ndarray,
                    tet_ids: np.ndarray):

    # Solution has shape (nEdges, nsols)
    nPoints = coords.shape[1]
    nEdges = edges.shape[1]

    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]

    dPdX = np.zeros((nPoints, ), dtype=np.complex128)
    dPdY = np.zeros((nPoints, ), dtype=np.complex128)
    dPdZ = np.zeros((nPoints, ), dtype=np.complex128)

    for itet in tet_ids:

        iv1, iv2, iv3, iv4 = tets[:, itet]

        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_edge[:, itet]]
        g_tri_ids = tris[:, tet_to_tri[:,itet]]

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

        inside = ((coords_local[0,:] + coords_local[1,:] + coords_local[2,:]) <= 1.0001) & (coords_local[0,:] >= -1e-5) & (coords_local[1,:] >= -1e-5) & (coords_local[2,:] >= -1e-5)

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
        
        Ev = Etet[0:4]
        Ee = Etet[4:]
        
        
        dpdx = np.zeros(x.shape, dtype=np.complex128)
        dpdy = np.zeros(x.shape, dtype=np.complex128)
        dpdz = np.zeros(x.shape, dtype=np.complex128)
        

        for ie in range(4):
            e = Ev[ie]
            
            a1 = a_s[ie]
            b1 = b_s[ie]
            c1 = c_s[ie]
            d1 = d_s[ie]

            dvdx = b1*e*(-3*V + 2*a1 + 2*b1*x + 2*c1*y + 2*d1*z)/(18*V**2)
            dvdy = c1*e*(-3*V + 2*a1 + 2*b1*x + 2*c1*y + 2*d1*z)/(18*V**2)
            dvdz = d1*e*(-3*V + 2*a1 + 2*b1*x + 2*c1*y + 2*d1*z)/(18*V**2)
            dpdx += dvdx
            dpdy += dvdy
            dpdz += dvdz

        for ie in range(6):
            e = Ee[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            d1, d2 = d_s[edgeids]

            dedx = e*(b1*(a2 + b2*x + c2*y + d2*z) + b2*(a1 + b1*x + c1*y + d1*z))/(9*V**2)
            dedy = e*(c1*(a2 + b2*x + c2*y + d2*z) + c2*(a1 + b1*x + c1*y + d1*z))/(9*V**2)
            dedz = e*(d1*(a2 + b2*x + c2*y + d2*z) + d2*(a1 + b1*x + c1*y + d1*z))/(9*V**2)
            dpdx += dedx
            dpdy += dedy
            dpdz += dedz
            
        
        
        dPdX[inside] = dpdx
        dPdY[inside] = dpdy
        dPdZ[inside] = dpdz
    return dPdX, dPdY, dPdZ


#@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], i8[:,:], f8, f8), cache=True)
def leg2_tet_stiff(tet_vertices, edge_lengths, local_edge_map, local_tri_map, C_stiffness):
    ''' Computes the curl dot curl submatrix terms
    
    Submatrix indexing:
    -------------------
    0, 1, 2, 3, 4, 5 = Edge_i mode 1 coefficients
    6, 7, 8, 9 = Face_i vector component 1
    10, 11, 12, 13, 14, 15 = Edge mode 2 coefficients
    16, 17, 18, 19 = Face_i vector component 2

    '''
    Dmat = np.zeros((10,10), dtype=np.complex128)

    xs, ys, zs = tet_vertices

    aas, bbs, ccs, dds, V = tet_coefficients(xs, ys, zs)
    a1, a2, a3, a4 = aas
    b1, b2, b3, b4 = bbs
    c1, c2, c3, c4 = ccs
    d1, d2, d3, d4 = dds
    
    Ds = compute_distances(xs, ys, zs)

    GL1 = np.array([b1, c1, d1])
    GL2 = np.array([b2, c2, d2])
    GL3 = np.array([b3, c3, d3])
    GL4 = np.array([b4, c4, d4])

    GLs = (GL1, GL2, GL3, GL4)

    letters = [1,2,3,4,5,6]

    V6 = 6*V
    for ei in range(4):
        #ei1, ei2 = local_edge_map[:, ei]
        for ej in range(4):
            #ej1, ej2 = local_edge_map[:, ej]
            
            A,B,C,D = letters[ei], letters[ej]#, letters[ej1], letters[ej2]
            GA = GLs[1j]
            GC = GLs[1j]

            KA = 1/(6*V)**4 
            
            dotGAGC = dot(GA,GC)

            VAC = V6*VOLUME_COEFF_CACHE[A,C,0,0]
            
            Dmat[ei,ej] += KA*(8*V6*VOLUME_COEFF_CACHE[A,A,0,0]*dotGAGC+8*VAC*dot(GA,GA)-6*A*dotGAGC-2*C*dot(GA,GA)+dot(GA,GC))
    for ei in range(4):
        for ej in range(6):
            ej1, ej2  = local_tri_map[:, ej]

            A,C,D = letters[ei],letters[ej1], letters[ej2]
            GA = GLs[ei]
            GC = GLs[ej1]
            GD = GLs[ej2]
            KA = 1/(6*V)**4 

            dotGAGC = dot(GA,GC)
            dotGAGD = dot(GA,GD)
            VAD = V6*VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = V6*VOLUME_COEFF_CACHE[A,C,0,0]

            Dmat[ei,ej+4] += KA*(16*VAD*dotGAGC+16*VAC*dotGAGD-4*D*dotGAGC-4*C*dotGAGD)
            Dmat[ej+4,ei] += KA*(16*VAD*dotGAGC+16*VAC*dotGAGD-4*D*dotGAGC-4*C*dotGAGD)
            
    for ei in range(6):
        ei1, ei2 = local_tri_map[:, ei]
        for ej in range(6):
            ej1, ej2 = local_tri_map[:, ej]
            
            A,B,C,D = letters[ei1], letters[ei2], letters[ej1], letters[ej2]
            GA = GLs[ei1]
            GB = GLs[ei2]
            GC = GLs[ej1]
            GD = GLs[ej2]

            KA = 1/(6*V)**4 
            dotGAGC = dot(GA,GC)
            dotGAGD = dot(GA,GD)
            dotGBGC = dot(GB,GC)
            dotGBGD = dot(GB,GD)
            VBD = V6*VOLUME_COEFF_CACHE[B,D,0,0]
            VBC = V6*VOLUME_COEFF_CACHE[B,C,0,0]
            VAD = V6*VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = V6*VOLUME_COEFF_CACHE[A,C,0,0]

            Dmat[ei+4,ej+4] += KA*(16*VBD*dotGAGC+16*VBC*dotGAGD+16*VAD*dotGBGC+16*VAC*dotGBGD)

    Dmat = Dmat/C_stiffness
    
    return Dmat

#########################################
###### NEDELEC 2 BASIS FUNCTIONS ########
#########################################

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


@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], i8[:,:], f8, f8), cache=True, parallel=True, fastmath=True)
def ned2_tet_stiff_mass_OLD(tet_vertices, edge_lengths, local_edge_map, local_tri_map, C_stiffness, C_mass):
    ''' Nedelec 2 tetrahedral stiffness and mass matrix submatrix Calculation

    '''
    Dmat = np.zeros((20,20), dtype=np.complex128)
    Fmat = np.zeros((20,20), dtype=np.complex128)

    xs, ys, zs = tet_vertices

    aas, bbs, ccs, dds, V = tet_coefficients(xs, ys, zs)
    b1, b2, b3, b4 = bbs
    c1, c2, c3, c4 = ccs
    d1, d2, d3, d4 = dds
    
    Ds = compute_distances(xs, ys, zs)

    GL1 = np.array([b1, c1, d1])
    GL2 = np.array([b2, c2, d2])
    GL3 = np.array([b3, c3, d3])
    GL4 = np.array([b4, c4, d4])

    GLs = (GL1, GL2, GL3, GL4)

    letters = [1,2,3,4,5,6]

    KA = 1/(6*V)**4
    KB = 1/(6*V)**2

    V6 = 6*V

    for ei in range(6):
        ei1, ei2 = local_edge_map[:, ei]
        GA = GLs[ei1]
        GB = GLs[ei2]
        A, B = letters[ei1], letters[ei2]
        Li = edge_lengths[ei]

        
        for ej in range(6):
            ej1, ej2 = local_edge_map[:, ej]
            
            C,D = letters[ej1], letters[ej2]

            GC = GLs[ej1]
            GD = GLs[ej2]

            GAxGB = cross(GA,GB)
            GCxGD = cross(GC,GD)
            
            dotGAGC = dot(GA,GC)
            dotGAGD = dot(GA,GD)
            dotGBGC = dot(GB,GC)
            dotGBGD = dot(GB,GD)
            dotGAxGBGCxGD = dot(GAxGB,GCxGD)
            
            VAD = V6*VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = V6*VOLUME_COEFF_CACHE[A,C,0,0]
            VBC = V6*VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = V6*VOLUME_COEFF_CACHE[B,D,0,0]
            VABCD = V6*VOLUME_COEFF_CACHE[A,B,C,D]
            VABCC = V6*VOLUME_COEFF_CACHE[A,B,C,C]
            VABDD = V6*VOLUME_COEFF_CACHE[A,B,D,D]
            VBBCD = V6*VOLUME_COEFF_CACHE[B,B,C,D]
            VABCD = V6*VOLUME_COEFF_CACHE[A,B,C,D]
            VBBCD = V6*VOLUME_COEFF_CACHE[B,B,C,D]

            Lj = edge_lengths[ej]

            Dmat[ei,ej] += Li*Lj*KA*(9*VAC*dotGAxGBGCxGD)
            Fmat[ei,ej] += Li*Lj*KB*(VABCD*dotGAGC-VABCC*dotGAGD-V6*VOLUME_COEFF_CACHE[A,A,C,D]*dotGBGC+V6*VOLUME_COEFF_CACHE[A,A,C,C]*dotGBGD)
            Dmat[ei,ej+10] += Li*Lj*KA*(9*VAD*dotGAxGBGCxGD)
            Fmat[ei,ej+10] += Li*Lj*KB*(VABDD*dotGAGC-VABCD*dotGAGD-V6*VOLUME_COEFF_CACHE[A,A,D,D]*dotGBGC+V6*VOLUME_COEFF_CACHE[A,A,C,D]*dotGBGD)
            Dmat[ei+10,ej] += Li*Lj*KA*(9*VBC*dotGAxGBGCxGD)
            Fmat[ei+10,ej] += Li*Lj*KB*(VBBCD*dotGAGC-V6*VOLUME_COEFF_CACHE[B,B,C,C]*dotGAGD-VABCD*dotGBGC+VABCC*dotGBGD)
            Dmat[ei+10,ej+10] += Li*Lj*KA*(9*VBD*dotGAxGBGCxGD)
            Fmat[ei+10,ej+10] += Li*Lj*KB*(V6*VOLUME_COEFF_CACHE[B,B,D,D]*dotGAGC-VBBCD*dotGAGD-VABDD*dotGBGC+VABCD*dotGBGD)
    
    for ei in range(6):
        ei1, ei2 = local_edge_map[:, ei]
        GA = GLs[ei1]
        GB = GLs[ei2]
        A,B = letters[ei1], letters[ei2]

        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]

            C,D,F = letters[ej1], letters[ej2], letters[fj]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            GF = GLs[fj]

            dotGAGC = dot(GA,GC)
            dotGAGD = dot(GA,GD)
            dotGBGC = dot(GB,GC)
            dotGBGD = dot(GB,GD)
            dotGBGF = dot(GB,GF)
            dotGAGF = dot(GA,GF)

            GCxGD = cross(GC,GD)
            GAxGB = cross(GA,GB)
            GCxGF = cross(GC,GF)
            GDxGF = cross(GD,GF)

            dotGAxGBGCxGD = dot(GAxGB,GCxGD)
            dotGaxGBGCxGF = dot(GAxGB,GCxGF)
            dotGAxGBGDxGF = dot(GAxGB,GDxGF)
            
            VABCD = V6*VOLUME_COEFF_CACHE[A,B,C,D]
            VBBCD = V6*VOLUME_COEFF_CACHE[B,B,C,D]
            VAD = V6*VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = V6*VOLUME_COEFF_CACHE[A,C,0,0]
            VAF = V6*VOLUME_COEFF_CACHE[A,F,0,0]
            VBF = V6*VOLUME_COEFF_CACHE[B,F,0,0]
            VBC = V6*VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = V6*VOLUME_COEFF_CACHE[B,D,0,0]
            VABDF = V6*VOLUME_COEFF_CACHE[A,B,D,F]
            VABFC = V6*VOLUME_COEFF_CACHE[A,B,F,C]
            VAADF = V6*VOLUME_COEFF_CACHE[A,A,D,F]

            Li = edge_lengths[ei]
            Lab = Ds[ej1, ej2]
            Lac = Ds[ej1, fj]

            Dmat[ei,ej+6] += Li*Lac*KA*(-6*VAD*dotGaxGBGCxGF-3*VAC*dotGAxGBGDxGF-3*VAF*dotGAxGBGCxGD)
            Fmat[ei,ej+6] += Li*Lac*KB*(VABCD*dotGAGF-VABDF*dotGAGC-V6*VOLUME_COEFF_CACHE[A,A,C,D]*dotGBGF+VAADF*dotGBGC)
            Dmat[ei,ej+16] += Li*Lab*KA*(6*VAF*dotGAxGBGCxGD+3*VAD*dotGaxGBGCxGF-3*VAC*dotGAxGBGDxGF)
            Fmat[ei,ej+16] += Li*Lab*KB*(VABDF*dotGAGC-VABFC*dotGAGD-VAADF*dotGBGC+V6*VOLUME_COEFF_CACHE[A,A,F,C]*dotGBGD)
            Dmat[ei+10,ej+6] += Li*Lac*KA*(-6*VBD*dotGaxGBGCxGF-3*VBC*dotGAxGBGDxGF-3*VBF*dotGAxGBGCxGD)
            Fmat[ei+10,ej+6] += Li*Lac*KB*(VBBCD*dotGAGF-V6*VOLUME_COEFF_CACHE[B,B,D,F]*dotGAGC-VABCD*dotGBGF+VABDF*dotGBGC)
            Dmat[ei+10,ej+16] += Li*Lab*KA*(6*VBF*dotGAxGBGCxGD+3*VBD*dotGaxGBGCxGF-3*VBC*dotGAxGBGDxGF)
            Fmat[ei+10,ej+16] += Li*Lab*KB*(V6*VOLUME_COEFF_CACHE[B,B,D,F]*dotGAGC-V6*VOLUME_COEFF_CACHE[B,B,F,C]*dotGAGD-VABDF*dotGBGC+VABFC*dotGBGD)

    ## Mirror the transpose part of the previous iteration as its symmetrical

    Dmat[6:10, :6] += Dmat[:6, 6:10].T
    Fmat[6:10, :6] += Fmat[:6, 6:10].T
    Dmat[16:20, :6] += Dmat[:6, 16:20].T
    Fmat[16:20, :6] += Fmat[:6, 16:20].T
    Dmat[6:10, 10:16] += Dmat[10:16, 6:10].T
    Fmat[6:10, 10:16] += Fmat[10:16, 6:10].T
    Dmat[16:20, 10:16] += Dmat[10:16, 16:20].T
    Fmat[16:20, 10:16] += Fmat[10:16, 16:20].T
    
    for ei in range(4):
        ei1, ei2, fi = local_tri_map[:, ei]
        A, B, E = letters[ei1], letters[ei2], letters[fi]
        GA = GLs[ei1]
        GB = GLs[ei2]
        GE = GLs[fi]
        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]
            
            C,D,F = letters[ej1], letters[ej2], letters[fj]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            GF = GLs[fj]

            GCxGD = cross(GC,GD)
            GCxGF = cross(GC,GF)
            GAxGB = cross(GA,GB)
            GAxGE = cross(GA,GE)
            GDxGF = cross(GD,GF)
            GBxGE = cross(GB,GE)

            dotGAGC = dot(GA,GC)
            dotGAGD = dot(GA,GD)
            dotGBGC = dot(GB,GC)
            dotGBGD = dot(GB,GD)
            dotGAGF = dot(GA,GF)
            dotGCGE = dot(GC,GE)
            dotGBGF = dot(GB,GF)
            

            dotGAxGBGCxGD = dot(GAxGB,GCxGD)
            dotGAxGEGCxGF = dot(GAxGE,GCxGF)
            dotGAxGEGDxGF = dot(GAxGE,GDxGF)
            dotGBxGEGCxGD = dot(GBxGE,GCxGD)
            dotGAxGEGCxGD = dot(GAxGE,GCxGD)
            dotGBxGEGDxGF = dot(GBxGE,GDxGF)
            dotGBxGEGCxGF = dot(GBxGE,GCxGF)
            dotGaxGBGCxGF = dot(GAxGB,GCxGF)
            dotGAxGBGDxGF = dot(GAxGB,GDxGF)

            VABCD = V6*VOLUME_COEFF_CACHE[A,B,C,D]
            VAD = V6*VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = V6*VOLUME_COEFF_CACHE[A,C,0,0]
            VAF = V6*VOLUME_COEFF_CACHE[A,F,0,0]
            VBF = V6*VOLUME_COEFF_CACHE[B,F,0,0]
            VBC = V6*VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = V6*VOLUME_COEFF_CACHE[B,D,0,0]
            VED = V6*VOLUME_COEFF_CACHE[E,D,0,0]
            VEF = V6*VOLUME_COEFF_CACHE[E,F,0,0]
            VEC = V6*VOLUME_COEFF_CACHE[E,C,0,0]
            VABDF = V6*VOLUME_COEFF_CACHE[A,B,D,F]
            VABFC = V6*VOLUME_COEFF_CACHE[A,B,F,C]
            VBECD = V6*VOLUME_COEFF_CACHE[B,C,D,F]
            VBEDF = V6*VOLUME_COEFF_CACHE[B,E,D,F]
            VEACD = V6*VOLUME_COEFF_CACHE[E,A,C,D]
            VBEFC = V6*VOLUME_COEFF_CACHE[B,E,F,C]
            VEADF = V6*VOLUME_COEFF_CACHE[E,A,D,F]

            Lac1 = Ds[ei1, fi]
            Lab1 = Ds[ei1, ei2]
            Lac2 = Ds[ej1, fj]
            Lab2 = Ds[ej1, ej2]

            Dmat[ei+6,ej+6] += Lac1*Lac2*KA*(4*VBD*dotGAxGEGCxGF+2*VBC*dotGAxGEGDxGF+2*VBF*dotGAxGEGCxGD+2*VAD*dotGBxGEGCxGF+VAC*dotGBxGEGDxGF+VAF*dotGBxGEGCxGD+2*VED*dotGaxGBGCxGF+VEC*dotGAxGBGDxGF+VEF*dotGAxGBGCxGD)
            Fmat[ei+6,ej+6] += Lac1*Lac2*KB*(VABCD*dot(GE,GF)-VABDF*dotGCGE-VBECD*dotGAGF+VBEDF*dotGAGC)
            Dmat[ei+6,ej+16] += Lac1*Lab2*KA*(-4*VBF*dotGAxGEGCxGD-2*VBD*dotGAxGEGCxGF+2*VBC*dotGAxGEGDxGF-2*VAF*dotGBxGEGCxGD-VAD*dotGBxGEGCxGF+VAC*dotGBxGEGDxGF-2*VEF*dotGAxGBGCxGD-VED*dotGaxGBGCxGF+VEC*dotGAxGBGDxGF)
            Fmat[ei+6,ej+16] += Lac1*Lab2*KB*(VABDF*dotGCGE-VABFC*dot(GD,GE)-VBEDF*dotGAGC+VBEFC*dotGAGD)
            Dmat[ei+16,ej+6] += Lab1*Lac2*KA*(-4*VED*dotGaxGBGCxGF-2*VEC*dotGAxGBGDxGF-2*VEF*dotGAxGBGCxGD-2*VBD*dotGAxGEGCxGF-VBC*dotGAxGEGDxGF-VBF*dotGAxGEGCxGD+2*VAD*dotGBxGEGCxGF+VAC*dotGBxGEGDxGF+VAF*dotGBxGEGCxGD)
            Fmat[ei+16,ej+6] += Lab1*Lac2*KB*(VBECD*dotGAGF-VBEDF*dotGAGC-VEACD*dotGBGF+VEADF*dotGBGC)
            Dmat[ei+16,ej+16] += Lab1*Lab2*KA*(4*VEF*dotGAxGBGCxGD+2*VED*dotGaxGBGCxGF-2*VEC*dotGAxGBGDxGF+2*VBF*dotGAxGEGCxGD+VBD*dotGAxGEGCxGF-VBC*dotGAxGEGDxGF-2*VAF*dotGBxGEGCxGD-VAD*dotGBxGEGCxGF+VAC*dotGBxGEGDxGF)
            Fmat[ei+16,ej+16] += Lab1*Lab2*KB*(VBEDF*dotGAGC-VBEFC*dotGAGD-VEADF*dotGBGC+V6*VOLUME_COEFF_CACHE[E,A,F,C]*dotGBGD)


    Dmat = Dmat*C_stiffness
    Fmat = Fmat*C_mass
    
    return Dmat, Fmat


@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], i8[:,:], c16[:,:], c16[:,:]), nogil=True, cache=True, parallel=False, fastmath=True)
def ned2_tet_stiff_mass(tet_vertices, edge_lengths, local_edge_map, local_tri_map, Ms, Mm):
    ''' Nedelec 2 tetrahedral stiffness and mass matrix submatrix Calculation

    '''
    
    Dmat = np.empty((20,20), dtype=np.complex128)
    Fmat = np.empty((20,20), dtype=np.complex128)

    xs, ys, zs = tet_vertices

    bbs, ccs, dds, V = tet_coefficients_bcd(xs, ys, zs)
    b1, b2, b3, b4 = bbs
    c1, c2, c3, c4 = ccs
    d1, d2, d3, d4 = dds
    
    Ds = compute_distances(xs, ys, zs)

    GL1 = np.array([b1, c1, d1]).astype(np.complex128)
    GL2 = np.array([b2, c2, d2]).astype(np.complex128)
    GL3 = np.array([b3, c3, d3]).astype(np.complex128)
    GL4 = np.array([b4, c4, d4]).astype(np.complex128)

    GLs = (GL1, GL2, GL3, GL4)

    letters = [1,2,3,4,5,6]

    KA = 1/(6*V)**4
    KB = 1/(6*V)**2

    V6 = 6*V

    VOLUME_COEFF_CACHE = VOLUME_COEFF_CACHE_BASE*V6
    for ei in range(6):
        ei1, ei2 = local_edge_map[:, ei]
        GA = GLs[ei1]
        GB = GLs[ei2]
        A, B = letters[ei1], letters[ei2]
        L1 = edge_lengths[ei]
        
        
        for ej in range(6):
            ej1, ej2 = local_edge_map[:, ej]
            
            C,D = letters[ej1], letters[ej2]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            
            VAD = VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = VOLUME_COEFF_CACHE[A,C,0,0]
            VBC = VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = VOLUME_COEFF_CACHE[B,D,0,0]
            VABCD = VOLUME_COEFF_CACHE[A,B,C,D]
            VABCC = VOLUME_COEFF_CACHE[A,B,C,C]
            VABDD = VOLUME_COEFF_CACHE[A,B,D,D]
            VBBCD = VOLUME_COEFF_CACHE[B,B,C,D]
            VABCD = VOLUME_COEFF_CACHE[A,B,C,D]
            VBBCD = VOLUME_COEFF_CACHE[B,B,C,D]
            VAACD = VOLUME_COEFF_CACHE[A,A,C,D]
            VAADD = VOLUME_COEFF_CACHE[A,A,D,D]
            VBBCC = VOLUME_COEFF_CACHE[B,B,C,C]
            VBBDD = VOLUME_COEFF_CACHE[B,B,D,D]
            VAACC = VOLUME_COEFF_CACHE[A,A,C,C]

            L2 = edge_lengths[ej]

            BB1 = matmul(Mm,GC)
            BC1 = matmul(Mm,GD)
            BD1 = dot_c(GA,BB1)
            BE1 = dot_c(GA,BC1)
            BF1 = dot_c(GB,BB1)
            BG1 = dot_c(GB,BC1)

            Q2 = L1*L2
            Q = Q2*9*dot_c(cross_c(GA,GB),matmul(Ms,cross_c(GC,GD)))
            Dmat[ei+0,ej+0] = Q*VAC
            Dmat[ei+0,ej+10] = Q*VAD
            Dmat[ei+10,ej+0] = Q*VBC
            Dmat[ei+10,ej+10] = Q*VBD

            
            Fmat[ei+0,ej+0] = Q2*(VABCD*BD1-VABCC*BE1-VAACD*BF1+VAACC*BG1)
            Fmat[ei+0,ej+10] = Q2*(VABDD*BD1-VABCD*BE1-VAADD*BF1+VAACD*BG1)
            Fmat[ei+10,ej+0] = Q2*(VBBCD*BD1-VBBCC*BE1-VABCD*BF1+VABCC*BG1)
            Fmat[ei+10,ej+10] = Q2*(VBBDD*BD1-VBBCD*BE1-VABDD*BF1+VABCD*BG1)       

        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]

            C,D,F = letters[ej1], letters[ej2], letters[fj]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            GF = GLs[fj]

            VABCD = VOLUME_COEFF_CACHE[A,B,C,D]
            VBBCD = VOLUME_COEFF_CACHE[B,B,C,D]
            VAD = VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = VOLUME_COEFF_CACHE[A,C,0,0]
            VAF = VOLUME_COEFF_CACHE[A,F,0,0]
            VBF = VOLUME_COEFF_CACHE[B,F,0,0]
            VBC = VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = VOLUME_COEFF_CACHE[B,D,0,0]
            VABDF = VOLUME_COEFF_CACHE[A,B,D,F]
            VABCF = VOLUME_COEFF_CACHE[A,B,F,C]
            VAADF = VOLUME_COEFF_CACHE[A,A,D,F]
            VAACD = VOLUME_COEFF_CACHE[A,A,C,D]
            VBBDF = VOLUME_COEFF_CACHE[B,B,D,F]
            VBBCD = VOLUME_COEFF_CACHE[B,B,C,D]
            VBBCF = VOLUME_COEFF_CACHE[B,B,F,C]
            VAACF = VOLUME_COEFF_CACHE[A,A,C,F]

            Lab2 = Ds[ej1, ej2]
            Lac2 = Ds[ej1, fj]
            
            AB1 = cross_c(GA,GB)
            AI1 = dot_c(AB1,matmul(Ms,cross_c(GC,GF)))
            AJ1 = dot_c(AB1,matmul(Ms,cross_c(GD,GF)))
            AK1 = dot_c(AB1,matmul(Ms,cross_c(GC,GD)))
            BB1 = matmul(Mm,GF)
            BC1 = matmul(Mm,GC)
            BD1 = matmul(Mm,GD)
            BE1 = dot_c(GA,BB1)
            BF1 = dot_c(GA,BC1)
            BG1 = dot_c(GB,BB1)
            BH1 = dot_c(GB,BC1)
            BI1 = dot_c(GA,BD1)
            BJ1 = dot_c(GB,BD1)

            
            Dmat[ei+0,ej+6] = L1*Lac2*(-6*VAD*AI1-3*VAC*AJ1-3*VAF*AK1)
            Dmat[ei+0,ej+16] = L1*Lab2*(6*VAF*AK1+3*VAD*AI1-3*VAC*AJ1)
            Dmat[ei+10,ej+6] = L1*Lac2*(-6*VBD*AI1-3*VBC*AJ1-3*VBF*AK1)
            Dmat[ei+10,ej+16] = L1*Lab2*(6*VBF*AK1+3*VBD*AI1-3*VBC*AJ1)

            Fmat[ei+0,ej+6] = L1*Lac2*(VABCD*BE1-VABDF*BF1-VAACD*BG1+VAADF*BH1)
            Fmat[ei+0,ej+16] = L1*Lab2*(VABDF*BF1-VABCF*BI1-VAADF*BH1+VAACF*BJ1)
            Fmat[ei+10,ej+6] = L1*Lac2*(VBBCD*BE1-VBBDF*BF1-VABCD*BG1+VABDF*BH1)
            Fmat[ei+10,ej+16] = L1*Lab2*(VBBDF*BF1-VBBCF*BI1-VABDF*BH1+VABCF*BJ1)
    
    ## Mirror the transpose part of the previous iteration as its symmetrical

    Dmat[6:10, :6] = Dmat[:6, 6:10].T
    Fmat[6:10, :6] = Fmat[:6, 6:10].T
    Dmat[16:20, :6] = Dmat[:6, 16:20].T
    Fmat[16:20, :6] = Fmat[:6, 16:20].T
    Dmat[6:10, 10:16] = Dmat[10:16, 6:10].T
    Fmat[6:10, 10:16] = Fmat[10:16, 6:10].T
    Dmat[16:20, 10:16] = Dmat[10:16, 16:20].T
    Fmat[16:20, 10:16] = Fmat[10:16, 16:20].T
    
    for ei in range(4):
        ei1, ei2, fi = local_tri_map[:, ei]
        A, B, E = letters[ei1], letters[ei2], letters[fi]
        GA = GLs[ei1]
        GB = GLs[ei2]
        GE = GLs[fi]
        Lac1 = Ds[ei1, fi]
        Lab1 = Ds[ei1, ei2]
        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]
            
            C,D,F = letters[ej1], letters[ej2], letters[fj]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            GF = GLs[fj]

            VABCD = VOLUME_COEFF_CACHE[A,B,C,D]
            VAD = VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = VOLUME_COEFF_CACHE[A,C,0,0]
            VAF = VOLUME_COEFF_CACHE[A,F,0,0]
            VBF = VOLUME_COEFF_CACHE[B,F,0,0]
            VBC = VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = VOLUME_COEFF_CACHE[B,D,0,0]
            VDE = VOLUME_COEFF_CACHE[E,D,0,0]
            VEF = VOLUME_COEFF_CACHE[E,F,0,0]
            VCE = VOLUME_COEFF_CACHE[E,C,0,0]
            VABDF = VOLUME_COEFF_CACHE[A,B,D,F]
            VACEF = VOLUME_COEFF_CACHE[A,C,E,F]
            VABCF = VOLUME_COEFF_CACHE[A,B,F,C]
            VBCDE = VOLUME_COEFF_CACHE[B,C,D,F]
            VBDEF = VOLUME_COEFF_CACHE[B,E,D,F]
            VACDE = VOLUME_COEFF_CACHE[E,A,C,D]
            VBCEF = VOLUME_COEFF_CACHE[B,E,F,C]
            VADEF = VOLUME_COEFF_CACHE[E,A,D,F]

            
            Lac2 = Ds[ej1, fj]
            Lab2 = Ds[ej1, ej2]

            AB1 = cross_c(GA,GE)
            AF1 = cross_c(GB,GE)
            AG1 = cross_c(GA,GB)
            AH1 = matmul(Ms,cross_c(GC,GF))
            AI1 = matmul(Ms,cross_c(GD,GF))
            AJ1 = matmul(Ms,cross_c(GC,GD))
            AK1 = dot_c(AB1,AH1)
            AL1 = dot_c(AB1,AI1)
            AM1 = dot_c(AB1,AJ1)
            AN1 = dot_c(AF1,AH1)
            AO1 = dot_c(AF1,AI1)
            AP1 = dot_c(AF1,AJ1)
            AQ1 = dot_c(AG1,AH1)
            AR1 = dot_c(AG1,AI1)
            AS1 = dot_c(AG1,AJ1)
            BB1 = matmul(Mm,GF)
            BC1 = matmul(Mm,GC)
            BD1 = matmul(Mm,GD)
            BE1 = dot_c(GE,BB1)
            BF1 = dot_c(GE,BC1)
            BG1 = dot_c(GA,BB1)
            BH1 = dot_c(GA,BC1)
            BI1 = dot_c(GE,BD1)
            BJ1 = dot_c(GA,BD1)
            BK1 = dot_c(GB,BB1)
            BL1 = dot_c(GB,BC1)
            BM1 = dot_c(GB,BD1)

            Q1 = 2*VAD*AN1+VAC*AO1+VAF*AP1
            Q2 = -2*VAF*AP1-VAD*AN1+VAC*AO1
            Dmat[ei+6,ej+6] = Lac1*Lac2*(4*VBD*AK1+2*VBC*AL1+2*VBF*AM1+Q1+2*VDE*AQ1+VCE*AR1+VEF*AS1)
            Dmat[ei+6,ej+16] = Lac1*Lab2*(-4*VBF*AM1-2*VBD*AK1+2*VBC*AL1+Q2-2*VEF*AS1-VDE*AQ1+VCE*AR1)
            Dmat[ei+16,ej+6] = Lab1*Lac2*(-4*VDE*AQ1-2*VCE*AR1-2*VEF*AS1-2*VBD*AK1-VBC*AL1-VBF*AM1+Q1)
            Dmat[ei+16,ej+16] = Lab1*Lab2*(4*VEF*AS1+2*VDE*AQ1-2*VCE*AR1+2*VBF*AM1+VBD*AK1-VBC*AL1+Q2)
            Fmat[ei+6,ej+6] = Lac1*Lac2*(VABCD*BE1-VABDF*BF1-VBCDE*BG1+VBDEF*BH1)
            Fmat[ei+6,ej+16] = Lac1*Lab2*(VABDF*BF1-VABCF*BI1-VBDEF*BH1+VBCEF*BJ1)
            Fmat[ei+16,ej+6] = Lab1*Lac2*(VBCDE*BG1-VBDEF*BH1-VACDE*BK1+VADEF*BL1)
            Fmat[ei+16,ej+16] = Lab1*Lab2*(VBDEF*BH1-VBCEF*BJ1-VADEF*BL1+VACEF*BM1)

    Dmat = Dmat*KA
    Fmat = Fmat*KB

    return Dmat, Fmat

