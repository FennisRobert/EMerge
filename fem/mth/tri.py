
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

from numba import njit, f8, i8, types, c16
import numpy as np

from .optimized import area_coeff, cross, compute_distances, tri_coefficients

# def njit(*args, **kwargs):
#     def wrap(x):
#         return x
#     return wrap

#########################################
###### LEGRANGE 2 BASIS FUNCTIONS #######
#########################################

@njit(f8(f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def dot(a: np.ndarray, b: np.ndarray):
    return a[0]*b[0] + a[1]*b[1]

def dirac(a,b):
    if a==b:
        return 1
    return 0

#@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], i8[:,:], f8), cache=True)
def leg2_tri_stiff(tri_vertices, local_edge_map, C_stiffness):
    ''' Computes the curl dot curl submatrix terms
    
    Submatrix indexing:
    -------------------
    0, 1, 2, 3, 4, 5 = Edge_i mode 1 coefficients
    6, 7, 8, 9 = Face_i vector component 1
    10, 11, 12, 13, 14, 15 = Edge mode 2 coefficients
    16, 17, 18, 19 = Face_i vector component 2

    '''
    Dmat = np.zeros((6,6), dtype=np.complex128)

    xs, ys, zs = tri_vertices

    aas, bbs, ccs, Area = tri_coefficients(xs, ys)
    a1, a2, a3, = aas
    b1, b2, b3 = bbs
    c1, c2, c3 = ccs
    
    Ds = compute_distances(xs, ys, zs)

    GL1 = np.array([b1, c1])
    GL2 = np.array([b2, c2])
    GL3 = np.array([b3, c3])

    GLs = (GL1, GL2, GL3)

    A = 1
    B = 2
    C = 3
    D = 4

    letters = [1,2,3,4,5,6]
    for ei in range(3):
        for ej in range(3):
            
            Dmat[ei,ej] = (4*dirac(ei,ej)-1)/(12*Area)*dot(GLs[ei],GLs[ej])
    
    
    

    Dmat[0, 3] = -4*Dmat[0,1]
    Dmat[1, 3] = -4*Dmat[0,1]
    Dmat[0, 5] = -4*Dmat[0,2]
    Dmat[2, 5] = -4*Dmat[0,2]
    Dmat[1, 4] = -4*Dmat[1,2]
    Dmat[2, 4] = -4*Dmat[1,2]

    Dmat[3, 0] = Dmat[0, 3]
    Dmat[3, 1] = Dmat[1, 3]
    Dmat[5,0] = Dmat[0, 5]
    Dmat[5,2] = Dmat[2, 5]
    Dmat[4,1] = Dmat[1, 4]
    Dmat[4,2] = Dmat[2, 4]

    
    Dmat[3,3] = 2/(3*Area)*((b1**2 + b1*b2 + b2**2) + (c1**2 + c1*c2 + c2**2))
    Dmat[4,4] = (2/(3*Area))*((b2**2 + b2*b3 + b3**2) + (c2**2 + c2*c3 + c3**2))
    Dmat[5,5] = (2/(3*Area))*((b3**2 + b1*b3 + b1**2) + (c3**2 + c1*c3 + c1**2))
    
    Dmat[3,4] = (1/(3*Area))*((b2*b3 + 2*b1*b3 + b1*b2 + b2**2) + (c2*c3 + 2*c1*c3 + c1*c2 + c2**2))
    Dmat[4,3] = Dmat[3,4]
    Dmat[3,5] = (1/(3*Area))*((b1*b3 + 2*b2*b3 + b1*b2 + b1**2) + (c1*c3 + 2*c2*c3 + c1*c2 + c1**2))
    Dmat[5,3] = Dmat[3,5]
    Dmat[4,5] = (1/(3*Area))*((b3*b1 + 2*b2*b1 + b2*b3 + b3**2) + (c3*c1 + 2*c2*c1 + c2*c3 + c3**2))
    Dmat[5,4] = Dmat[4,5]

    Dmat = Dmat*C_stiffness
    
    return Dmat


#########################################
###### NEDELEC 2 BASIS FUNCTIONS ########
#########################################

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

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6)

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

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6)

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

        mult = np.array([L1,L2,L3,L3,L1,L2,L3,L1,1,1,1,1,1,1])

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri*mult

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s

        ex =  (e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        ey =  (e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        ez =  (a1 + b1*x + c1*y)*(e10*(-A + a1 + b1*x + c1*y) + e11*(-A + a1 + b1*x + c1*y) + e12*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y) + e14*(a2 + b2*x + c2*y) + e9*(-A + a1 + b1*x + c1*y))/(2*A**2)
        
        Ex[inside] = ex
        Ey[inside] = ey
        Ez[inside] = ez
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], f8[:,:], i8[:,:], c16[:,:,:], f8), cache=True, nogil=True)
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

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TETRAHEDRON #########
        
        x = xs[inside==1]
        y = ys[inside==1]

        xvs = nodes[0, tris[:,itri]]
        yvs = nodes[1, tris[:,itri]]

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)

        Ds = compute_distances(xvs, yvs, 0*xvs)

        L1 = Ds[0,1]
        L2 = Ds[1,2]
        L3 = Ds[0,2]

        mult = np.array([L1,L2,L3,L3,L1,L2,L3,L1,1,1,1,1,1,1])
        
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri*mult

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s

        hx =  (4*A*(c1*(e10*(-A + a1 + b1*x + c1*y) + e11*(-A + a1 + b1*x + c1*y) + e12*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y) + e14*(a2 + b2*x + c2*y) + e9*(-A + a1 + b1*x + c1*y)) + (a1 + b1*x + c1*y)*(c1*e10 + c1*e11 + c1*e9 + c2*e12 + c2*e13 + c2*e14)) + jB*(e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y)))/(8*A**3)
        hy =  (4*A*(b1*(e10*(-A + a1 + b1*x + c1*y) + e11*(-A + a1 + b1*x + c1*y) + e12*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y) + e14*(a2 + b2*x + c2*y) + e9*(-A + a1 + b1*x + c1*y)) + (a1 + b1*x + c1*y)*(b1*e10 + b1*e11 + b1*e9 + b2*e12 + b2*e13 + b2*e14)) - jB*(e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 + b2*x + c2*y) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 + b1*x + c1*y) - e4*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a3 + b3*x + c3*y) + e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + e8*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y)))/(8*A**3)
        hz =  (-3*a1*b1*c2*e1 - 3*a1*b1*c3*e3 + 3*a1*b2*c1*e1 + a1*b2*c3*e4 + a1*b2*c3*e8 + 3*a1*b3*c1*e3 - a1*b3*c2*e4 - a1*b3*c2*e8 - 3*a2*b1*c2*e5 + 2*a2*b1*c3*e4 - a2*b1*c3*e8 + 3*a2*b2*c1*e5 - 3*a2*b2*c3*e2 - 2*a2*b3*c1*e4 + a2*b3*c1*e8 + 3*a2*b3*c2*e2 + a3*b1*c2*e4 - 2*a3*b1*c2*e8 - 3*a3*b1*c3*e7 - a3*b2*c1*e4 + 2*a3*b2*c1*e8 - 3*a3*b2*c3*e6 + 3*a3*b3*c1*e7 + 3*a3*b3*c2*e6 - 3*b1**2*c2*e1*x - 3*b1**2*c3*e3*x + 3*b1*b2*c1*e1*x - 3*b1*b2*c2*e5*x + 3*b1*b2*c3*e4*x + 3*b1*b3*c1*e3*x - 3*b1*b3*c2*e8*x - 3*b1*b3*c3*e7*x - 3*b1*c1*c2*e1*y - 3*b1*c1*c3*e3*y - 3*b1*c2**2*e5*y + 3*b1*c2*c3*e4*y - 3*b1*c2*c3*e8*y - 3*b1*c3**2*e7*y + 3*b2**2*c1*e5*x - 3*b2**2*c3*e2*x - 3*b2*b3*c1*e4*x + 3*b2*b3*c1*e8*x + 3*b2*b3*c2*e2*x - 3*b2*b3*c3*e6*x + 3*b2*c1**2*e1*y + 3*b2*c1*c2*e5*y + 3*b2*c1*c3*e8*y - 3*b2*c2*c3*e2*y - 3*b2*c3**2*e6*y + 3*b3**2*c1*e7*x + 3*b3**2*c2*e6*x + 3*b3*c1**2*e3*y - 3*b3*c1*c2*e4*y + 3*b3*c1*c3*e7*y + 3*b3*c2**2*e2*y + 3*b3*c2*c3*e6*y)/(8*A**3)
        
        Ex[inside] = hx*dc[0,0]
        Ey[inside] = hy*dc[1,1]
        Ez[inside] = hz*dc[2,2]
    return Ex, Ey, Ez


NFILL = 5
AREA_COEFF_CACHE_BASE = np.zeros((NFILL,NFILL,NFILL,NFILL), dtype=np.float64)
for I in range(NFILL):
    for J in range(NFILL):
        for K in range(NFILL):
            for L in range(NFILL):
                AREA_COEFF_CACHE_BASE[I,J,K,L] = area_coeff(I,J,K,L)

@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], c16[:,:], c16[:,:]), cache=True, nogil=True)
def ned2_tri_stiff_mass(tri_vertices, edge_lengths, local_edge_map, C_stiffness, C_mass):
    '''Nedelec-2 Triangle stiffness and mass submatrix'''
    Dmat = np.zeros((8,8), dtype=np.complex128)
    Fmat = np.zeros((8,8), dtype=np.complex128)

    xs = tri_vertices[0,:]
    ys = tri_vertices[1,:]
    #zs = tri_vertices[2,:]

    aas, bbs, ccs, Area = tri_coefficients(xs, ys)
    a1, a2, a3 = aas
    b1, b2, b3 = bbs
    c1, c2, c3 = ccs
    
    Ds = compute_distances(xs, ys, 0*xs)

    GL1 = np.array([b1, c1, 0])
    GL2 = np.array([b2, c2, 0])
    GL3 = np.array([b3, c3, 0])

    GLs = (GL1, GL2, GL3)

    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    letters = [1,2,3,4,5,6]

    AREA_COEFF = AREA_COEFF_CACHE_BASE * Area

    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        for ej in range(3):
            ej1, ej2 = local_edge_map[:, ej]
            
            A,B,C,D = letters[ei1], letters[ei2], letters[ej1], letters[ej2]
            GA = GLs[ei1]
            GB = GLs[ei2]
            GC = GLs[ej1]
            GD = GLs[ej2]

            GAxGB = cross(GA,GB)
            GCxGD = cross(GC,GD)
            
            Li = edge_lengths[ei]
            Lj = edge_lengths[ej]

            CEE = 1/(2*Area)**4 
            CFEE = 1/(2*Area)**2
            
            Dmat[ei,ej] += Li*Lj*CEE*(9*AREA_COEFF[A,C,0,0]*dot(GAxGB,GCxGD))
            Dmat[ei,ej+4] += Li*Lj*CEE*(9*AREA_COEFF[A,D,0,0]*dot(GAxGB,GCxGD))
            Dmat[ei+4,ej] += Li*Lj*CEE*(9*AREA_COEFF[B,C,0,0]*dot(GAxGB,GCxGD))
            Dmat[ei+4,ej+4] += Li*Lj*CEE*(9*AREA_COEFF[B,D,0,0]*dot(GAxGB,GCxGD))
            
            Fmat[ei,ej] += Li*Lj*CFEE*(AREA_COEFF[A,B,C,D]*dot(GA,GC)-AREA_COEFF[A,B,C,C]*dot(GA,GD)-AREA_COEFF[A,A,C,D]*dot(GB,GC)+AREA_COEFF[A,A,C,C]*dot(GB,GD))
            Fmat[ei,ej+4] += Li*Lj*CFEE*(AREA_COEFF[A,B,D,D]*dot(GA,GC)-AREA_COEFF[A,B,C,D]*dot(GA,GD)-AREA_COEFF[A,A,D,D]*dot(GB,GC)+AREA_COEFF[A,A,C,D]*dot(GB,GD))
            Fmat[ei+4,ej] += Li*Lj*CFEE*(AREA_COEFF[B,B,C,D]*dot(GA,GC)-AREA_COEFF[B,B,C,C]*dot(GA,GD)-AREA_COEFF[A,B,C,D]*dot(GB,GC)+AREA_COEFF[A,B,C,C]*dot(GB,GD))
            Fmat[ei+4,ej+4] += Li*Lj*CFEE*(AREA_COEFF[B,B,D,D]*dot(GA,GC)-AREA_COEFF[B,B,C,D]*dot(GA,GD)-AREA_COEFF[A,B,D,D]*dot(GB,GC)+AREA_COEFF[A,B,C,D]*dot(GB,GD))

    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        ej1, ej2, fj = 0, 1, 2

        A,B,C,D,F = letters[ei1], letters[ei2], letters[ej1], letters[ej2], letters[fj]
        GA = GLs[ei1]
        GB = GLs[ei2]
        GC = GLs[ej1]
        GD = GLs[ej2]
        GF = GLs[fj]
        
        GCxGD = cross(GC,GD)
        GAxGB = cross(GA,GB)
        GCxGF = cross(GC,GF)
        GDxGF = cross(GD,GF)
        
        Li = edge_lengths[ei]
        Lab = Ds[ej1, ej2]
        Lac = Ds[ej1, fj]

        CEF = 1/(2*Area)**4
        CFEF = 1/(2*Area)**2 

        Dmat[ei,3] += Li*Lac*CEF*(-6*AREA_COEFF[A,D,0,0]*dot(GAxGB,GCxGF)-3*AREA_COEFF[A,C,0,0]*dot(GAxGB,GDxGF)-3*AREA_COEFF[A,F,0,0]*dot(GAxGB,GCxGD))
        Fmat[ei,3] += Li*Lac*CFEF*(AREA_COEFF[A,B,C,D]*dot(GA,GF)-AREA_COEFF[A,B,D,F]*dot(GA,GC)-AREA_COEFF[A,A,C,D]*dot(GB,GF)+AREA_COEFF[A,A,D,F]*dot(GB,GC))
        Dmat[ei,7] += Li*Lab*CEF*(6*AREA_COEFF[A,F,0,0]*dot(GAxGB,GCxGD)+3*AREA_COEFF[A,D,0,0]*dot(GAxGB,GCxGF)-3*AREA_COEFF[A,C,0,0]*dot(GAxGB,GDxGF))
        Fmat[ei,7] += Li*Lab*CFEF*(AREA_COEFF[A,B,D,F]*dot(GA,GC)-AREA_COEFF[A,B,F,C]*dot(GA,GD)-AREA_COEFF[A,A,D,F]*dot(GB,GC)+AREA_COEFF[A,A,F,C]*dot(GB,GD))
        Dmat[ei+4,3] += Li*Lac*CEF*(-6*AREA_COEFF[B,D,0,0]*dot(GAxGB,GCxGF)-3*AREA_COEFF[B,C,0,0]*dot(GAxGB,GDxGF)-3*AREA_COEFF[B,F,0,0]*dot(GAxGB,GCxGD))
        Fmat[ei+4,3] += Li*Lac*CFEF*(AREA_COEFF[B,B,C,D]*dot(GA,GF)-AREA_COEFF[B,B,D,F]*dot(GA,GC)-AREA_COEFF[A,B,C,D]*dot(GB,GF)+AREA_COEFF[A,B,D,F]*dot(GB,GC))
        Dmat[ei+4,7] += Li*Lab*CEF*(6*AREA_COEFF[B,F,0,0]*dot(GAxGB,GCxGD)+3*AREA_COEFF[B,D,0,0]*dot(GAxGB,GCxGF)-3*AREA_COEFF[B,C,0,0]*dot(GAxGB,GDxGF))
        Fmat[ei+4,7] += Li*Lab*CFEF*(AREA_COEFF[B,B,D,F]*dot(GA,GC)-AREA_COEFF[B,B,F,C]*dot(GA,GD)-AREA_COEFF[A,B,D,F]*dot(GB,GC)+AREA_COEFF[A,B,F,C]*dot(GB,GD))


    for ej in range(3):
        ei1, ei2, fi = 0, 1, 2
        ej1, ej2 = local_edge_map[:, ej]

        A,B,C,D,E = letters[ei1], letters[ei2], letters[ej1], letters[ej2], letters[fi]
        GA = GLs[ei1]
        GB = GLs[ei2]
        GC = GLs[ej1]
        GD = GLs[ej2]
        GE = GLs[fi]

        GCxGD = cross(GC,GD)
        GAxGB = cross(GA,GB)
        GAxGE = cross(GA,GE)
        GBxGE = cross(GB,GE)

        Lj = edge_lengths[ej]
        Lab = Ds[ei1, ei2]
        Lac = Ds[ei1, fi]

        CFE = 1/(2*Area)**4
        CFFE = 1/(2*Area)**2 

        Dmat[3,ej] += Lj*Lac*CFE*(-6*AREA_COEFF[B,C,0,0]*dot(GAxGE,GCxGD)-3*AREA_COEFF[A,C,0,0]*dot(GBxGE,GCxGD)-3*AREA_COEFF[E,C,0,0]*dot(GAxGB,GCxGD))
        Fmat[3,ej] += Lj*Lac*CFFE*(AREA_COEFF[A,B,C,D]*dot(GC,GE)-AREA_COEFF[A,B,C,C]*dot(GD,GE)-AREA_COEFF[B,E,C,D]*dot(GA,GC)+AREA_COEFF[B,E,C,C]*dot(GA,GD))
        Dmat[3,ej+4] += Lj*Lac*CFE*(-6*AREA_COEFF[B,D,0,0]*dot(GAxGE,GCxGD)-3*AREA_COEFF[A,D,0,0]*dot(GBxGE,GCxGD)-3*AREA_COEFF[E,D,0,0]*dot(GAxGB,GCxGD))
        Fmat[3,ej+4] += Lj*Lac*CFFE*(AREA_COEFF[A,B,D,D]*dot(GC,GE)-AREA_COEFF[A,B,C,D]*dot(GD,GE)-AREA_COEFF[B,E,D,D]*dot(GA,GC)+AREA_COEFF[B,E,C,D]*dot(GA,GD))
        Dmat[7,ej] += Lj*Lab*CFE*(6*AREA_COEFF[E,C,0,0]*dot(GAxGB,GCxGD)+3*AREA_COEFF[B,C,0,0]*dot(GAxGE,GCxGD)-3*AREA_COEFF[A,C,0,0]*dot(GBxGE,GCxGD))
        Fmat[7,ej] += Lj*Lab*CFFE*(AREA_COEFF[B,E,C,D]*dot(GA,GC)-AREA_COEFF[B,E,C,C]*dot(GA,GD)-AREA_COEFF[E,A,C,D]*dot(GB,GC)+AREA_COEFF[E,A,C,C]*dot(GB,GD))
        Dmat[7,ej+4] += Lj*Lab*CFE*(6*AREA_COEFF[E,D,0,0]*dot(GAxGB,GCxGD)+3*AREA_COEFF[B,D,0,0]*dot(GAxGE,GCxGD)-3*AREA_COEFF[A,D,0,0]*dot(GBxGE,GCxGD))
        Fmat[7,ej+4] += Lj*Lab*CFFE*(AREA_COEFF[B,E,D,D]*dot(GA,GC)-AREA_COEFF[B,E,C,D]*dot(GA,GD)-AREA_COEFF[E,A,D,D]*dot(GB,GC)+AREA_COEFF[E,A,C,D]*dot(GB,GD))

    ei1, ei2, fi = 0, 1, 2
    ej1, ej2, fj = 0, 1, 2
        
    A,B,C,D,E,F = letters[ei1], letters[ei2], letters[ej1], letters[ej2], letters[fi], letters[fj]

    GA = GLs[ei1]
    GB = GLs[ei2]
    GC = GLs[ej1]
    GD = GLs[ej2]
    GE = GLs[fi]
    GF = GLs[fj]

    GCxGD = cross(GC,GD)
    GCxGF = cross(GC,GF)
    GAxGB = cross(GA,GB)
    GAxGE = cross(GA,GE)
    GDxGF = cross(GD,GF)
    GBxGE = cross(GB,GE)

    Lac1 = Ds[ei1, fi]
    Lab1 = Ds[ei1, ei2]
    Lac2 = Ds[ej1, fj]
    Lab2 = Ds[ej1, ej2]

    CFF = 1/(2*Area)**4
    CFFF = 1/(2*Area)**2

    Dmat[3,3] += Lac1*Lac2*CFF*(4*AREA_COEFF[B,D,0,0]*dot(GAxGE,GCxGF)+2*AREA_COEFF[B,C,0,0]*dot(GAxGE,GDxGF)+2*AREA_COEFF[B,F,0,0]*dot(GAxGE,GCxGD)+2*AREA_COEFF[A,D,0,0]*dot(GBxGE,GCxGF)+AREA_COEFF[A,C,0,0]*dot(GBxGE,GDxGF)+AREA_COEFF[A,F,0,0]*dot(GBxGE,GCxGD)+2*AREA_COEFF[E,D,0,0]*dot(GAxGB,GCxGF)+AREA_COEFF[E,C,0,0]*dot(GAxGB,GDxGF)+AREA_COEFF[E,F,0,0]*dot(GAxGB,GCxGD))
    Fmat[3,3] += Lac1*Lac2*CFFF*(AREA_COEFF[A,B,C,D]*dot(GE,GF)-AREA_COEFF[A,B,D,F]*dot(GC,GE)-AREA_COEFF[B,E,C,D]*dot(GA,GF)+AREA_COEFF[B,E,D,F]*dot(GA,GC))
    Dmat[3,7] += Lac1*Lab2*CFF*(-4*AREA_COEFF[B,F,0,0]*dot(GAxGE,GCxGD)-2*AREA_COEFF[B,D,0,0]*dot(GAxGE,GCxGF)+2*AREA_COEFF[B,C,0,0]*dot(GAxGE,GDxGF)-2*AREA_COEFF[A,F,0,0]*dot(GBxGE,GCxGD)-AREA_COEFF[A,D,0,0]*dot(GBxGE,GCxGF)+AREA_COEFF[A,C,0,0]*dot(GBxGE,GDxGF)-2*AREA_COEFF[E,F,0,0]*dot(GAxGB,GCxGD)-AREA_COEFF[E,D,0,0]*dot(GAxGB,GCxGF)+AREA_COEFF[E,C,0,0]*dot(GAxGB,GDxGF))
    Fmat[3,7] += Lac1*Lab2*CFFF*(AREA_COEFF[A,B,D,F]*dot(GC,GE)-AREA_COEFF[A,B,F,C]*dot(GD,GE)-AREA_COEFF[B,E,D,F]*dot(GA,GC)+AREA_COEFF[B,E,F,C]*dot(GA,GD))
    Dmat[7,3] += Lab1*Lac2*CFF*(-4*AREA_COEFF[E,D,0,0]*dot(GAxGB,GCxGF)-2*AREA_COEFF[E,C,0,0]*dot(GAxGB,GDxGF)-2*AREA_COEFF[E,F,0,0]*dot(GAxGB,GCxGD)-2*AREA_COEFF[B,D,0,0]*dot(GAxGE,GCxGF)-AREA_COEFF[B,C,0,0]*dot(GAxGE,GDxGF)-AREA_COEFF[B,F,0,0]*dot(GAxGE,GCxGD)+2*AREA_COEFF[A,D,0,0]*dot(GBxGE,GCxGF)+AREA_COEFF[A,C,0,0]*dot(GBxGE,GDxGF)+AREA_COEFF[A,F,0,0]*dot(GBxGE,GCxGD))
    Fmat[7,3] += Lab1*Lac2*CFFF*(AREA_COEFF[B,E,C,D]*dot(GA,GF)-AREA_COEFF[B,E,D,F]*dot(GA,GC)-AREA_COEFF[E,A,C,D]*dot(GB,GF)+AREA_COEFF[E,A,D,F]*dot(GB,GC))
    Dmat[7,7] += Lab1*Lab2*CFF*(4*AREA_COEFF[E,F,0,0]*dot(GAxGB,GCxGD)+2*AREA_COEFF[E,D,0,0]*dot(GAxGB,GCxGF)-2*AREA_COEFF[E,C,0,0]*dot(GAxGB,GDxGF)+2*AREA_COEFF[B,F,0,0]*dot(GAxGE,GCxGD)+AREA_COEFF[B,D,0,0]*dot(GAxGE,GCxGF)-AREA_COEFF[B,C,0,0]*dot(GAxGE,GDxGF)-2*AREA_COEFF[A,F,0,0]*dot(GBxGE,GCxGD)-AREA_COEFF[A,D,0,0]*dot(GBxGE,GCxGF)+AREA_COEFF[A,C,0,0]*dot(GBxGE,GDxGF))
    Fmat[7,7] += Lab1*Lab2*CFFF*(AREA_COEFF[B,E,D,F]*dot(GA,GC)-AREA_COEFF[B,E,F,C]*dot(GA,GD)-AREA_COEFF[E,A,D,F]*dot(GB,GC)+AREA_COEFF[E,A,F,C]*dot(GB,GD))
    

    Dmat = Dmat*C_stiffness
    Fmat = Fmat/C_mass
    
    return Dmat, Fmat



@njit(types.Tuple((c16[:,:],c16[:]))(f8[:,:], c16, c16[:,:], f8[:,:]), cache=True, nogil=True)
def ned2_tri_stiff_force(lcs_vertices, gamma, lcs_Uinc, DPTs):
    ''' Nedelec-2 Triangle Stiffness matrix and forcing vector (For Boundary Condition of the Third Kind)

    '''
    local_edge_map = np.array([[0,1,0],[1,2,2]])
    Bmat = np.zeros((8,8), dtype=np.complex128)
    bvec = np.zeros((8,), dtype=np.complex128)

    xs = lcs_vertices[0,:]
    ys = lcs_vertices[1,:]

    x1, x2, x3 = xs
    y1, y2, y3 = ys

    a1 = x2*y3-y2*x3
    a2 = x3*y1-y3*x1
    a3 = x1*y2-y1*x2
    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    As = np.array([a1, a2, a3])
    Bs = np.array([b1, b2, b3])
    Cs = np.array([c1, c2, c3])

    Ds = compute_distances(xs, ys, np.zeros_like(xs))

    GL1 = np.array([b1, c1])
    GL2 = np.array([b2, c2])
    GL3 = np.array([b3, c3])

    GLs = (GL1, GL2, GL3)

    Area = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
    signA = -np.sign((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    letters = [1,2,3,4,5,6]

    tA, tB, tC = letters[0], letters[1], letters[2]
    GtA, GtB, GtC = GLs[0], GLs[1], GLs[2]
    
    Lt1, Lt2 = Ds[2, 0], Ds[1, 0]
    
    Ux = lcs_Uinc[0,:]
    Uy = lcs_Uinc[1,:]

    x = x1*DPTs[1,:] + x2*DPTs[2,:] + x3*DPTs[3,:]
    y = y1*DPTs[1,:] + y2*DPTs[2,:] + y3*DPTs[3,:]

    Ws = DPTs[0,:]

    COEFF = gamma/(2*Area)**2
    AREA_COEFF = AREA_COEFF_CACHE_BASE * Area
    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        Li = Ds[ei1, ei2]
        
        A = letters[ei1]
        B = letters[ei2]

        GA = GLs[ei1]
        GB = GLs[ei2]

        for ej in range(3):
            ej1, ej2 = local_edge_map[:, ej]
            Lj = Ds[ej1, ej2]

            C = letters[ej1]
            D = letters[ej2]

            GC = GLs[ej1]
            GD = GLs[ej2]

            Bmat[ei,ej] += Li*Lj*(AREA_COEFF[A,B,C,D]*dot(GA,GC)-AREA_COEFF[A,B,C,C]*dot(GA,GD)-AREA_COEFF[A,A,C,D]*dot(GB,GC)+AREA_COEFF[A,A,C,C]*dot(GB,GD))
            Bmat[ei,ej+4] += Li*Lj*(AREA_COEFF[A,B,D,D]*dot(GA,GC)-AREA_COEFF[A,B,C,D]*dot(GA,GD)-AREA_COEFF[A,A,D,D]*dot(GB,GC)+AREA_COEFF[A,A,C,D]*dot(GB,GD))
            Bmat[ei+4,ej] += Li*Lj*(AREA_COEFF[B,B,C,D]*dot(GA,GC)-AREA_COEFF[B,B,C,C]*dot(GA,GD)-AREA_COEFF[A,B,C,D]*dot(GB,GC)+AREA_COEFF[A,B,C,C]*dot(GB,GD))
            Bmat[ei+4,ej+4] += Li*Lj*(AREA_COEFF[B,B,D,D]*dot(GA,GC)-AREA_COEFF[B,B,C,D]*dot(GA,GD)-AREA_COEFF[A,B,D,D]*dot(GB,GC)+AREA_COEFF[A,B,C,D]*dot(GB,GD))
            
        Bmat[ei,3] += Li*Lt1*(AREA_COEFF[A,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,A,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,A,tB,tC]*dot(GB,GtA))
        Bmat[ei,7] += Li*Lt2*(AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,A,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,A,tC,tA]*dot(GB,GtB))
        Bmat[3,ei] += Lt1*Li*(AREA_COEFF[tA,tB,A,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,A]*dot(GB,GtC)-AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,A]*dot(GB,GtA))
        Bmat[7,ei] += Lt2*Li*(AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,A]*dot(GB,GtA)-AREA_COEFF[tC,tA,A,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,A]*dot(GB,GtB))
        Bmat[ei+4,3] += Li*Lt1*(AREA_COEFF[B,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,B,tB,tC]*dot(GB,GtA))
        Bmat[ei+4,7] += Li*Lt2*(AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[B,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,B,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,B,tC,tA]*dot(GB,GtB))
        Bmat[3,ei+4] += Lt1*Li*(AREA_COEFF[tA,tB,B,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,B]*dot(GB,GtC)-AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,B]*dot(GB,GtA))
        Bmat[7,ei+4] += Lt2*Li*(AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,B]*dot(GB,GtA)-AREA_COEFF[tC,tA,B,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,B]*dot(GB,GtB))
            
        A1, A2 = As[ei1], As[ei2]
        B1, B2 = Bs[ei1], Bs[ei2]
        C1, C2 = Cs[ei1], Cs[ei2]

        Ee1x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee1y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee2x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)
        Ee2y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)

        bvec[ei] += signA*Area*np.sum(Ws*(Ee1x*Ux + Ee1y*Uy))
        bvec[ei+4] += signA*Area*np.sum(Ws*(Ee2x*Ux + Ee2y*Uy))
    
    Bmat[3,3] += Lt1*Lt1*(AREA_COEFF[tA,tB,tA,tB]*dot(GtC,GtC)-AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)+AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA))
    Bmat[3,7] += Lt1*Lt2*(AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tA,tB,tC,tA]*dot(GtB,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)+AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB))
    Bmat[7,3] += Lt2*Lt1*(AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tC,tA,tA,tB]*dot(GtB,GtC)+AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB))
    Bmat[7,7] += Lt2*Lt2*(AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB)-AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB)+AREA_COEFF[tC,tA,tC,tA]*dot(GtB,GtB))
    
    A1, A2, A3 = As
    B1, B2, B3 = Bs
    C1, C2, C3 = Cs
   
    Ef1x = Lt1*(-B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + B3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef1y = Lt1*(-C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + C3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef2x = Lt2*(B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - B2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    Ef2y = Lt2*(C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - C2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    
    bvec[3] += signA*Area*np.sum(Ws*(Ef1x*Ux + Ef1y*Uy))
    bvec[7] += signA*Area*np.sum(Ws*(Ef2x*Ux + Ef2y*Uy))
    Bmat = Bmat * COEFF
    return Bmat, bvec

@njit(c16[:,:](f8[:,:], f8[:], c16), cache=True, nogil=True)
def ned2_tri_stiff(vertices, edge_lengths, gamma):
    ''' Nedelec-2 Triangle Stiffness matrix and forcing vector (For Boundary Condition of the Third Kind)

    '''
    local_edge_map = np.array([[0,1,0],[1,2,2]])
    Bmat = np.zeros((8,8), dtype=np.complex128)

    xs = vertices[0,:]
    ys = vertices[1,:]
    zs = vertices[2,:]

    ax1 = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
    ax2 = np.array([xs[2]-xs[0], ys[2]-ys[0], zs[2]-zs[0]])
    ax1 = ax1/np.linalg.norm(ax1)
    ax2 = ax2/np.linalg.norm(ax2)

    axn = cross(ax1, ax2)
    ax2 = -cross(axn, ax1)
    basis = np.zeros((3,3), dtype=np.float64)
    basis[:,0] = ax1
    basis[:,1] = ax2
    basis[:,2] = axn
    basis = np.linalg.pinv(basis)
    lcs_vertices = basis @ vertices

    xs = lcs_vertices[0,:]
    ys = lcs_vertices[1,:]

    x1, x2, x3 = xs
    y1, y2, y3 = ys

    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    Ds = compute_distances(xs, ys, np.zeros_like(xs))

    GL1 = np.array([b1, c1])
    GL2 = np.array([b2, c2])
    GL3 = np.array([b3, c3])

    GLs = (GL1, GL2, GL3)

    Area = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    letters = [1,2,3,4,5,6]

    tA, tB, tC = letters[0], letters[1], letters[2]
    GtA, GtB, GtC = GLs[0], GLs[1], GLs[2]
    
    Lt1, Lt2 = Ds[2, 0], Ds[1, 0]
    
    COEFF = gamma/(2*Area)**2
    AREA_COEFF = AREA_COEFF_CACHE_BASE * Area
    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        Li = edge_lengths[ei]
        
        A = letters[ei1]
        B = letters[ei2]

        GA = GLs[ei1]
        GB = GLs[ei2]

        for ej in range(3):
            ej1, ej2 = local_edge_map[:, ej]
            Lj = edge_lengths[ej]

            C = letters[ej1]
            D = letters[ej2]

            GC = GLs[ej1]
            GD = GLs[ej2]

            Bmat[ei,ej] += Li*Lj*COEFF*(AREA_COEFF[A,B,C,D]*dot(GA,GC)-AREA_COEFF[A,B,C,C]*dot(GA,GD)-AREA_COEFF[A,A,C,D]*dot(GB,GC)+AREA_COEFF[A,A,C,C]*dot(GB,GD))
            Bmat[ei,ej+4] += Li*Lj*COEFF*(AREA_COEFF[A,B,D,D]*dot(GA,GC)-AREA_COEFF[A,B,C,D]*dot(GA,GD)-AREA_COEFF[A,A,D,D]*dot(GB,GC)+AREA_COEFF[A,A,C,D]*dot(GB,GD))
            Bmat[ei+4,ej] += Li*Lj*COEFF*(AREA_COEFF[B,B,C,D]*dot(GA,GC)-AREA_COEFF[B,B,C,C]*dot(GA,GD)-AREA_COEFF[A,B,C,D]*dot(GB,GC)+AREA_COEFF[A,B,C,C]*dot(GB,GD))
            Bmat[ei+4,ej+4] += Li*Lj*COEFF*(AREA_COEFF[B,B,D,D]*dot(GA,GC)-AREA_COEFF[B,B,C,D]*dot(GA,GD)-AREA_COEFF[A,B,D,D]*dot(GB,GC)+AREA_COEFF[A,B,C,D]*dot(GB,GD))
            
        Bmat[ei,3] += Li*Lt1*COEFF*(AREA_COEFF[A,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,A,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,A,tB,tC]*dot(GB,GtA))
        Bmat[ei,7] += Li*Lt2*COEFF*(AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,A,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,A,tC,tA]*dot(GB,GtB))
        Bmat[3,ei] += Lt1*Li*COEFF*(AREA_COEFF[tA,tB,A,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,A]*dot(GB,GtC)-AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,A]*dot(GB,GtA))
        Bmat[7,ei] += Lt2*Li*COEFF*(AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,A]*dot(GB,GtA)-AREA_COEFF[tC,tA,A,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,A]*dot(GB,GtB))
        Bmat[ei+4,3] += Li*Lt1*COEFF*(AREA_COEFF[B,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,B,tB,tC]*dot(GB,GtA))
        Bmat[ei+4,7] += Li*Lt2*COEFF*(AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[B,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,B,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,B,tC,tA]*dot(GB,GtB))
        Bmat[3,ei+4] += Lt1*Li*COEFF*(AREA_COEFF[tA,tB,B,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,B]*dot(GB,GtC)-AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,B]*dot(GB,GtA))
        Bmat[7,ei+4] += Lt2*Li*COEFF*(AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,B]*dot(GB,GtA)-AREA_COEFF[tC,tA,B,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,B]*dot(GB,GtB))
      
    
    Bmat[3,3] += Lt1*Lt1*COEFF*(AREA_COEFF[tA,tB,tA,tB]*dot(GtC,GtC)-AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)+AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA))
    Bmat[3,7] += Lt1*Lt2*COEFF*(AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tA,tB,tC,tA]*dot(GtB,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)+AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB))
    Bmat[7,3] += Lt2*Lt1*COEFF*(AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tC,tA,tA,tB]*dot(GtB,GtC)+AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB))
    Bmat[7,7] += Lt2*Lt2*COEFF*(AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB)-AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB)+AREA_COEFF[tC,tA,tC,tA]*dot(GtB,GtB))


    return Bmat


@njit(c16(f8[:,:], f8[:], c16[:,:], f8[:,:]), cache=True, nogil=True)
def ned2_tri_surface_integral(lcs_vertices, edge_lengths, lcs_Uinc, DPTs):
    ''' Nedelec-2 Triangle surface integral

    '''
    local_edge_map = np.array([[0,1,0],[1,2,2]])
    bvec = np.zeros((8,), dtype=np.complex128)

    xs = lcs_vertices[0,:]
    ys = lcs_vertices[1,:]
    
    x1, x2, x3 = xs
    y1, y2, y3 = ys

    a1 = x2*y3-y2*x3
    a2 = x3*y1-y3*x1
    a3 = x1*y2-y1*x2
    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    As = np.array([a1, a2, a3])
    Bs = np.array([b1, b2, b3])
    Cs = np.array([c1, c2, c3])

    Ds = compute_distances(xs, ys, np.zeros_like(xs))

    Area = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
    signA = np.sign((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    Lt1, Lt2 = Ds[2, 0], Ds[1, 0]
    
    Ux = lcs_Uinc[0,:]
    Uy = lcs_Uinc[1,:]

    x = x1*DPTs[1,:] + x2*DPTs[2,:] + x3*DPTs[3,:]
    y = y1*DPTs[1,:] + y2*DPTs[2,:] + y3*DPTs[3,:]

    Ws = DPTs[0,:]

    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        Li = edge_lengths[ei]
           
        A1, A2 = As[ei1], As[ei2]
        B1, B2 = Bs[ei1], Bs[ei2]
        C1, C2 = Cs[ei1], Cs[ei2]

        Ee1x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee1y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee2x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)
        Ee2y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)

        bvec[ei] += signA*Area*np.sum(Ws*(Ee1x*Ux + Ee1y*Uy))
        bvec[ei+4] += signA*Area*np.sum(Ws*(Ee2x*Ux + Ee2y*Uy))

    A1, A2, A3 = As
    B1, B2, B3 = Bs
    C1, C2, C3 = Cs
   
    Ef1x = Lt1*(-B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + B3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef1y = Lt1*(-C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + C3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef2x = Lt2*(B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - B2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    Ef2y = Lt2*(C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - C2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    
    bvec[3] += signA*Area*np.sum(Ws*(Ef1x*Ux + Ef1y*Uy))
    bvec[7] += signA*Area*np.sum(Ws*(Ef2x*Ux + Ef2y*Uy))

    return np.sum(bvec)