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

import numpy as np
from ....elements.nedleg2 import NedelecLegrange2
from scipy.sparse import csr_matrix
from ....mth.optimized import local_mapping, matinv, compute_distances, gaus_quad_tri
from numba import c16, types, f8, i8, njit, prange



############################################################
#                      FIELD MAPPING                      #
############################################################

@njit(i8[:,:](i8, i8[:,:], i8[:,:], i8[:,:]), cache=True, nogil=True)
def local_tri_to_edgeid(itri: int, tris, edges, tri_to_edge) -> np.ndarray:
    global_edge_map = edges[:, tri_to_edge[:,itri]]
    return local_mapping(tris[:, itri], global_edge_map)



############################################################
#                     PYTHON INTERFACE                     #
############################################################

def generelized_eigenvalue_matrix(field: NedelecLegrange2,
                           er: np.ndarray, 
                           ur: np.ndarray,
                           basis: np.ndarray,
                           k0: float,) -> tuple[csr_matrix, csr_matrix]:
    
    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes
    
    nT = tris.shape[1]
    tri_to_field = field.tri_to_field

    nodes = field.local_nodes
    
    dataE, dataB, rows, cols = _matrix_builder(nodes, tris, edges, tri_to_field, ur, er, k0)
    
    nfield = field.n_field

    E = csr_matrix((dataE, (rows, cols)), shape=(nfield, nfield))
    B = csr_matrix((dataB, (rows, cols)), shape=(nfield, nfield))

    return E, B


############################################################
#                   MATRIX MULTIPLICATION                  #
############################################################

@njit(c16[:,:](c16[:,:], c16[:,:]), cache=True, nogil=True)
def matmul(a, b):
    out = np.empty((2,b.shape[1]), dtype=np.complex128)
    out[0,:] = a[0,0]*b[0,:] + a[0,1]*b[1,:]
    out[1,:] = a[1,0]*b[0,:] + a[1,1]*b[1,:]
    return out


############################################################
#              GAUSS QUADRATURE IMPLEMENTATION             #
############################################################

@njit(c16(c16[:], c16[:], types.Array(types.float64, 1, 'A', readonly=True)), cache=True, nogil=True)
def _gqi(v1, v2, W):
    return np.sum(v1*v2*W)

@njit(c16(c16[:,:], c16[:,:], types.Array(types.float64, 1, 'A', readonly=True)), cache=True, nogil=True)
def _gqi2(v1, v2, W):
    return np.sum(W*np.sum(v1*v2,axis=0))

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _ne1(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = (-b1*(a2 + b2*xs + c2*ys) + b2*(a1 + b1*xs + c1*ys))
    out[1,:] = (-c1*(a2 + b2*xs + c2*ys) + c2*(a1 + b1*xs + c1*ys))
    return out

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _ne2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = (-b1*(a2 + b2*xs + c2*ys) + b2*(a1 + b1*xs + c1*ys))*(a1 - a2 + b1*xs - b2*xs + c1*ys - c2*ys)
    out[1,:] = (-c1*(a2 + b2*xs + c2*ys) + c2*(a1 + b1*xs + c1*ys))*(a1 - a2 + b1*xs - b2*xs + c1*ys - c2*ys)
    return out

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf1(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = -(b1*(a3 + b3*xs + c3*ys) - b3*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys) - (b2*(a3 + b3*xs + c3*ys) - b3*(a2 + b2*xs + c2*ys))*(a1 + b1*xs + c1*ys)
    out[1,:] = -(c1*(a3 + b3*xs + c3*ys) - c3*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys) + (-c2*(a3 + b3*xs + c3*ys) + c3*(a2 + b2*xs + c2*ys))*(a1 + b1*xs + c1*ys)
    return out

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = (b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys))*(a3 + b3*xs + c3*ys) + (b1*(a3 + b3*xs + c3*ys) - b3*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys)
    out[1,:] = -(-c1*(a2 + b2*xs + c2*ys) + c2*(a1 + b1*xs + c1*ys))*(a3 + b3*xs + c3*ys) + (c1*(a3 + b3*xs + c3*ys) - c3*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys)
    return out

@njit(c16[:](f8[:], f8[:,:]), cache=True, nogil=True)
def _lv(coeff, coords):
    a1, b1, c1 = coeff
    xs = coords[0,:]
    ys = coords[1,:]
    return -a1 - b1*xs - c1*ys + 2*(a1 + b1*xs + c1*ys)**2 + 0*1j

@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _le(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    return 4*(a1 + b1*xs + c1*ys)*(a2 + b2*xs + c2*ys)+ 0*1j

@njit(c16[:,:](f8[:], f8[:,:]), cache=True, nogil=True)
def _lv_grad(coeff, coords):
    a1, b1, c1 = coeff
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = b1*(4*a1 + 4*b1*xs + 4*c1*ys - 1)
    out[1,:] = c1*(4*a1 + 4*b1*xs + 4*c1*ys - 1)
    return out

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _le_grad(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = 4*b1*(a2 + b2*xs + c2*ys) + 4*b2*(a1 + b1*xs + c1*ys)
    out[1,:] = 4*c1*(a2 + b2*xs + c2*ys) + 4*c2*(a1 + b1*xs + c1*ys)
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _ne1_curl(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    return (2*b1*c2 - 2*b2*c1)*np.ones_like(xs) + 0j#-3*a1*b1*c2 + 3*a1*b2*c1 - 3*b1**2*c2*xs + 3*b1*b2*c1*xs - 3*b1*c1*c2*ys + 3*b2*c1**2*ys + 0j


@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _ne2_curl(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    return (-(b1 - b2)*(c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys)) + (c1 - c2)*(b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys)) + 2*(b1*c2 - b2*c1)*(a1 - a2 + b1*xs - b2*xs + c1*ys - c2*ys)) + 0j

@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf1_curl(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    return 3*a1*b2*c3 - 3*a1*b3*c2 + 3*a2*b1*c3 - 3*a2*b3*c1 + 6*b1*b2*c3*xs - 3*b1*b3*c2*xs + 3*b1*c2*c3*ys - 3*b2*b3*c1*xs + 3*b2*c1*c3*ys - 6*b3*c1*c2*ys + 0*1j
           
@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf2_curl(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    return -3*a2*b1*c3 + 3*a2*b3*c1 - 3*a3*b1*c2 + 3*a3*b2*c1 - 3*b1*b2*c3*xs - 3*b1*b3*c2*xs - 6*b1*c2*c3*ys + 6*b2*b3*c1*xs + 3*b2*c1*c3*ys + 3*b3*c1*c2*ys + 0*1j
           

############################################################
#     TRIANGLE BARYCENTRIC COORDINATE LIN. COEFFICIENTS    #
############################################################


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


############################################################
#                    CONSTANT DEFINITION                   #
############################################################


DPTS = np.array([[0.22338159, 0.22338159, 0.22338159, 0.10995174, 0.10995174, 0.10995174],
                 [0.10810302, 0.44594849, 0.44594849, 0.81684757, 0.09157621, 0.09157621],
                 [0.44594849, 0.44594849, 0.10810302, 0.09157621, 0.09157621, 0.81684757],
                 [0.44594849, 0.10810302, 0.44594849, 0.09157621, 0.81684757, 0.09157621]], dtype=np.float64)


############################################################
#                 NUMBA OPTIMIZED ASSEMBLER                #
############################################################


@njit(types.Tuple((c16[:,:], c16[:,:]))(f8[:,:], i8[:,:], c16[:,:], c16[:,:], f8), cache=True, nogil=True)
def generalized_matrix_GQ(tri_vertices, local_edge_map, Ms, Mm, k0):
    '''Nedelec-2 Triangle stiffness and mass submatrix'''
    Att = np.zeros((8,8), dtype=np.complex128)
    Btt = np.zeros((8,8), dtype=np.complex128)

    Dtt = np.zeros((8,8), dtype=np.complex128)
    Dzt = np.zeros((6,8), dtype=np.complex128)

    Dzz1 = np.zeros((6,6), dtype=np.complex128)
    Dzz2 = np.zeros((6,6), dtype=np.complex128)
    
    Ls = np.ones((14,14), dtype=np.float64)

    WEIGHTS = DPTS[0,:]
    DPTS1 = DPTS[1,:]
    DPTS2 = DPTS[2,:]
    DPTS3 = DPTS[3,:]

    txs = tri_vertices[0,:]
    tys = tri_vertices[1,:]

    Ds = compute_distances(txs, tys, 0*txs)

    xs = txs[0]*DPTS1 + txs[1]*DPTS2 + txs[2]*DPTS3
    ys = tys[0]*DPTS1 + tys[1]*DPTS2 + tys[2]*DPTS3
    
    cs = np.empty((2,xs.shape[0]), dtype=np.float64)
    cs[0,:] = xs
    cs[1,:] = ys

    aas, bbs, ccs, Area = tri_coefficients(txs, tys)

    coeff = np.empty((3,3), dtype=np.float64)
    coeff[0,:] = aas/(2*Area)
    coeff[1,:] = bbs/(2*Area)
    coeff[2,:] = ccs/(2*Area)

    Msz = Ms[2,2]
    Mmz = Mm[2,2]
    Ms = Ms[:2,:2]
    Mm = Mm[:2,:2]

    Ls[3,:] *= Ds[0,2]
    Ls[7,:] *= Ds[0,1]
    Ls[:,3] *= Ds[0,2]
    Ls[:,7] *= Ds[0,1]
    
    for iv1 in range(3):
        ie1 = local_edge_map[:, iv1]
        
        Le = Ds[ie1[0], ie1[1]]
        Ls[iv1,:] *= Le
        Ls[:,iv1] *= Le
        Ls[iv1+4,:] *= Le
        Ls[:,iv1+4] *= Le
        F1 = _ne1_curl(coeff[:,ie1], cs)
        F2 = _ne2_curl(coeff[:,ie1], cs)
        F3 = _ne1(coeff[:,ie1], cs)
        F4 = _ne2(coeff[:,ie1], cs)
        F5 = _lv_grad(coeff[:,iv1],cs)
        F6 = _le_grad(coeff[:,ie1],cs)
        
        for iv2 in range(3):
            ei2 = local_edge_map[:, iv2]
            
            H1 = matmul(Ms,_ne1(coeff[:,ei2],cs))
            H2 = matmul(Ms,_ne2(coeff[:,ei2],cs))
            
            Att[iv1,iv2]     = _gqi(F1, Msz * _ne1_curl(coeff[:,ei2],cs), WEIGHTS)
            Att[iv1+4,iv2]   = _gqi(F2, Msz * _ne1_curl(coeff[:,ei2],cs), WEIGHTS)
            Att[iv1,iv2+4]   = _gqi(F1, Msz * _ne2_curl(coeff[:,ei2],cs), WEIGHTS)
            Att[iv1+4,iv2+4] = _gqi(F2, Msz * _ne2_curl(coeff[:,ei2],cs), WEIGHTS)

            Btt[iv1,iv2]     = _gqi2(F3, matmul(Mm,_ne1(coeff[:,ei2],cs)), WEIGHTS)
            Btt[iv1+4,iv2]   = _gqi2(F4, matmul(Mm,_ne1(coeff[:,ei2],cs)), WEIGHTS)
            Btt[iv1,iv2+4]   = _gqi2(F3, matmul(Mm,_ne2(coeff[:,ei2],cs)), WEIGHTS)
            Btt[iv1+4,iv2+4] = _gqi2(F4, matmul(Mm,_ne2(coeff[:,ei2],cs)), WEIGHTS)

            Dtt[iv1,iv2]     = _gqi2(F3, H1, WEIGHTS)
            Dtt[iv1+4,iv2]   = _gqi2(F4, H1, WEIGHTS)
            Dtt[iv1,iv2+4]   = _gqi2(F3, H2, WEIGHTS)
            Dtt[iv1+4,iv2+4] = _gqi2(F4, H2, WEIGHTS)

            Dzt[iv1, iv2]     = _gqi2(F5, H1, WEIGHTS)
            Dzt[iv1+3, iv2]   = _gqi2(F6, H1, WEIGHTS)
            Dzt[iv1, iv2+4]   = _gqi2(F5, H2, WEIGHTS)
            Dzt[iv1+3, iv2+4] = _gqi2(F6, H2, WEIGHTS)

            Dzz1[iv1, iv2]     = _gqi2(_lv_grad(coeff[:,iv1], cs), matmul(Ms,_lv_grad(coeff[:,iv2],cs)), WEIGHTS)
            Dzz1[iv1, iv2+3]   = _gqi2(_lv_grad(coeff[:,iv1], cs), matmul(Ms,_le_grad(coeff[:,ei2],cs)), WEIGHTS)
            Dzz1[iv1+3, iv2]   = _gqi2(_le_grad(coeff[:,ie1], cs), matmul(Ms,_lv_grad(coeff[:,iv2],cs)), WEIGHTS)
            Dzz1[iv1+3, iv2+3] = _gqi2(_le_grad(coeff[:,ie1], cs), matmul(Ms,_le_grad(coeff[:,ei2],cs)), WEIGHTS)

            Dzz2[iv1, iv2]     = _gqi(_lv(coeff[:,iv1], cs), Mmz * _lv(coeff[:,iv2],cs), WEIGHTS)
            Dzz2[iv1, iv2+3]   = _gqi(_lv(coeff[:,iv1], cs), Mmz * _le(coeff[:,ei2],cs), WEIGHTS)
            Dzz2[iv1+3, iv2]   = _gqi(_le(coeff[:,ie1], cs), Mmz * _lv(coeff[:,iv2],cs), WEIGHTS)
            Dzz2[iv1+3, iv2+3] = _gqi(_le(coeff[:,ie1], cs), Mmz * _le(coeff[:,ei2],cs), WEIGHTS)


        G1 = matmul(Mm,_nf1(coeff,cs))
        G2 = matmul(Mm,_nf2(coeff,cs))
        G3 = matmul(Ms,_nf1(coeff,cs))
        G4 = matmul(Ms,_nf2(coeff,cs))
        
        Att[iv1,3]   = _gqi(F1, Msz * _nf1_curl(coeff,cs), WEIGHTS)
        Att[iv1+4,3] = _gqi(_ne2_curl(coeff[:,ie1], cs), Msz * _nf1_curl(coeff,cs), WEIGHTS)
        Att[iv1,7]   = _gqi(F1, Msz * _nf2_curl(coeff,cs), WEIGHTS)
        Att[iv1+4,7] = _gqi(_ne2_curl(coeff[:,ie1], cs), Msz * _nf2_curl(coeff,cs), WEIGHTS)
        
        Att[3, iv1]   = Att[iv1,3]
        Att[7, iv1]   = Att[iv1,7]
        Att[3, iv1+4] = Att[iv1+4,3]
        Att[7, iv1+4] = Att[iv1+4,7]

        Btt[iv1,3]   = _gqi2(F3, G1, WEIGHTS)
        Btt[iv1+4,3] = _gqi2(F4, G1, WEIGHTS)
        Btt[iv1,7]   = _gqi2(F3, G2, WEIGHTS)
        Btt[iv1+4,7] = _gqi2(F4, G2, WEIGHTS)

        Btt[3, iv1]   = Btt[iv1,3]
        Btt[7, iv1]   = Btt[iv1,7]
        Btt[3, iv1+4] = Btt[iv1+4,3]
        Btt[7, iv1+4] = Btt[iv1+4,7]

        Dtt[iv1,3]   = _gqi2(F3, G3, WEIGHTS)
        Dtt[iv1+4,3] = _gqi2(F4, G3, WEIGHTS)
        Dtt[iv1,7]   = _gqi2(F3, G4, WEIGHTS)
        Dtt[iv1+4,7] = _gqi2(F4, G4, WEIGHTS)

        Dtt[3, iv1]   = Dtt[iv1,3]
        Dtt[7, iv1]   = Dtt[iv1,7]
        Dtt[3, iv1+4] = Dtt[iv1+4,3]
        Dtt[7, iv1+4] = Dtt[iv1+4,7]

        Dzt[iv1, 3]   = _gqi2(F5, G3, WEIGHTS)
        Dzt[iv1, 7]   = _gqi2(F5, G4, WEIGHTS)
        Dzt[iv1+3, 3] = _gqi2(F6, G3, WEIGHTS)
        Dzt[iv1+3, 7] = _gqi2(F6, G4, WEIGHTS)

    Att[3,3] = _gqi(_nf1_curl(coeff, cs), Msz * _nf1_curl(coeff,cs), WEIGHTS)
    Att[7,3] = _gqi(_nf2_curl(coeff, cs), Msz * _nf1_curl(coeff,cs), WEIGHTS)
    Att[3,7] = _gqi(_nf1_curl(coeff, cs), Msz * _nf2_curl(coeff,cs), WEIGHTS)
    Att[7,7] = _gqi(_nf2_curl(coeff, cs), Msz * _nf2_curl(coeff,cs), WEIGHTS)

    Btt[3,3] = _gqi2(_nf1(coeff, cs), G1, WEIGHTS)
    Btt[7,3] = _gqi2(_nf2(coeff, cs), G1, WEIGHTS)
    Btt[3,7] = _gqi2(_nf1(coeff, cs), G2, WEIGHTS)
    Btt[7,7] = _gqi2(_nf2(coeff, cs), G2, WEIGHTS)

    A = np.zeros((14, 14), dtype = np.complex128)
    B = np.zeros((14, 14), dtype = np.complex128)
    
    A[:8,:8] = (Att - k0**2 * Btt)

    B[:8,:8] = Dtt
    B[8:,:8] = Dzt
    B[:8,8:] = Dzt.T
    B[8:,8:] = Dzz1 - k0**2 * Dzz2

    Ls = np.ones((14,14), dtype=np.float64)
    
    B = Ls*B*np.abs(Area)
    A = Ls*A*np.abs(Area)
    return A, B


@njit(types.Tuple((c16[:], c16[:], i8[:], i8[:]))(f8[:,:],
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      i8[:,:],
                                                      c16[:,:,:], 
                                                      c16[:,:,:], 
                                                      f8), cache=True, nogil=True, parallel=True)
def _matrix_builder(nodes, tris, edges, tri_to_field, ur, er, k0):

    ntritot = tris.shape[1]
    nnz = ntritot*196

    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    dataE = np.zeros_like(rows, dtype=np.complex128)
    dataB = np.zeros_like(rows, dtype=np.complex128)

    tri_to_edge = tri_to_field[:3,:]
    
    for itri in prange(ntritot): # type: ignore
        p = itri*196
        urt = ur[:,:,itri]
        ert = er[:,:,itri]

        # Construct a local mapping to global triangle orientations
        local_tri_map = local_tri_to_edgeid(itri, tris, edges, tri_to_edge)

        # Construct the local edge map
        tri_nodes = nodes[:, tris[:,itri]]
        Esub, Bsub = generalized_matrix_GQ(tri_nodes,local_tri_map, matinv(urt), ert, k0)
        
        indices = tri_to_field[:, itri]
        for ii in range(14):
            rows[p+14*ii:p+14*(ii+1)] = indices[ii]
            cols[p+ii:p+ii+196:14] = indices[ii]

        dataE[p:p+196] = Esub.ravel()
        dataB[p:p+196] = Bsub.ravel()
    return dataE, dataB, rows, cols