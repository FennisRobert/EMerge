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
from ...elements.nedleg2 import NedelecLegrange2
from scipy.sparse import coo_matrix
from numba_progress import ProgressBar, ProgressBarType
from ...mth.optimized import local_mapping, matinv, compute_distances
from numba import c16, types, f8, i8, njit, prange

@njit(i8[:,:](i8, i8[:,:], i8[:,:], i8[:,:]), cache=True, nogil=True)
def local_tri_to_edgeid(itri: int, tris, edges, tri_to_edge) -> np.ndarray:
    global_edge_map = edges[:, tri_to_edge[:,itri]]
    return local_mapping(tris[:, itri], global_edge_map)

def generelized_eigenvalue_matrix(field: NedelecLegrange2,
                           er: np.ndarray, 
                           ur: np.ndarray,
                           basis: np.ndarray,
                           k0: float,) -> tuple[coo_matrix, coo_matrix]:
    
    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes
    
    nT = tris.shape[1]
    tri_to_field = field.tri_to_field

    nodes = np.linalg.pinv(basis) @ nodes
    
    with ProgressBar(total=nT, ncols=100, dynamic_ncols=False) as pgb:
        dataE, dataB, rows, cols = _matrix_builder(nodes, tris, edges, tri_to_field, ur, er,k0, pgb)
    
    nfield = field.n_field

    E = coo_matrix((dataE, (rows, cols)), shape=(nfield, nfield)).tocsr()
    B = coo_matrix((dataB, (rows, cols)), shape=(nfield, nfield)).tocsr()

    return E, B

@njit(c16[:,:](c16[:,:], c16[:,:]), cache=True, nogil=True)
def matmul(a, b):
    out = np.empty((2,b.shape[1]), dtype=np.complex128)
    out[0,:] = a[0,0]*b[0,:] + a[0,1]*b[1,:]
    out[1,:] = a[1,0]*b[0,:] + a[1,1]*b[1,:]
    return out

### GAUSS QUADRATURE IMPLEMENTATION

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
    out[0,:] = (b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys))*(a1 + b1*xs + c1*ys)
    out[1,:] = (c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys))*(a1 + b1*xs + c1*ys)
    return out

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _ne2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = (b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys)
    out[1,:] = (c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys)
    return out

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf1(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = -(b1*(a3 + b3*xs + c3*ys) - b3*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys)
    out[1,:] = -(c1*(a3 + b3*xs + c3*ys) - c3*(a1 + b1*xs + c1*ys))*(a2 + b2*xs + c2*ys)
    return out

@njit(c16[:,:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out=np.empty((2,xs.shape[0]), dtype=np.complex128)
    out[0,:] = (b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys))*(a3 + b3*xs + c3*ys)
    out[1,:] = (c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys))*(a3 + b3*xs + c3*ys)
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
    return -3*a1*b1*c2 + 3*a1*b2*c1 - 3*b1**2*c2*xs + 3*b1*b2*c1*xs - 3*b1*c1*c2*ys + 3*b2*c1**2*ys + 0*1j

@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _ne2_curl(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    return -3*a2*b1*c2 + 3*a2*b2*c1 - 3*b1*b2*c2*xs - 3*b1*c2**2*ys + 3*b2**2*c1*xs + 3*b2*c1*c2*ys+ 0*1j

@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf1_curl(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    return -b2*(c1*(a3 + b3*xs + c3*ys) - c3*(a1 + b1*xs + c1*ys)) + c2*(b1*(a3 + b3*xs + c3*ys) - b3*(a1 + b1*xs + c1*ys)) + 2*(b1*c3 - b3*c1)*(a2 + b2*xs + c2*ys) + 0*1j

@njit(c16[:](f8[:,:], f8[:,:]), cache=True, nogil=True)
def _nf2_curl(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    return b3*(c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys)) - c3*(b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys)) - 2*(b1*c2 - b2*c1)*(a3 + b3*xs + c3*ys) + 0*1j


####

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

#DPTS = gaus_quad_tri(4).astype(np.float64)

DPTS = np.array([[0.22338159, 0.22338159, 0.22338159, 0.10995174, 0.10995174, 0.10995174],
                        [0.10810302, 0.44594849, 0.44594849, 0.81684757, 0.09157621, 0.09157621],
                        [0.44594849, 0.44594849, 0.10810302, 0.09157621, 0.09157621, 0.81684757],
                        [0.44594849, 0.10810302, 0.44594849, 0.09157621, 0.81684757, 0.09157621]], dtype=np.float64)

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
    #Ls2 = np.ones((14,14), dtype=np.float64)

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

    fid = np.array([0,1,2], dtype=np.int64)

    Ls[3,:] *= Ds[0,2]
    Ls[7,:] *= Ds[0,1]
    Ls[:,3] *= Ds[0,2]
    Ls[:,7] *= Ds[0,1]

    for ei in range(3):
        eis = local_edge_map[:, ei]
        
        Le = Ds[eis[0], eis[1]]
        Ls[ei,:] *= Le
        Ls[:,ei] *= Le
        Ls[ei+4,:] *= Le
        Ls[:,ei+4] *= Le
        
        for ej in range(3):
            ejs = local_edge_map[:, ej]

            Att[ei,ej]     = _gqi(_ne1_curl(coeff[:,eis], cs), Msz * _ne1_curl(coeff[:,ejs],cs), WEIGHTS)
            Att[ei+4,ej]   = _gqi(_ne2_curl(coeff[:,eis], cs), Msz * _ne1_curl(coeff[:,ejs],cs), WEIGHTS)
            Att[ei,ej+4]   = _gqi(_ne1_curl(coeff[:,eis], cs), Msz * _ne2_curl(coeff[:,ejs],cs), WEIGHTS)
            Att[ei+4,ej+4] = _gqi(_ne2_curl(coeff[:,eis], cs), Msz * _ne2_curl(coeff[:,ejs],cs), WEIGHTS)

            Btt[ei,ej]     = _gqi2(_ne1(coeff[:,eis], cs), matmul(Mm,_ne1(coeff[:,ejs],cs)), WEIGHTS)
            Btt[ei+4,ej]   = _gqi2(_ne2(coeff[:,eis], cs), matmul(Mm,_ne1(coeff[:,ejs],cs)), WEIGHTS)
            Btt[ei,ej+4]   = _gqi2(_ne1(coeff[:,eis], cs), matmul(Mm,_ne2(coeff[:,ejs],cs)), WEIGHTS)
            Btt[ei+4,ej+4] = _gqi2(_ne2(coeff[:,eis], cs), matmul(Mm,_ne2(coeff[:,ejs],cs)), WEIGHTS)

            Dtt[ei,ej]     = _gqi2(_ne1(coeff[:,eis], cs), matmul(Ms,_ne1(coeff[:,ejs],cs)), WEIGHTS)
            Dtt[ei+4,ej]   = _gqi2(_ne2(coeff[:,eis], cs), matmul(Ms,_ne1(coeff[:,ejs],cs)), WEIGHTS)
            Dtt[ei,ej+4]   = _gqi2(_ne1(coeff[:,eis], cs), matmul(Ms,_ne2(coeff[:,ejs],cs)), WEIGHTS)
            Dtt[ei+4,ej+4] = _gqi2(_ne2(coeff[:,eis], cs), matmul(Ms,_ne2(coeff[:,ejs],cs)), WEIGHTS)

            Dzt[ei, ej]     = _gqi2(_lv_grad(coeff[:,ei],cs), matmul(Ms,_ne1(coeff[:,ejs],cs)), WEIGHTS)
            Dzt[ei, ej+4]   = _gqi2(_lv_grad(coeff[:,ei],cs), matmul(Ms,_ne2(coeff[:,ejs],cs)), WEIGHTS)
            Dzt[ei+3, ej]   = _gqi2(_le_grad(coeff[:,eis],cs), matmul(Ms,_ne1(coeff[:,ejs],cs)), WEIGHTS)
            Dzt[ei+3, ej+4] = _gqi2(_le_grad(coeff[:,eis],cs), matmul(Ms,_ne2(coeff[:,ejs],cs)), WEIGHTS)

            Dzz1[ei, ej]     = _gqi2(_lv_grad(coeff[:,ei], cs), matmul(Ms,_lv_grad(coeff[:,ej],cs)), WEIGHTS)
            Dzz1[ei, ej+3]   = _gqi2(_lv_grad(coeff[:,ei], cs), matmul(Ms,_le_grad(coeff[:,ejs],cs)), WEIGHTS)
            Dzz1[ei+3, ej]   = _gqi2(_le_grad(coeff[:,eis], cs), matmul(Ms,_lv_grad(coeff[:,ej],cs)), WEIGHTS)
            Dzz1[ei+3, ej+3] = _gqi2(_le_grad(coeff[:,eis], cs), matmul(Ms,_le_grad(coeff[:,ejs],cs)), WEIGHTS)

            Dzz2[ei, ej]     = _gqi(_lv(coeff[:,ei], cs), Mmz * _lv(coeff[:,ej],cs), WEIGHTS)
            Dzz2[ei, ej+3]   = _gqi(_lv(coeff[:,ei], cs), Mmz * _le(coeff[:,ejs],cs), WEIGHTS)
            Dzz2[ei+3, ej]   = _gqi(_le(coeff[:,eis], cs), Mmz * _lv(coeff[:,ej],cs), WEIGHTS)
            Dzz2[ei+3, ej+3] = _gqi(_le(coeff[:,eis], cs), Mmz * _le(coeff[:,ejs],cs), WEIGHTS)


        Att[ei,3]   = _gqi(_ne1_curl(coeff[:,eis], cs), Msz * _nf1_curl(coeff[:,fid],cs), WEIGHTS)
        Att[ei+4,3] = _gqi(_ne2_curl(coeff[:,eis], cs), Msz * _nf1_curl(coeff[:,fid],cs), WEIGHTS)
        Att[ei,7]   = _gqi(_ne1_curl(coeff[:,eis], cs), Msz * _nf2_curl(coeff[:,fid],cs), WEIGHTS)
        Att[ei+4,7] = _gqi(_ne2_curl(coeff[:,eis], cs), Msz * _nf2_curl(coeff[:,fid],cs), WEIGHTS)
        
        Att[3, ei]   = Att[ei,3]
        Att[7, ei]   = Att[ei,7]
        Att[3, ei+4] = Att[ei+4,3]
        Att[7, ei+4] = Att[ei+4,7]

        Btt[ei,3]   = _gqi2(_ne1(coeff[:,eis], cs), matmul(Mm,_nf1(coeff[:,fid],cs)), WEIGHTS)
        Btt[ei+4,3] = _gqi2(_ne2(coeff[:,eis], cs), matmul(Mm,_nf1(coeff[:,fid],cs)), WEIGHTS)
        Btt[ei,7]   = _gqi2(_ne1(coeff[:,eis], cs), matmul(Mm,_nf2(coeff[:,fid],cs)), WEIGHTS)
        Btt[ei+4,7] = _gqi2(_ne2(coeff[:,eis], cs), matmul(Mm,_nf2(coeff[:,fid],cs)), WEIGHTS)

        Btt[3, ei]   = Btt[ei,3]
        Btt[7, ei]   = Btt[ei,7]
        Btt[3, ei+4] = Btt[ei+4,3]
        Btt[7, ei+4] = Btt[ei+4,7]

        Dtt[ei,3]   = _gqi2(_ne1(coeff[:,eis], cs), matmul(Ms,_nf1(coeff[:,fid],cs)), WEIGHTS)
        Dtt[ei+4,3] = _gqi2(_ne2(coeff[:,eis], cs), matmul(Ms,_nf1(coeff[:,fid],cs)), WEIGHTS)
        Dtt[ei,7]   = _gqi2(_ne1(coeff[:,eis], cs), matmul(Ms,_nf2(coeff[:,fid],cs)), WEIGHTS)
        Dtt[ei+4,7] = _gqi2(_ne2(coeff[:,eis], cs), matmul(Ms,_nf2(coeff[:,fid],cs)), WEIGHTS)

        Dtt[3, ei]   = Dtt[ei,3]
        Dtt[7, ei]   = Dtt[ei,7]
        Dtt[3, ei+4] = Dtt[ei+4,3]
        Dtt[7, ei+4] = Dtt[ei+4,7]

        Dzt[ei, 3]   = _gqi2(_lv_grad(coeff[:,ei],cs), matmul(Ms,_nf1(coeff[:,fid],cs)), WEIGHTS)
        Dzt[ei, 7]   = _gqi2(_lv_grad(coeff[:,ei],cs), matmul(Ms,_nf2(coeff[:,fid],cs)), WEIGHTS)
        Dzt[ei+3, 3] = _gqi2(_le_grad(coeff[:,eis],cs), matmul(Ms,_nf1(coeff[:,fid],cs)), WEIGHTS)
        Dzt[ei+3, 7] = _gqi2(_le_grad(coeff[:,eis],cs), matmul(Ms,_nf2(coeff[:,fid],cs)), WEIGHTS)

    Att[3,3] = _gqi(_nf1_curl(coeff[:,fid], cs), Msz * _nf1_curl(coeff[:,fid],cs), WEIGHTS)
    Att[7,3] = _gqi(_nf2_curl(coeff[:,fid], cs), Msz * _nf1_curl(coeff[:,fid],cs), WEIGHTS)
    Att[3,7] = _gqi(_nf1_curl(coeff[:,fid], cs), Msz * _nf2_curl(coeff[:,fid],cs), WEIGHTS)
    Att[7,7] = _gqi(_nf2_curl(coeff[:,fid], cs), Msz * _nf2_curl(coeff[:,fid],cs), WEIGHTS)

    Btt[3,3] = _gqi2(_nf1(coeff[:,fid], cs), matmul(Mm,_nf1(coeff[:,fid],cs)), WEIGHTS)
    Btt[7,3] = _gqi2(_nf2(coeff[:,fid], cs), matmul(Mm,_nf1(coeff[:,fid],cs)), WEIGHTS)
    Btt[3,7] = _gqi2(_nf1(coeff[:,fid], cs), matmul(Mm,_nf2(coeff[:,fid],cs)), WEIGHTS)
    Btt[7,7] = _gqi2(_nf2(coeff[:,fid], cs), matmul(Mm,_nf2(coeff[:,fid],cs)), WEIGHTS)

    A = np.zeros((14, 14), dtype = np.complex128)
    B = np.zeros((14, 14), dtype = np.complex128)
    
    A[:8,:8] = (Att - k0**2 * Btt)

    B[:8,:8] = Dtt
    B[8:,:8] = Dzt
    B[:8,8:] = Dzt.T
    B[8:,8:] = Dzz1 - k0**2 * Dzz2

    B = Ls*B*np.abs(Area)
    A = Ls*A*np.abs(Area)
    return A, B


@njit(types.Tuple((c16[:], c16[:], i8[:], i8[:]))(f8[:,:],
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      i8[:,:],
                                                      c16[:,:,:], 
                                                      c16[:,:,:], 
                                                      f8,
                                                      ProgressBarType), cache=True, nogil=True, parallel=True)
def _matrix_builder(nodes, tris, edges, tri_to_field, ur, er, k0, pgb: ProgressBar):

    ntritot = tris.shape[1]
    nnz = ntritot*196

    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    dataE = np.zeros_like(rows, dtype=np.complex128)
    dataB = np.zeros_like(rows, dtype=np.complex128)

    tri_to_edge = tri_to_field[:3,:]
    
    for itri in prange(ntritot):
        p = itri*196
        if np.mod(itri,10)==0:
            pgb.update(10)
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
            cols[p+ii:p+196:14] = indices[ii]

        dataE[p:p+196] = Esub.ravel()
        dataB[p:p+196] = Bsub.ravel()
    return dataE, dataB, rows, cols