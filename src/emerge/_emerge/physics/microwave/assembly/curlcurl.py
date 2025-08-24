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
from ....elements import Nedelec2
from scipy.sparse import csr_matrix, coo_matrix
from ....mth.optimized import local_mapping, matinv, dot_c, cross_c, compute_distances
from numba import c16, types, f8, i8, njit, prange

############################################################
#                  CACHED FACTORIAL VALUES                 #
############################################################

_FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=np.int64)

############################################################
#                  INDEX MAPPING FUNCTIONS                 #
############################################################

# These mapping functions return edge and face coordinates in the appropriate order.

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8, i8), cache=True, nogil=True)
def local_tet_to_triid(tet_to_field, tets, tris, itet, nedges) -> np.ndarray:
    """Returns the triangle node indices in the right order given a tet-index"""
    tri_ids = tet_to_field[6:10, itet] - nedges
    global_tri_map = tris[:, tri_ids]
    return local_mapping(tets[:, itet], global_tri_map)

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def local_tet_to_edgeid(tets, edges, tet_to_field, itet) -> np.ndarray:
    """Returns the edge node indices in the right order given a tet-index"""
    global_edge_map = edges[:, tet_to_field[:6,itet]]
    return local_mapping(tets[:, itet], global_edge_map)

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def local_tri_to_edgeid(tris, edges, tri_to_field, itri: int) -> np.ndarray:
    """Returns the edge node indices in the right order given a triangle-index"""
    global_edge_map = edges[:, tri_to_field[:3,itri]]
    return local_mapping(tris[:, itri], global_edge_map)

@njit(c16[:](c16[:,:], c16[:]), cache=True, nogil=True)
def matmul(Mat, Vec):
    ## Matrix multiplication of a 3D vector
    Vout = np.empty((3,), dtype=np.complex128)
    Vout[0] = Mat[0,0]*Vec[0] + Mat[0,1]*Vec[1] + Mat[0,2]*Vec[2]
    Vout[1] = Mat[1,0]*Vec[0] + Mat[1,1]*Vec[1] + Mat[1,2]*Vec[2]
    Vout[2] = Mat[2,0]*Vec[0] + Mat[2,1]*Vec[1] + Mat[2,2]*Vec[2]
    return Vout

@njit(f8(i8, i8, i8, i8), cache=True, fastmath=True, nogil=True)
def volume_coeff(a: int, b: int, c: int, d: int):
    """ Computes the appropriate matrix coefficients given a list of
    barycentric coordinate functions mentioned.
    Example:
      - L1^2 * L2 - volume_coeff(1,1,2,0) """
    klmn = np.array([0,0,0,0,0,0,0])
    klmn[a] += 1
    klmn[b] += 1
    klmn[c] += 1
    klmn[d] += 1
    output = (_FACTORIALS[klmn[1]]*_FACTORIALS[klmn[2]]*_FACTORIALS[klmn[3]]
                  *_FACTORIALS[klmn[4]]*_FACTORIALS[klmn[5]]*_FACTORIALS[klmn[6]])/_FACTORIALS[(np.sum(klmn[1:])+3)]
    return output


############################################################
#        PRECOMPUTATION OF INTEGRATION COEFFICIENTS       #
############################################################

NFILL = 5
VOLUME_COEFF_CACHE_BASE = np.zeros((NFILL,NFILL,NFILL,NFILL), dtype=np.float64)
for I in range(NFILL):
    for J in range(NFILL):
        for K in range(NFILL):
            for L in range(NFILL):
                VOLUME_COEFF_CACHE_BASE[I,J,K,L] = volume_coeff(I,J,K,L)

VOLUME_COEFF_CACHE = VOLUME_COEFF_CACHE_BASE


############################################################
#  COMPUTATION OF THE BARYCENTRIC COORDINATE COEFFICIENTS #
############################################################

@njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]), cache = True, nogil=True)
def tet_coefficients_bcd(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Computes the a,b,c and d coefficients of a tet barycentric coordinate functions and the volume

    Args:
        xs (np.ndarray): The tetrahedron X-coordinates
        ys (np.ndarray): The tetrahedron Y-coordinates
        zs (np.ndrray): The tetrahedron Z-coordinates

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]: The a, b, c, d coefficients and volume
    """
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(-x1*y2*z3/6 + x1*y2*z4/6 + x1*y3*z2/6 - x1*y3*z4/6 - x1*y4*z2/6 + \
                x1*y4*z3/6 + x2*y1*z3/6 - x2*y1*z4/6 - x2*y3*z1/6 + x2*y3*z4/6 + \
                x2*y4*z1/6 - x2*y4*z3/6 - x3*y1*z2/6 + x3*y1*z4/6 + x3*y2*z1/6 - \
                x3*y2*z4/6 - x3*y4*z1/6 + x3*y4*z2/6 + x4*y1*z2/6 - x4*y1*z3/6 - \
                x4*y2*z1/6 + x4*y2*z3/6 + x4*y3*z1/6 - x4*y3*z2/6)
    
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

    return bbs, ccs, dds, V


############################################################
#              MAIN CURL-CURL MATRIX ASSEMBLY             #
############################################################

def tet_mass_stiffness_matrices(field: Nedelec2,
                           er: np.ndarray, 
                           ur: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    """Computes the curl-curl Nedelec-2 mass and stiffness matrices

    Args:
        field (Nedelec2): The Nedelec2 Field object
        er (np.ndarray): a 3x3xN array with permittivity tensors
        ur (np.ndarray): a 3x3xN array with permeability tensors

    Returns:
        tuple[csr_matrix, csr_matrix]: The stiffness and mass matrix.
    """
    tets = field.mesh.tets
    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes

    nT = tets.shape[1]
    tet_to_field = field.tet_to_field
    tet_to_edge = field.mesh.tet_to_edge
    nE = edges.shape[1]
    nTri = tris.shape[1]

    dataE, dataB, rows, cols = _matrix_builder(nodes, tets, tris, edges, field.mesh.edge_lengths, tet_to_field, tet_to_edge, ur, er)
        
    E = coo_matrix((dataE, (rows, cols)), shape=(nE*2 + nTri*2, nE*2 + nTri*2)).tocsr()
    B = coo_matrix((dataB, (rows, cols)), shape=(nE*2 + nTri*2, nE*2 + nTri*2)).tocsr()

    return E, B


############################################################
#           NUMBA ACCELLERATE SUB-MATRIX ASSEMBLY          #
############################################################

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


############################################################
#             NUMBA ACCELLERATED MATRIX BUILDER            #
############################################################

@njit(types.Tuple((c16[:], c16[:], i8[:], i8[:]))(f8[:,:], 
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      f8[:], 
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      c16[:,:,:], 
                                                      c16[:,:,:]), cache=True, nogil=True, parallel=True)
def _matrix_builder(nodes, tets, tris, edges, all_edge_lengths, tet_to_field, tet_to_edge, ur, er):
    nT = tets.shape[1]
    nedges = edges.shape[1]

    nnz = nT*400

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    dataE = np.empty_like(rows, dtype=np.complex128)
    dataB = np.empty_like(rows, dtype=np.complex128)

    
    for itet in prange(nT): # ty: ignore
        p = itet*400
        urt = ur[:,:,itet]
        ert = er[:,:,itet]

        # Construct a local mapping to global triangle orientations
        
        local_tri_map = local_tet_to_triid(tet_to_field, tets, tris, itet, nedges)
        local_edge_map = local_tet_to_edgeid(tets, edges, tet_to_field, itet)
        edge_lengths = all_edge_lengths[tet_to_edge[:,itet]]

        # Construct the local edge map

        Esub, Bsub = ned2_tet_stiff_mass(nodes[:,tets[:,itet]], 
                                                edge_lengths, 
                                                local_edge_map, 
                                                local_tri_map, 
                                                matinv(urt), ert)
        
        indices = tet_to_field[:, itet]
        for ii in range(20):
            rows[p+20*ii:p+20*(ii+1)] = indices[ii]
            cols[p+ii:p+400+ii:20] = indices[ii]

        dataE[p:p+400] = Esub.ravel()
        dataB[p:p+400] = Bsub.ravel()
    return dataE, dataB, rows, cols


