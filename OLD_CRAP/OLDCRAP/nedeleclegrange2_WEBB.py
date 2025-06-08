import numpy as np
from ....elements.nedleg2 import NedelecLegrange2
from scipy.sparse import csr_matrix, coo_matrix
from numba_progress import ProgressBar, ProgressBarType
from ....mth.optimized import local_mapping, matinv, compute_distances, tri_coefficients, area_coeff
from numba import c16, types, f8, i8, njit, prange

def local_tri_to_edgeid(itri: int, tris, edges, tri_to_edge) -> np.ndarray:
    global_edge_map = edges[:, tri_to_edge[:,itri]]
    return local_mapping(tris[:, itri], global_edge_map)

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

    nodes = np.linalg.pinv(basis) @ nodes
    
    with ProgressBar(total=nT, ncols=100, dynamic_ncols=False) as pgb:
        dataE, dataB, rows, cols = _matrix_builder(nodes, tris, edges, tri_to_field, ur, er,k0, pgb)
    
    nfield = field.n_field

    E = coo_matrix((dataE, (rows, cols)), shape=(nfield, nfield)).tocsr()
    B = coo_matrix((dataB, (rows, cols)), shape=(nfield, nfield)).tocsr()

    # for i in range(field.n_xy, field.n_field):
    #     E[i,i] = 1e-5
    #B[field.n_xy:, fiel]
    return E, B

def dot(a,b):
    '''Dot product of two vectors'''
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    '''Cross product of two vectors'''
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])

def matmul(a, b):
    return a @ b

# @njit(types.Tuple((c16[:], c16[:], i8[:], i8[:]))(f8[:,:],
#                                                       i8[:,:], 
#                                                       i8[:,:], 
#                                                       f8[:], 
#                                                       i8[:,:], 
#                                                       i8[:,:], 
#                                                       c16[:,:,:], 
#                                                       c16[:,:,:], 
#                                                       ProgressBarType), cache=True, nogil=True, parallel=True)
def _matrix_builder(nodes, tris, edges, tri_to_field, ur, er, k0, pgb: ProgressBar):

    ntritot = tris.shape[1]
    nnz = ntritot*196

    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    dataE = np.zeros_like(rows, dtype=np.complex128)
    dataB = np.zeros_like(rows, dtype=np.complex128)

    tri_to_edge = tri_to_field[:3,:]
    
    for itri in range(ntritot):
        p = itri*196
        if np.mod(itri,10)==0:
            pgb.update(10)
        urt = ur[:,:,itri]
        ert = er[:,:,itri]

        # Construct a local mapping to global triangle orientations
        local_tri_map = local_tri_to_edgeid(itri, tris, edges, tri_to_edge)

        # Construct the local edge map
        tri_nodes = nodes[:, tris[:,itri]]
        Esub, Bsub = generalized_matrix(tri_nodes,local_tri_map, matinv(urt), ert, k0)
        
        indices = tri_to_field[:, itri]
        for ii in range(14):
            rows[p+14*ii:p+14*(ii+1)] = indices[ii]
            cols[p+ii:p+196:14] = indices[ii]

        dataE[p:p+196] = Esub.ravel()
        dataB[p:p+196] = Bsub.ravel()
    return dataE, dataB, rows, cols

NFILL = 7
AREA_COEFF_BASE = np.zeros((NFILL,NFILL,NFILL,NFILL), dtype=np.float64)
for I in range(NFILL):
    for J in range(NFILL):
        for K in range(NFILL):
            for L in range(NFILL):
                AREA_COEFF_BASE[I,J,K,L] = area_coeff(I,J,K,L)

def generalized_matrix(tri_vertices, local_edge_map, Ms, Mm, k0):
    '''Nedelec-2 Triangle stiffness and mass submatrix'''
    Att = np.zeros((8,8), dtype=np.complex128)
    Btt = np.zeros((8,8), dtype=np.complex128)

    Dtt = np.zeros((8,8), dtype=np.complex128)
    Dzt = np.zeros((6,8), dtype=np.complex128)

    Dzz1 = np.zeros((6,6), dtype=np.complex128)
    Dzz2 = np.zeros((6,6), dtype=np.complex128)
    

    xs = tri_vertices[0,:]
    ys = tri_vertices[1,:]

    ermn = (Mm[0,0]+Mm[1,1]+Mm[2,2])/3
    urmn = (Ms[0,0]+Ms[1,1]+Ms[2,2])/3
    #print(ermn, urmn)
    # print('1/urt:', Ms[0,0])
    # print('ert:', Mm[0,0])
    
    aas, bbs, ccs, Area = tri_coefficients(xs, ys)

    a1, a2, a3 = aas
    b1, b2, b3 = bbs
    c1, c2, c3 = ccs
    
    GL1 = np.array([b1, c1, 0])/(2*Area)
    GL2 = np.array([b2, c2, 0])/(2*Area)
    GL3 = np.array([b3, c3, 0])/(2*Area)

    GLs = (GL1, GL2, GL3)

    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6

    letters = [1,2,3,4,5,6]

    ArC = AREA_COEFF_BASE

    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]

        for ej in range(3):
            ej1, ej2 = local_edge_map[:, ej]
            
            A,B,C,D = letters[ei1], letters[ei2], letters[ej1], letters[ej2]
            
            GA = GLs[ei1]
            GB = GLs[ei2]
            GC = GLs[ej1]
            GD = GLs[ej2]

            ## DTT Terms Edge To Edge

            # ∇xNe . ∇xNe ✅
            AB1 = cross(GA,GB)
            AC1 = cross(GC,GD)
            AD1 = matmul(Ms,AC1)
            AE1 = dot(AB1,AD1)

            Att[ei+0,ej+0] = (4*AE1)

            # # Ne . Ne  ✅
            AB1 = matmul(Ms,GD)
            AC1 = matmul(Ms,GC)
            AD1 = dot(GB,AB1)
            AE1 = dot(GB,AC1)
            AF1 = dot(GA,AB1)
            AG1 = dot(GA,AC1)

            Dtt[ei+0,ej+0] = 1*(ArC[A,C,0,0]*AD1-ArC[A,D,0,0]*AE1-ArC[B,C,0,0]*AF1+ArC[B,D,0,0]*AG1)
            Dtt[ei+0,ej+4] = 1*(ArC[A,D,0,0]*AE1+ArC[A,C,0,0]*AD1-ArC[B,D,0,0]*AG1-ArC[B,C,0,0]*AF1)
            Dtt[ei+4,ej+0] = 1*(ArC[B,C,0,0]*AF1-ArC[B,D,0,0]*AG1+ArC[A,C,0,0]*AD1-ArC[A,D,0,0]*AE1)
            Dtt[ei+4,ej+4] = 1*(ArC[B,D,0,0]*AG1+ArC[B,C,0,0]*AF1+ArC[A,D,0,0]*AE1+ArC[A,C,0,0]*AD1)
            
            AB1 = matmul(Mm,GD)
            AC1 = matmul(Mm,GC)
            AD1 = dot(GB,AB1)
            AE1 = dot(GB,AC1)
            AF1 = dot(GA,AB1)
            AG1 = dot(GA,AC1)

            Btt[ei+0,ej+0] = 1*(ArC[A,C,0,0]*AD1-ArC[A,D,0,0]*AE1-ArC[B,C,0,0]*AF1+ArC[B,D,0,0]*AG1)
            Btt[ei+0,ej+4] = 1*(ArC[A,D,0,0]*AE1+ArC[A,C,0,0]*AD1-ArC[B,D,0,0]*AG1-ArC[B,C,0,0]*AF1)
            Btt[ei+4,ej+0] = 1*(ArC[B,C,0,0]*AF1-ArC[B,D,0,0]*AG1+ArC[A,C,0,0]*AD1-ArC[A,D,0,0]*AE1)
            Btt[ei+4,ej+4] = 1*(ArC[B,D,0,0]*AG1+ArC[B,C,0,0]*AF1+ArC[A,D,0,0]*AE1+ArC[A,C,0,0]*AD1)
            # ## PARTIAL DZT matrix from Each Z to the edge XY components

            # # ∇Nz . Ne ✅

            BB1 = matmul(Ms,GD)
            BC1 = matmul(Ms,GC)
            BD1 = dot(GA,BB1)
            BE1 = dot(GA,BC1)

            Dzt[ei+0,ej+0] = (4*ArC[A,C,0,0]*BD1-4*ArC[A,D,0,0]*BE1-ArC[C,0,0,0]*BD1+ArC[D,0,0,0]*BE1)
            Dzt[ei+0,ej+4] = (4*ArC[A,D,0,0]*BE1+4*ArC[A,C,0,0]*BD1-ArC[D,0,0,0]*BE1-ArC[C,0,0,0]*BD1)
            
            BB1 = matmul(Ms,GD)
            BC1 = matmul(Ms,GC)
            BD1 = dot(GA,BB1)
            BE1 = dot(GA,BC1)
            BF1 = dot(GB,BB1)
            BG1 = dot(GB,BC1)

            Dzt[ei+3,ej+0] = (4*ArC[B,C,0,0]*BD1-4*ArC[B,D,0,0]*BE1+4*ArC[A,C,0,0]*BF1-4*ArC[A,D,0,0]*BG1)
            Dzt[ei+3,ej+4] = (4*ArC[B,D,0,0]*BE1+4*ArC[B,C,0,0]*BD1+4*ArC[A,D,0,0]*BG1+4*ArC[A,C,0,0]*BF1)
            # ## ENTIRE DZZ MATRIX ASEMBLY 

            # ## ∇Nz . ∇Nz✅

            DB1 = matmul(Ms,GC)
            DC1 = dot(GA,DB1)

            Dzz1[ei+0,ej+0] = (16*ArC[A,C,0,0]*DC1-4*ArC[A,0,0,0]*DC1-4*ArC[C,0,0,0]*DC1 + DC1*ArC[0,0,0,0])

            DB1 = matmul(Ms,GC)
            DC1 = matmul(Ms,GD)
            DD1 = dot(GA,DB1)
            DE1 = dot(GA,DC1)

            Dzz1[ei+0,ej+3] = (16*ArC[A,D,0,0]*DD1+16*ArC[A,C,0,0]*DE1-4*ArC[D,0,0,0]*DD1-4*ArC[C,0,0,0]*DE1)

            DB1 = matmul(Ms,GC)
            DC1 = dot(GA,DB1)
            DD1 = dot(GB,DB1)

            Dzz1[ei+3,ej+0] = (16*ArC[B,C,0,0]*DC1-4*ArC[B,0,0,0]*DC1+16*ArC[A,C,0,0]*DD1-4*ArC[A,0,0,0]*DD1)

            DB1 = matmul(Ms,GC)
            DC1 = matmul(Ms,GD)
            DD1 = dot(GA,DB1)
            DE1 = dot(GA,DC1)
            DF1 = dot(GB,DB1)
            DG1 = dot(GB,DC1)

            Dzz1[ei+3,ej+3] = (16*ArC[B,D,0,0]*DD1+16*ArC[B,C,0,0]*DE1+16*ArC[A,D,0,0]*DF1+16*ArC[A,C,0,0]*DG1)
            
            # ## Nz . Nz ✅

            Dzz2[ei+0,ej+0] = Mm[2,2]*(4*ArC[A,A,C,C]-2*ArC[A,A,C,0]-2*ArC[A,C,C,0]+ArC[A,C,0,0])
            Dzz2[ei+0,ej+3] = Mm[2,2]*(8*ArC[A,A,C,D]-4*ArC[A,C,D,0])
            Dzz2[ei+3,ej+0] = Mm[2,2]*(8*ArC[A,B,C,C]-4*ArC[A,B,C,0])
            Dzz2[ei+3,ej+3] = Mm[2,2]*(16*ArC[A,B,C,D])

    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        ej1, ej2, fj = 0, 1, 2
        ej = 0

        A,B,C,D,F = letters[ei1], letters[ei2], letters[ej1], letters[ej2], letters[fj]
        GA = GLs[ei1]
        GB = GLs[ei2]
        GC = GLs[ej1]
        GD = GLs[ej2]
        GF = GLs[fj]

        ## Dtt edge to Face modes

        ## ∇xNe . ∇xNf ✅

        AB1 = cross(GA,GB)
        AC1 = cross(GC,GF)
        AD1 = cross(GD,GF)
        AE1 = cross(GC,GD)
        AF1 = matmul(Ms,AC1)
        AG1 = matmul(Ms,AD1)
        AH1 = matmul(Ms,AE1)
        AI1 = dot(AB1,AF1)
        AJ1 = dot(AB1,AG1)
        AK1 = dot(AB1,AH1)

        Att[ei+0,ej+3] = (-6*ArC[D,0,0,0]*AI1 - 6*ArC[C,0,0,0]*AJ1)
        Att[ei+0,ej+7] = (6*ArC[F,0,0,0]*AK1 + 6*ArC[D,0,0,0]*AI1)
        
        Att[ej+3, ei+0] = Att[ei+0,ej+3]
        Att[ej+7, ei+0] = Att[ei+0,ej+7]
        

        # ##  er * Ne . Nf ✅
        AB1 = matmul(Mm,GC)
        AC1 = matmul(Mm,GD)
        AD1 = matmul(Mm,GF)
        AE1 = dot(GB,AB1)
        AF1 = dot(GB,AC1)
        AG1 = dot(GB,AD1)
        AH1 = dot(GA,AB1)
        AI1 = dot(GA,AC1)
        AJ1 = dot(GA,AD1)

        Btt[ei+0,ej+3] = 1*(ArC[A,D,F,0]*AE1+ArC[A,C,F,0]*AF1-2*ArC[A,C,D,0]*AG1-ArC[B,D,F,0]*AH1-ArC[B,C,F,0]*AI1+2*ArC[B,C,D,0]*AJ1)
        Btt[ei+0,ej+7] = 1*(ArC[A,C,F,0]*AF1+ArC[A,C,D,0]*AG1-2*ArC[A,D,F,0]*AE1-ArC[B,C,F,0]*AI1-ArC[B,C,D,0]*AJ1+2*ArC[B,D,F,0]*AH1)
        Btt[ei+4,ej+3] = 1*(ArC[B,D,F,0]*AH1+ArC[B,C,F,0]*AI1-2*ArC[B,C,D,0]*AJ1+ArC[A,D,F,0]*AE1+ArC[A,C,F,0]*AF1-2*ArC[A,C,D,0]*AG1)
        Btt[ei+4,ej+7] = 1*(ArC[B,C,F,0]*AI1+ArC[B,C,D,0]*AJ1-2*ArC[B,D,F,0]*AH1+ArC[A,C,F,0]*AF1+ArC[A,C,D,0]*AG1-2*ArC[A,D,F,0]*AE1)
        
        Btt[ej+3, ei+0] = Btt[ei+0,ej+3]
        Btt[ej+7, ei+0] = Btt[ei+0,ej+7]
        Btt[ej+3, ei+4] = Btt[ei+4,ej+3]
        Btt[ej+7, ei+4] = Btt[ei+4,ej+7]

        # ##  1/ur * Ne . Nf ✅
        AB1 = matmul(Ms,GC)
        AC1 = matmul(Ms,GD)
        AD1 = matmul(Ms,GF)
        AE1 = dot(GB,AB1)
        AF1 = dot(GB,AC1)
        AG1 = dot(GB,AD1)
        AH1 = dot(GA,AB1)
        AI1 = dot(GA,AC1)
        AJ1 = dot(GA,AD1)

        Dtt[ei+0,ej+3] = 1*(ArC[A,D,F,0]*AE1+ArC[A,C,F,0]*AF1-2*ArC[A,C,D,0]*AG1-ArC[B,D,F,0]*AH1-ArC[B,C,F,0]*AI1+2*ArC[B,C,D,0]*AJ1)
        Dtt[ei+0,ej+7] = 1*(ArC[A,C,F,0]*AF1+ArC[A,C,D,0]*AG1-2*ArC[A,D,F,0]*AE1-ArC[B,C,F,0]*AI1-ArC[B,C,D,0]*AJ1+2*ArC[B,D,F,0]*AH1)
        Dtt[ei+4,ej+3] = 1*(ArC[B,D,F,0]*AH1+ArC[B,C,F,0]*AI1-2*ArC[B,C,D,0]*AJ1+ArC[A,D,F,0]*AE1+ArC[A,C,F,0]*AF1-2*ArC[A,C,D,0]*AG1)
        Dtt[ei+4,ej+7] = 1*(ArC[B,C,F,0]*AI1+ArC[B,C,D,0]*AJ1-2*ArC[B,D,F,0]*AH1+ArC[A,C,F,0]*AF1+ArC[A,C,D,0]*AG1-2*ArC[A,D,F,0]*AE1)
        Dtt[ej+3, ei+0] = Dtt[ei+0,ej+3]
        Dtt[ej+7, ei+0] = Dtt[ei+0,ej+7]
        Dtt[ej+3, ei+4] = Dtt[ei+4,ej+3]
        Dtt[ej+7, ei+4] = Dtt[ei+4,ej+7]

        # ## Ez to all Et face modes 
        # ## 1/ur ∇Nz . Nf✅

        BB1 = matmul(Ms,GC)
        BC1 = matmul(Ms,GD)
        BD1 = matmul(Ms,GF)
        BE1 = dot(GA,BB1)
        BF1 = dot(GA,BC1)
        BG1 = dot(GA,BD1)

        Dzt[ei+0,ej+3] = (4*ArC[A,D,F,0]*BE1+4*ArC[A,C,F,0]*BF1-8*ArC[A,C,D,0]*BG1-ArC[D,F,0,0]*BE1-ArC[C,F,0,0]*BF1+2*ArC[C,D,0,0]*BG1)
        Dzt[ei+0,ej+7] = (4*ArC[A,C,F,0]*BF1+4*ArC[A,C,D,0]*BG1-8*ArC[A,D,F,0]*BE1-ArC[C,F,0,0]*BF1-ArC[C,D,0,0]*BG1+2*ArC[D,F,0,0]*BE1)
        # ✅

        BB1 = matmul(Ms,GC)
        BC1 = matmul(Ms,GD)
        BD1 = matmul(Ms,GF)
        BE1 = dot(GA,BB1)
        BF1 = dot(GA,BC1)
        BG1 = dot(GA,BD1)
        BH1 = dot(GB,BB1)
        BI1 = dot(GB,BC1)
        BJ1 = dot(GB,BD1)

        Dzt[ei+3,ej+3] = (4*ArC[B,D,F,0]*BE1+4*ArC[B,C,F,0]*BF1-8*ArC[B,C,D,0]*BG1+4*ArC[A,D,F,0]*BH1+4*ArC[A,C,F,0]*BI1-8*ArC[A,C,D,0]*BJ1)
        Dzt[ei+3,ej+7] = (4*ArC[B,C,F,0]*BF1+4*ArC[B,C,D,0]*BG1-8*ArC[B,D,F,0]*BE1+4*ArC[A,C,F,0]*BI1+4*ArC[A,C,D,0]*BJ1-8*ArC[A,D,F,0]*BH1)
        
    ei1, ei2, fi = 0, 1, 2
    ej1, ej2, fj = 0, 1, 2

    ei, ej = 0,0

    A,B,C,D,E,F = letters[ei1], letters[ei2], letters[ej1], letters[ej2], letters[fi], letters[fj]

    GA = GLs[ei1]
    GB = GLs[ei2]
    GC = GLs[ej1]
    GD = GLs[ej2]
    GE = GLs[fi]
    GF = GLs[fj]

    # Cross Curl Nf .  Curl Nf  ✅
    AB1 = cross(GA,GE)
    AC1 = cross(GC,GF)
    AD1 = cross(GD,GF)
    AE1 = cross(GB,GE)
    AF1 = cross(GC,GD)
    AG1 = cross(GA,GB)
    AH1 = matmul(Ms,AC1)
    AI1 = matmul(Ms,AD1)
    AJ1 = matmul(Ms,AF1)
    AK1 = dot(AB1,AH1)
    AL1 = dot(AB1,AI1)
    AM1 = dot(AE1,AH1)
    AN1 = dot(AE1,AI1)
    AO1 = dot(AB1,AJ1)
    AP1 = dot(AE1,AJ1)
    AQ1 = dot(AG1,AH1)
    AR1 = dot(AG1,AI1)
    AS1 = dot(AG1,AJ1)

    Att[ei+3,ej+3] = 1*(9*ArC[B,D,0,0]*AK1+9*ArC[B,C,0,0]*AL1+9*ArC[A,D,0,0]*AM1+9*ArC[A,C,0,0]*AN1)
    Att[ei+3,ej+7] = 1*(-9*ArC[B,F,0,0]*AO1-9*ArC[B,D,0,0]*AK1-9*ArC[A,F,0,0]*AP1-9*ArC[A,D,0,0]*AM1)
    Att[ei+7,ej+3] = 1*(-9*ArC[D,E,0,0]*AQ1-9*ArC[C,E,0,0]*AR1-9*ArC[B,D,0,0]*AK1-9*ArC[B,C,0,0]*AL1)
    Att[ei+7,ej+7] = 1*(9*ArC[E,F,0,0]*AS1+9*ArC[D,E,0,0]*AQ1+9*ArC[B,F,0,0]*AO1+9*ArC[B,D,0,0]*AK1)
    
    # ## 1/ur Nf . Nf ✅
    AB1 = matmul(Ms,GC)
    AC1 = matmul(Ms,GD)
    AD1 = matmul(Ms,GF)
    AE1 = dot(GA,AB1)
    AF1 = dot(GA,AC1)
    AG1 = dot(GA,AD1)
    AH1 = dot(GB,AB1)
    AI1 = dot(GB,AC1)
    AJ1 = dot(GB,AD1)
    AK1 = dot(GE,AB1)
    AL1 = dot(GE,AC1)
    AM1 = dot(GE,AD1)

    Dtt[ei+3,ej+3] = 1*(ArC[B,D,E,F]*AE1+ArC[B,C,E,F]*AF1-2*ArC[B,C,D,E]*AG1+ArC[A,D,E,F]*AH1+ArC[A,C,E,F]*AI1-2*ArC[A,C,D,E]*AJ1-2*ArC[A,B,D,F]*AK1-2*ArC[A,B,C,F]*AL1+4*ArC[A,B,C,D]*AM1)
    Dtt[ei+3,ej+7] = 1*(ArC[B,C,E,F]*AF1+ArC[B,C,D,E]*AG1-2*ArC[B,D,E,F]*AE1+ArC[A,C,E,F]*AI1+ArC[A,C,D,E]*AJ1-2*ArC[A,D,E,F]*AH1-2*ArC[A,B,C,F]*AL1-2*ArC[A,B,C,D]*AM1+4*ArC[A,B,D,F]*AK1)
    Dtt[ei+7,ej+3] = 1*(ArC[A,D,E,F]*AH1+ArC[A,C,E,F]*AI1-2*ArC[A,C,D,E]*AJ1+ArC[A,B,D,F]*AK1+ArC[A,B,C,F]*AL1-2*ArC[A,B,C,D]*AM1-2*ArC[B,D,E,F]*AE1-2*ArC[B,C,E,F]*AF1+4*ArC[B,C,D,E]*AG1)
    Dtt[ei+7,ej+7] = 1*(ArC[A,C,E,F]*AI1+ArC[A,C,D,E]*AJ1-2*ArC[A,D,E,F]*AH1+ArC[A,B,C,F]*AL1+ArC[A,B,C,D]*AM1-2*ArC[A,B,D,F]*AK1-2*ArC[B,C,E,F]*AF1-2*ArC[B,C,D,E]*AG1+4*ArC[B,D,E,F]*AE1)
    
    # ## er Nf . Nf: ✅
    AB1 = matmul(Mm,GC)
    AC1 = matmul(Mm,GD)
    AD1 = matmul(Mm,GF)
    AE1 = dot(GA,AB1)
    AF1 = dot(GA,AC1)
    AG1 = dot(GA,AD1)
    AH1 = dot(GB,AB1)
    AI1 = dot(GB,AC1)
    AJ1 = dot(GB,AD1)
    AK1 = dot(GE,AB1)
    AL1 = dot(GE,AC1)
    AM1 = dot(GE,AD1)

    Btt[ei+3,ej+3] = 1*(ArC[B,D,E,F]*AE1+ArC[B,C,E,F]*AF1-2*ArC[B,C,D,E]*AG1+ArC[A,D,E,F]*AH1+ArC[A,C,E,F]*AI1-2*ArC[A,C,D,E]*AJ1-2*ArC[A,B,D,F]*AK1-2*ArC[A,B,C,F]*AL1+4*ArC[A,B,C,D]*AM1)
    Btt[ei+3,ej+7] = 1*(ArC[B,C,E,F]*AF1+ArC[B,C,D,E]*AG1-2*ArC[B,D,E,F]*AE1+ArC[A,C,E,F]*AI1+ArC[A,C,D,E]*AJ1-2*ArC[A,D,E,F]*AH1-2*ArC[A,B,C,F]*AL1-2*ArC[A,B,C,D]*AM1+4*ArC[A,B,D,F]*AK1)
    Btt[ei+7,ej+3] = 1*(ArC[A,D,E,F]*AH1+ArC[A,C,E,F]*AI1-2*ArC[A,C,D,E]*AJ1+ArC[A,B,D,F]*AK1+ArC[A,B,C,F]*AL1-2*ArC[A,B,C,D]*AM1-2*ArC[B,D,E,F]*AE1-2*ArC[B,C,E,F]*AF1+4*ArC[B,C,D,E]*AG1)
    Btt[ei+7,ej+7] = 1*(ArC[A,C,E,F]*AI1+ArC[A,C,D,E]*AJ1-2*ArC[A,D,E,F]*AH1+ArC[A,B,C,F]*AL1+ArC[A,B,C,D]*AM1-2*ArC[A,B,D,F]*AK1-2*ArC[B,C,E,F]*AF1-2*ArC[B,C,D,E]*AG1+4*ArC[B,D,E,F]*AE1)

    if True:
        A = np.zeros((14, 14), dtype = np.complex128)
        B = np.zeros((14, 14), dtype = np.complex128)
        C = np.zeros((14, 14), dtype = np.complex128)
        
        A[:8,:8] = (Att - k0**2 * Btt)

        B[:8,:8] = Dtt
        B[8:,:8] = Dzt
        B[:8,8:] = Dzt.T
        B[8:,8:] = Dzz1 - k0**2 * Dzz2

        #A = A + C*ermn*k0**2

        B = B*Area
        A = A*Area
    else:
        A = np.zeros((14, 14), dtype = np.complex128)
        B = np.zeros((14, 14), dtype = np.complex128)
        
        A[:8,:8] = Att

        B[:8,:8] = Btt
        
        B = B*Area
        A = A*Area
    return A, B