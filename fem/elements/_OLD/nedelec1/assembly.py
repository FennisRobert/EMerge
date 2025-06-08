import numpy as np
from ...bc import PEC, BoundaryCondition, WaveguidePort
from ...mesh3d import Mesh3D
from loguru import logger
from numba import njit, f8, i8, c16, types

c0 = 299792458

def plot_matrix(matrix, title='Matrix Visualization', cmap='viridis'):
    """
    Quickly visualize a matrix to inspect sparsity, symmetry, etc.
    
    Parameters:
    - matrix: np.ndarray
    - title: str, optional
    - cmap: str, optional (colormap, e.g., 'viridis', 'gray', 'hot', etc.)
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(matrix), cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.grid(False)
    plt.show()

# TETS
# [0,  1,  2, 3]
# [4, 34, 12, 8]
# [5,  1,  3, 4]
#
#
def matprint(mat: np.ndarray) -> None:
    factor = np.max(np.abs(mat.flatten()))
    if factor == 0:
        factor = 1
    print(mat.real/factor)

def select_bc(bcs: list[BoundaryCondition], bctype):
    return [bc for bc in bcs if isinstance(bc,bctype)]

def _iterate_segments(bcs: list[BoundaryCondition]) -> tuple[BoundaryCondition, list[np.ndarray]]:
    indices_list = []
    for bc in bcs:
        for indices in bc.node_indices:
            indices_list.append((bc,indices))
    return indices_list

TET_E2V = ((0,1),(0,2),(0,3),(1,2),(3,1),(2,3))
ITERIJ = [(i, j) for i in range(6) for j in range(i, 6)]

def diagnose_matrix(mat: np.ndarray) -> None:
    ''' Prints all indices of Nan's and infinities in a matrix '''
    logger.info('Diagnosing matrix')
    ids = np.where(np.isnan(mat))
    if len(ids[0]) > 0:
        logger.error(f'Found NaN at {ids}')
    ids = np.where(np.isinf(mat))
    if len(ids[0]) > 0:
        logger.error(f'Found Inf at {ids}')

@njit(cache=True)
def compute_coeffs(xs, ys, zs):
    ## THIS FUNCTION WORKS
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    V = np.abs(-x1*y2*z3/6 + x1*y2*z4/6 + x1*y3*z2/6 - x1*y3*z4/6 - x1*y4*z2/6 + x1*y4*z3/6 + x2*y1*z3/6 - x2*y1*z4/6 - x2*y3*z1/6 + x2*y3*z4/6 + x2*y4*z1/6 - x2*y4*z3/6 - x3*y1*z2/6 + x3*y1*z4/6 + x3*y2*z1/6 - x3*y2*z4/6 - x3*y4*z1/6 + x3*y4*z2/6 + x4*y1*z2/6 - x4*y1*z3/6 - x4*y2*z1/6 + x4*y2*z3/6 + x4*y3*z1/6 - x4*y3*z2/6)
    
    a1 = x2*y3*z4 - x2*y4*z3 - x3*y2*z4 + x3*y4*z2 + x4*y2*z3 - x4*y3*z2
    a2 = -x1*y3*z4 + x1*y4*z3 + x3*y1*z4 - x3*y4*z1 - x4*y1*z3 + x4*y3*z1
    a3 = x1*y2*z4 - x1*y4*z2 - x2*y1*z4 + x2*y4*z1 + x4*y1*z2 - x4*y2*z1
    a4 = -x1*y2*z3 + x1*y3*z2 + x2*y1*z3 - x2*y3*z1 - x3*y1*z2 + x3*y2*z1
    b1 = -y2*z3 + y2*z4 + y3*z2 - y3*z4 - y4*z2 + y4*z3
    b2 = y1*z3 - y1*z4 - y3*z1 + y3*z4 + y4*z1 - y4*z3
    b3 = -y1*z2 + y1*z4 + y2*z1 - y2*z4 - y4*z1 + y4*z2
    b4 = y1*z2 - y1*z3 - y2*z1 + y2*z3 + y3*z1 - y3*z2
    c1 = x2*z3 - x2*z4 - x3*z2 + x3*z4 + x4*z2 - x4*z3
    c2 = -x1*z3 + x1*z4 + x3*z1 - x3*z4 - x4*z1 + x4*z3
    c3 = x1*z2 - x1*z4 - x2*z1 + x2*z4 + x4*z1 - x4*z2
    c4 = -x1*z2 + x1*z3 + x2*z1 - x2*z3 - x3*z1 + x3*z2
    d1 = -x2*y3 + x2*y4 + x3*y2 - x3*y4 - x4*y2 + x4*y3
    d2 = x1*y3 - x1*y4 - x3*y1 + x3*y4 + x4*y1 - x4*y3
    d3 = -x1*y2 + x1*y4 + x2*y1 - x2*y4 - x4*y1 + x4*y2
    d4 = x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2
    
    return a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4, V


@njit((types.Tuple((c16[:,:], c16[:,:])))(f8[:,:], i8[:,:], i8[:,:], i8[:,:], i8[:,:], f8[:], f8[:,]))
def assemble_base_matrices(vertices: np.ndarray, 
                           tets: np.ndarray, 
                           edges: np.ndarray,
                           tet2edge: np.ndarray,
                           sgn_tet2edge: np.ndarray,
                           er: np.ndarray, 
                           ur: np.ndarray):
    
    nT = tets.shape[1]
    nE = edges.shape[1]


    D = np.zeros((nE, nE)).astype(np.complex128)
    F = np.zeros((nE, nE)).astype(np.complex128)
    
    ev1 = vertices[:,edges[0, :]]
    ev2 = vertices[:,edges[1, :]]

    edge_lengths = np.sqrt((ev1[0,:] - ev2[0,:])**2 + (ev1[1,:] - ev2[1,:])**2 + (ev1[2,:] - ev2[2,:])**2)

    for it in range(nT):
        urt = ur[it]
        ert = er[it]

        ie1, ie2, ie3, ie4, ie5, ie6 = tet2edge[:,it]
        ies = np.array([ie1, ie2, ie3, ie4, ie5, ie6])

        xs = vertices[0, tets[:,it]]
        ys = vertices[1, tets[:,it]]
        zs = vertices[2, tets[:,it]]

        a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4, V = compute_coeffs(xs, ys, zs)

        b = ((b1, b2),(b1,b3),(b1,b4),(b2,b3),(b4,b2),(b3,b4))
        c = ((c1, c2),(c1,c3),(c1,c4),(c2,c3),(c4,c2),(c3,c4))
        d = ((d1, d2),(d1,d3),(d1,d4),(d2,d3),(d4,d2),(d3,d4))

        bes = (b1, b2, b3, b4)
        ces = (c1, c2, c3, c4)
        des = (d1, d2, d3, d4)

        fs = np.zeros((5,5))

        for i in range(6):
            for j in range(i, 6):
                iei = ies[i]
                iej = ies[j]
                lei = edge_lengths[iei]
                lej = edge_lengths[iej]

                Dije = (lei*lej/(324*V**3)) *  (
                                                (c[i][0]*d[i][1] - d[i][0]*c[i][1]) * (c[j][0]*d[j][1] - d[j][0]*c[j][1]) 
                                            + (d[i][0]*b[i][1] - b[i][0]*d[i][1]) * (d[j][0]*b[j][1] - b[j][0]*d[j][1])
                                            + (b[i][0]*c[i][1] - c[i][0]*b[i][1]) * (b[j][0]*c[j][1] - c[j][0]*b[j][1])
                                            )
                s = (1/urt) * Dije * sgn_tet2edge[i,it] * sgn_tet2edge[j,it]
                D[iei, iej] += s                 # add once
                if iei != iej:                   # mirror explicitly
                    D[iej, iei] += s

        signs = np.outer(sgn_tet2edge[:,it], sgn_tet2edge[:,it])
        for (i,j) in [(ii,jj) for ii in range(1,5) for jj in range(1,5)]:
            fs[i,j] = bes[i-1]*bes[j-1] + ces[i-1]*ces[j-1] + des[i-1]*des[j-1]
            
        Ft = np.zeros((7,7)).astype(np.complex128)
        Q = ert
        Ft[1,1] = Q*edge_lengths[ie1]*edge_lengths[ie1]/(360*V) * (fs[2,2]-fs[1,2]+fs[1,1])
        Ft[1,2] = Q*edge_lengths[ie1]*edge_lengths[ie2]/(720*V) * (2*fs[2,3]-fs[2,1]-fs[1,3]+fs[1,1])
        Ft[1,3] = Q*edge_lengths[ie1]*edge_lengths[ie3]/(720*V) * (2*fs[2,4]-fs[2,1]-fs[1,4]+fs[1,1])
        Ft[1,4] = Q*edge_lengths[ie1]*edge_lengths[ie4]/(720*V) * (fs[2,3]-fs[2,2]-2*fs[1,3]+fs[1,2])
        Ft[1,5] = Q*edge_lengths[ie1]*edge_lengths[ie5]/(720*V) * (fs[2,2]-fs[2,4]-fs[1,2]+2*fs[1,4])
        Ft[1,6] = Q*edge_lengths[ie1]*edge_lengths[ie6]/(720*V) * (fs[2,4]-fs[2,3]-fs[1,4]+fs[1,3])

        Ft[2,1] = Ft[1, 2]
        Ft[2,2] = Q*edge_lengths[ie2]*edge_lengths[ie2]/(360*V) * (fs[3,3]-fs[1,3]+fs[1,1])
        Ft[2,3] = Q*edge_lengths[ie2]*edge_lengths[ie3]/(720*V) * (2*fs[3,4]-fs[1,3]-fs[1,4]+fs[1,1])
        Ft[2,4] = Q*edge_lengths[ie2]*edge_lengths[ie4]/(720*V) * (fs[3,3]-fs[2,3]-fs[1,3]+2*fs[1,2])
        Ft[2,5] = Q*edge_lengths[ie2]*edge_lengths[ie5]/(720*V) * (fs[2,3]-fs[3,4]-fs[1,2]+fs[1,4])
        Ft[2,6] = Q*edge_lengths[ie2]*edge_lengths[ie6]/(720*V) * (fs[1,3]-fs[3,3]-2*fs[1,4]+fs[3,4])

        Ft[3,1] = Ft[1, 3]
        Ft[3,2] = Ft[2, 3]
        Ft[3,3] = Q*edge_lengths[ie3]*edge_lengths[ie3]/(360*V) * (fs[4,4]-fs[1,4]+fs[1,1])
        Ft[3,4] = Q*edge_lengths[ie3]*edge_lengths[ie4]/(720*V) * (fs[3,4]-fs[2,4]-fs[1,3]+fs[1,2])
        Ft[3,5] = Q*edge_lengths[ie3]*edge_lengths[ie5]/(720*V) * (fs[2,4]-fs[4,4]-2*fs[1,2]+fs[1,4])
        Ft[3,6] = Q*edge_lengths[ie3]*edge_lengths[ie6]/(720*V) * (fs[4,4]-fs[3,4]-fs[1,4]+2*fs[1,3])

        Ft[4,1] = Ft[1, 4]
        Ft[4,2] = Ft[2, 4]
        Ft[4,3] = Ft[3, 4]
        Ft[4,4] = Q*edge_lengths[ie4]*edge_lengths[ie4]/(360*V) * (fs[3,3]-fs[2,3]+fs[2,2])
        Ft[4,5] = Q*edge_lengths[ie4]*edge_lengths[ie5]/(720*V) * (fs[2,3]-2*fs[3,4]-fs[2,2]+fs[2,4])
        Ft[4,6] = Q*edge_lengths[ie4]*edge_lengths[ie6]/(720*V) * (fs[3,4]-fs[3,3]-2*fs[2,4]+fs[2,3])

        Ft[5,1] = Ft[1, 5]
        Ft[5,2] = Ft[2, 5]
        Ft[5,3] = Ft[3, 5]
        Ft[5,4] = Ft[4, 5]
        Ft[5,5] = Q*edge_lengths[ie5]*edge_lengths[ie5]/(360*V) * (fs[2,2]-fs[2,4]+fs[4,4])
        Ft[5,6] = Q*edge_lengths[ie5]*edge_lengths[ie6]/(720*V) * (fs[2,4]-2*fs[2,3]-fs[4,4]+fs[3,4])

        Ft[6,1] = Ft[1, 6]
        Ft[6,2] = Ft[2, 6]
        Ft[6,3] = Ft[3, 6]
        Ft[6,4] = Ft[4, 6]
        Ft[6,5] = Ft[5, 6]
        Ft[6,6] = Q*edge_lengths[ie6]*edge_lengths[ie6]/(360*V) * (fs[4,4]-fs[3,4]+fs[3,3])

        for i in range(6):
            for j in range(6):
                F[ies[i],ies[j]] += Ft[i+1,j+1]*signs[i,j]
        
        
    return D, F

#@njit((types.Tuple((c16[:,:], c16[:])))(f8[:,:], i8[:,:], i8[:,:], i8[:,:], i8[:,:], f8[:,:], c16, c16[:,:]))
def assemble_surf_matrices(vertices: np.ndarray, 
                           surftris: np.ndarray,
                           triangles: np.ndarray, 
                           edges: np.ndarray,
                           tri2edge: np.ndarray,
                           sgn_tri2edge: np.ndarray,
                           local_basis: np.ndarray,
                           gamma: float,
                           U: np.ndarray = None):
    
    nT = surftris.shape[0]
    nE = edges.shape[1]


    B = np.zeros((nE, nE)).astype(np.complex128)
    bv = np.zeros((nE,), dtype=np.complex128)

    vertices_local = np.linalg.pinv(local_basis) @ vertices
    Ulocal = np.linalg.pinv(local_basis) @ U
    Ux = Ulocal[0,:]
    Uy = Ulocal[1,:]

    for it in range(nT):
        indtri = surftris[it]
        ie1, ie2, ie3, = tri2edge[:,indtri]
        ies = np.array([ie1, ie2, ie3])

        iv1, iv2, iv3 = triangles[:,indtri]
        x1, x2, x3 = vertices_local[0, triangles[:,indtri]]
        y1, y2, y3 = vertices_local[1, triangles[:,indtri]]
        
        U1x, U2x, U3x = Ux[ies]
        U1y, U2y, U3y = Uy[ies]

        a1 = x2*y3-y2*x3
        a2 = x3*y1-y3*x1
        a3 = x1*y2-y1*x2
        b1 = y2-y3
        b2 = y3-y1
        b3 = y1-y2
        c1 = x3-x2
        c2 = x1-x3
        c3 = x2-x1

        l1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        l2 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        l3 = np.sqrt((x1-x3)**2 + (y1-y3)**2)

        Area = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
        f11 = b1*b1 + c1*c1
        f12 = b1*b2 + c1*c2
        f13 = b1*b3 + c1*c3
        f21 = b2*b1 + c2*c1
        f22 = b2*b2 + c2*c2
        f23 = b2*b3 + c2*c3
        f31 = b3*b1 + c3*c1
        #f32 = b3*b2 + c3*c2
        f33 = b3*b3 + c3*c3

        F11 = l1*l1/(24*Area) * (f22 - f12 + f11)
        F12 = l1*l2/(48*Area) * (f23 - f22 - 2*f13 + f12)
        F13 = l1*l3/(48*Area) * (f21 - 2*f23 - f11 + f13)
        F21 = F12
        F22 = l2*l2/(24*Area) * (f33 - f23 + f22)
        F23 = l2*l3/(48*Area) * (f31 - f33 - 2*f21 + f23)
        F31 = F13
        F32 = F23
        F33 = l3*l3/(24*Area) * (f11 - f13 + f33)

        F = np.array([[F11, F12, F13],[F21, F22, F23],[F31, F32, F33]])
        
        signs = np.outer(sgn_tri2edge[:,indtri], sgn_tri2edge[:,indtri])
        for i in range(3):
            for j in range(3):
                B[ies[i],ies[j]] += F[i,j]*gamma*signs[i,j]
        
        B1 = l1*(-3*U1x*a1*b2 + 3*U1x*a2*b1 + U1x*b1*c2*y1 + U1x*b1*c2*y2 + U1x*b1*c2*y3 - U1x*b2*c1*y1 - U1x*b2*c1*y2 - U1x*b2*c1*y3 - 3*U1y*a1*c2 + 3*U1y*a2*c1 - U1y*b1*c2*x1 - U1y*b1*c2*x2 - U1y*b1*c2*x3 + U1y*b2*c1*x1 + U1y*b2*c1*x2 + U1y*b2*c1*x3)/(12*Area)

        B2 = l2*(-3*U2x*a2*b3 + 3*U2x*a3*b2 + U2x*b2*c3*y1 + U2x*b2*c3*y2 + U2x*b2*c3*y3 - U2x*b3*c2*y1 - U2x*b3*c2*y2 - U2x*b3*c2*y3 - 3*U2y*a2*c3 + 3*U2y*a3*c2 - U2y*b2*c3*x1 - U2y*b2*c3*x2 - U2y*b2*c3*x3 + U2y*b3*c2*x1 + U2y*b3*c2*x2 + U2y*b3*c2*x3)/(12*Area)

        B3 = l3*(3*U3x*a1*b3 - 3*U3x*a3*b1 - U3x*b1*c3*y1 - U3x*b1*c3*y2 - U3x*b1*c3*y3 + U3x*b3*c1*y1 + U3x*b3*c1*y2 + U3x*b3*c1*y3 + 3*U3y*a1*c3 - 3*U3y*a3*c1 + U3y*b1*c3*x1 + U3y*b1*c3*x2 + U3y*b1*c3*x3 - U3y*b3*c1*x1 - U3y*b3*c1*x2 - U3y*b3*c1*x3)/(12*Area)
        bv[ie1] += B1*sgn_tri2edge[0,indtri]
        bv[ie2] += B2*sgn_tri2edge[1,indtri]
        bv[ie3] += B3*sgn_tri2edge[2,indtri]
    return B, bv



@logger.catch
def assemble_eig_matrix(mesh: Mesh3D,
                     er: np.ndarray, 
                     ur: np.ndarray, 
                     bcs: list[BoundaryCondition]):
    
    logger.debug('Assembling base matrix')
    D, B = assemble_base_matrices(mesh.nodes, mesh.tets, mesh.edges, mesh.tet_to_edge, mesh.tet_to_edge_sign, er, ur)

    logger.debug('Starting boundary conditions.')
    ne = mesh.edges.shape[1]

    pecs = [bc for bc in bcs if isinstance(bc, PEC)]
    
    # Process all PEC Boundary Conditions
    pec_ids = []
    for pec in pecs:
        face_tags = pec.tags
        tri_ids = mesh.get_triangles(face_tags)
        edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

        for ii in edge_ids:
            i = abs(ii)
            D[i, :] = 0
            D[:, i] = 0
            B[i, :] = 0
            B[:, i] = 0
        pec_ids += list(edge_ids)

    ve = pe.Viewer()
    
    pec_ids = [int(x) for x in pec_ids]
    
    solve_ids = np.array([i for i in range(ne) if i not in pec_ids])

    with ve.new3d('Edges') as v:
        v.scatter(mesh.nodes[0,:], mesh.nodes[1,:], mesh.nodes[2,:], size=0.0001)
        v.scatter(mesh.edge_centers[0,pec_ids], mesh.edge_centers[1,pec_ids], mesh.edge_centers[2,pec_ids], size=0.001)
        v.scatter(mesh.edge_centers[0,solve_ids], mesh.edge_centers[1,solve_ids], mesh.edge_centers[2,solve_ids], color=(0,1,0),size=0.001)
    
    return D, B, solve_ids

@logger.catch
def assemble_freq_matrix(mesh: Mesh3D, 
                     er: np.ndarray, 
                     ur: np.ndarray, 
                     bcs: list[BoundaryCondition],
                     frequency: float):
    
    k0 = 2*np.pi*frequency/299792458
    logger.debug('Assembling base matrix')
    D, B = assemble_base_matrices(mesh.nodes, mesh.tets, mesh.edges, mesh.tet_to_edge, mesh.tet_to_edge_sign, er, ur)

    K = D - B*(k0**2)
    
    logger.debug('Starting boundary conditions.')
    
    b = np.zeros((D.shape[0],)).astype(np.complex128)


    pecs = [bc for bc in bcs if isinstance(bc,PEC)]
    ports = [bc for bc in bcs if isinstance(bc,WaveguidePort)]

    # Process all PEC Boundary Conditions
    pec_ids = []
    for pec in pecs:
        face_tags = pec.tags
        tri_ids = mesh.get_triangles(face_tags)
        edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

        for ii in edge_ids:
            i = abs(ii)
            K[i, :] = 0
            K[:, i] = 0
        pec_ids += list(edge_ids)

    # Process all port boundary Conditions
    for port in ports:
        face_tags = port.tags
        tri_ids = mesh.get_triangles(face_tags)
        edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

        x,y,z = mesh.edge_centers[:,edge_ids]

        E, beta, basis = port.port_mode_3d(x, y, z, k0)

        Uinc = -2*1j*beta*E

        Uinc_tot = np.zeros((3, mesh.n_edges), dtype=np.complex128)
        Uinc_tot[:, edge_ids] = Uinc
        
        #port._field_amplitude = np.zeros_like(b)
        #port._field_amplitude[edge_ids] = Ez # dot product with edge

        port._edge_ids = edge_ids
        port._edge_amp = np.sqrt(np.abs(E[0,:])**2 + np.abs(E[1,:])**2 + np.abs(E[2,:])**2)
        gamma = 1j*beta
        B_p, b_p = assemble_surf_matrices(mesh.nodes, tri_ids, mesh.tris, mesh.edges, mesh.tri_to_edge, mesh.tri_to_edge_sign, basis, gamma, Uinc_tot)
        
        ei1s = mesh.edges[0, edge_ids]
        ei2s = mesh.edges[1, edge_ids]
        signs = np.array([mesh.get_edge_sign(i1, i2) for i1, i2 in zip(ei1s, ei2s)])
        ex = (mesh.nodes[0, ei2s] - mesh.nodes[0, ei1s]) * signs
        ey = (mesh.nodes[1, ei2s] - mesh.nodes[1, ei1s]) * signs
        ez = (mesh.nodes[2, ei2s] - mesh.nodes[2, ei1s]) * signs
        length = np.sqrt(ex**2 + ey**2 + ez**2)
        ex = ex/length
        ey = ey/length
        ez = ez/length

        port._b = b_p[edge_ids].imag * np.array([ex, ey, ez])
        if port.active:
            b = b + b_p
            K = K + B_p
        else:
            K = K + B_p

    # pecids = [i for i in pecids if i not in portids]

    solve_ids = np.array([i for i in range(D.shape[0]) if i not in pec_ids])
    
    diagnose_matrix(K)
    diagnose_matrix(b)
    return K, b, solve_ids
        

def interpolate_solution(coords: np.ndarray,
                         solutions: np.ndarray, 
                         tets: np.ndarray, 
                         nodes: np.ndarray, 
                         tet_to_edge: np.ndarray,
                         tet_to_edge_sign: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]

    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]

    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)

    for itet in range(tets.shape[1]):
        iv1, iv2, iv3, iv4 = tets[:, itet]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]
        v4 = nodes[:,iv4]

        bv1 = v2 - v1
        bv2 = v3 - v1
        bv3 = v4 - v1

        blocal = np.array([bv1, bv2, bv3]).T
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        inside = ((coords_local[0,:] + coords_local[1,:] + coords_local[2,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6) & (coords_local[2,:] >= -1e-6)

        xvs = nodes[0, tets[:,itet]]
        yvs = nodes[1, tets[:,itet]]
        zvs = nodes[2, tets[:,itet]]

        x1, x2, x3, x4 = xvs
        y1, y2, y3, y4 = yvs
        z1, z2, z3, z4 = zvs

        a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4, V = compute_coeffs(xvs, yvs, zvs)

        e1, e2, e3, e4, e5, e6 = solutions[tet_to_edge[:, itet]] * tet_to_edge_sign[:, itet]

        x = xs[inside==1]
        y = ys[inside==1]
        z = zs[inside==1]

        ex = (-e1*(b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - e2*(b1*(a3 + b3*x + c3*y + d3*z) - b3*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2) - e3*(b1*(a4 + b4*x + c4*y + d4*z) - b4*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x4)**2 + (y1 - y4)**2 + (z1 - z4)**2) - e4*(b2*(a3 + b3*x + c3*y + d3*z) - b3*(a2 + b2*x + c2*y + d2*z))*np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2) + e5*(b2*(a4 + b4*x + c4*y + d4*z) - b4*(a2 + b2*x + c2*y + d2*z))*np.sqrt((x2 - x4)**2 + (y2 - y4)**2 + (z2 - z4)**2) - e6*(b3*(a4 + b4*x + c4*y + d4*z) - b4*(a3 + b3*x + c3*y + d3*z))*np.sqrt((x3 - x4)**2 + (y3 - y4)**2 + (z3 - z4)**2))/(36*V**2)
        ey = (-e1*(c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - e2*(c1*(a3 + b3*x + c3*y + d3*z) - c3*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2) - e3*(c1*(a4 + b4*x + c4*y + d4*z) - c4*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x4)**2 + (y1 - y4)**2 + (z1 - z4)**2) - e4*(c2*(a3 + b3*x + c3*y + d3*z) - c3*(a2 + b2*x + c2*y + d2*z))*np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2) + e5*(c2*(a4 + b4*x + c4*y + d4*z) - c4*(a2 + b2*x + c2*y + d2*z))*np.sqrt((x2 - x4)**2 + (y2 - y4)**2 + (z2 - z4)**2) - e6*(c3*(a4 + b4*x + c4*y + d4*z) - c4*(a3 + b3*x + c3*y + d3*z))*np.sqrt((x3 - x4)**2 + (y3 - y4)**2 + (z3 - z4)**2))/(36*V**2)
        ez = (-e1*(d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - e2*(d1*(a3 + b3*x + c3*y + d3*z) - d3*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2) - e3*(d1*(a4 + b4*x + c4*y + d4*z) - d4*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x4)**2 + (y1 - y4)**2 + (z1 - z4)**2) - e4*(d2*(a3 + b3*x + c3*y + d3*z) - d3*(a2 + b2*x + c2*y + d2*z))*np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2) + e5*(d2*(a4 + b4*x + c4*y + d4*z) - d4*(a2 + b2*x + c2*y + d2*z))*np.sqrt((x2 - x4)**2 + (y2 - y4)**2 + (z2 - z4)**2) - e6*(d3*(a4 + b4*x + c4*y + d4*z) - d4*(a3 + b3*x + c3*y + d3*z))*np.sqrt((x3 - x4)**2 + (y3 - y4)**2 + (z3 - z4)**2))/(36*V**2)
        Ex[inside] = ex
        Ey[inside] = ey
        Ez[inside] = ez

    return Ex, Ey, Ez
        
class Assembler:

    pass