from numba import njit, f8, i8, types, c16
import numpy as np

from ...optimized import local_mapping, cross, dot, compute_distances, tet_coefficients, volume_coeff, area_coeff, tri_coefficients

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], i8[:,:], i8[:,:], f8[:,:], i8[:,:]), cache=True)
def interpolate_solution(coords: np.ndarray,
                         solutions: np.ndarray, 
                         tets: np.ndarray, 
                         tris: np.ndarray,
                         edges: np.ndarray,
                         nodes: np.ndarray,
                         tet_to_field: np.ndarray):

    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    nEdges = edges.shape[1]

    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]

    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)

    for itet in range(tets.shape[1]):

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

            ex = (Em1*(a1 + b1*x + c1*y + d1*z) + Em2*(a2 + b2*x + c2*y + d2*z))*(b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)/(216*V**3)
            ey = (Em1*(a1 + b1*x + c1*y + d1*z) + Em2*(a2 + b2*x + c2*y + d2*z))*(c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)/(216*V**3)
            ez = (Em1*(a1 + b1*x + c1*y + d1*z) + Em2*(a2 + b2*x + c2*y + d2*z))*(d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)/(216*V**3)
            
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

            efx = 1*(-Em1*(b1*(a3 + b3*x + c3*y + d3*z) - b3*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2)*(a2 + b2*x + c2*y + d2*z) + Em2*(b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)*(a3 + b3*x + c3*y + d3*z))/(216*V**3)
            efy = 1*(-Em1*(c1*(a3 + b3*x + c3*y + d3*z) - c3*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2)*(a2 + b2*x + c2*y + d2*z) + Em2*(c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)*(a3 + b3*x + c3*y + d3*z))/(216*V**3)
            efz = 1*(-Em1*(d1*(a3 + b3*x + c3*y + d3*z) - d3*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2)*(a2 + b2*x + c2*y + d2*z) + Em2*(d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)*(a3 + b3*x + c3*y + d3*z))/(216*V**3)
            Exl += efx
            Eyl += efy
            Ezl += efz
        Ex[inside] = Exl
        Ey[inside] = Eyl
        Ez[inside] = Ezl
    return Ex, Ey, Ez


@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], i8[:,:], f8, f8), cache=True)
def tet_stiff_submatrix(tet_vertices, edge_lengths, local_edge_map, local_tri_map, C_stiffness):
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

    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    letters = [1,2,3,4,5,6]

    for ei in range(4):
        #ei1, ei2 = local_edge_map[:, ei]
        for ej in range(4):
            #ej1, ej2 = local_edge_map[:, ej]
            
            A,B,C,D = letters[ei], letters[ej]#, letters[ej1], letters[ej2]
            GA = GLs[1j]
            #GB = GLs[ei2]
            GC = GLs[1j]
            #GD = GLs[ej2]

            CEE = 1/(6*V)**4 
            
            Dmat[ei,ej] += CEE*(8*volume_coeff(V,A,A,0,0)*dot(GA,GC)+8*volume_coeff(V,A,C,0,0)*dot(GA,GA)-6*A*dot(GA,GC)-2*C*dot(GA,GA)+dot(GA,GC))
    for ei in range(4):
        for ej in range(6):
            ej1, ej2  = local_tri_map[:, ej]

            A,C,D = letters[ei],letters[ej1], letters[ej2]
            GA = GLs[ei]
            GC = GLs[ej1]
            GD = GLs[ej2]
            CEE = 1/(6*V)**4 

            Dmat[ei,ej+4] += CEE*(16*volume_coeff(V,A,D,0,0)*dot(GA,GC)+16*volume_coeff(V,A,C,0,0)*dot(GA,GD)-4*D*dot(GA,GC)-4*C*dot(GA,GD))
            Dmat[ej+4,ei] += CEE*(16*volume_coeff(V,A,D,0,0)*dot(GA,GC)+16*volume_coeff(V,A,C,0,0)*dot(GA,GD)-4*D*dot(GA,GC)-4*C*dot(GA,GD))
            
    for ei in range(6):
        ei1, ei2 = local_tri_map[:, ei]
        for ej in range(6):
            ej1, ej2 = local_tri_map[:, ej]
            
            A,B,C,D = letters[ei1], letters[ei2], letters[ej1], letters[ej2]
            GA = GLs[ei1]
            GB = GLs[ei2]
            GC = GLs[ej1]
            GD = GLs[ej2]

            CEE = 1/(6*V)**4 
            Dmat[ei+4,ej+4] += CEE*(16*volume_coeff(V,B,D,0,0)*dot(GA,GC)+16*volume_coeff(V,B,C,0,0)*dot(GA,GD)+16*volume_coeff(V,A,D,0,0)*dot(GB,GC)+16*volume_coeff(V,A,C,0,0)*dot(GB,GD))

    Dmat = Dmat/C_stiffness
    
    return Dmat
