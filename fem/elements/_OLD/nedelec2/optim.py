from numba import njit, f8, i8, types, c16
import numpy as np

from ...math.optimized import local_mapping, cross, dot, compute_distances, tet_coefficients, volume_coeff, area_coeff, tri_coefficients
@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], i8[:,:], i8[:,:], f8[:,:], i8[:,:]), cache=True, nogil=True)
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


@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], i8[:,:], f8, f8), cache=True, nogil=True)
def tet_stiff_mass_submatrix(tet_vertices, edge_lengths, local_edge_map, local_tri_map, C_stiffness, C_mass):
    ''' Computes the curl dot curl submatrix terms
    
    Submatrix indexing:
    -------------------
    0, 1, 2, 3, 4, 5 = Edge_i mode 1 coefficients
    6, 7, 8, 9 = Face_i vector component 1
    10, 11, 12, 13, 14, 15 = Edge mode 2 coefficients
    16, 17, 18, 19 = Face_i vector component 2

    '''
    Dmat = np.zeros((20,20), dtype=np.complex128)
    Fmat = np.zeros((20,20), dtype=np.complex128)

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

    for ei in range(6):
        ei1, ei2 = local_edge_map[:, ei]
        for ej in range(6):
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

            CEE = 1/(6*V)**4 
            CFEE = 1/(6*V)**2
            
            Dmat[ei,ej] += Li*Lj*CEE*(9*volume_coeff(V,A,C,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei,ej] += Li*Lj*CFEE*(volume_coeff(V,A,B,C,D)*dot(GA,GC)-volume_coeff(V,A,B,C,C)*dot(GA,GD)-volume_coeff(V,A,A,C,D)*dot(GB,GC)+volume_coeff(V,A,A,C,C)*dot(GB,GD))
            Dmat[ei,ej+10] += Li*Lj*CEE*(9*volume_coeff(V,A,D,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei,ej+10] += Li*Lj*CFEE*(volume_coeff(V,A,B,D,D)*dot(GA,GC)-volume_coeff(V,A,B,C,D)*dot(GA,GD)-volume_coeff(V,A,A,D,D)*dot(GB,GC)+volume_coeff(V,A,A,C,D)*dot(GB,GD))
            Dmat[ei+10,ej] += Li*Lj*CEE*(9*volume_coeff(V,B,C,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+10,ej] += Li*Lj*CFEE*(volume_coeff(V,B,B,C,D)*dot(GA,GC)-volume_coeff(V,B,B,C,C)*dot(GA,GD)-volume_coeff(V,A,B,C,D)*dot(GB,GC)+volume_coeff(V,A,B,C,C)*dot(GB,GD))
            Dmat[ei+10,ej+10] += Li*Lj*CEE*(9*volume_coeff(V,B,D,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+10,ej+10] += Li*Lj*CFEE*(volume_coeff(V,B,B,D,D)*dot(GA,GC)-volume_coeff(V,B,B,C,D)*dot(GA,GD)-volume_coeff(V,A,B,D,D)*dot(GB,GC)+volume_coeff(V,A,B,C,D)*dot(GB,GD))
    
    for ei in range(6):
        ei1, ei2 = local_edge_map[:, ei]
        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]

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

            CEF = 1/(6*V)**4
            CFEF = 1/(6*V)**2 

            Dmat[ei,ej+6] += Li*Lac*CEF*(-6*volume_coeff(V,A,D,0,0)*dot(GAxGB,GCxGF)-3*volume_coeff(V,A,C,0,0)*dot(GAxGB,GDxGF)-3*volume_coeff(V,A,F,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei,ej+6] += Li*Lac*CFEF*(volume_coeff(V,A,B,C,D)*dot(GA,GF)-volume_coeff(V,A,B,D,F)*dot(GA,GC)-volume_coeff(V,A,A,C,D)*dot(GB,GF)+volume_coeff(V,A,A,D,F)*dot(GB,GC))
            Dmat[ei,ej+16] += Li*Lab*CEF*(6*volume_coeff(V,A,F,0,0)*dot(GAxGB,GCxGD)+3*volume_coeff(V,A,D,0,0)*dot(GAxGB,GCxGF)-3*volume_coeff(V,A,C,0,0)*dot(GAxGB,GDxGF))
            Fmat[ei,ej+16] += Li*Lab*CFEF*(volume_coeff(V,A,B,D,F)*dot(GA,GC)-volume_coeff(V,A,B,F,C)*dot(GA,GD)-volume_coeff(V,A,A,D,F)*dot(GB,GC)+volume_coeff(V,A,A,F,C)*dot(GB,GD))
            Dmat[ei+10,ej+6] += Li*Lac*CEF*(-6*volume_coeff(V,B,D,0,0)*dot(GAxGB,GCxGF)-3*volume_coeff(V,B,C,0,0)*dot(GAxGB,GDxGF)-3*volume_coeff(V,B,F,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+10,ej+6] += Li*Lac*CFEF*(volume_coeff(V,B,B,C,D)*dot(GA,GF)-volume_coeff(V,B,B,D,F)*dot(GA,GC)-volume_coeff(V,A,B,C,D)*dot(GB,GF)+volume_coeff(V,A,B,D,F)*dot(GB,GC))
            Dmat[ei+10,ej+16] += Li*Lab*CEF*(6*volume_coeff(V,B,F,0,0)*dot(GAxGB,GCxGD)+3*volume_coeff(V,B,D,0,0)*dot(GAxGB,GCxGF)-3*volume_coeff(V,B,C,0,0)*dot(GAxGB,GDxGF))
            Fmat[ei+10,ej+16] += Li*Lab*CFEF*(volume_coeff(V,B,B,D,F)*dot(GA,GC)-volume_coeff(V,B,B,F,C)*dot(GA,GD)-volume_coeff(V,A,B,D,F)*dot(GB,GC)+volume_coeff(V,A,B,F,C)*dot(GB,GD))

    for ei in range(4):
        ei1, ei2, fi = local_tri_map[:, ei]
        for ej in range(6):
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

            CFE = 1/(6*V)**4
            CFFE = 1/(6*V)**2 

            Dmat[ei+6,ej] += Lj*Lac*CFE*(-6*volume_coeff(V,B,C,0,0)*dot(GAxGE,GCxGD)-3*volume_coeff(V,A,C,0,0)*dot(GBxGE,GCxGD)-3*volume_coeff(V,E,C,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+6,ej] += Lj*Lac*CFFE*(volume_coeff(V,A,B,C,D)*dot(GC,GE)-volume_coeff(V,A,B,C,C)*dot(GD,GE)-volume_coeff(V,B,E,C,D)*dot(GA,GC)+volume_coeff(V,B,E,C,C)*dot(GA,GD))
            Dmat[ei+6,ej+10] += Lj*Lac*CFE*(-6*volume_coeff(V,B,D,0,0)*dot(GAxGE,GCxGD)-3*volume_coeff(V,A,D,0,0)*dot(GBxGE,GCxGD)-3*volume_coeff(V,E,D,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+6,ej+10] += Lj*Lac*CFFE*(volume_coeff(V,A,B,D,D)*dot(GC,GE)-volume_coeff(V,A,B,C,D)*dot(GD,GE)-volume_coeff(V,B,E,D,D)*dot(GA,GC)+volume_coeff(V,B,E,C,D)*dot(GA,GD))
            Dmat[ei+16,ej] += Lj*Lab*CFE*(6*volume_coeff(V,E,C,0,0)*dot(GAxGB,GCxGD)+3*volume_coeff(V,B,C,0,0)*dot(GAxGE,GCxGD)-3*volume_coeff(V,A,C,0,0)*dot(GBxGE,GCxGD))
            Fmat[ei+16,ej] += Lj*Lab*CFFE*(volume_coeff(V,B,E,C,D)*dot(GA,GC)-volume_coeff(V,B,E,C,C)*dot(GA,GD)-volume_coeff(V,E,A,C,D)*dot(GB,GC)+volume_coeff(V,E,A,C,C)*dot(GB,GD))
            Dmat[ei+16,ej+10] += Lj*Lab*CFE*(6*volume_coeff(V,E,D,0,0)*dot(GAxGB,GCxGD)+3*volume_coeff(V,B,D,0,0)*dot(GAxGE,GCxGD)-3*volume_coeff(V,A,D,0,0)*dot(GBxGE,GCxGD))
            Fmat[ei+16,ej+10] += Lj*Lab*CFFE*(volume_coeff(V,B,E,D,D)*dot(GA,GC)-volume_coeff(V,B,E,C,D)*dot(GA,GD)-volume_coeff(V,E,A,D,D)*dot(GB,GC)+volume_coeff(V,E,A,C,D)*dot(GB,GD))

    for ei in range(4):
        ei1, ei2, fi = local_tri_map[:, ei]
        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]
            
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

            CFF = 1/(6*V)**4
            CFFF = 1/(6*V)**2

            Dmat[ei+6,ej+6] += Lac1*Lac2*CFF*(4*volume_coeff(V,B,D,0,0)*dot(GAxGE,GCxGF)+2*volume_coeff(V,B,C,0,0)*dot(GAxGE,GDxGF)+2*volume_coeff(V,B,F,0,0)*dot(GAxGE,GCxGD)+2*volume_coeff(V,A,D,0,0)*dot(GBxGE,GCxGF)+volume_coeff(V,A,C,0,0)*dot(GBxGE,GDxGF)+volume_coeff(V,A,F,0,0)*dot(GBxGE,GCxGD)+2*volume_coeff(V,E,D,0,0)*dot(GAxGB,GCxGF)+volume_coeff(V,E,C,0,0)*dot(GAxGB,GDxGF)+volume_coeff(V,E,F,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+6,ej+6] += Lac1*Lac2*CFFF*(volume_coeff(V,A,B,C,D)*dot(GE,GF)-volume_coeff(V,A,B,D,F)*dot(GC,GE)-volume_coeff(V,B,E,C,D)*dot(GA,GF)+volume_coeff(V,B,E,D,F)*dot(GA,GC))
            Dmat[ei+6,ej+16] += Lac1*Lab2*CFF*(-4*volume_coeff(V,B,F,0,0)*dot(GAxGE,GCxGD)-2*volume_coeff(V,B,D,0,0)*dot(GAxGE,GCxGF)+2*volume_coeff(V,B,C,0,0)*dot(GAxGE,GDxGF)-2*volume_coeff(V,A,F,0,0)*dot(GBxGE,GCxGD)-volume_coeff(V,A,D,0,0)*dot(GBxGE,GCxGF)+volume_coeff(V,A,C,0,0)*dot(GBxGE,GDxGF)-2*volume_coeff(V,E,F,0,0)*dot(GAxGB,GCxGD)-volume_coeff(V,E,D,0,0)*dot(GAxGB,GCxGF)+volume_coeff(V,E,C,0,0)*dot(GAxGB,GDxGF))
            Fmat[ei+6,ej+16] += Lac1*Lab2*CFFF*(volume_coeff(V,A,B,D,F)*dot(GC,GE)-volume_coeff(V,A,B,F,C)*dot(GD,GE)-volume_coeff(V,B,E,D,F)*dot(GA,GC)+volume_coeff(V,B,E,F,C)*dot(GA,GD))
            Dmat[ei+16,ej+6] += Lab1*Lac2*CFF*(-4*volume_coeff(V,E,D,0,0)*dot(GAxGB,GCxGF)-2*volume_coeff(V,E,C,0,0)*dot(GAxGB,GDxGF)-2*volume_coeff(V,E,F,0,0)*dot(GAxGB,GCxGD)-2*volume_coeff(V,B,D,0,0)*dot(GAxGE,GCxGF)-volume_coeff(V,B,C,0,0)*dot(GAxGE,GDxGF)-volume_coeff(V,B,F,0,0)*dot(GAxGE,GCxGD)+2*volume_coeff(V,A,D,0,0)*dot(GBxGE,GCxGF)+volume_coeff(V,A,C,0,0)*dot(GBxGE,GDxGF)+volume_coeff(V,A,F,0,0)*dot(GBxGE,GCxGD))
            Fmat[ei+16,ej+6] += Lab1*Lac2*CFFF*(volume_coeff(V,B,E,C,D)*dot(GA,GF)-volume_coeff(V,B,E,D,F)*dot(GA,GC)-volume_coeff(V,E,A,C,D)*dot(GB,GF)+volume_coeff(V,E,A,D,F)*dot(GB,GC))
            Dmat[ei+16,ej+16] += Lab1*Lab2*CFF*(4*volume_coeff(V,E,F,0,0)*dot(GAxGB,GCxGD)+2*volume_coeff(V,E,D,0,0)*dot(GAxGB,GCxGF)-2*volume_coeff(V,E,C,0,0)*dot(GAxGB,GDxGF)+2*volume_coeff(V,B,F,0,0)*dot(GAxGE,GCxGD)+volume_coeff(V,B,D,0,0)*dot(GAxGE,GCxGF)-volume_coeff(V,B,C,0,0)*dot(GAxGE,GDxGF)-2*volume_coeff(V,A,F,0,0)*dot(GBxGE,GCxGD)-volume_coeff(V,A,D,0,0)*dot(GBxGE,GCxGF)+volume_coeff(V,A,C,0,0)*dot(GBxGE,GDxGF))
            Fmat[ei+16,ej+16] += Lab1*Lab2*CFFF*(volume_coeff(V,B,E,D,F)*dot(GA,GC)-volume_coeff(V,B,E,F,C)*dot(GA,GD)-volume_coeff(V,E,A,D,F)*dot(GB,GC)+volume_coeff(V,E,A,F,C)*dot(GB,GD))


    Dmat = Dmat/C_mass
    Fmat = Fmat*C_stiffness
    
    return Dmat, Fmat

@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], f8[:,:], i8[:,:], f8, f8), cache=True, nogil=True)
def tri_stiff_mass_submatrix(tri_vertices, edge_lengths, local_edge_map, C_stiffness, C_mass):
    ''' Computes the curl dot curl submatrix terms
    
    Submatrix indexing:
    -------------------
    0, 1, 2, 3, 4, 5 = Edge_i mode 1 coefficients
    6, 7, 8, 9 = Face_i vector component 1
    10, 11, 12, 13, 14, 15 = Edge mode 2 coefficients
    16, 17, 18, 19 = Face_i vector component 2

    '''
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
            
            Dmat[ei,ej] += Li*Lj*CEE*(9*area_coeff(Area,A,C,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei,ej] += Li*Lj*CFEE*(area_coeff(Area,A,B,C,D)*dot(GA,GC)-area_coeff(Area,A,B,C,C)*dot(GA,GD)-area_coeff(Area,A,A,C,D)*dot(GB,GC)+area_coeff(Area,A,A,C,C)*dot(GB,GD))
            Dmat[ei,ej+4] += Li*Lj*CEE*(9*area_coeff(Area,A,D,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei,ej+4] += Li*Lj*CFEE*(area_coeff(Area,A,B,D,D)*dot(GA,GC)-area_coeff(Area,A,B,C,D)*dot(GA,GD)-area_coeff(Area,A,A,D,D)*dot(GB,GC)+area_coeff(Area,A,A,C,D)*dot(GB,GD))
            Dmat[ei+4,ej] += Li*Lj*CEE*(9*area_coeff(Area,B,C,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+4,ej] += Li*Lj*CFEE*(area_coeff(Area,B,B,C,D)*dot(GA,GC)-area_coeff(Area,B,B,C,C)*dot(GA,GD)-area_coeff(Area,A,B,C,D)*dot(GB,GC)+area_coeff(Area,A,B,C,C)*dot(GB,GD))
            Dmat[ei+4,ej+4] += Li*Lj*CEE*(9*area_coeff(Area,B,D,0,0)*dot(GAxGB,GCxGD))
            Fmat[ei+4,ej+4] += Li*Lj*CFEE*(area_coeff(Area,B,B,D,D)*dot(GA,GC)-area_coeff(Area,B,B,C,D)*dot(GA,GD)-area_coeff(Area,A,B,D,D)*dot(GB,GC)+area_coeff(Area,A,B,C,D)*dot(GB,GD))
    
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

        Dmat[ei,3] += Li*Lac*CEF*(-6*area_coeff(Area,A,D,0,0)*dot(GAxGB,GCxGF)-3*area_coeff(Area,A,C,0,0)*dot(GAxGB,GDxGF)-3*area_coeff(Area,A,F,0,0)*dot(GAxGB,GCxGD))
        Fmat[ei,3] += Li*Lac*CFEF*(area_coeff(Area,A,B,C,D)*dot(GA,GF)-area_coeff(Area,A,B,D,F)*dot(GA,GC)-area_coeff(Area,A,A,C,D)*dot(GB,GF)+area_coeff(Area,A,A,D,F)*dot(GB,GC))
        Dmat[ei,7] += Li*Lab*CEF*(6*area_coeff(Area,A,F,0,0)*dot(GAxGB,GCxGD)+3*area_coeff(Area,A,D,0,0)*dot(GAxGB,GCxGF)-3*area_coeff(Area,A,C,0,0)*dot(GAxGB,GDxGF))
        Fmat[ei,7] += Li*Lab*CFEF*(area_coeff(Area,A,B,D,F)*dot(GA,GC)-area_coeff(Area,A,B,F,C)*dot(GA,GD)-area_coeff(Area,A,A,D,F)*dot(GB,GC)+area_coeff(Area,A,A,F,C)*dot(GB,GD))
        Dmat[ei+4,3] += Li*Lac*CEF*(-6*area_coeff(Area,B,D,0,0)*dot(GAxGB,GCxGF)-3*area_coeff(Area,B,C,0,0)*dot(GAxGB,GDxGF)-3*area_coeff(Area,B,F,0,0)*dot(GAxGB,GCxGD))
        Fmat[ei+4,3] += Li*Lac*CFEF*(area_coeff(Area,B,B,C,D)*dot(GA,GF)-area_coeff(Area,B,B,D,F)*dot(GA,GC)-area_coeff(Area,A,B,C,D)*dot(GB,GF)+area_coeff(Area,A,B,D,F)*dot(GB,GC))
        Dmat[ei+4,7] += Li*Lab*CEF*(6*area_coeff(Area,B,F,0,0)*dot(GAxGB,GCxGD)+3*area_coeff(Area,B,D,0,0)*dot(GAxGB,GCxGF)-3*area_coeff(Area,B,C,0,0)*dot(GAxGB,GDxGF))
        Fmat[ei+4,7] += Li*Lab*CFEF*(area_coeff(Area,B,B,D,F)*dot(GA,GC)-area_coeff(Area,B,B,F,C)*dot(GA,GD)-area_coeff(Area,A,B,D,F)*dot(GB,GC)+area_coeff(Area,A,B,F,C)*dot(GB,GD))


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

        Dmat[3,ej] += Lj*Lac*CFE*(-6*area_coeff(Area,B,C,0,0)*dot(GAxGE,GCxGD)-3*area_coeff(Area,A,C,0,0)*dot(GBxGE,GCxGD)-3*area_coeff(Area,E,C,0,0)*dot(GAxGB,GCxGD))
        Fmat[3,ej] += Lj*Lac*CFFE*(area_coeff(Area,A,B,C,D)*dot(GC,GE)-area_coeff(Area,A,B,C,C)*dot(GD,GE)-area_coeff(Area,B,E,C,D)*dot(GA,GC)+area_coeff(Area,B,E,C,C)*dot(GA,GD))
        Dmat[3,ej+4] += Lj*Lac*CFE*(-6*area_coeff(Area,B,D,0,0)*dot(GAxGE,GCxGD)-3*area_coeff(Area,A,D,0,0)*dot(GBxGE,GCxGD)-3*area_coeff(Area,E,D,0,0)*dot(GAxGB,GCxGD))
        Fmat[3,ej+4] += Lj*Lac*CFFE*(area_coeff(Area,A,B,D,D)*dot(GC,GE)-area_coeff(Area,A,B,C,D)*dot(GD,GE)-area_coeff(Area,B,E,D,D)*dot(GA,GC)+area_coeff(Area,B,E,C,D)*dot(GA,GD))
        Dmat[7,ej] += Lj*Lab*CFE*(6*area_coeff(Area,E,C,0,0)*dot(GAxGB,GCxGD)+3*area_coeff(Area,B,C,0,0)*dot(GAxGE,GCxGD)-3*area_coeff(Area,A,C,0,0)*dot(GBxGE,GCxGD))
        Fmat[7,ej] += Lj*Lab*CFFE*(area_coeff(Area,B,E,C,D)*dot(GA,GC)-area_coeff(Area,B,E,C,C)*dot(GA,GD)-area_coeff(Area,E,A,C,D)*dot(GB,GC)+area_coeff(Area,E,A,C,C)*dot(GB,GD))
        Dmat[7,ej+4] += Lj*Lab*CFE*(6*area_coeff(Area,E,D,0,0)*dot(GAxGB,GCxGD)+3*area_coeff(Area,B,D,0,0)*dot(GAxGE,GCxGD)-3*area_coeff(Area,A,D,0,0)*dot(GBxGE,GCxGD))
        Fmat[7,ej+4] += Lj*Lab*CFFE*(area_coeff(Area,B,E,D,D)*dot(GA,GC)-area_coeff(Area,B,E,C,D)*dot(GA,GD)-area_coeff(Area,E,A,D,D)*dot(GB,GC)+area_coeff(Area,E,A,C,D)*dot(GB,GD))

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

    Dmat[3,3] += Lac1*Lac2*CFF*(4*area_coeff(Area,B,D,0,0)*dot(GAxGE,GCxGF)+2*area_coeff(Area,B,C,0,0)*dot(GAxGE,GDxGF)+2*area_coeff(Area,B,F,0,0)*dot(GAxGE,GCxGD)+2*area_coeff(Area,A,D,0,0)*dot(GBxGE,GCxGF)+area_coeff(Area,A,C,0,0)*dot(GBxGE,GDxGF)+area_coeff(Area,A,F,0,0)*dot(GBxGE,GCxGD)+2*area_coeff(Area,E,D,0,0)*dot(GAxGB,GCxGF)+area_coeff(Area,E,C,0,0)*dot(GAxGB,GDxGF)+area_coeff(Area,E,F,0,0)*dot(GAxGB,GCxGD))
    Fmat[3,3] += Lac1*Lac2*CFFF*(area_coeff(Area,A,B,C,D)*dot(GE,GF)-area_coeff(Area,A,B,D,F)*dot(GC,GE)-area_coeff(Area,B,E,C,D)*dot(GA,GF)+area_coeff(Area,B,E,D,F)*dot(GA,GC))
    Dmat[3,7] += Lac1*Lab2*CFF*(-4*area_coeff(Area,B,F,0,0)*dot(GAxGE,GCxGD)-2*area_coeff(Area,B,D,0,0)*dot(GAxGE,GCxGF)+2*area_coeff(Area,B,C,0,0)*dot(GAxGE,GDxGF)-2*area_coeff(Area,A,F,0,0)*dot(GBxGE,GCxGD)-area_coeff(Area,A,D,0,0)*dot(GBxGE,GCxGF)+area_coeff(Area,A,C,0,0)*dot(GBxGE,GDxGF)-2*area_coeff(Area,E,F,0,0)*dot(GAxGB,GCxGD)-area_coeff(Area,E,D,0,0)*dot(GAxGB,GCxGF)+area_coeff(Area,E,C,0,0)*dot(GAxGB,GDxGF))
    Fmat[3,7] += Lac1*Lab2*CFFF*(area_coeff(Area,A,B,D,F)*dot(GC,GE)-area_coeff(Area,A,B,F,C)*dot(GD,GE)-area_coeff(Area,B,E,D,F)*dot(GA,GC)+area_coeff(Area,B,E,F,C)*dot(GA,GD))
    Dmat[7,3] += Lab1*Lac2*CFF*(-4*area_coeff(Area,E,D,0,0)*dot(GAxGB,GCxGF)-2*area_coeff(Area,E,C,0,0)*dot(GAxGB,GDxGF)-2*area_coeff(Area,E,F,0,0)*dot(GAxGB,GCxGD)-2*area_coeff(Area,B,D,0,0)*dot(GAxGE,GCxGF)-area_coeff(Area,B,C,0,0)*dot(GAxGE,GDxGF)-area_coeff(Area,B,F,0,0)*dot(GAxGE,GCxGD)+2*area_coeff(Area,A,D,0,0)*dot(GBxGE,GCxGF)+area_coeff(Area,A,C,0,0)*dot(GBxGE,GDxGF)+area_coeff(Area,A,F,0,0)*dot(GBxGE,GCxGD))
    Fmat[7,3] += Lab1*Lac2*CFFF*(area_coeff(Area,B,E,C,D)*dot(GA,GF)-area_coeff(Area,B,E,D,F)*dot(GA,GC)-area_coeff(Area,E,A,C,D)*dot(GB,GF)+area_coeff(Area,E,A,D,F)*dot(GB,GC))
    Dmat[7,7] += Lab1*Lab2*CFF*(4*area_coeff(Area,E,F,0,0)*dot(GAxGB,GCxGD)+2*area_coeff(Area,E,D,0,0)*dot(GAxGB,GCxGF)-2*area_coeff(Area,E,C,0,0)*dot(GAxGB,GDxGF)+2*area_coeff(Area,B,F,0,0)*dot(GAxGE,GCxGD)+area_coeff(Area,B,D,0,0)*dot(GAxGE,GCxGF)-area_coeff(Area,B,C,0,0)*dot(GAxGE,GDxGF)-2*area_coeff(Area,A,F,0,0)*dot(GBxGE,GCxGD)-area_coeff(Area,A,D,0,0)*dot(GBxGE,GCxGF)+area_coeff(Area,A,C,0,0)*dot(GBxGE,GDxGF))
    Fmat[7,7] += Lab1*Lab2*CFFF*(area_coeff(Area,B,E,D,F)*dot(GA,GC)-area_coeff(Area,B,E,F,C)*dot(GA,GD)-area_coeff(Area,E,A,D,F)*dot(GB,GC)+area_coeff(Area,E,A,F,C)*dot(GB,GD))
    

    Dmat = Dmat*C_stiffness
    Fmat = Fmat/C_mass
    
    return Dmat, Fmat

@njit(types.Tuple((c16[:,:],c16[:]))(f8[:,:], f8[:], c16, c16[:,:], f8[:,:]), cache=True, nogil=True)
def tri_stiff_vec_matrix(lcs_vertices, edge_lengths, gamma, lcs_Uinc, DPTs):
    ''' Computes the curl dot curl submatrix terms
    
    Submatrix indexing:
    -------------------
    0, 1, 2, 3, 4, 5 = Edge_i mode 1 coefficients
    6, 7, 8, 9 = Face_i vector component 1
    10, 11, 12, 13, 14, 15 = Edge mode 2 coefficients
    16, 17, 18, 19 = Face_i vector component 2

    '''
    local_edge_map = np.array([[0,1,0],[1,2,2]])
    Bmat = np.zeros((8,8), dtype=np.complex128)
    bvec = np.zeros((8,), dtype=np.complex128)

    xs = lcs_vertices[0,:]
    ys = lcs_vertices[1,:]

    #v1, v2, v3 = lcs_vertices[:,0], lcs_vertices[:,1], lcs_vertices[:,2]

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

            Bmat[ei,ej] += Li*Lj*COEFF*(area_coeff(Area,A,B,C,D)*dot(GA,GC)-area_coeff(Area,A,B,C,C)*dot(GA,GD)-area_coeff(Area,A,A,C,D)*dot(GB,GC)+area_coeff(Area,A,A,C,C)*dot(GB,GD))
            Bmat[ei,ej+4] += Li*Lj*COEFF*(area_coeff(Area,A,B,D,D)*dot(GA,GC)-area_coeff(Area,A,B,C,D)*dot(GA,GD)-area_coeff(Area,A,A,D,D)*dot(GB,GC)+area_coeff(Area,A,A,C,D)*dot(GB,GD))
            Bmat[ei+4,ej] += Li*Lj*COEFF*(area_coeff(Area,B,B,C,D)*dot(GA,GC)-area_coeff(Area,B,B,C,C)*dot(GA,GD)-area_coeff(Area,A,B,C,D)*dot(GB,GC)+area_coeff(Area,A,B,C,C)*dot(GB,GD))
            Bmat[ei+4,ej+4] += Li*Lj*COEFF*(area_coeff(Area,B,B,D,D)*dot(GA,GC)-area_coeff(Area,B,B,C,D)*dot(GA,GD)-area_coeff(Area,A,B,D,D)*dot(GB,GC)+area_coeff(Area,A,B,C,D)*dot(GB,GD))
            
        Bmat[ei,3] += Li*Lt1*COEFF*(area_coeff(Area,A,B,tA,tB)*dot(GA,GtC)-area_coeff(Area,A,B,tB,tC)*dot(GA,GtA)-area_coeff(Area,A,A,tA,tB)*dot(GB,GtC)+area_coeff(Area,A,A,tB,tC)*dot(GB,GtA))
        Bmat[ei,7] += Li*Lt2*COEFF*(area_coeff(Area,A,B,tB,tC)*dot(GA,GtA)-area_coeff(Area,A,B,tC,tA)*dot(GA,GtB)-area_coeff(Area,A,A,tB,tC)*dot(GB,GtA)+area_coeff(Area,A,A,tC,tA)*dot(GB,GtB))
        Bmat[3,ei] += Lt1*Li*COEFF*(area_coeff(Area,tA,tB,A,B)*dot(GA,GtC)-area_coeff(Area,tA,tB,A,A)*dot(GB,GtC)-area_coeff(Area,tB,tC,A,B)*dot(GA,GtA)+area_coeff(Area,tB,tC,A,A)*dot(GB,GtA))
        Bmat[7,ei] += Lt2*Li*COEFF*(area_coeff(Area,tB,tC,A,B)*dot(GA,GtA)-area_coeff(Area,tB,tC,A,A)*dot(GB,GtA)-area_coeff(Area,tC,tA,A,B)*dot(GA,GtB)+area_coeff(Area,tC,tA,A,A)*dot(GB,GtB))
        Bmat[ei+4,3] += Li*Lt1*COEFF*(area_coeff(Area,B,B,tA,tB)*dot(GA,GtC)-area_coeff(Area,B,B,tB,tC)*dot(GA,GtA)-area_coeff(Area,A,B,tA,tB)*dot(GB,GtC)+area_coeff(Area,A,B,tB,tC)*dot(GB,GtA))
        Bmat[ei+4,7] += Li*Lt2*COEFF*(area_coeff(Area,B,B,tB,tC)*dot(GA,GtA)-area_coeff(Area,B,B,tC,tA)*dot(GA,GtB)-area_coeff(Area,A,B,tB,tC)*dot(GB,GtA)+area_coeff(Area,A,B,tC,tA)*dot(GB,GtB))
        Bmat[3,ei+4] += Lt1*Li*COEFF*(area_coeff(Area,tA,tB,B,B)*dot(GA,GtC)-area_coeff(Area,tA,tB,A,B)*dot(GB,GtC)-area_coeff(Area,tB,tC,B,B)*dot(GA,GtA)+area_coeff(Area,tB,tC,A,B)*dot(GB,GtA))
        Bmat[7,ei+4] += Lt2*Li*COEFF*(area_coeff(Area,tB,tC,B,B)*dot(GA,GtA)-area_coeff(Area,tB,tC,A,B)*dot(GB,GtA)-area_coeff(Area,tC,tA,B,B)*dot(GA,GtB)+area_coeff(Area,tC,tA,A,B)*dot(GB,GtB))
            
        A1, A2 = As[ei1], As[ei2]
        B1, B2 = Bs[ei1], Bs[ei2]
        C1, C2 = Cs[ei1], Cs[ei2]

        Ee1x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee1y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee2x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)
        Ee2y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)

        bvec[ei] += signA*Area*np.sum(Ws*(Ee1x*Ux + Ee1y*Uy))
        bvec[ei+4] += signA*Area*np.sum(Ws*(Ee2x*Ux + Ee2y*Uy))
    
    Bmat[3,3] += Lt1*Lt1*COEFF*(area_coeff(Area,tA,tB,tA,tB)*dot(GtC,GtC)-area_coeff(Area,tA,tB,tB,tC)*dot(GtA,GtC)-area_coeff(Area,tB,tC,tA,tB)*dot(GtA,GtC)+area_coeff(Area,tB,tC,tB,tC)*dot(GtA,GtA))
    Bmat[3,7] += Lt1*Lt2*COEFF*(area_coeff(Area,tA,tB,tB,tC)*dot(GtA,GtC)-area_coeff(Area,tA,tB,tC,tA)*dot(GtB,GtC)-area_coeff(Area,tB,tC,tB,tC)*dot(GtA,GtA)+area_coeff(Area,tB,tC,tC,tA)*dot(GtA,GtB))
    Bmat[7,3] += Lt2*Lt1*COEFF*(area_coeff(Area,tB,tC,tA,tB)*dot(GtA,GtC)-area_coeff(Area,tB,tC,tB,tC)*dot(GtA,GtA)-area_coeff(Area,tC,tA,tA,tB)*dot(GtB,GtC)+area_coeff(Area,tC,tA,tB,tC)*dot(GtA,GtB))
    Bmat[7,7] += Lt2*Lt2*COEFF*(area_coeff(Area,tB,tC,tB,tC)*dot(GtA,GtA)-area_coeff(Area,tB,tC,tC,tA)*dot(GtA,GtB)-area_coeff(Area,tC,tA,tB,tC)*dot(GtA,GtB)+area_coeff(Area,tC,tA,tC,tA)*dot(GtB,GtB))
    
    A1, A2, A3 = As
    B1, B2, B3 = Bs
    C1, C2, C3 = Cs
   
    Ef1x = Lt1*(-B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + B3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef1y = Lt1*(-C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + C3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef2x = Lt2*(B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - B2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    Ef2y = Lt2*(C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - C2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    
    bvec[3] += signA*Area*np.sum(Ws*(Ef1x*Ux + Ef1y*Uy))
    bvec[7] += signA*Area*np.sum(Ws*(Ef2x*Ux + Ef2y*Uy))

    return Bmat, bvec


@njit(c16(f8[:,:], f8[:], c16[:,:], f8[:,:]), cache=True, nogil=True)
def tri_surf_integral(lcs_vertices, edge_lengths, lcs_Uinc, DPTs):
    ''' Computes the curl dot curl submatrix terms
    
    Submatrix indexing:
    -------------------
    0, 1, 2, 3, 4, 5 = Edge_i mode 1 coefficients
    6, 7, 8, 9 = Face_i vector component 1
    10, 11, 12, 13, 14, 15 = Edge mode 2 coefficients
    16, 17, 18, 19 = Face_i vector component 2

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