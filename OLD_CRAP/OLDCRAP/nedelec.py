from numba import njit, f8, c16, types, i8
import numpy as np
from ....mth.optimized import generate_int_points_tet, gaus_quad_tet, tet_coefficients, compute_distances

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_1(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = (b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))*(a1 + b1*x + c1*y + d1*z)
    fs[1,:] = (c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))*(a1 + b1*x + c1*y + d1*z)
    fs[2,:] = (d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))*(a1 + b1*x + c1*y + d1*z)
    return L*fs

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_2(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = (b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z)
    fs[1,:] = (c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z)
    fs[2,:] = (d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z)
    return L*fs

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_3(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = (-b1*(a3 + b3*x + c3*y + d3*z) + b3*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z)
    fs[1,:] = (-c1*(a3 + b3*x + c3*y + d3*z) + c3*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z)
    fs[2,:] = (-d1*(a3 + b3*x + c3*y + d3*z) + d3*(a1 + b1*x + c1*y + d1*z))*(a2 + b2*x + c2*y + d2*z)
    return L*fs

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_4(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = (b1*(a2 + b2*x + c2*y + d2*z) - b2*(a1 + b1*x + c1*y + d1*z))*(a3 + b3*x + c3*y + d3*z)
    fs[1,:] = (c1*(a2 + b2*x + c2*y + d2*z) - c2*(a1 + b1*x + c1*y + d1*z))*(a3 + b3*x + c3*y + d3*z)
    fs[2,:] = (d1*(a2 + b2*x + c2*y + d2*z) - d2*(a1 + b1*x + c1*y + d1*z))*(a3 + b3*x + c3*y + d3*z)
    return L*fs

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_1_curl(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = -3*a1*c1*d2 + 3*a1*c2*d1 - 3*b1*c1*d2*x + 3*b1*c2*d1*x - 3*c1**2*d2*y + 3*c1*c2*d1*y - 3*c1*d1*d2*z + 3*c2*d1**2*z
    fs[1,:] = 3*a1*b1*d2 - 3*a1*b2*d1 + 3*b1**2*d2*x - 3*b1*b2*d1*x + 3*b1*c1*d2*y + 3*b1*d1*d2*z - 3*b2*c1*d1*y - 3*b2*d1**2*z
    fs[2,:] = -3*a1*b1*c2 + 3*a1*b2*c1 - 3*b1**2*c2*x + 3*b1*b2*c1*x - 3*b1*c1*c2*y - 3*b1*c2*d1*z + 3*b2*c1**2*y + 3*b2*c1*d1*z
    return L*fs

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_2_curl(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = -3*a2*c1*d2 + 3*a2*c2*d1 - 3*b2*c1*d2*x + 3*b2*c2*d1*x - 3*c1*c2*d2*y - 3*c1*d2**2*z + 3*c2**2*d1*y + 3*c2*d1*d2*z
    fs[1,:] = 3*a2*b1*d2 - 3*a2*b2*d1 + 3*b1*b2*d2*x + 3*b1*c2*d2*y + 3*b1*d2**2*z - 3*b2**2*d1*x - 3*b2*c2*d1*y - 3*b2*d1*d2*z
    fs[2,:] = -3*a2*b1*c2 + 3*a2*b2*c1 - 3*b1*b2*c2*x - 3*b1*c2**2*y - 3*b1*c2*d2*z + 3*b2**2*c1*x + 3*b2*c1*c2*y + 3*b2*c1*d2*z
    return L*fs

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_3_curl(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = c1*d2*(a3 + b3*x + c3*y + d3*z) + 2*c1*d3*(a2 + b2*x + c2*y + d2*z) - c2*d1*(a3 + b3*x + c3*y + d3*z) + c2*d3*(a1 + b1*x + c1*y + d1*z) - 2*c3*d1*(a2 + b2*x + c2*y + d2*z) - c3*d2*(a1 + b1*x + c1*y + d1*z)
    fs[1,:] = -b1*d2*(a3 + b3*x + c3*y + d3*z) - 2*b1*d3*(a2 + b2*x + c2*y + d2*z) + b2*d1*(a3 + b3*x + c3*y + d3*z) - b2*d3*(a1 + b1*x + c1*y + d1*z) + 2*b3*d1*(a2 + b2*x + c2*y + d2*z) + b3*d2*(a1 + b1*x + c1*y + d1*z)
    fs[2,:] = b1*c2*(a3 + b3*x + c3*y + d3*z) + 2*b1*c3*(a2 + b2*x + c2*y + d2*z) - b2*c1*(a3 + b3*x + c3*y + d3*z) + b2*c3*(a1 + b1*x + c1*y + d1*z) - 2*b3*c1*(a2 + b2*x + c2*y + d2*z) - b3*c2*(a1 + b1*x + c1*y + d1*z)
    return L*fs

@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)
def basis_4_curl(C1, C2, C3, x, y, z, L):
    a1, b1, c1, d1 = C1
    a2, b2, c2, d2 = C2
    a3, b3, c3, d3 = C3
    fs = np.empty((3, x.shape[0]))
    fs[0,:] = -2*c1*d2*(a3 + b3*x + c3*y + d3*z) - c1*d3*(a2 + b2*x + c2*y + d2*z) + 2*c2*d1*(a3 + b3*x + c3*y + d3*z) + c2*d3*(a1 + b1*x + c1*y + d1*z) + c3*d1*(a2 + b2*x + c2*y + d2*z) - c3*d2*(a1 + b1*x + c1*y + d1*z)
    fs[1,:] = 2*b1*d2*(a3 + b3*x + c3*y + d3*z) + b1*d3*(a2 + b2*x + c2*y + d2*z) - 2*b2*d1*(a3 + b3*x + c3*y + d3*z) - b2*d3*(a1 + b1*x + c1*y + d1*z) - b3*d1*(a2 + b2*x + c2*y + d2*z) + b3*d2*(a1 + b1*x + c1*y + d1*z)
    fs[2,:] = -2*b1*c2*(a3 + b3*x + c3*y + d3*z) - b1*c3*(a2 + b2*x + c2*y + d2*z) + 2*b2*c1*(a3 + b3*x + c3*y + d3*z) + b2*c3*(a1 + b1*x + c1*y + d1*z) + b3*c1*(a2 + b2*x + c2*y + d2*z) - b3*c2*(a1 + b1*x + c1*y + d1*z)
    return L*fs

TET_PTS = gaus_quad_tet(4)

@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], i8[:,:], c16[:,:], c16[:,:]), nogil=True, cache=True, parallel=False, fastmath=True)
def matrix_curl_curl(tet_vertices, edge_lengths, local_edge_map, local_tri_map, Ms, Mm):
    weights = TET_PTS[0,:]
    x1, x2, x3, x4 = tet_vertices[0, :]
    y1, y2, y3, y4 = tet_vertices[1, :]
    z1, z2, z3, z4 = tet_vertices[2, :]

    xi = x1*TET_PTS[1,:] + x2*TET_PTS[2,:] + x3*TET_PTS[3,:] + x4*TET_PTS[4,:]
    yi = y1*TET_PTS[1,:] + y2*TET_PTS[2,:] + y3*TET_PTS[3,:] + y4*TET_PTS[4,:]
    zi = z1*TET_PTS[1,:] + z2*TET_PTS[2,:] + z3*TET_PTS[3,:] + z4*TET_PTS[4,:]

    ni = xi.shape[0]
    Dmat = np.empty((20,20), dtype=np.complex128)
    Fmat = np.empty((20,20), dtype=np.complex128)

    xs, ys, zs = tet_vertices
    Ds = compute_distances(xs, ys, zs)
    aas, bbs, ccs, dds, V = tet_coefficients(xs, ys, zs)
    coeefs = np.empty((4,4), dtype=np.float64)
    coeefs[:,0] = aas
    coeefs[:,1] = bbs
    coeefs[:,2] = ccs
    coeefs[:,3] = dds

    dxs = np.empty((20, ni), dtype=np.complex128)
    dys = np.empty((20, ni), dtype=np.complex128)
    dzs = np.empty((20, ni), dtype=np.complex128)
    fxs = np.empty((20, ni), dtype=np.complex128)
    fys = np.empty((20, ni), dtype=np.complex128)
    fzs = np.empty((20, ni), dtype=np.complex128)

    funcnum = (1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4)
    elemnum = (0,1,2,3,4,5,0,1,2,3,0,1,2,3,4,5,0,1,2,3)

    for i in range(20):
        
        if funcnum[i]==1:
            ei, ej = local_edge_map[:,elemnum[i]]
            L = edge_lengths[elemnum[i]]
            fi = 0
            dx, dy, dz = basis_1_curl(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi,L)
            fx, fy, fz = basis_1(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi,L)
        elif funcnum[i]==2:
            ei, ej, fi = local_tri_map[:, elemnum[i]]
            L = Ds[ei,fi]
            dx, dy, dz = basis_3_curl(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi,L)
            fx, fy, fz = basis_3(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi,L)
        elif funcnum[i]==3:
            ei, ej = local_edge_map[:,elemnum[i]]
            L = edge_lengths[elemnum[i]]
            fi = 0
            dx, dy, dz = basis_2_curl(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi, L)
            fx, fy, fz = basis_2(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi, L)
        elif funcnum[i]==4:
            ei, ej, fi = local_tri_map[:, elemnum[i]]
            L = Ds[ei, ej]
            dx, dy, dz = basis_4_curl(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi,L)
            fx, fy, fz = basis_4(coeefs[ei,:], coeefs[ej,:], coeefs[fi,:], xi, yi, zi,L)
        
        dxs[i,:] = dx
        dys[i,:] = dy
        dzs[i,:] = dz
        fxs[i,:] = fx
        fys[i,:] = fy
        fzs[i,:] = fz
    
    KA = 1/(6*V)**4
    KB = 1/(6*V)**2

    F1 = np.empty((3,ni), dtype=np.complex128)
    F2 = np.empty((3,ni), dtype=np.complex128)   
    for i in range(20):
        for j in range(i,20):
            F1[0,:] = dxs[i,:]
            F1[1,:] = dys[i,:]
            F1[2,:] = dzs[i,:]
            F2[0,:] = Ms[0,0]*dxs[j,:] + Ms[0,1]*dys[j,:] + Ms[0,2]*dzs[j,:]
            F2[1,:] = Ms[1,0]*dxs[j,:] + Ms[1,1]*dys[j,:] + Ms[1,2]*dzs[j,:]
            F2[2,:] = Ms[2,0]*dxs[j,:] + Ms[2,1]*dys[j,:] + Ms[2,2]*dzs[j,:]
            Dmat[i,j] = V*np.sum(weights*(F1[0,:]*F2[0,:] + F1[1,:]*F2[1,:] + F1[2,:]*F2[2,:]))
            F1[0,:] = fxs[i,:]
            F1[1,:] = fys[i,:]
            F1[2,:] = fzs[i,:]
            F2[0,:] = Mm[0,0]*fxs[j,:] + Mm[0,1]*fys[j,:] + Mm[0,2]*fzs[j,:]
            F2[1,:] = Mm[1,0]*fxs[j,:] + Mm[1,1]*fys[j,:] + Mm[1,2]*fzs[j,:]
            F2[2,:] = Mm[2,0]*fxs[j,:] + Mm[2,1]*fys[j,:] + Mm[2,2]*fzs[j,:]
            Fmat[i,j] = V*np.sum(weights*(F1[0,:]*F2[0,:] + F1[1,:]*F2[1,:] + F1[2,:]*F2[2,:]))

            if i!=j:
                Dmat[j,i] = Dmat[i,j]
                Fmat[j,i] = Fmat[i,j]
    Dmat = Dmat*KA
    Fmat = Fmat*KB
    return Dmat, Fmat