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



############################################################
#             TRIANGLE GAUSS QUADRATURE POINTS            #
############################################################

_GAUSQUADTRI = {
    1: [(1, 1, 1/3, 1/3, 1/3),],
    2: [(3, 1/3, 2/3, 1/6, 1/6),],
    3: [(1, -0.562500000000000, 1/3, 1/3, 1/3),
        (3, 0.520833333333333, 0.6, 0.2, 0.2)],
    4: [(3, 0.223381589678011, 0.108103018168070, 0.445948490915965,0.445948490915965),
        (3, 0.109951743655322, 0.816847572980459, 0.091576213509771, 0.091576213509771)],
    5: [(1, 0.225000000000000, 0.333333333333333, 0.333333333333333, 0.333333333333333),
        (3, 0.132394152788506, 0.059715871789770, 0.470142064105115, 0.470142064105115),
        (3, 0.125939180544827, 0.797426985353087, 0.101286507323456, 0.101286507323456),],
    6: [(3, 0.116786275726379, 0.501426509658179, 0.249286745170910, 0.249286745170910),
        (3, 0.050844906370207, 0.873821971016996, 0.063089014491502, 0.063089014491502),
        (6, 0.082851075618374, 0.053145049844817, 0.310352451033784, 0.636502499121399)],
    7: [(1, -0.149570044467682, 0.333333333333333, 0.333333333333333, 0.333333333333333),
        (3, 0.175615257433208, 0.479308067841920, 0.260345966079040, 0.260345966079040),
        (3, 0.053347235608838, 0.869739794195568, 0.065130102902216, 0.065130102902216),
        (6, 0.077113760890257, 0.048690315425316, 0.312865496004874, 0.638444188569810)],
    8: [(1, 0.144315607677787, 0.333333333333333, 0.333333333333333, 0.333333333333333),
        (3, 0.095091634267285, 0.081414823414554, 0.459292588292723, 0.459292588292723),
        (3, 0.103217370534718, 0.658861384496480, 0.170569307751760, 0.170569307751760),
        (3, 0.032458497623198, 0.898905543365938, 0.050547228317031, 0.050547228317031),
        (6, 0.027230314174435, 0.008394777409958, 0.263112829634638, 0.728492392955404)],
    9: [(1, 0.097135796282799, 0.333333333333333, 0.333333333333333, 0.333333333333333),
        (3, 0.031334700227139, 0.020634961602525, 0.489682519198738, 0.489682519198738),
        (3, 0.077827541004774, 0.125820817014127, 0.437089591492937, 0.437089591492937),
        (3, 0.079647738927210, 0.623592928761935, 0.188203535619033, 0.188203535619033),
        (3, 0.025577675658698, 0.910540973211095, 0.044729513394453, 0.044729513394453),
        (6, 0.043283539377289, 0.036838412054736, 0.221962989160766, 0.741198598784498)],
    10: [(1, 0.090817990382754, 0.333333333333333, 0.333333333333333, 0.333333333333333),
         (3, 0.036725957756467, 0.028844733232685, 0.485577633383657, 0.485577633383657),
         (3, 0.045321059435528, 0.781036849029926, 0.109481575485037, 0.109481575485037),
         (6, 0.072757916845420, 0.141707219414880, 0.307939838764121, 0.550352941820999),
         (6, 0.028327242531057, 0.025003534762686, 0.246672560639903, 0.728323904597411),
         (6, 0.009421666963733, 0.009540815400299, 0.066803251012200, 0.923655933587500)]
}

############################################################
#            TETRAHEDRON GAUSS QUADRATURE POINTS           #
############################################################

_GAUSQUADTET = {
    1: [(1, 1, 0.25, 0.25, 0.25, 0.25),],
    2: [(4, 0.25, 0.5584510197, 0.1381966011, 0.1381966011, 0.1381966011),],
    3: [(1, -0.8, 0.25, 0.25, 0.25, 0.25),
        (4, 0.45, 0.5, 0.166666667, 0.166666667, 0.166666667)],
    4: [(1, -0.078933, 0.25, 0.25, 0.25, 0.25),
        (4, 0.0457333333, 0.7857142857, 0.0714285714, 0.0714285714, 0.0714285714),
        (1, 0.1493333333, 0.3994035762, 0.1005964238, 0.3994035762, 0.1005964238),
        (1, 0.1493333333, 0.3994035762, 0.1005964238, 0.1005964238, 0.3994035762),
        (1, 0.1493333333, 0.3994035762, 0.3994035762, 0.1005964238, 0.1005964238),
        (1, 0.1493333333, 0.1005964238, 0.3994035762, 0.3994035762, 0.1005964238),
        (1, 0.1493333333, 0.1005964238, 0.3994035762, 0.1005964238, 0.3994035762),
        (1, 0.1493333333, 0.1005964238, 0.1005964238, 0.3994035762, 0.3994035762),],
    5: None
}


############################################################
#                         FUNCTIONS                        #
############################################################

def gaus_quad_tri(p: int) -> np.ndarray:
    """
    Returns the duvanant quadrature triangle sample points W, L1, L2, L3, coordinates for a given order p.

    Parameters
    ----------
    p : int
        The order of the quadrature rule.
    Returns
    -------
    pts : np.ndarray
        The sample points W, L1, L2, L3.
    -------

    P = dunavant_points(p)
    P[0,:] = Weights
    P[1,:] = L1 values
    P[2,:] = L2 values
    P[3,:] = L3 values
    """

    Pts = []
    for N, W, L1, L2, L3 in _GAUSQUADTRI[p]:
        l1, l2, l3 = L1, L2, L3
        for n in range(N):
            if n==3:
                l1, l2, l3 = L1, L3, L2
            
            Pts.append([W,l1, l2, l3])
            l1, l2, l3 = l2, l3, l1
    pts = np.array(Pts).T
    return pts

def gaus_quad_tet(p: int) -> np.ndarray:
    """
    Returns the duvanant quadrature tetrahedron sample points W, L1, L2, L3, L4, coordinates for a given order p.

    Parameters
    ----------
    p : int
        The order of the quadrature rule.
    Returns
    -------
    pts : np.ndarray
        The sample points W, L1, L2, L3.
    -------

    P = dunavant_points(p)
    P[0,:] = Weights
    P[1,:] = L1 values
    P[2,:] = L2 values
    P[3,:] = L3 values
    """

    Pts = []
    for N, W, L1, L2, L3, L4 in _GAUSQUADTET[p]:
        l1, l2, l3, l4 = L1, L2, L3, L4
        for n in range(N):
            Pts.append([W, l1, l2, l3, l4])
            l1, l2, l3, l4 = l2, l3, l4, l1
    pts = np.array(Pts).T
    return pts



############################################################
#                      NUMBA COMPILED                     #
############################################################

@njit(types.Tuple((f8[:], f8[:], f8[:], i8[:]))(f8[:,:], i8[:,:], f8[:,:]), cache=True, nogil=True)
def generate_int_points_tri(nodes: np.ndarray,
                                triangles: np.ndarray,
                                PTS: np.ndarray):

    nDPTs = PTS.shape[1]
    xall = np.zeros((nDPTs, triangles.shape[1]))
    yall = np.zeros((nDPTs, triangles.shape[1]))
    zall = np.zeros((nDPTs, triangles.shape[1]))

    for it in range(triangles.shape[1]):
        
        vertex_ids = triangles[:, it]

        x1, x2, x3 = nodes[0, vertex_ids]
        y1, y2, y3 = nodes[1, vertex_ids]
        z1, z2, z3 = nodes[2, vertex_ids]

        xspts = x1*PTS[1,:] + x2*PTS[2,:] + x3*PTS[3,:]
        yspts = y1*PTS[1,:] + y2*PTS[2,:] + y3*PTS[3,:]
        zspts = z1*PTS[1,:] + z2*PTS[2,:] + z3*PTS[3,:]

        xall[:, it] = xspts
        yall[:, it] = yspts
        zall[:, it] = zspts

    xall_flat = xall.flatten()
    yall_flat = yall.flatten()
    zall_flat = zall.flatten()
    shape = np.array((nDPTs, triangles.shape[1]))

    return xall_flat, yall_flat, zall_flat, shape

@njit(types.Tuple((f8[:], f8[:], f8[:], i8[:]))(f8[:,:], i8[:,:], f8[:,:]), cache=True, nogil=True)
def generate_int_points_tet(nodes: np.ndarray,
                            tets: np.ndarray,
                            PTS: np.ndarray):

    nPTS = PTS.shape[1]
    xall = np.zeros((nPTS, tets.shape[1]))
    yall = np.zeros((nPTS, tets.shape[1]))
    zall = np.zeros((nPTS, tets.shape[1]))

    for it in range(tets.shape[1]):
        
        vertex_ids = tets[:, it]

        x1, x2, x3, x4 = nodes[0, vertex_ids]
        y1, y2, y3, y4 = nodes[1, vertex_ids]
        z1, z2, z3, z4 = nodes[2, vertex_ids]

        xspts = x1*PTS[1,:] + x2*PTS[2,:] + x3*PTS[3,:] + x4*PTS[4,:]
        yspts = y1*PTS[1,:] + y2*PTS[2,:] + y3*PTS[3,:] + y4*PTS[4,:]
        zspts = z1*PTS[1,:] + z2*PTS[2,:] + z3*PTS[3,:] + z4*PTS[4,:]

        xall[:, it] = xspts
        yall[:, it] = yspts
        zall[:, it] = zspts

    xall_flat = xall.flatten()
    yall_flat = yall.flatten()
    zall_flat = zall.flatten()
    shape = np.array((nPTS, tets.shape[1]))

    return xall_flat, yall_flat, zall_flat, shape

@njit(f8(f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Computes a dot product of two 3D vectors

    Args:
        a (np.ndarray): (3,) array
        b (np.ndarray): (3,) array

    Returns:
        float: a · b
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(c16(c16[:], c16[:]), cache=True, fastmath=True, nogil=True)
def dot_c(a: np.ndarray, b: np.ndarray) -> complex:
    """Computes the complex dot product of two 3D vectors

    Args:
        a (np.ndarray): (3,) array
        b (np.ndarray): (3,) array

    Returns:
        complex: a · b
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(f8[:](f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized single vector cross product

    Args:
        a (np.ndarray): (3,) vector a
        b (np.ndarray): (3,) vector b

    Returns:
        np.ndarray: a ⨉ b
    """
    crossv = np.empty((3,), dtype=np.float64)
    crossv[0] = a[1]*b[2] - a[2]*b[1]
    crossv[1] = a[2]*b[0] - a[0]*b[2]
    crossv[2] = a[0]*b[1] - a[1]*b[0]
    return crossv

@njit(c16[:](c16[:], c16[:]), cache=True, fastmath=True, nogil=True)
def cross_c(a: np.ndarray, b: np.ndarray):
    """Optimized complex single vector cross product

    Args:
        a (np.ndarray): (3,) vector a
        b (np.ndarray): (3,) vector b

    Returns:
        np.ndarray: a ⨉ b
    """
    crossv = np.empty((3,), dtype=np.complex128)
    crossv[0] = a[1]*b[2] - a[2]*b[1]
    crossv[1] = a[2]*b[0] - a[0]*b[2]
    crossv[2] = a[0]*b[1] - a[1]*b[0]
    return crossv

@njit(f8[:](f8[:], f8[:], f8[:], f8[:]), cache=True, nogil=True)
def outward_normal(n1: np.ndarray, n2: np.ndarray, n3: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Copmutes an outward normal vector of a triangle spanned by 3 points.

    Computes the triangle surface (n1, n2, n3) which have normal n. 
    The normal is aligned with respect to an origin.

    Args:
        n1 (np.ndarray): Node 1 (3,) array
        n2 (np.ndarray): Node 2 (3,) array
        n3 (np.ndarray): Node 3 (3,) array
        o (np.ndarray): The alignment origin

    Returns:
        Node 1 (3,) array: The outward pointing normal
    """
    e1 = n2-n1
    e2 = n3-n1
    n = cross(e1, e2)
    n = n/np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    sgn = 1
    if dot(n,(n1+n2+n3)/3.0 - o) < 0:
        sgn = -1
    return n*sgn
    
@njit(f8(f8[:], f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def calc_area(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> float:
    """Computes the tirangle surface area of the traingle spun by three nodes.

    Args:
        x1 (np.ndarray): Node 1 (3,) array
        x2 (np.ndarray): Node 2 (3,) array
        x3 (np.ndarray): Node 3 (3,) array

    Returns:
        float: The surface area
    """
    e1 = x2 - x1
    e2 = x3 - x1
    av = cross(e1, e2)
    return np.sqrt(av[0]**2 + av[1]**2 + av[2]**2)/2

@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex 0.1005964238ers of one tetrahedron, in *its* order
        (v0, v1, v2, v3).

    triangle_ids : 2-D int64 array (nTri × 3)
        Each row is a global-ID triple of one face that belongs to this tet.

    Returns
    -------
    local_tris   : 2-D int64 array (nTri × 3)
        Same triangles, but every entry replaced by the local index
        0,1,2,3 that the vertex has inside this tetrahedron.
        (Guaranteed to be ∈{0,1,2,3}; no -1 ever appears if the input
        really belongs to the tet.)
    """
    ndim = triangle_ids.shape[0]
    ntri = triangle_ids.shape[1]
    out  = np.zeros(triangle_ids.shape, dtype=np.int64)

    for t in range(ntri):                 # each triangle
        for j in range(ndim):                # each vertex in that triangle
            gid = triangle_ids[j, t]      # global ID to look up

            # linear search over the four tet vertices
            for k in range(4):
                if vertex_ids[k] == gid:
                    out[j, t] = k         # store local index 0-3
                    break                 # stop the k-loop

    return out


@njit(f8[:,:](f8[:], f8[:], f8[:]), cache=True, nogil=True, fastmath=True)
def compute_distances(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    N = xs.shape[0]
    Ds = np.empty((N,N), dtype=np.float64)
    for i in range(N):
        for j in range(i,N):
            Ds[i,j] = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2) 
            Ds[j,i] = Ds[i,j]  
    return Ds

ids = np.array([[0, 0, 1, 1, 2, 2],[1,2,0, 2, 0, 1]], dtype=np.int64)

@njit(c16[:,:](c16[:,:]), cache=True, nogil=True)
def matinv(M: np.ndarray) -> np.ndarray:
    """Optimized matrix inverse of 3x3 matrix

    Args:
        M (np.ndarray): Input matrix M of shape (3,3)

    Returns:
        np.ndarray: The matrix inverse inv(M)
    """
    out = np.zeros((3,3), dtype=np.complex128)
   
    if M[0,1]==0 and M[0,2]==0 and M[1,0]==0 and M[1,2]==0 and M[2,0]==0 and M[2,1]==0:
        out[0,0] = 1/M[0,0]
        out[1,1] = 1/M[1,1]
        out[2,2] = 1/M[2,2]
    else:
        det = M[0,0]*M[1,1]*M[2,2] - M[0,0]*M[1,2]*M[2,1] - M[0,1]*M[1,0]*M[2,2] + M[0,1]*M[1,2]*M[2,0] + M[0,2]*M[1,0]*M[2,1] - M[0,2]*M[1,1]*M[2,0]
        out[0,0] = M[1,1]*M[2,2] - M[1,2]*M[2,1]
        out[0,1] = -M[0,1]*M[2,2] + M[0,2]*M[2,1]
        out[0,2] = M[0,1]*M[1,2] - M[0,2]*M[1,1]
        out[1,0] = -M[1,0]*M[2,2] + M[1,2]*M[2,0]
        out[1,1] = M[0,0]*M[2,2] - M[0,2]*M[2,0]
        out[1,2] = -M[0,0]*M[1,2] + M[0,2]*M[1,0]
        out[2,0] = M[1,0]*M[2,1] - M[1,1]*M[2,0]
        out[2,1] = -M[0,0]*M[2,1] + M[0,1]*M[2,0]
        out[2,2] = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        out = out*det
    return out

@njit(f8[:,:](f8[:,:]), cache=True, nogil=True)
def matinv_r(M: np.ndarray) -> np.ndarray:
    """Optimized matrix inverse of 3x3 matrix

    Args:
        M (np.ndarray): Input matrix M of shape (3,3)

    Returns:
        np.ndarray: The matrix inverse inv(M)
    """
    out = np.zeros((3,3), dtype=np.float64)
   
    if M[0,1]==0 and M[0,2]==0 and M[1,0]==0 and M[1,2]==0 and M[2,0]==0 and M[2,1]==0:
        out[0,0] = 1/M[0,0]
        out[1,1] = 1/M[1,1]
        out[2,2] = 1/M[2,2]
    else:
        det = M[0,0]*M[1,1]*M[2,2] - M[0,0]*M[1,2]*M[2,1] - M[0,1]*M[1,0]*M[2,2] + M[0,1]*M[1,2]*M[2,0] + M[0,2]*M[1,0]*M[2,1] - M[0,2]*M[1,1]*M[2,0]
        out[0,0] = M[1,1]*M[2,2] - M[1,2]*M[2,1]
        out[0,1] = -M[0,1]*M[2,2] + M[0,2]*M[2,1]
        out[0,2] = M[0,1]*M[1,2] - M[0,2]*M[1,1]
        out[1,0] = -M[1,0]*M[2,2] + M[1,2]*M[2,0]
        out[1,1] = M[0,0]*M[2,2] - M[0,2]*M[2,0]
        out[1,2] = -M[0,0]*M[1,2] + M[0,2]*M[1,0]
        out[2,0] = M[1,0]*M[2,1] - M[1,1]*M[2,0]
        out[2,1] = -M[0,0]*M[2,1] + M[0,1]*M[2,0]
        out[2,2] = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        out = out*det
    return out

@njit(cache=True, nogil=True)
def matmul(M: np.ndarray, vecs: np.ndarray):
    """Executes a basis transformation of vectors (3,N) with a basis matrix M

    Args:
        M (np.ndarray): A (3,3) basis matrix
        vec (np.ndarray): A (3,N) set of coordinates

    Returns:
        np.ndarray: The transformed (3,N) set of vectors
    """
    out = np.empty((3,vecs.shape[1]), dtype=vecs.dtype)
    out[0,:] = M[0,0]*vecs[0,:] + M[0,1]*vecs[1,:] + M[0,2]*vecs[2,:]
    out[1,:] = M[1,0]*vecs[0,:] + M[1,1]*vecs[1,:] + M[1,2]*vecs[2,:]
    out[2,:] = M[2,0]*vecs[0,:] + M[2,1]*vecs[1,:] + M[2,2]*vecs[2,:]
    return out