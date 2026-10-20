//! Integer index mapping functions between mesh entities.
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Replaces every global vertex id in triangle_ids (ndim, ntri) by its local index 0-3 within vertex_ids.
pub fn local_mapping(vertex_ids: &[i64], triangle_ids: &Array2<i64>) -> Array2<i64> {
    let (ndim, ntri) = triangle_ids.dim();
    let mut out = Array2::<i64>::zeros((ndim, ntri));
    for t in 0..ntri {
        for j in 0..ndim {
            let gid = triangle_ids[[j, t]];
            for k in 0..4.min(vertex_ids.len()) {
                if vertex_ids[k] == gid {
                    out[[j, t]] = k as i64;
                    break;
                }
            }
        }
    }
    out
}

/// Gathers the columns `ids` of a (k, N) index array.
pub fn gather_cols(a: &ArrayView2<i64>, ids: &[i64]) -> Array2<i64> {
    let mut out = Array2::<i64>::zeros((a.shape()[0], ids.len()));
    for (c, &id) in ids.iter().enumerate() {
        for r in 0..a.shape()[0] {
            out[[r, c]] = a[[r, id as usize]];
        }
    }
    out
}

/// Returns column `i` of an index array as a Vec.
pub fn col(a: &ArrayView2<i64>, i: usize) -> Vec<i64> {
    a.column(i).to_vec()
}

/// Local vertex indices of the four triangles of tet itet, in tet_to_tri order.
pub fn local_tet_to_triid(tet_to_tri: &ArrayView2<i64>, tets: &ArrayView2<i64>, tris: &ArrayView2<i64>, itet: usize) -> Array2<i64> {
    let global_tri_map = gather_cols(tris, &col(tet_to_tri, itet));
    local_mapping(&col(tets, itet), &global_tri_map)
}

/// Local vertex indices of the six edges of tet itet, in tet_to_edge order.
pub fn local_tet_to_edgeid(tet_to_edge: &ArrayView2<i64>, tets: &ArrayView2<i64>, edges: &ArrayView2<i64>, itet: usize) -> Array2<i64> {
    let global_edge_map = gather_cols(edges, &col(tet_to_edge, itet));
    local_mapping(&col(tets, itet), &global_edge_map)
}

/// Local vertex indices of the three edges of triangle itri, in tri_to_edge order.
pub fn local_tri_to_edgeid(tri_to_edge: &ArrayView2<i64>, tris: &ArrayView2<i64>, edges: &ArrayView2<i64>, itri: usize) -> Array2<i64> {
    let global_edge_map = gather_cols(edges, &col(tri_to_edge, itri));
    local_mapping(&col(tris, itri), &global_edge_map)
}

/// Stable argsort of an integer array.
pub fn argsort(a: &ArrayView1<i64>) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by_key(|&i| a[i]);
    idx
}

/// numpy.searchsorted with side='left' on a sorted slice.
pub fn searchsorted_left(sorted: &[i64], v: i64) -> usize {
    sorted.partition_point(|&x| x < v)
}

/// numpy.searchsorted with side='right' on a sorted slice.
pub fn searchsorted_right(sorted: &[i64], v: i64) -> usize {
    sorted.partition_point(|&x| x <= v)
}
