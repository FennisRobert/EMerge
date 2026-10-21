//! Field interpolation of Nedelec-2 and Lagrange-2 solutions on tetrahedral and triangular meshes.
use super::ccbf::{_eval_curl_f_2d, _eval_curl_f_3d, _eval_f_2d, _eval_f_3d, parse_dofcode};
use super::legrange::{_ne_grad_tri, _ne_tri, _nv_grad_tri, _nv_tri};
use crate::mth::float::{inv2, inv3, tet_coefficients, tri_coefficients};
use crate::mth::index::{argsort, col, gather_cols, local_mapping, searchsorted_left, searchsorted_right};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const EPS: f64 = 1e-9;
const SCALE_LENGTH: bool = false;
const NAN_C: c64 = c64::new(f64::NAN, 0.0);

type C3Arr = (Array1<c64>, Array1<c64>, Array1<c64>);
type F3Arr = (Array1<f64>, Array1<f64>, Array1<f64>);

/// Resets all lengths to one unless length scaling is enabled.
pub fn incl_length(lengths: &mut ArrayViewMut2<f64>) {
    if !SCALE_LENGTH {
        lengths.fill(1.0);
    }
}

/// Unique values of a sorted array with their first and one-past-last positions.
pub fn get_group_indices(assigned_sorted: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    if assigned_sorted.is_empty() {
        return (vec![], vec![], vec![]);
    }
    let mut unique_tets = vec![assigned_sorted[0]];
    let mut first_indices = vec![0i64];
    let mut last_indices = vec![];
    for i in 1..assigned_sorted.len() {
        if assigned_sorted[i] != assigned_sorted[i - 1] {
            last_indices.push(i as i64);
            unique_tets.push(assigned_sorted[i]);
            first_indices.push(i as i64);
        }
    }
    last_indices.push(assigned_sorted.len() as i64);
    (unique_tets, first_indices, last_indices)
}

/// Sorts query points by their assigned tet and returns (sort_idx, start, end) offsets per tet id.
fn group_points(tet_mapping: &ArrayView1<i64>, tetids: &ArrayView1<i64>) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let sort_idx = argsort(tet_mapping);
    let assigned_sorted: Vec<i64> = sort_idx.iter().map(|&i| tet_mapping[i]).collect();
    let offsets = tetids.iter().map(|&t| searchsorted_left(&assigned_sorted, t)).collect();
    let offsets_end = tetids.iter().map(|&t| searchsorted_right(&assigned_sorted, t)).collect();
    (sort_idx, offsets, offsets_end)
}

/// Tet vertex coordinates and normalised barycentric coefficient matrix (4, 4).
fn tet_coeff(nodes: &ArrayView2<f64>, tets: &ArrayView2<i64>, itet: usize, normalise: bool) -> (Array2<f64>, f64) {
    let ids = col(tets, itet);
    let xs = [0, 1, 2, 3].map(|v| nodes[[0, ids[v] as usize]]);
    let ys = [0, 1, 2, 3].map(|v| nodes[[1, ids[v] as usize]]);
    let zs = [0, 1, 2, 3].map(|v| nodes[[2, ids[v] as usize]]);
    let (a, b, c, d, v) = tet_coefficients(&xs, &ys, &zs);
    let f = if normalise { 6.0 * v } else { 1.0 };
    let coeff = Array2::from_shape_fn((4, 4), |(r, m)| [a, b, c, d][r][m] / f);
    (coeff, v)
}

/// Local (i, j, k) vertex indices of DoF idof from its type/index and the local edge/face maps.
fn dof_ijk(i_type: i64, i_index: i64, l_edge_ids: &Array2<i64>, l_tri_ids: &Array2<i64>) -> (usize, usize, usize) {
    let ii = i_index as usize;
    if i_type == 0 {
        (l_edge_ids[[0, ii]] as usize, l_edge_ids[[1, ii]] as usize, 0)
    } else {
        (
            l_tri_ids[[0, ii]] as usize,
            l_tri_ids[[1, ii]] as usize,
            l_tri_ids[[2, ii]] as usize,
        )
    }
}

/// Shared Nedelec-2 tet interpolation of either the field or its (scaled) curl.
#[allow(clippy::too_many_arguments)]
fn ned2_tet_interp_generic(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<c64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tet_to_field: &ArrayView2<i64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    c: Option<&ArrayView1<c64>>,
    tetids: &ArrayView1<i64>,
    tet_mapping: &ArrayView1<i64>,
    dofcodes: &ArrayView1<i64>,
) -> C3Arr {
    let n_nodes = coords.shape()[1];
    let (sort_idx, offsets, offsets_end) = group_points(tet_mapping, tetids);
    let codes = dofcodes.to_vec();
    let (typearry, indexarry) = parse_dofcode(&codes);
    let nidof = codes.len();

    let results: Vec<Vec<(usize, [c64; 3])>> = (0..tetids.len())
        .into_par_iter()
        .map(|i_iter| {
            let itet = tetids[i_iter] as usize;
            let (start, end) = (offsets[i_iter], offsets_end[i_iter]);
            if start == end {
                return vec![];
            }
            let (coeff, _v) = tet_coeff(nodes, tets, itet, true);
            let g_node_ids = col(tets, itet);
            let l_edge_ids = local_mapping(&g_node_ids, &gather_cols(edges, &col(tet_to_edge, itet)));
            let l_tri_ids = local_mapping(&g_node_ids, &gather_cols(tris, &col(tet_to_tri, itet)));
            let etet: Vec<c64> = (0..nidof).map(|f| solutions[tet_to_field[[f, itet]] as usize]).collect();
            let npts = end - start;
            let pt_coords = Array2::from_shape_fn((3, npts), |(r, p)| coords[[r, sort_idx[start + p]]]);
            let mut f = Array2::<c64>::zeros((3, npts));
            let mut acc = vec![[c64::new(0.0, 0.0); 3]; npts];
            for idof in 0..nidof {
                let (i1, j1, k1) = dof_ijk(typearry[idof], indexarry[idof], &l_edge_ids, &l_tri_ids);
                match c {
                    None => _eval_f_3d(&coeff.view(), &pt_coords.view(), i1, j1, k1, codes[idof], &mut f.view_mut()),
                    Some(_) => _eval_curl_f_3d(&coeff.view(), &pt_coords.view(), i1, j1, k1, codes[idof], &mut f.view_mut()),
                }
                for (i, a) in acc.iter_mut().enumerate() {
                    for r in 0..3 {
                        a[r] += match c {
                            None => f[[r, i]] * etet[idof],
                            Some(cc) => f[[r, i]] * cc[itet] * etet[idof],
                        };
                    }
                }
            }
            acc.into_iter().enumerate().map(|(i, a)| (sort_idx[start + i], a)).collect()
        })
        .collect();

    let mut ex = Array1::<c64>::zeros(n_nodes);
    let mut ey = Array1::<c64>::zeros(n_nodes);
    let mut ez = Array1::<c64>::zeros(n_nodes);
    let mut setnan = vec![false; n_nodes];
    for tet_res in results {
        for (idx, a) in tet_res {
            ex[idx] += a[0];
            ey[idx] += a[1];
            ez[idx] += a[2];
            setnan[idx] = true;
        }
    }
    for i in 0..n_nodes {
        if !setnan[i] {
            ex[i] = NAN_C;
            ey[i] = NAN_C;
            ez[i] = NAN_C;
        }
    }
    (ex, ey, ez)
}

/// Nedelec-2 tetrahedral interpolation of the field at coords.
#[allow(clippy::too_many_arguments)]
pub fn ned2_tet_interp(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<c64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tet_to_field: &ArrayView2<i64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    tetids: &ArrayView1<i64>,
    tet_mapping: &ArrayView1<i64>,
    dofcodes: &ArrayView1<i64>,
) -> C3Arr {
    ned2_tet_interp_generic(
        coords,
        solutions,
        tets,
        tris,
        edges,
        nodes,
        tet_to_field,
        tet_to_edge,
        tet_to_tri,
        None,
        tetids,
        tet_mapping,
        dofcodes,
    )
}

/// Nedelec-2 tetrahedral interpolation of the analytic curl scaled by c per tet.
#[allow(clippy::too_many_arguments)]
pub fn ned2_tet_interp_curl(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<c64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tet_to_field: &ArrayView2<i64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    c: &ArrayView1<c64>,
    tetids: &ArrayView1<i64>,
    tet_mapping: &ArrayView1<i64>,
    dofcodes: &ArrayView1<i64>,
) -> C3Arr {
    ned2_tet_interp_generic(
        coords,
        solutions,
        tets,
        tris,
        edges,
        nodes,
        tet_to_field,
        tet_to_edge,
        tet_to_tri,
        Some(c),
        tetids,
        tet_mapping,
        dofcodes,
    )
}

const LEG_EDGES: [[usize; 6]; 2] = [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]];

/// Per tet barycentric data (a, b, c, d, inv6V) used by the Lagrange interpolators.
fn leg_tet_data(nodes: &ArrayView2<f64>, tets: &ArrayView2<i64>, itet: usize) -> ([[f64; 4]; 4], f64) {
    let ids = col(tets, itet);
    let xs = [0, 1, 2, 3].map(|v| nodes[[0, ids[v] as usize]]);
    let ys = [0, 1, 2, 3].map(|v| nodes[[1, ids[v] as usize]]);
    let zs = [0, 1, 2, 3].map(|v| nodes[[2, ids[v] as usize]]);
    let (a, b, c, d, v) = tet_coefficients(&xs, &ys, &zs);
    ([a, b, c, d], 1.0 / (6.0 * v))
}

/// Lagrange-2 tetrahedral scalar interpolation at coords.
#[allow(clippy::too_many_arguments)]
pub fn leg2_tet_interp(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<f64>,
    tets: &ArrayView2<i64>,
    _edges: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tet_to_field: &ArrayView2<i64>,
    tetids: &ArrayView1<i64>,
    tet_mapping: &ArrayView1<i64>,
) -> Array1<f64> {
    let n_nodes = coords.shape()[1];
    let (sort_idx, offsets, offsets_end) = group_points(tet_mapping, tetids);
    let results: Vec<Vec<(usize, f64)>> = (0..tetids.len())
        .into_par_iter()
        .map(|i_iter| {
            let itet = tetids[i_iter] as usize;
            let (start, end) = (offsets[i_iter], offsets_end[i_iter]);
            if start == end {
                return vec![];
            }
            let ([a_s, b_s, c_s, d_s], inv6v) = leg_tet_data(nodes, tets, itet);
            let tloc: Vec<f64> = (0..10).map(|f| solutions[tet_to_field[[f, itet]] as usize]).collect();
            (start..end)
                .map(|i| {
                    let idx = sort_idx[i];
                    let (x, y, z) = (coords[[0, idx]], coords[[1, idx]], coords[[2, idx]]);
                    let ls = [0, 1, 2, 3].map(|m| (a_s[m] + b_s[m] * x + c_s[m] * y + d_s[m] * z) * inv6v);
                    let mut val = tloc[0] * ls[0] * (2.0 * ls[0] - 1.0)
                        + tloc[1] * ls[1] * (2.0 * ls[1] - 1.0)
                        + tloc[2] * ls[2] * (2.0 * ls[2] - 1.0)
                        + tloc[3] * ls[3] * (2.0 * ls[3] - 1.0);
                    for ie in 0..6 {
                        val += tloc[4 + ie] * 4.0 * ls[LEG_EDGES[0][ie]] * ls[LEG_EDGES[1][ie]];
                    }
                    (idx, val)
                })
                .collect()
        })
        .collect();
    let mut result = Array1::<f64>::from_elem(n_nodes, f64::NAN);
    for (idx, val) in results.into_iter().flatten() {
        result[idx] = val;
    }
    result
}

/// Lagrange-2 tetrahedral gradient interpolation (dT/dx, dT/dy, dT/dz) at coords.
#[allow(clippy::too_many_arguments)]
pub fn leg2_tet_interp_grad(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<f64>,
    tets: &ArrayView2<i64>,
    _edges: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tet_to_field: &ArrayView2<i64>,
    tetids: &ArrayView1<i64>,
    tet_mapping: &ArrayView1<i64>,
) -> F3Arr {
    let n_nodes = coords.shape()[1];
    let (sort_idx, offsets, offsets_end) = group_points(tet_mapping, tetids);
    let results: Vec<Vec<(usize, [f64; 3])>> = (0..tetids.len())
        .into_par_iter()
        .map(|i_iter| {
            let itet = tetids[i_iter] as usize;
            let (start, end) = (offsets[i_iter], offsets_end[i_iter]);
            if start == end {
                return vec![];
            }
            let ([a_s, b_s, c_s, d_s], inv6v) = leg_tet_data(nodes, tets, itet);
            let gb = b_s.map(|v| v * inv6v);
            let gc = c_s.map(|v| v * inv6v);
            let gd = d_s.map(|v| v * inv6v);
            let tloc: Vec<f64> = (0..10).map(|f| solutions[tet_to_field[[f, itet]] as usize]).collect();
            (start..end)
                .map(|i| {
                    let idx = sort_idx[i];
                    let (x, y, z) = (coords[[0, idx]], coords[[1, idx]], coords[[2, idx]]);
                    let ls = [0, 1, 2, 3].map(|m| (a_s[m] + b_s[m] * x + c_s[m] * y + d_s[m] * z) * inv6v);
                    let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
                    for iv in 0..4 {
                        let q = (4.0 * ls[iv] - 1.0) * tloc[iv];
                        vx += q * gb[iv];
                        vy += q * gc[iv];
                        vz += q * gd[iv];
                    }
                    for ie in 0..6 {
                        let (ei, ej) = (LEG_EDGES[0][ie], LEG_EDGES[1][ie]);
                        let t_e = tloc[4 + ie];
                        let (li, lj) = (ls[ei], ls[ej]);
                        vx += t_e * 4.0 * (lj * gb[ei] + li * gb[ej]);
                        vy += t_e * 4.0 * (lj * gc[ei] + li * gc[ej]);
                        vz += t_e * 4.0 * (lj * gd[ei] + li * gd[ej]);
                    }
                    (idx, [vx, vy, vz])
                })
                .collect()
        })
        .collect();
    let mut gx = Array1::<f64>::from_elem(n_nodes, f64::NAN);
    let mut gy = Array1::<f64>::from_elem(n_nodes, f64::NAN);
    let mut gz = Array1::<f64>::from_elem(n_nodes, f64::NAN);
    for (idx, g) in results.into_iter().flatten() {
        gx[idx] = g[0];
        gy[idx] = g[1];
        gz[idx] = g[2];
    }
    (gx, gy, gz)
}

/// Triangle data shared by the 2D interpolators: points inside triangle itri and its coefficient matrix.
fn tri_inside(
    coords: &ArrayView2<f64>,
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    itri: usize,
) -> Option<(Vec<usize>, Array2<f64>, Array2<f64>)> {
    let iv = [0, 1, 2].map(|v| tris[[v, itri]] as usize);
    let v1 = [nodes[[0, iv[0]]], nodes[[1, iv[0]]]];
    let bv1 = [nodes[[0, iv[1]]] - v1[0], nodes[[1, iv[1]]] - v1[1]];
    let bv2 = [nodes[[0, iv[2]]] - v1[0], nodes[[1, iv[2]]] - v1[1]];
    let basis = inv2(&[[bv1[0], bv2[0]], [bv1[1], bv2[1]]]);
    let inside: Vec<usize> = (0..coords.shape()[1])
        .filter(|&p| {
            let (dx, dy) = (coords[[0, p]] - v1[0], coords[[1, p]] - v1[1]);
            let u = basis[0][0] * dx + basis[0][1] * dy;
            let v = basis[1][0] * dx + basis[1][1] * dy;
            (u + v) <= 1.0 + EPS && u >= -EPS && v >= -EPS
        })
        .collect();
    if inside.is_empty() {
        return None;
    }
    let coords_inside = Array2::from_shape_fn((2, inside.len()), |(r, p)| coords[[r, inside[p]]]);
    let xvs = iv.map(|v| nodes[[0, v]]);
    let yvs = iv.map(|v| nodes[[1, v]]);
    let (a_s, b_s, c_s, _a) = tri_coefficients(&xvs, &yvs);
    let coeff = Array2::from_shape_fn((3, 3), |(r, m)| [a_s, b_s, c_s][r][m]);
    Some((inside, coords_inside, coeff))
}

const IVEC: [usize; 3] = [0, 1, 0];
const JVEC: [usize; 3] = [1, 2, 2];
const IVEC_LEG: [usize; 6] = [0, 1, 2, 0, 1, 0];
const JVEC_LEG: [usize; 6] = [0, 0, 0, 1, 2, 2];

/// Local (i, j, k) of a triangle Nedelec DoF.
fn tri_dof_ijk(i_type: i64, i_index: i64) -> (usize, usize, usize) {
    if i_type == 0 {
        (IVEC[i_index as usize], JVEC[i_index as usize], 0)
    } else {
        (0, 1, 2)
    }
}

/// Applies per triangle (inside, values) results in triangle order onto the output arrays.
fn apply_tri_results(results: Vec<Option<(Vec<usize>, Vec<[c64; 3]>)>>, n_nodes: usize) -> C3Arr {
    let mut ex = Array1::<c64>::from_elem(n_nodes, NAN_C);
    let mut ey = Array1::<c64>::from_elem(n_nodes, NAN_C);
    let mut ez = Array1::<c64>::from_elem(n_nodes, NAN_C);
    for (inside, vals) in results.into_iter().flatten() {
        for (p, idx) in inside.into_iter().enumerate() {
            ex[idx] = vals[p][0];
            ey[idx] = vals[p][1];
            ez[idx] = vals[p][2];
        }
    }
    (ex, ey, ez)
}

/// Nedelec-2 triangle interpolation of the transverse field and the Lagrange longitudinal component.
pub fn ned2_tri_interp_full(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<c64>,
    tris: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tri_to_field: &ArrayView2<i64>,
    dofcodes: &ArrayView1<i64>,
) -> C3Arr {
    let nodes = nodes.slice(s![..2, ..]);
    let codes = dofcodes.to_vec();
    let (typearry, indexarry) = parse_dofcode(&codes);
    let nidof = codes.len();
    let results: Vec<_> = (0..tris.shape()[1])
        .into_par_iter()
        .map(|itri| {
            let (inside, ci, coeff) = tri_inside(coords, &nodes, tris, itri)?;
            let etri: Vec<c64> = (0..tri_to_field.shape()[0])
                .map(|f| solutions[tri_to_field[[f, itri]] as usize])
                .collect();
            let n = inside.len();
            let (mut ex, mut ey, mut ez) = (
                vec![c64::new(0.0, 0.0); n],
                vec![c64::new(0.0, 0.0); n],
                vec![c64::new(0.0, 0.0); n],
            );
            let mut evec = Array2::<c64>::zeros((2, n));
            for idof in 0..(nidof + 6) {
                if idof < nidof {
                    let (i, j, k) = tri_dof_ijk(typearry[idof], indexarry[idof]);
                    _eval_f_2d(&coeff.view(), &ci.view(), i, j, k, codes[idof], &mut evec.view_mut());
                    for p in 0..n {
                        ex[p] += etri[idof] * evec[[0, p]];
                        ey[p] += etri[idof] * evec[[1, p]];
                    }
                } else {
                    let ileg = idof - nidof;
                    let (i, j) = (IVEC_LEG[ileg], JVEC_LEG[ileg]);
                    let ezc = if ileg < 3 {
                        _nv_tri(&coeff.view(), &ci.view(), i, j, 0)
                    } else {
                        _ne_tri(&coeff.view(), &ci.view(), i, j, 0)
                    };
                    for p in 0..n {
                        ez[p] += etri[idof] * ezc[p];
                    }
                }
            }
            Some((inside, (0..n).map(|p| [ex[p], ey[p], ez[p]]).collect()))
        })
        .collect();
    apply_tri_results(results, coords.shape()[1])
}

/// Gradient of the Lagrange longitudinal component of a Nedelec-Lagrange triangle solution.
pub fn ned2_tri_interp_grad(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<c64>,
    tris: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tri_to_field: &ArrayView2<i64>,
    dofcodes: &ArrayView1<i64>,
) -> (Array1<c64>, Array1<c64>) {
    let nodes = nodes.slice(s![..2, ..]);
    let nidof = dofcodes.len();
    let results: Vec<_> = (0..tris.shape()[1])
        .into_par_iter()
        .map(|itri| {
            let (inside, ci, coeff) = tri_inside(coords, &nodes, tris, itri)?;
            let etri: Vec<c64> = (0..tri_to_field.shape()[0])
                .map(|f| solutions[tri_to_field[[f, itri]] as usize])
                .collect();
            let n = inside.len();
            let (mut ex, mut ey) = (vec![c64::new(0.0, 0.0); n], vec![c64::new(0.0, 0.0); n]);
            for idof in 0..6 {
                let (i, j) = (IVEC_LEG[idof], JVEC_LEG[idof]);
                let evec = if idof < 3 {
                    _nv_grad_tri(&coeff.view(), &ci.view(), i, j, 0)
                } else {
                    _ne_grad_tri(&coeff.view(), &ci.view(), i, j, 0)
                };
                for p in 0..n {
                    ex[p] += etri[idof + nidof] * evec[[0, p]];
                    ey[p] += etri[idof + nidof] * evec[[1, p]];
                }
            }
            Some((inside, (0..n).map(|p| [ex[p], ey[p], NAN_C]).collect()))
        })
        .collect();
    let (ex, ey, _) = apply_tri_results(results, coords.shape()[1]);
    (ex, ey)
}

/// Curl of a Nedelec-Lagrange triangle mode with propagation constant beta, multiplied by the local-frame tensor diadic per triangle.
#[allow(clippy::too_many_arguments)]
pub fn ned2_tri_interp_curl(
    coords: &ArrayView2<f64>,
    solutions: &ArrayView1<c64>,
    tris: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tri_to_field: &ArrayView2<i64>,
    diadic: &ArrayView3<c64>,
    beta: c64,
    dofcodes: &ArrayView1<i64>,
) -> C3Arr {
    let nodes = nodes.slice(s![..2, ..]);
    let jb = -c64::new(0.0, 1.0) * beta;
    let codes = dofcodes.to_vec();
    let (typearry, indexarry) = parse_dofcode(&codes);
    let nidof = codes.len();
    let results: Vec<_> = (0..tris.shape()[1])
        .into_par_iter()
        .map(|itri| {
            let (inside, ci, coeff) = tri_inside(coords, &nodes, tris, itri)?;
            let etri: Vec<c64> = (0..tri_to_field.shape()[0])
                .map(|f| solutions[tri_to_field[[f, itri]] as usize])
                .collect();
            let n = inside.len();
            let zero = c64::new(0.0, 0.0);
            let (mut ex, mut ey, mut ez) = (vec![zero; n], vec![zero; n], vec![zero; n]);
            let mut evec = Array2::<c64>::zeros((2, n));
            let mut evecz = Array1::<c64>::zeros(n);
            for idof in 0..(nidof + 6) {
                if idof < nidof {
                    let (i, j, k) = tri_dof_ijk(typearry[idof], indexarry[idof]);
                    _eval_f_2d(&coeff.view(), &ci.view(), i, j, k, codes[idof], &mut evec.view_mut());
                    _eval_curl_f_2d(&coeff.view(), &ci.view(), i, j, k, codes[idof], &mut evecz.view_mut());
                    for p in 0..n {
                        ex[p] += -jb * etri[idof] * evec[[1, p]];
                        ey[p] += jb * etri[idof] * evec[[0, p]];
                        ez[p] += etri[idof] * evecz[p];
                    }
                } else {
                    let ileg = idof - nidof;
                    let (i, j) = (IVEC_LEG[ileg], JVEC_LEG[ileg]);
                    let g = if ileg < 3 {
                        _nv_grad_tri(&coeff.view(), &ci.view(), i, j, 0)
                    } else {
                        _ne_grad_tri(&coeff.view(), &ci.view(), i, j, 0)
                    };
                    for p in 0..n {
                        ex[p] += etri[idof] * g[[1, p]];
                        ey[p] += -etri[idof] * g[[0, p]];
                    }
                }
            }
            let d = |r: usize, c: usize| diadic[[r, c, itri]];
            Some((
                inside,
                (0..n)
                    .map(|p| {
                        let v = [ex[p], ey[p], ez[p]];
                        [0, 1, 2].map(|r| d(r, 0) * v[0] + d(r, 1) * v[1] + d(r, 2) * v[2])
                    })
                    .collect(),
            ))
        })
        .collect();
    apply_tri_results(results, coords.shape()[1])
}

/// Assigns value[itet] to every coordinate inside each tet of tetids (zero elsewhere).
pub fn constant_interp(
    coords: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tetids: &ArrayView1<i64>,
    value: &ArrayView1<f64>,
) -> Array1<f64> {
    let n_nodes = coords.shape()[1];
    let mut prop = Array1::<f64>::zeros(n_nodes);
    for &itet in tetids.iter() {
        let itet = itet as usize;
        let iv = [0, 1, 2, 3].map(|v| tets[[v, itet]] as usize);
        let v1 = [nodes[[0, iv[0]]], nodes[[1, iv[0]]], nodes[[2, iv[0]]]];
        let mut b = [[0.0; 3]; 3];
        for (c, &ivc) in iv[1..].iter().enumerate() {
            for r in 0..3 {
                b[r][c] = nodes[[r, ivc]] - v1[r];
            }
        }
        let basis = inv3(&b);
        for p in 0..n_nodes {
            let d = [coords[[0, p]] - v1[0], coords[[1, p]] - v1[1], coords[[2, p]] - v1[2]];
            let l = [0, 1, 2].map(|r| basis[r][0] * d[0] + basis[r][1] * d[1] + basis[r][2] * d[2]);
            if (l[0] + l[1] + l[2]) <= 1.00000001 && l[0] >= -1e-6 && l[1] >= -1e-6 && l[2] >= -1e-6 {
                prop[p] = value[itet];
            }
        }
    }
    prop
}

type PyC3<'py> = (Bound<'py, PyArray1<c64>>, Bound<'py, PyArray1<c64>>, Bound<'py, PyArray1<c64>>);

fn c3_py(py: Python<'_>, r: C3Arr) -> PyC3<'_> {
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py), r.2.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "ned2_tet_interp")]
#[allow(clippy::too_many_arguments)]
fn py_ned2_tet_interp<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    solutions: PyReadonlyArray1<c64>,
    tets: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tet_to_field: PyReadonlyArray2<i64>,
    tet_to_edge: PyReadonlyArray2<i64>,
    tet_to_tri: PyReadonlyArray2<i64>,
    tetids: PyReadonlyArray1<i64>,
    tet_mapping: PyReadonlyArray1<i64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> PyC3<'py> {
    let (co, so, te, tr, ed, no) = (
        coords.as_array(),
        solutions.as_array(),
        tets.as_array(),
        tris.as_array(),
        edges.as_array(),
        nodes.as_array(),
    );
    let (tf, tte, ttt, ti, tm, dc) = (
        tet_to_field.as_array(),
        tet_to_edge.as_array(),
        tet_to_tri.as_array(),
        tetids.as_array(),
        tet_mapping.as_array(),
        dofcodes.as_array(),
    );
    let r = py.allow_threads(|| ned2_tet_interp(&co, &so, &te, &tr, &ed, &no, &tf, &tte, &ttt, &ti, &tm, &dc));
    c3_py(py, r)
}

#[pyfunction]
#[pyo3(name = "ned2_tet_interp_curl")]
#[allow(clippy::too_many_arguments)]
fn py_ned2_tet_interp_curl<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    solutions: PyReadonlyArray1<c64>,
    tets: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tet_to_field: PyReadonlyArray2<i64>,
    tet_to_edge: PyReadonlyArray2<i64>,
    tet_to_tri: PyReadonlyArray2<i64>,
    c: PyReadonlyArray1<c64>,
    tetids: PyReadonlyArray1<i64>,
    tet_mapping: PyReadonlyArray1<i64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> PyC3<'py> {
    let (co, so, te, tr, ed, no) = (
        coords.as_array(),
        solutions.as_array(),
        tets.as_array(),
        tris.as_array(),
        edges.as_array(),
        nodes.as_array(),
    );
    let (tf, tte, ttt, ti, tm, dc) = (
        tet_to_field.as_array(),
        tet_to_edge.as_array(),
        tet_to_tri.as_array(),
        tetids.as_array(),
        tet_mapping.as_array(),
        dofcodes.as_array(),
    );
    let cc = c.as_array();
    let r = py.allow_threads(|| ned2_tet_interp_curl(&co, &so, &te, &tr, &ed, &no, &tf, &tte, &ttt, &cc, &ti, &tm, &dc));
    c3_py(py, r)
}

#[pyfunction]
#[pyo3(name = "leg2_tet_interp")]
#[allow(clippy::too_many_arguments)]
fn py_leg2_tet_interp<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    solutions: PyReadonlyArray1<f64>,
    tets: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tet_to_field: PyReadonlyArray2<i64>,
    tetids: PyReadonlyArray1<i64>,
    tet_mapping: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray1<f64>> {
    let (co, so, te, ed, no) = (
        coords.as_array(),
        solutions.as_array(),
        tets.as_array(),
        edges.as_array(),
        nodes.as_array(),
    );
    let (tf, ti, tm) = (tet_to_field.as_array(), tetids.as_array(), tet_mapping.as_array());
    py.allow_threads(|| leg2_tet_interp(&co, &so, &te, &ed, &no, &tf, &ti, &tm))
        .into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(name = "leg2_tet_interp_grad")]
#[allow(clippy::too_many_arguments)]
fn py_leg2_tet_interp_grad<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    solutions: PyReadonlyArray1<f64>,
    tets: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tet_to_field: PyReadonlyArray2<i64>,
    tetids: PyReadonlyArray1<i64>,
    tet_mapping: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (co, so, te, ed, no) = (
        coords.as_array(),
        solutions.as_array(),
        tets.as_array(),
        edges.as_array(),
        nodes.as_array(),
    );
    let (tf, ti, tm) = (tet_to_field.as_array(), tetids.as_array(), tet_mapping.as_array());
    let r = py.allow_threads(|| leg2_tet_interp_grad(&co, &so, &te, &ed, &no, &tf, &ti, &tm));
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py), r.2.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "ned2_tri_interp_full")]
fn py_ned2_tri_interp_full<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    solutions: PyReadonlyArray1<c64>,
    tris: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tri_to_field: PyReadonlyArray2<i64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> PyC3<'py> {
    let (co, so, tr, no, tf, dc) = (
        coords.as_array(),
        solutions.as_array(),
        tris.as_array(),
        nodes.as_array(),
        tri_to_field.as_array(),
        dofcodes.as_array(),
    );
    let r = py.allow_threads(|| ned2_tri_interp_full(&co, &so, &tr, &no, &tf, &dc));
    c3_py(py, r)
}

#[pyfunction]
#[pyo3(name = "ned2_tri_interp_grad")]
fn py_ned2_tri_interp_grad<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    solutions: PyReadonlyArray1<c64>,
    tris: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tri_to_field: PyReadonlyArray2<i64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<c64>>, Bound<'py, PyArray1<c64>>) {
    let (co, so, tr, no, tf, dc) = (
        coords.as_array(),
        solutions.as_array(),
        tris.as_array(),
        nodes.as_array(),
        tri_to_field.as_array(),
        dofcodes.as_array(),
    );
    let r = py.allow_threads(|| ned2_tri_interp_grad(&co, &so, &tr, &no, &tf, &dc));
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "ned2_tri_interp_curl")]
#[allow(clippy::too_many_arguments)]
fn py_ned2_tri_interp_curl<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    solutions: PyReadonlyArray1<c64>,
    tris: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tri_to_field: PyReadonlyArray2<i64>,
    diadic: PyReadonlyArray3<c64>,
    beta: c64,
    dofcodes: PyReadonlyArray1<i64>,
) -> PyC3<'py> {
    let (co, so, tr, no, tf, dc) = (
        coords.as_array(),
        solutions.as_array(),
        tris.as_array(),
        nodes.as_array(),
        tri_to_field.as_array(),
        dofcodes.as_array(),
    );
    let di = diadic.as_array();
    let r = py.allow_threads(|| ned2_tri_interp_curl(&co, &so, &tr, &no, &tf, &di, beta, &dc));
    c3_py(py, r)
}

#[pyfunction]
#[pyo3(name = "constant_interp")]
fn py_constant_interp<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tetids: PyReadonlyArray1<i64>,
    value: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let (co, te, no, ti, va) = (
        coords.as_array(),
        tets.as_array(),
        nodes.as_array(),
        tetids.as_array(),
        value.as_array(),
    );
    py.allow_threads(|| constant_interp(&co, &te, &no, &ti, &va)).into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(name = "incl_length")]
fn py_incl_length(mut lengths: PyReadwriteArray2<f64>) {
    incl_length(&mut lengths.as_array_mut());
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_ned2_tet_interp, m)?)?;
    m.add_function(wrap_pyfunction!(py_ned2_tet_interp_curl, m)?)?;
    m.add_function(wrap_pyfunction!(py_leg2_tet_interp, m)?)?;
    m.add_function(wrap_pyfunction!(py_leg2_tet_interp_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_ned2_tri_interp_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_ned2_tri_interp_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_ned2_tri_interp_curl, m)?)?;
    m.add_function(wrap_pyfunction!(py_constant_interp, m)?)?;
    m.add_function(wrap_pyfunction!(py_incl_length, m)?)?;
    Ok(())
}
