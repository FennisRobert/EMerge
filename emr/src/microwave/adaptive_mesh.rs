//! A-posteriori error estimation of Nedelec-2 solutions and mesh size field helpers for adaptive refinement.
use crate::compiled::ccbf::{_eval_curl_f_3d, _eval_curlcurl_f_3d, _eval_div_f_3d, _eval_f_3d, parse_dofcode};
use crate::constants::{C0, EPS0, MU0, PI};
use crate::mth::complex::{cross_c_arry, dot_c_arry, C3, CM3};
use crate::mth::float::{calc_inward_normal, col3, compute_volume, tet_coefficients};
use crate::mth::index::{col, gather_cols, local_mapping};
use crate::mth::quadrature::{keast, strang};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const DEPS: f64 = 1e-20;
const MISSING_ID: i64 = -1234;

const N2D: usize = 6;
const N3D: usize = 11;

/// Local (i, j, k) vertex indices of a tet DoF.
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

/// Barycentric coefficient matrix (4, 4) of a tet given its (3, 4) vertices, divided by `div`.
fn coeff_matrix(vertices: &ArrayView2<f64>, normalise: bool) -> Array2<f64> {
    let xs = [0, 1, 2, 3].map(|v| vertices[[0, v]]);
    let ys = [0, 1, 2, 3].map(|v| vertices[[1, v]]);
    let zs = [0, 1, 2, 3].map(|v| vertices[[2, v]]);
    let (a, b, c, d, v) = tet_coefficients(&xs, &ys, &zs);
    let rows = [a, b, c, d];
    let div = if normalise { 6.0 * v } else { 1.0 };
    Array2::from_shape_fn((4, 4), |(r, m)| rows[r][m] / div)
}

/// Nedelec field (3, N) of the tet DoF vector Etet at coords (unnormalised barycentric coefficients).
pub fn compute_field(
    coords: &ArrayView2<f64>,
    vertices: &ArrayView2<f64>,
    etet: &[c64],
    l_edge_ids: &Array2<i64>,
    l_tri_ids: &Array2<i64>,
    dofcodes: &[i64],
) -> Array2<c64> {
    compute_field_or_curl(coords, vertices, etet, l_edge_ids, l_tri_ids, dofcodes, false)
}

/// Nedelec curl (3, N) of the tet DoF vector Etet at coords (unnormalised barycentric coefficients).
pub fn compute_curl(
    coords: &ArrayView2<f64>,
    vertices: &ArrayView2<f64>,
    etet: &[c64],
    l_edge_ids: &Array2<i64>,
    l_tri_ids: &Array2<i64>,
    dofcodes: &[i64],
) -> Array2<c64> {
    compute_field_or_curl(coords, vertices, etet, l_edge_ids, l_tri_ids, dofcodes, true)
}

fn compute_field_or_curl(
    coords: &ArrayView2<f64>,
    vertices: &ArrayView2<f64>,
    etet: &[c64],
    l_edge_ids: &Array2<i64>,
    l_tri_ids: &Array2<i64>,
    dofcodes: &[i64],
    curl: bool,
) -> Array2<c64> {
    let n = coords.shape()[1];
    let coeff = coeff_matrix(vertices, false);
    let (typearray, indexarray) = parse_dofcode(dofcodes);
    let mut out = Array2::<c64>::zeros((3, n));
    let mut f = Array2::<c64>::zeros((3, n));
    for idof in 0..dofcodes.len() {
        let (i1, j1, k1) = dof_ijk(typearray[idof], indexarray[idof], l_edge_ids, l_tri_ids);
        if curl {
            _eval_curl_f_3d(&coeff.view(), coords, i1, j1, k1, dofcodes[idof], &mut f.view_mut());
        } else {
            _eval_f_3d(&coeff.view(), coords, i1, j1, k1, dofcodes[idof], &mut f.view_mut());
        }
        for r in 0..3 {
            for p in 0..n {
                out[[r, p]] += f[[r, p]] * etet[idof];
            }
        }
    }
    out
}

/// Divergence of Um[0, 0] times the Nedelec field at coords (isotropic Um).
pub fn compute_div(
    coords: &ArrayView2<f64>,
    vertices: &ArrayView2<f64>,
    etet: &[c64],
    l_edge_ids: &Array2<i64>,
    l_tri_ids: &Array2<i64>,
    um: &CM3,
    dofcodes: &[i64],
) -> Array1<c64> {
    let n = coords.shape()[1];
    let coeff = coeff_matrix(vertices, true);
    let u_scalar = um[0][0];
    let (typearray, indexarray) = parse_dofcode(dofcodes);
    let mut dif_e = Array1::<c64>::zeros(n);
    let mut fdiv = Array1::<c64>::zeros(n);
    for idof in 0..dofcodes.len() {
        let (i1, j1, k1) = dof_ijk(typearray[idof], indexarray[idof], l_edge_ids, l_tri_ids);
        _eval_div_f_3d(&coeff.view(), coords, i1, j1, k1, dofcodes[idof], &mut fdiv.view_mut());
        for p in 0..n {
            dif_e[p] += etet[idof] * fdiv[p];
        }
    }
    dif_e.map(|v| u_scalar * v)
}

/// Um times the (constant) curl-curl of the Nedelec field in the tet.
pub fn compute_curl_curl(
    vertices: &ArrayView2<f64>,
    etet: &[c64],
    l_edge_ids: &Array2<i64>,
    l_tri_ids: &Array2<i64>,
    um: &CM3,
    dofcodes: &[i64],
) -> C3 {
    let coeff = coeff_matrix(vertices, false);
    let dummy_coords = Array2::<f64>::zeros((3, 1));
    let (typearray, indexarray) = parse_dofcode(dofcodes);
    let mut out = [c64::new(0.0, 0.0); 3];
    let mut f = Array2::<c64>::zeros((3, 1));
    for idof in 0..dofcodes.len() {
        let (i1, j1, k1) = dof_ijk(typearray[idof], indexarray[idof], l_edge_ids, l_tri_ids);
        _eval_curlcurl_f_3d(&coeff.view(), &dummy_coords.view(), i1, j1, k1, dofcodes[idof], &mut f.view_mut());
        let (fx, fy, fz) = (f[[0, 0]], f[[1, 0]], f[[2, 0]]);
        for (r, o) in out.iter_mut().enumerate() {
            *o += etet[idof] * (um[r][0] * fx + um[r][1] * fy + um[r][2] * fz);
        }
    }
    out
}

/// Per tet results of the parallel error estimate pass, including the writes into neighbour face slots.
struct TetResidual {
    alpha_t: c64,
    max_elem_size: f64,
    is_nan: bool,
    rf: [f64; 4],
    area: [f64; 4],
    adj: [i64; 4],
    qf1: [[c64; N2D]; 4],
    jf1: [[[c64; N2D]; 3]; 4],
    neighbour: Vec<(usize, usize, [c64; N2D], [[c64; N2D]; 3])>,
}

/// Computes the residual based error indicator and the maximum edge length of every tet.
#[allow(clippy::too_many_arguments)]
pub fn compute_error_single(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    centers: &ArrayView2<f64>,
    efield: &ArrayView1<c64>,
    edge_lengths: &ArrayView1<f64>,
    areas: &ArrayView1<f64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    _tri_centers: &ArrayView2<f64>,
    tri_to_tet: &ArrayView2<i64>,
    tet_to_field: &ArrayView2<i64>,
    er: &ArrayView1<c64>,
    ur: &ArrayView1<c64>,
    pec_tris: &ArrayView1<i64>,
    k0: f64,
    dofcodes3d: &ArrayView1<i64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let dpts_3d = keast(4);
    let dpts_2d = strang(5);
    let mut tet_is_pec = vec![false; tris.shape()[1]];
    for &t in pec_tris.iter() {
        tet_is_pec[t as usize] = true;
    }
    let n_tets = tets.shape()[1];
    let codes = dofcodes3d.to_vec();
    let w0 = k0 * C0;
    let y0 = (1.0 / MU0).sqrt();
    let zero = c64::new(0.0, 0.0);
    let ii = c64::new(0.0, 1.0);

    let results: Vec<TetResidual> = (0..n_tets)
        .into_par_iter()
        .map(|itet| {
            let (erc, urc) = (er[itet], ur[itet]);
            let mut uinv = [[zero; 3]; 3];
            let mut ermat = [[zero; 3]; 3];
            for d in 0..3 {
                uinv[d][d] = 1.0 / urc;
                ermat[d][d] = erc;
            }
            let vertices = Array2::from_shape_fn((3, 4), |(r, v)| nodes[[r, tets[[v, itet]] as usize]]);
            let vv = vertices.view();
            let intpts = Array2::from_shape_fn((3, N3D), |(k, q)| {
                dpts_3d[[1, q]] * vertices[[k, 0]]
                    + dpts_3d[[2, q]] * vertices[[k, 1]]
                    + dpts_3d[[3, q]] * vertices[[k, 2]]
                    + dpts_3d[[4, q]] * vertices[[k, 3]]
            });
            let g_node_ids = col(tets, itet);
            let l_edge_ids = local_mapping(&g_node_ids, &gather_cols(edges, &col(tet_to_edge, itet)));
            let l_tri_ids = local_mapping(&g_node_ids, &gather_cols(tris, &col(tet_to_tri, itet)));
            let triids = col(tet_to_tri, itet);
            let size_max = (0..6)
                .map(|e| edge_lengths[tet_to_edge[[e, itet]] as usize])
                .fold(f64::NEG_INFINITY, f64::max);
            let xs = [0, 1, 2, 3].map(|v| vertices[[0, v]]);
            let ys = [0, 1, 2, 3].map(|v| vertices[[1, v]]);
            let zs = [0, 1, 2, 3].map(|v| vertices[[2, v]]);
            let tet_volume = compute_volume(&xs, &ys, &zs);
            let rt = ((3.0 * tet_volume) / (4.0 * PI)).powf(1.0 / 3.0);
            let ef: Vec<c64> = (0..tet_to_field.shape()[0])
                .map(|f| efield[tet_to_field[[f, itet]] as usize])
                .collect();

            let div = compute_div(&intpts.view(), &vv, &ef, &l_edge_ids, &l_tri_ids, &ermat, &codes);
            let mut qsum = zero;
            for q in 0..N3D {
                qsum += div[q] * dpts_3d[[0, q]];
            }
            let qt = tet_volume * EPS0 * qsum;

            let rv1 = compute_curl_curl(&vv, &ef, &l_edge_ids, &l_tri_ids, &uinv, &codes);
            let field = compute_field(&intpts.view(), &vv, &ef, &l_edge_ids, &l_tri_ids, &codes);
            let pre = -(k0.powi(2)) * erc;
            let cj = 1.0 / (ii * w0 * MU0);
            let mut jt = [zero; 3];
            for k in 0..3 {
                let mut acc = zero;
                for q in 0..N3D {
                    let rv = (pre * field[[k, q]] + rv1[k]) * dpts_3d[[0, q]];
                    acc += cj * rv;
                }
                jt[k] = -tet_volume * acc;
            }
            let gt = ii * w0 * c64::new(0.0, -k0 * rt).exp() / (4.0 * PI * rt + DEPS);
            let alpha_t = -gt / (erc * EPS0 + DEPS) * qt * qt - gt * urc * MU0 * (jt[0] * jt[0] + jt[1] * jt[1] + jt[2] * jt[2]);
            let is_nan = alpha_t.norm().is_nan();

            let mut all_face_coords = Array2::<f64>::zeros((3, 4 * N2D));
            for itri in 0..4 {
                let tn = [0, 1, 2].map(|v| tris[[v, triids[itri] as usize]] as usize);
                for k in 0..3 {
                    for q in 0..N2D {
                        all_face_coords[[k, itri * N2D + q]] =
                            dpts_2d[[1, q]] * nodes[[k, tn[0]]] + dpts_2d[[2, q]] * nodes[[k, tn[1]]] + dpts_2d[[3, q]] * nodes[[k, tn[2]]];
                    }
                }
            }
            let e_face_all = compute_field(&all_face_coords.view(), &vv, &ef, &l_edge_ids, &l_tri_ids, &codes);
            let qf_all = e_face_all.map(|v| erc * EPS0 * v);
            let curl_all = compute_curl(&all_face_coords.view(), &vv, &ef, &l_edge_ids, &l_tri_ids, &codes);
            let cjf = -1.0 / (ii * MU0 * w0);
            let jf_all = curl_all.map(|v| cjf * ((1.0 / urc) * v));
            let tetc = col3(centers, itet);

            let mut res = TetResidual {
                alpha_t,
                max_elem_size: size_max,
                is_nan,
                rf: [0.0; 4],
                area: [0.0; 4],
                adj: [-1; 4],
                qf1: [[zero; N2D]; 4],
                jf1: [[[zero; N2D]; 3]; 4],
                neighbour: Vec::new(),
            };
            for iface in 0..4 {
                let tri_index = triids[iface] as usize;
                let tv = [0, 1, 2].map(|v| col3(nodes, tris[[v, tri_index]] as usize));
                let normal_r = calc_inward_normal(&tv[0], &tv[1], &tv[2], &tetc);
                let normal: C3 = normal_r.map(|v| c64::new(v, 0.0));
                let area = areas[tri_index];
                res.rf[iface] = (area / PI).powf(0.5);
                res.area[iface] = area;
                let mut itet_adj = -1i64;
                for j in 0..2 {
                    let t = tri_to_tet[[j, tri_index]];
                    if t != itet as i64 && t != MISSING_ID {
                        itet_adj = t;
                        break;
                    }
                }
                if itet_adj == -1 {
                    continue;
                }
                let slc = (iface * N2D, iface * N2D + N2D);
                if tet_is_pec[tri_index] {
                    let ef_slc = e_face_all.slice(s![.., slc.0..slc.1]);
                    let inner = cross_c_arry(&normal, &ef_slc).map(|v| -v);
                    let jtan = cross_c_arry(&normal, &inner.view()).map(|v| y0 * (1.0 / urc).sqrt() * v);
                    for c in 0..3 {
                        for q in 0..N2D {
                            res.jf1[iface][c][q] = jtan[[c, q]];
                        }
                    }
                    res.adj[iface] = itet_adj;
                    continue;
                }
                let mut iface_adj = 0usize;
                for j in 0..4 {
                    if tet_to_tri[[j, itet_adj as usize]] == tri_index as i64 {
                        iface_adj = j;
                        break;
                    }
                }
                let jcross = cross_c_arry(&normal, &jf_all.slice(s![.., slc.0..slc.1]));
                let qdot = dot_c_arry(&normal, &qf_all.slice(s![.., slc.0..slc.1]));
                let mut jf2 = [[zero; N2D]; 3];
                let mut qf2 = [zero; N2D];
                for q in 0..N2D {
                    for c in 0..3 {
                        res.jf1[iface][c][q] = jcross[[c, q]];
                        jf2[c][q] = -jcross[[c, q]];
                    }
                    res.qf1[iface][q] = qdot[q];
                    qf2[q] = -qdot[q];
                }
                res.neighbour.push((iface_adj, itet_adj as usize, qf2, jf2));
                res.adj[iface] = itet_adj;
            }
            res
        })
        .collect();

    for (itet, r) in results.iter().enumerate() {
        if r.is_nan {
            return Err(format!(
                "NaN detected in the a-posteriori error estimate (tet {}, er = {}, ur = {})",
                itet, er[itet], ur[itet]
            ));
        }
    }

    let mut qf_face2 = vec![[[zero; N2D]; 4]; n_tets];
    let mut jf_face2 = vec![[[[zero; N2D]; 3]; 4]; n_tets];
    for r in results.iter() {
        for (iface_adj, itet_adj, qf2, jf2) in r.neighbour.iter() {
            qf_face2[*itet_adj][*iface_adj] = *qf2;
            jf_face2[*itet_adj][*iface_adj] = *jf2;
        }
    }

    let error: Vec<f64> = (0..n_tets)
        .into_par_iter()
        .map(|it| {
            let r = &results[it];
            let mut alpha_f = zero;
            for iface in 0..4 {
                let area = r.area[iface];
                let (mut qf_int, mut jx, mut jy, mut jz) = (zero, zero, zero, zero);
                for q in 0..N2D {
                    let w = dpts_2d[[0, q]];
                    qf_int += (r.qf1[iface][q] - qf_face2[it][iface][q]) * area * w;
                    jx += (r.jf1[iface][0][q] - jf_face2[it][iface][0][q]) * area * w;
                    jy += (r.jf1[iface][1][q] - jf_face2[it][iface][1][q]) * area * w;
                    jz += (r.jf1[iface][2][q] - jf_face2[it][iface][2][q]) * area * w;
                }
                let rf = r.rf[iface];
                let gf = ii * w0 * c64::new(0.0, -k0 * rf).exp() / (4.0 * PI * rf + DEPS);
                let alpha_df = -gf / (er[it] * EPS0) * (qf_int * qf_int) - gf * (ur[it] * MU0) * (jx * jx + jy * jy + jz * jz);
                let alpha_nf = if r.adj[iface] == -1 {
                    zero
                } else {
                    results[r.adj[iface] as usize].alpha_t
                };
                alpha_f += (r.alpha_t / (r.alpha_t + alpha_nf + DEPS)) * alpha_df;
            }
            (r.alpha_t + alpha_f).norm().powf(0.5)
        })
        .collect();
    let max_elem_size = Array1::from_iter(results.iter().map(|r| r.max_elem_size));
    Ok((Array1::from(error), max_elem_size))
}

#[pyfunction]
#[pyo3(name = "compute_error_single")]
#[allow(clippy::too_many_arguments)]
fn py_compute_error_single<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    centers: PyReadonlyArray2<f64>,
    efield: PyReadonlyArray1<c64>,
    edge_lengths: PyReadonlyArray1<f64>,
    areas: PyReadonlyArray1<f64>,
    tet_to_edge: PyReadonlyArray2<i64>,
    tet_to_tri: PyReadonlyArray2<i64>,
    tri_centers: PyReadonlyArray2<f64>,
    tri_to_tet: PyReadonlyArray2<i64>,
    tet_to_field: PyReadonlyArray2<i64>,
    er: PyReadonlyArray1<c64>,
    ur: PyReadonlyArray1<c64>,
    pec_tris: PyReadonlyArray1<i64>,
    k0: f64,
    dofcodes3d: PyReadonlyArray1<i64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (no, te, tr, ed, ce, ef) = (
        nodes.as_array(),
        tets.as_array(),
        tris.as_array(),
        edges.as_array(),
        centers.as_array(),
        efield.as_array(),
    );
    let (el, ar, tte, ttt, tc, ttt2) = (
        edge_lengths.as_array(),
        areas.as_array(),
        tet_to_edge.as_array(),
        tet_to_tri.as_array(),
        tri_centers.as_array(),
        tri_to_tet.as_array(),
    );
    let (ttf, e, u, pt, dc) = (
        tet_to_field.as_array(),
        er.as_array(),
        ur.as_array(),
        pec_tris.as_array(),
        dofcodes3d.as_array(),
    );
    let r = py.allow_threads(|| {
        compute_error_single(
            &no, &te, &tr, &ed, &ce, &ef, &el, &ar, &tte, &ttt, &tc, &ttt2, &ttf, &e, &u, &pt, k0, &dc,
        )
    });
    let (error, sizes) = r.map_err(PyValueError::new_err)?;
    Ok((error.into_pyarray_bound(py), sizes.into_pyarray_bound(py)))
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compute_error_single, m)?)?;
    Ok(())
}
