//! Per tet local blocks of the AMS discrete gradient and Nedelec interpolation operators.
use crate::compiled::ccbf::_eval_f_3d;
use crate::compiled::legrange::{_ne, _ne_grad, _nv, _nv_grad};
use crate::microwave::curlcurl::{bary_coeff, tet_vertices};
use crate::mth::index::{local_tet_to_edgeid, local_tet_to_triid};
use crate::SyncPtr;
use nalgebra::DMatrix;
use ndarray::{Array2, ArrayView1, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Local (a, b, c) vertex indices of DoF j from its type and index.
fn dof_abc(typearr: &ArrayView1<i64>, indexarr: &ArrayView1<i64>, j: usize, lem: &Array2<i64>, ltm: &Array2<i64>) -> (usize, usize, usize) {
    let idx = indexarr[j] as usize;
    if typearr[j] == 0 {
        (lem[[0, idx]] as usize, lem[[1, idx]] as usize, 0)
    } else {
        (ltm[[0, idx]] as usize, ltm[[1, idx]] as usize, ltm[[2, idx]] as usize)
    }
}

/// Least squares solution of A x = R with the numpy default singular value cutoff.
fn lstsq(a: DMatrix<f64>, r: &DMatrix<f64>) -> DMatrix<f64> {
    let (m, n) = a.shape();
    let svd = a.svd(true, true);
    let smax = svd.singular_values.iter().cloned().fold(0.0, f64::max);
    let eps = f64::EPSILON * (m.max(n) as f64) * smax;
    svd.solve(r, eps).expect("SVD solve failed")
}

/// Discrete gradient blocks (least squares fit of grad(P2) in the Nedelec-2 span) with the relative residual per tet.
#[allow(clippy::too_many_arguments)]
pub fn gradient_kernel(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    tet_to_field: &ArrayView2<i64>,
    dofcodes: &ArrayView1<i64>,
    typearr: &ArrayView1<i64>,
    indexarr: &ArrayView1<i64>,
    bary: &ArrayView2<f64>,
    out_vals: &mut [f64],
    out_rows: &mut [i64],
    out_cols: &mut [i64],
    out_res: &mut [f64],
) {
    let n_tets = tets.shape()[1];
    let ndof = dofcodes.len();
    let npts = bary.shape()[1];
    let n_nodes = nodes.shape()[1];
    let (pv, pr, pc, pres) = (
        SyncPtr(out_vals.as_mut_ptr()),
        SyncPtr(out_rows.as_mut_ptr()),
        SyncPtr(out_cols.as_mut_ptr()),
        SyncPtr(out_res.as_mut_ptr()),
    );
    (0..n_tets).into_par_iter().for_each(|itet| {
        let verts = tet_vertices(nodes, tets, itet);
        let (coeff, _v) = bary_coeff(&verts.view());
        let coords = Array2::from_shape_fn((3, npts), |(c, q)| {
            let mut acc = 0.0;
            for m in 0..4 {
                acc += verts[[c, m]] * bary[[m, q]];
            }
            acc
        });
        let lem = local_tet_to_edgeid(tet_to_edge, tets, edges, itet);
        let ltm = local_tet_to_triid(tet_to_tri, tets, tris, itet);
        let mut a = DMatrix::<f64>::zeros(3 * npts, ndof);
        let mut buf = Array2::<c64>::zeros((3, npts));
        for j in 0..ndof {
            let (ia, ib, ic) = dof_abc(typearr, indexarr, j, &lem, &ltm);
            _eval_f_3d(&coeff.view(), &coords.view(), ia, ib, ic, dofcodes[j], &mut buf.view_mut());
            for c in 0..3 {
                for q in 0..npts {
                    a[(c * npts + q, j)] = buf[[c, q]].re;
                }
            }
        }
        let mut r = DMatrix::<f64>::zeros(3 * npts, 10);
        for v in 0..4 {
            let g = _nv_grad(&coeff.view(), &coords.view(), v, v, 0);
            for c in 0..3 {
                for q in 0..npts {
                    r[(c * npts + q, v)] = g[[c, q]].re;
                }
            }
        }
        for e in 0..6 {
            let g = _ne_grad(&coeff.view(), &coords.view(), lem[[0, e]] as usize, lem[[1, e]] as usize, 0);
            for c in 0..3 {
                for q in 0..npts {
                    r[(c * npts + q, 4 + e)] = g[[c, q]].re;
                }
            }
        }
        let sol = lstsq(a.clone(), &r);
        let fit = &a * &sol;
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..3 * npts {
            for m in 0..10 {
                let d = fit[(p, m)] - r[(p, m)];
                num += d * d;
                den += r[(p, m)] * r[(p, m)];
            }
        }
        let base = itet * ndof * 10;
        unsafe {
            pres.write(itet, if den > 0.0 { (num / den).sqrt() } else { 0.0 });
            for j in 0..ndof {
                let grow = tet_to_field[[j, itet]];
                for m in 0..10 {
                    let gcol = if m < 4 {
                        tets[[m, itet]]
                    } else {
                        tet_to_edge[[m - 4, itet]] + n_nodes as i64
                    };
                    let k = base + j * 10 + m;
                    pr.write(k, grow);
                    pc.write(k, gcol);
                    pv.write(k, sol[(j, m)]);
                }
            }
        }
    });
}

/// Nedelec interpolation blocks Pi_loc = Q^-1 m(phi_a e_d) from entity-local moments with cond(Q) per tet.
#[allow(clippy::too_many_arguments)]
pub fn interpolation_kernel(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    tet_to_field: &ArrayView2<i64>,
    dofcodes: &ArrayView1<i64>,
    typearr: &ArrayView1<i64>,
    indexarr: &ArrayView1<i64>,
    gs: &ArrayView1<f64>,
    gws: &ArrayView1<f64>,
    tb: &ArrayView2<f64>,
    tw: &ArrayView1<f64>,
    n_scalar: i64,
    out_vals: &mut [f64],
    out_rows: &mut [i64],
    out_cols: &mut [i64],
    out_cond: &mut [f64],
) {
    let n_tets = tets.shape()[1];
    let ndof = dofcodes.len();
    let (nq_e, nq_f) = (gs.len(), tb.shape()[0]);
    let n_nodes = nodes.shape()[1] as i64;
    let (pv, pr, pc, pcond) = (
        SyncPtr(out_vals.as_mut_ptr()),
        SyncPtr(out_rows.as_mut_ptr()),
        SyncPtr(out_cols.as_mut_ptr()),
        SyncPtr(out_cond.as_mut_ptr()),
    );
    (0..n_tets).into_par_iter().for_each(|itet| {
        let verts = tet_vertices(nodes, tets, itet);
        let (coeff, _v) = bary_coeff(&verts.view());
        let lem = local_tet_to_edgeid(tet_to_edge, tets, edges, itet);
        let ltm = local_tet_to_triid(tet_to_tri, tets, tris, itet);
        let mut qm = DMatrix::<f64>::zeros(ndof, ndof);
        let mut fm = DMatrix::<f64>::zeros(ndof, 30);
        for i in 0..ndof {
            let (pts, wts, dirv) = if i < 12 {
                let (e, wid) = (i % 6, i / 6);
                let (l0, l1) = (lem[[0, e]] as usize, lem[[1, e]] as usize);
                let dirv = [0, 1, 2].map(|c| verts[[c, l1]] - verts[[c, l0]]);
                let pts = Array2::from_shape_fn((3, nq_e), |(c, q)| verts[[c, l0]] + gs[q] * dirv[c]);
                let wts: Vec<f64> = (0..nq_e).map(|q| gws[q] * if wid == 0 { 1.0 } else { 1.0 - 2.0 * gs[q] }).collect();
                (pts, wts, dirv)
            } else {
                let (f, wid) = ((i - 12) % 4, (i - 12) / 4);
                let t = [0, 1, 2].map(|k| ltm[[k, f]] as usize);
                let dirv = [0, 1, 2].map(|c| {
                    if wid == 0 {
                        verts[[c, t[1]]] - verts[[c, t[0]]]
                    } else {
                        verts[[c, t[2]]] - verts[[c, t[0]]]
                    }
                });
                let pts = Array2::from_shape_fn((3, nq_f), |(c, q)| {
                    tb[[q, 0]] * verts[[c, t[0]]] + tb[[q, 1]] * verts[[c, t[1]]] + tb[[q, 2]] * verts[[c, t[2]]]
                });
                let wts: Vec<f64> = (0..nq_f).map(|q| tw[q]).collect();
                (pts, wts, dirv)
            };
            let npts = wts.len();
            let mut buf = Array2::<c64>::zeros((3, npts));
            for j in 0..ndof {
                let (ia, ib, ic) = dof_abc(typearr, indexarr, j, &lem, &ltm);
                _eval_f_3d(&coeff.view(), &pts.view(), ia, ib, ic, dofcodes[j], &mut buf.view_mut());
                let mut acc = 0.0;
                for q in 0..npts {
                    let mut s = 0.0;
                    for c in 0..3 {
                        s += buf[[c, q]].re * dirv[c];
                    }
                    acc += wts[q] * s;
                }
                qm[(i, j)] = acc;
            }
            for v in 0..4 {
                let sc = _nv(&coeff.view(), &pts.view(), v, v, 0);
                let mut acc = 0.0;
                for q in 0..npts {
                    acc += wts[q] * sc[q].re;
                }
                for d in 0..3 {
                    fm[(i, d * 10 + v)] = acc * dirv[d];
                }
            }
            for e2 in 0..6 {
                let sc = _ne(&coeff.view(), &pts.view(), lem[[0, e2]] as usize, lem[[1, e2]] as usize, 0);
                let mut acc = 0.0;
                for q in 0..npts {
                    acc += wts[q] * sc[q].re;
                }
                for d in 0..3 {
                    fm[(i, d * 10 + 4 + e2)] = acc * dirv[d];
                }
            }
        }
        let sv = qm.clone().singular_values();
        let smax = sv.iter().cloned().fold(0.0, f64::max);
        let smin = sv.iter().cloned().fold(f64::INFINITY, f64::min);
        let pi_loc = qm.lu().solve(&fm).unwrap_or_else(|| DMatrix::from_element(ndof, 30, f64::NAN));
        let base = itet * ndof * 30;
        unsafe {
            pcond.write(itet, smax / smin);
            for j in 0..ndof {
                let grow = tet_to_field[[j, itet]];
                for d in 0..3 {
                    for m in 0..10 {
                        let gcol = (if m < 4 {
                            tets[[m, itet]]
                        } else {
                            tet_to_edge[[m - 4, itet]] + n_nodes
                        }) + d as i64 * n_scalar;
                        let k = base + j * 30 + d * 10 + m;
                        pr.write(k, grow);
                        pc.write(k, gcol);
                        pv.write(k, pi_loc[(j, d * 10 + m)]);
                    }
                }
            }
        }
    });
}

fn slice_mut<'a, T: numpy::Element>(a: &'a mut PyReadwriteArray1<T>, name: &str) -> PyResult<&'a mut [T]> {
    a.as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("{name} must be contiguous")))
}

#[pyfunction]
#[pyo3(name = "gradient_kernel")]
#[allow(clippy::too_many_arguments)]
fn py_gradient_kernel(
    py: Python<'_>,
    nodes: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    tet_to_edge: PyReadonlyArray2<i64>,
    tet_to_tri: PyReadonlyArray2<i64>,
    tet_to_field: PyReadonlyArray2<i64>,
    dofcodes: PyReadonlyArray1<i64>,
    typearr: PyReadonlyArray1<i64>,
    indexarr: PyReadonlyArray1<i64>,
    bary: PyReadonlyArray2<f64>,
    mut out_vals: PyReadwriteArray1<f64>,
    mut out_rows: PyReadwriteArray1<i64>,
    mut out_cols: PyReadwriteArray1<i64>,
    mut out_res: PyReadwriteArray1<f64>,
) -> PyResult<()> {
    let (no, te, tr, ed, tte, ttt, ttf) = (
        nodes.as_array(),
        tets.as_array(),
        tris.as_array(),
        edges.as_array(),
        tet_to_edge.as_array(),
        tet_to_tri.as_array(),
        tet_to_field.as_array(),
    );
    let (dc, ta, ia, ba) = (dofcodes.as_array(), typearr.as_array(), indexarr.as_array(), bary.as_array());
    let (v, r, c, res) = (
        slice_mut(&mut out_vals, "out_vals")?,
        slice_mut(&mut out_rows, "out_rows")?,
        slice_mut(&mut out_cols, "out_cols")?,
        slice_mut(&mut out_res, "out_res")?,
    );
    py.allow_threads(|| gradient_kernel(&no, &te, &tr, &ed, &tte, &ttt, &ttf, &dc, &ta, &ia, &ba, v, r, c, res));
    Ok(())
}

#[pyfunction]
#[pyo3(name = "interpolation_kernel")]
#[allow(clippy::too_many_arguments)]
fn py_interpolation_kernel(
    py: Python<'_>,
    nodes: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    tet_to_edge: PyReadonlyArray2<i64>,
    tet_to_tri: PyReadonlyArray2<i64>,
    tet_to_field: PyReadonlyArray2<i64>,
    dofcodes: PyReadonlyArray1<i64>,
    typearr: PyReadonlyArray1<i64>,
    indexarr: PyReadonlyArray1<i64>,
    gs: PyReadonlyArray1<f64>,
    gws: PyReadonlyArray1<f64>,
    tb: PyReadonlyArray2<f64>,
    tw: PyReadonlyArray1<f64>,
    n_scalar: i64,
    mut out_vals: PyReadwriteArray1<f64>,
    mut out_rows: PyReadwriteArray1<i64>,
    mut out_cols: PyReadwriteArray1<i64>,
    mut out_cond: PyReadwriteArray1<f64>,
) -> PyResult<()> {
    let (no, te, tr, ed, tte, ttt, ttf) = (
        nodes.as_array(),
        tets.as_array(),
        tris.as_array(),
        edges.as_array(),
        tet_to_edge.as_array(),
        tet_to_tri.as_array(),
        tet_to_field.as_array(),
    );
    let (dc, ta, ia) = (dofcodes.as_array(), typearr.as_array(), indexarr.as_array());
    let (g, gw, t, w) = (gs.as_array(), gws.as_array(), tb.as_array(), tw.as_array());
    let (v, r, c, cond) = (
        slice_mut(&mut out_vals, "out_vals")?,
        slice_mut(&mut out_rows, "out_rows")?,
        slice_mut(&mut out_cols, "out_cols")?,
        slice_mut(&mut out_cond, "out_cond")?,
    );
    py.allow_threads(|| {
        interpolation_kernel(
            &no, &te, &tr, &ed, &tte, &ttt, &ttf, &dc, &ta, &ia, &g, &gw, &t, &w, n_scalar, v, r, c, cond,
        )
    });
    Ok(())
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_gradient_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(py_interpolation_kernel, m)?)?;
    Ok(())
}
