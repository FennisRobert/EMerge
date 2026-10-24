// [Written by Claude] All code in this module was written by Claude (AI).
//! Nedelec-2 curl-curl and mass element matrices of curved (isoparametric, 10-node geometry) tetrahedra.
//!
//! A curved tetrahedron is the straight tetrahedron y plus a quadratic displacement of its edge midpoints:
//!     x(y) = y + sum_e d_e phi_e(y),   phi_e = 4 l_i l_j,   D = dx/dy.
//! With the covariant Piola map E = D^-T E_s the curved element becomes the straight one with point-wise
//! effective materials
//!     curl-curl:  mu^-1  ->  D^T mu^-1 D / det D
//!     mass:       eps    ->  det D * D^-1 eps D^-T
//! so the basis functions of `curlcurl.rs` are reused unchanged. Faces without displaced edges keep their
//! tangential traces, so curved and straight neighbours stay conforming.
use crate::compiled::ccbf::{_eval_curl_f_3d, _eval_f_3d, parse_dofcode};
use crate::microwave::curlcurl::{bary_coeff, quad_coords, tet_dof_ijk, tet_vertices};
use crate::mth::complex::{mat3_at, matinv, CM3};
use crate::mth::index::{local_tet_to_edgeid, local_tet_to_triid};
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Local vertex pairs of the six edges, the order of the displacements d (6, 3).
pub const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Element stiffness and mass matrices of one curved tet.
///
/// tv (3, 4) are the straight vertices, d (6, 3) the edge midpoint displacements in PAIRS order and
/// qp (5, nq) a quadrature rule (weights summing to 1, then the 4 barycentric rows). The displacement adds
/// polynomial degree to the integrand, so qp should be exact to about degree 7 (a 4x4x4 conical product rule).
#[allow(clippy::too_many_arguments)]
pub fn ned2_tet_stiff_mass_iso(
    tv: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    local_edge_map: &Array2<i64>,
    local_tri_map: &Array2<i64>,
    ms: &CM3,
    mm: &CM3,
    dofcodes: &[i64],
    typearry: &[i64],
    indexarry: &[i64],
    qp: &ArrayView2<f64>,
) -> (Array2<c64>, Array2<c64>) {
    let ndof = dofcodes.len();
    let (coeff, v) = bary_coeff(tv);
    // tet_coefficients divides by |V|: for a negatively oriented tet the gradients flip sign, which the
    // basis functions do not mind but the geometry map does
    let e = |k: usize, c: usize| tv[[c, k]] - tv[[c, 0]];
    let orient = e(1, 0) * (e(2, 1) * e(3, 2) - e(2, 2) * e(3, 1)) - e(1, 1) * (e(2, 0) * e(3, 2) - e(2, 2) * e(3, 0))
        + e(1, 2) * (e(2, 0) * e(3, 1) - e(2, 1) * e(3, 0));
    let sg = if orient > 0.0 { 1.0 } else { -1.0 };
    let coords = quad_coords(tv, qp);
    let nq = qp.ncols();
    let zero = c64::new(0.0, 0.0);

    // point-wise effective materials
    let mut ms_e = vec![[[zero; 3]; 3]; nq];
    let mut mm_e = vec![[[zero; 3]; 3]; nq];
    for q in 0..nq {
        let mut dm = [[0.0f64; 3]; 3];
        for (r, row) in dm.iter_mut().enumerate() {
            row[r] = 1.0;
        }
        for (ie, &(i, j)) in PAIRS.iter().enumerate() {
            let (li, lj) = (qp[[1 + i, q]], qp[[1 + j, q]]);
            for s in 0..3 {
                let gp = sg * 4.0 * (li * coeff[[1 + s, j]] + lj * coeff[[1 + s, i]]);
                for r in 0..3 {
                    dm[r][s] += d[[ie, r]] * gp;
                }
            }
        }
        let det = dm[0][0] * (dm[1][1] * dm[2][2] - dm[1][2] * dm[2][1])
            - dm[0][1] * (dm[1][0] * dm[2][2] - dm[1][2] * dm[2][0])
            + dm[0][2] * (dm[1][0] * dm[2][1] - dm[1][1] * dm[2][0]);
        let di = [
            [
                (dm[1][1] * dm[2][2] - dm[1][2] * dm[2][1]) / det,
                (dm[0][2] * dm[2][1] - dm[0][1] * dm[2][2]) / det,
                (dm[0][1] * dm[1][2] - dm[0][2] * dm[1][1]) / det,
            ],
            [
                (dm[1][2] * dm[2][0] - dm[1][0] * dm[2][2]) / det,
                (dm[0][0] * dm[2][2] - dm[0][2] * dm[2][0]) / det,
                (dm[0][2] * dm[1][0] - dm[0][0] * dm[1][2]) / det,
            ],
            [
                (dm[1][0] * dm[2][1] - dm[1][1] * dm[2][0]) / det,
                (dm[0][1] * dm[2][0] - dm[0][0] * dm[2][1]) / det,
                (dm[0][0] * dm[1][1] - dm[0][1] * dm[1][0]) / det,
            ],
        ];
        for r in 0..3 {
            for s in 0..3 {
                let mut a1 = zero;
                let mut a2 = zero;
                for k in 0..3 {
                    for m in 0..3 {
                        a1 += ms[k][m] * (dm[k][r] * dm[m][s]);
                        a2 += mm[k][m] * (di[r][k] * di[s][m]);
                    }
                }
                ms_e[q][r][s] = a1 / det;
                mm_e[q][r][s] = a2 * det;
            }
        }
    }

    let mut f = Array3::<c64>::zeros((ndof, 3, nq));
    let mut cf = Array3::<c64>::zeros((ndof, 3, nq));
    for idof in 0..ndof {
        let (i1, j1, k1) = tet_dof_ijk(typearry[idof], indexarry[idof], local_edge_map, local_tri_map);
        let cv = coords.view();
        _eval_f_3d(&coeff.view(), &cv, i1, j1, k1, dofcodes[idof], &mut f.index_axis_mut(ndarray::Axis(0), idof));
        _eval_curl_f_3d(&coeff.view(), &cv, i1, j1, k1, dofcodes[idof], &mut cf.index_axis_mut(ndarray::Axis(0), idof));
    }

    let mut k_mat = Array2::<c64>::zeros((ndof, ndof));
    let mut m_mat = Array2::<c64>::zeros((ndof, ndof));
    for i in 0..ndof {
        for j in i..ndof {
            let mut a_s = zero;
            let mut a_m = zero;
            for q in 0..nq {
                let w = qp[[0, q]];
                for r in 0..3 {
                    let mut t_s = zero;
                    let mut t_m = zero;
                    for s in 0..3 {
                        t_s += ms_e[q][r][s] * cf[[i, s, q]];
                        t_m += mm_e[q][r][s] * f[[i, s, q]];
                    }
                    a_s += t_s * cf[[j, r, q]] * w;
                    a_m += t_m * f[[j, r, q]] * w;
                }
            }
            k_mat[[i, j]] = a_s * v;
            m_mat[[i, j]] = a_m * v;
            k_mat[[j, i]] = a_s * v;
            m_mat[[j, i]] = a_m * v;
        }
    }
    (k_mat, m_mat)
}

/// Row-major element blocks (dataE, dataB), each (nc * ndof * ndof), of the curved tets ctets with
/// displacements disp (nc, 6, 3), in the same layout as the blocks of `curlcurl::_matrix_builder`.
#[allow(clippy::too_many_arguments)]
pub fn _matrix_builder_iso(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    ur: &ArrayView3<c64>,
    er: &ArrayView3<c64>,
    ctets: &ArrayView1<i64>,
    disp: &ArrayView3<f64>,
    dofcodes: &ArrayView1<i64>,
    qp: &ArrayView2<f64>,
) -> (Vec<c64>, Vec<c64>) {
    let codes = dofcodes.to_vec();
    let ndof = codes.len();
    let nsq = (ndof * ndof).max(1);
    let (typearry, indexarry) = parse_dofcode(&codes);
    let mut data_e = vec![c64::new(0.0, 0.0); ctets.len() * nsq];
    let mut data_b = vec![c64::new(0.0, 0.0); ctets.len() * nsq];
    data_e
        .par_chunks_mut(nsq)
        .zip(data_b.par_chunks_mut(nsq))
        .enumerate()
        .for_each(|(c, (de, db))| {
            let itet = ctets[c] as usize;
            let local_tri_map = local_tet_to_triid(tet_to_tri, tets, tris, itet);
            let local_edge_map = local_tet_to_edgeid(tet_to_edge, tets, edges, itet);
            let verts = tet_vertices(nodes, tets, itet);
            let d = disp.index_axis(ndarray::Axis(0), c);
            let (k, m) = ned2_tet_stiff_mass_iso(
                &verts.view(),
                &d,
                &local_edge_map,
                &local_tri_map,
                &matinv(&mat3_at(ur, itet)),
                &mat3_at(er, itet),
                &codes,
                &typearry,
                &indexarry,
                qp,
            );
            for (n, (e, b)) in k.iter().zip(m.iter()).enumerate() {
                de[n] = *e;
                db[n] = *b;
            }
        });
    (data_e, data_b)
}

// [Written by Claude]
#[pyfunction]
#[pyo3(name = "_matrix_builder_iso")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_matrix_builder_iso<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    tet_to_edge: PyReadonlyArray2<i64>,
    tet_to_tri: PyReadonlyArray2<i64>,
    ur: PyReadonlyArray3<c64>,
    er: PyReadonlyArray3<c64>,
    ctets: PyReadonlyArray1<i64>,
    disp: PyReadonlyArray3<f64>,
    dofcodes: PyReadonlyArray1<i64>,
    qp: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray1<c64>>, Bound<'py, PyArray1<c64>>) {
    let (no, te, tr, ed) = (nodes.as_array(), tets.as_array(), tris.as_array(), edges.as_array());
    let (tte, ttt, u, e) = (tet_to_edge.as_array(), tet_to_tri.as_array(), ur.as_array(), er.as_array());
    let (ct, di, dc, q) = (ctets.as_array(), disp.as_array(), dofcodes.as_array(), qp.as_array());
    let r = py.allow_threads(|| _matrix_builder_iso(&no, &te, &tr, &ed, &tte, &ttt, &u, &e, &ct, &di, &dc, &q));
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py))
}

// [Written by Claude]
/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_matrix_builder_iso, m)?)?;
    Ok(())
}
