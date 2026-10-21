//! In-plane conduction stiffness of thin conductive sheets on Lagrange-2 triangles.
use crate::heat::grad::coo_f_py;
use crate::heat::heatflux::TRI_EDGE_MAP;
use crate::microwave::curlcurl::write_block_rowcol;
use crate::microwave::robinbc::tri_global_vertices;
use crate::mth::float::{construct_local_vertices, tri_coefficients_signed};
use crate::mth::quadrature::strang;
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// In-plane gradient (2, N) of the triangle vertex basis with coefficients (a, b, c).
pub fn grad_n1(coeff: &[f64; 3], coords: &ArrayView2<f64>) -> Array2<f64> {
    let [a, b, c] = *coeff;
    let n = coords.shape()[1];
    let mut out = Array2::<f64>::zeros((2, n));
    for p in 0..n {
        let q = 4.0 * (b * coords[[0, p]] + c * coords[[1, p]]) + (4.0 * a - 1.0);
        out[[0, p]] = q * b;
        out[[1, p]] = q * c;
    }
    out
}

/// In-plane gradient (2, N) of the triangle edge basis spanned by vertex coefficients c1 and c2.
pub fn grad_n2(c1: &[f64; 3], c2: &[f64; 3], coords: &ArrayView2<f64>) -> Array2<f64> {
    let n = coords.shape()[1];
    let mut out = Array2::<f64>::zeros((2, n));
    for p in 0..n {
        let (x, y) = (coords[[0, p]], coords[[1, p]]);
        let l1 = 4.0 * (c1[0] + c1[1] * x + c1[2] * y);
        let l2 = 4.0 * (c2[0] + c2[1] * x + c2[2] * y);
        out[[0, p]] = l2 * c1[1] + l1 * c2[1];
        out[[1, p]] = l2 * c1[2] + l1 * c2[2];
    }
    out
}

/// 6x6 sheet stiffness kappa_t ∫ ∇N_i · ∇N_j dA of one triangle in its local frame.
pub fn leg2_tri_stiff_triangle(glob_vertices: &ArrayView2<f64>, kappa_t: f64) -> Array2<f64> {
    let tri_dpts = strang(3);
    let (_basis, lcs) = construct_local_vertices(glob_vertices);
    let xs = [lcs[[0, 0]], lcs[[0, 1]], lcs[[0, 2]]];
    let ys = [lcs[[1, 0]], lcs[[1, 1]], lcs[[1, 2]]];
    let (aas, bbs, ccs, a) = tri_coefficients_signed(&xs, &ys);
    let cols = [0, 1, 2].map(|m| [aas[m] / (2.0 * a), bbs[m] / (2.0 * a), ccs[m] / (2.0 * a)]);
    let coords = Array2::from_shape_fn((2, tri_dpts.ncols()), |(r, q)| {
        let t = if r == 0 { &xs } else { &ys };
        t[0] * tri_dpts[[1, q]] + t[1] * tri_dpts[[2, q]] + t[2] * tri_dpts[[3, q]]
    });
    let grad = |iv: usize| {
        if iv < 3 {
            grad_n1(&cols[iv], &coords.view())
        } else {
            grad_n2(&cols[TRI_EDGE_MAP[0][iv - 3]], &cols[TRI_EDGE_MAP[1][iv - 3]], &coords.view())
        }
    };
    let mut kmat = Array2::<f64>::zeros((6, 6));
    for iv1 in 0..6 {
        let g1 = grad(iv1);
        for iv2 in 0..6 {
            let g2 = grad(iv2);
            let mut acc = 0.0;
            for q in 0..tri_dpts.ncols() {
                acc += (g1[[0, q]] * g2[[0, q]] + g1[[1, q]] * g2[[1, q]]) * tri_dpts[[0, q]];
            }
            kmat[[iv1, iv2]] = acc;
        }
    }
    kmat * kappa_t * a
}

/// COO sheet stiffness entries of the selected triangles.
pub fn leg2_tri_stiff(
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    tri_to_field: &ArrayView2<i64>,
    tri_ids_2d: &ArrayView2<i64>,
    kappa_t: f64,
) -> (Array1<f64>, Array1<i64>, Array1<i64>) {
    let n_sel = tri_ids_2d.shape()[1];
    let mut values = vec![0.0; n_sel * 36];
    let mut rows = vec![0i64; n_sel * 36];
    let mut cols = vec![0i64; n_sel * 36];
    values
        .par_chunks_mut(36)
        .zip(rows.par_chunks_mut(36))
        .zip(cols.par_chunks_mut(36))
        .enumerate()
        .for_each(|(idx, ((v, r), c))| {
            let itri = tri_ids_2d[[0, idx]] as usize;
            let ksub = leg2_tri_stiff_triangle(&tri_global_vertices(nodes, tris, itri).view(), kappa_t);
            let fids: Vec<i64> = (0..6).map(|i| tri_to_field[[i, itri]]).collect();
            write_block_rowcol(&fids, r, c);
            for (k, x) in ksub.iter().enumerate() {
                v[k] = *x;
            }
        });
    (Array1::from(values), Array1::from(rows), Array1::from(cols))
}

#[pyfunction]
#[pyo3(name = "leg2_tri_stiff")]
fn py_leg2_tri_stiff<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    tri_to_field: PyReadonlyArray2<i64>,
    tri_ids_2d: PyReadonlyArray2<i64>,
    kappa_t: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    coo_f_py(
        py,
        leg2_tri_stiff(
            &nodes.as_array(),
            &tris.as_array(),
            &tri_to_field.as_array(),
            &tri_ids_2d.as_array(),
            kappa_t,
        ),
    )
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_leg2_tri_stiff, m)?)?;
    Ok(())
}
