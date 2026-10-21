//! Second order absorbing boundary condition surface curl and divergence matrices on Nedelec-2 triangles.
use crate::compiled::ccbf::{_eval_curl_f_2d, _eval_div_f_2d, parse_dofcode};
use crate::mth::complex::_gqi;
use crate::mth::float::{construct_local_vertices, tri_coefficients_signed};
use crate::mth::quadrature::strang;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const IVEC: [usize; 3] = [0, 1, 0];
const JVEC: [usize; 3] = [1, 2, 2];

/// Surface curl [0] and surface divergence [1] matrices (2, ndof, ndof) of one triangle.
pub fn _abc_order_2_terms(tri_vertices: &ArrayView2<f64>, dofcodes: &[i64]) -> Array3<c64> {
    let dpts = strang(6);
    let (typearry, indexarry) = parse_dofcode(dofcodes);
    let ndof = dofcodes.len();
    let (_basis, local_vertices) = construct_local_vertices(tri_vertices);
    let xpts = [local_vertices[[0, 0]], local_vertices[[0, 1]], local_vertices[[0, 2]]];
    let ypts = [local_vertices[[1, 0]], local_vertices[[1, 1]], local_vertices[[1, 2]]];
    let (aas, bbs, ccs, area) = tri_coefficients_signed(&xpts, &ypts);
    let area = area.abs();
    let rows = [aas, bbs, ccs];
    let bary_coeff = Array2::from_shape_fn((3, 3), |(r, m)| rows[r][m] / (2.0 * area));
    let weights = dpts.row(0).to_slice().unwrap();
    let coords = Array2::from_shape_fn((2, dpts.ncols()), |(r, q)| {
        let t = if r == 0 { &xpts } else { &ypts };
        t[0] * dpts[[1, q]] + t[1] * dpts[[2, q]] + t[2] * dpts[[3, q]]
    });
    let (bc, cv) = (bary_coeff.view(), coords.view());
    let n = coords.shape()[1];
    let eval = |idof: usize, fc: &mut Array1<c64>, fd: &mut Array1<c64>| {
        let (i, j, k) = if typearry[idof] == 0 {
            (IVEC[indexarry[idof] as usize], JVEC[indexarry[idof] as usize], 0)
        } else {
            (0, 1, 2)
        };
        _eval_curl_f_2d(&bc, &cv, i, j, k, dofcodes[idof], &mut fc.view_mut());
        _eval_div_f_2d(&bc, &cv, i, j, k, dofcodes[idof], &mut fd.view_mut());
    };
    let mut out = Array3::<c64>::zeros((2, ndof, ndof));
    for idof1 in 0..ndof {
        let (mut fc1, mut fd1) = (Array1::<c64>::zeros(n), Array1::<c64>::zeros(n));
        let (mut fc2, mut fd2) = (Array1::<c64>::zeros(n), Array1::<c64>::zeros(n));
        eval(idof1, &mut fc1, &mut fd1);
        for idof2 in 0..ndof {
            eval(idof2, &mut fc2, &mut fd2);
            out[[0, idof1, idof2]] = _gqi(&fc1.view(), &fc2.view(), weights) * area;
            out[[1, idof1, idof2]] = _gqi(&fd1.view(), &fd2.view(), weights) * area;
        }
    }
    out
}

/// Compact (Curl, Div) COO data of the second order ABC terms of the given triangles (ndof² per triangle).
pub fn _matrix_builder(
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    tri_ids: &ArrayView1<i64>,
    dofcodes: &ArrayView1<i64>,
) -> (Array1<c64>, Array1<c64>) {
    let codes = dofcodes.to_vec();
    let nsq = codes.len() * codes.len();
    let ntris = tri_ids.len();
    let mut curl = vec![c64::new(0.0, 0.0); ntris * nsq];
    let mut div = vec![c64::new(0.0, 0.0); ntris * nsq];
    curl.par_chunks_mut(nsq.max(1))
        .zip(div.par_chunks_mut(nsq.max(1)))
        .enumerate()
        .for_each(|(itri_sub, (cc, dd))| {
            let itri = tri_ids[itri_sub] as usize;
            let tri_nodes = Array2::from_shape_fn((3, 3), |(r, v)| nodes[[r, tris[[v, itri]] as usize]]);
            let sub = _abc_order_2_terms(&tri_nodes.view(), &codes);
            for (k, v) in sub.index_axis(ndarray::Axis(0), 0).iter().enumerate() {
                cc[k] += *v;
            }
            for (k, v) in sub.index_axis(ndarray::Axis(0), 1).iter().enumerate() {
                dd[k] += *v;
            }
        });
    (Array1::from(curl), Array1::from(div))
}

#[pyfunction]
#[pyo3(name = "_matrix_builder")]
fn py_matrix_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    tri_ids: PyReadonlyArray1<i64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<c64>>, Bound<'py, PyArray1<c64>>) {
    let (no, tr, ti, dc) = (nodes.as_array(), tris.as_array(), tri_ids.as_array(), dofcodes.as_array());
    let r = py.allow_threads(|| _matrix_builder(&no, &tr, &ti, &dc));
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py))
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_matrix_builder, m)?)?;
    Ok(())
}
