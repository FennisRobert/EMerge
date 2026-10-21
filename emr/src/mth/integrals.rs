//! Surface integrals over triangulated surfaces and periodic coordinate pairing.
use super::float::{calc_area, col3};
use ndarray::{Array2, ArrayView1, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Complex surface integral Σ C·f·A using quadrature weights DPTs[0] and per triangle field samples.
pub fn _fast_integral_c(
    nodes: &ArrayView2<f64>,
    triangles: &ArrayView2<i64>,
    constants: &ArrayView1<c64>,
    dpts: &ArrayView2<f64>,
    field_values: &ArrayView2<c64>,
) -> c64 {
    let mut tot = c64::new(0.0, 0.0);
    for it in 0..triangles.shape()[1] {
        let a = tri_area(nodes, triangles, it);
        let mut field = c64::new(0.0, 0.0);
        for q in 0..dpts.shape()[1] {
            field += field_values[[q, it]] * dpts[[0, q]];
        }
        tot += constants[it] * field * a;
    }
    tot
}

/// Real surface integral Σ C·f·A using quadrature weights DPTs[0] and per triangle field samples.
pub fn _fast_integral_f(
    nodes: &ArrayView2<f64>,
    triangles: &ArrayView2<i64>,
    constants: &ArrayView1<f64>,
    dpts: &ArrayView2<f64>,
    field_values: &ArrayView2<f64>,
) -> f64 {
    let mut tot = 0.0;
    for it in 0..triangles.shape()[1] {
        let a = tri_area(nodes, triangles, it);
        let mut field = 0.0;
        for q in 0..dpts.shape()[1] {
            field += dpts[[0, q]] * field_values[[q, it]];
        }
        tot += constants[it] * field * a;
    }
    tot
}

fn tri_area(nodes: &ArrayView2<f64>, triangles: &ArrayView2<i64>, it: usize) -> f64 {
    calc_area(
        &col3(nodes, triangles[[0, it]] as usize),
        &col3(nodes, triangles[[1, it]] as usize),
        &col3(nodes, triangles[[2, it]] as usize),
    )
}

/// Maps every id in ids1 to the closest not yet used id in ids2 after displacing by disp (within dsmax).
#[allow(clippy::mut_range_bound)]
pub fn link_coords(
    coords: &ArrayView2<f64>,
    ids1: &ArrayView1<i64>,
    ids2: &ArrayView1<i64>,
    disp: &ArrayView1<f64>,
    dsmax: f64,
) -> Array2<i64> {
    let d = dsmax.powi(2);
    let n = ids1.len();
    let mut ids2_mapped = vec![0i64; ids2.len()];
    let mut available = vec![1i64; ids2.len()];
    let mut id_start = 0usize;
    for i1 in 0..n {
        let mut ictr = 0;
        let c1 = col3(coords, ids1[i1] as usize);
        for i2 in id_start..n {
            if available[i2] == 0 {
                continue;
            }
            let c2 = col3(coords, ids2[i2] as usize);
            let c2 = [c2[0] - disp[0], c2[1] - disp[1], c2[2] - disp[2]];
            let dist = (c2[0] - c1[0]).powi(2) + (c2[1] - c1[1]).powi(2) + (c2[2] - c1[2]).powi(2);
            if dist > d {
                ictr += 1;
                continue;
            }
            if ictr == 0 {
                id_start += 1;
            }
            ids2_mapped[i1] = ids2[i2];
            available[i2] = 0;
            break;
        }
    }
    let mut out = Array2::<i64>::zeros((2, n));
    for i in 0..n {
        out[[0, i]] = ids1[i];
        out[[1, i]] = ids2_mapped[i];
    }
    out
}

#[pyfunction]
#[pyo3(name = "_fast_integral_c")]
fn py_fast_integral_c(
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    constants: PyReadonlyArray1<c64>,
    dpts: PyReadonlyArray2<f64>,
    field_values: PyReadonlyArray2<c64>,
) -> c64 {
    _fast_integral_c(
        &nodes.as_array(),
        &triangles.as_array(),
        &constants.as_array(),
        &dpts.as_array(),
        &field_values.as_array(),
    )
}

#[pyfunction]
#[pyo3(name = "_fast_integral_f")]
fn py_fast_integral_f(
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    constants: PyReadonlyArray1<f64>,
    dpts: PyReadonlyArray2<f64>,
    field_values: PyReadonlyArray2<f64>,
) -> f64 {
    _fast_integral_f(
        &nodes.as_array(),
        &triangles.as_array(),
        &constants.as_array(),
        &dpts.as_array(),
        &field_values.as_array(),
    )
}

#[pyfunction]
#[pyo3(name = "link_coords")]
fn py_link_coords<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    ids1: PyReadonlyArray1<i64>,
    ids2: PyReadonlyArray1<i64>,
    disp: PyReadonlyArray1<f64>,
    dsmax: f64,
) -> Bound<'py, PyArray2<i64>> {
    link_coords(&coords.as_array(), &ids1.as_array(), &ids2.as_array(), &disp.as_array(), dsmax).into_pyarray_bound(py)
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_fast_integral_c, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_integral_f, m)?)?;
    m.add_function(wrap_pyfunction!(py_link_coords, m)?)?;
    Ok(())
}
