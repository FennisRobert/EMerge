//! Thermal contact conductance coupling between the two DoF sides of a split interface.
use crate::heat::convection::_tri_mass_stencil;
use crate::heat::grad::coo_f_py;
use crate::heat::heatflux::tri_area;
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// COO entries of the AA, BB, AB and BA contact blocks h_c ∫ N_i N_j dA of every interface triangle.
pub fn _thermal_contact_builder(
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    tri_to_field_a: &ArrayView2<i64>,
    tri_to_field_b: &ArrayView2<i64>,
    tri_ids: &ArrayView1<i64>,
    h_c: f64,
) -> (Array1<f64>, Array1<i64>, Array1<i64>) {
    let n_tri = tri_ids.len();
    let nnz = n_tri * 144;
    let mut values = Array1::<f64>::zeros(nnz);
    let mut rows = Array1::<i64>::zeros(nnz);
    let mut cols = Array1::<i64>::zeros(nnz);
    let m = _tri_mass_stencil();
    for idx in 0..n_tri {
        let p = idx * 144;
        let scale = h_c * tri_area(nodes, tris, tri_ids[idx] as usize);
        for i in 0..6 {
            for j in 0..6 {
                let mij = m[i][j] * scale;
                let stride = 6 * i + j;
                let (ai, aj, bi, bj) = (
                    tri_to_field_a[[i, idx]],
                    tri_to_field_a[[j, idx]],
                    tri_to_field_b[[i, idx]],
                    tri_to_field_b[[j, idx]],
                );
                for (off, r, c, v) in [(0, ai, aj, mij), (36, bi, bj, mij), (72, ai, bj, -mij), (108, bi, aj, -mij)] {
                    rows[p + off + stride] = r;
                    cols[p + off + stride] = c;
                    values[p + off + stride] = v;
                }
            }
        }
    }
    (values, rows, cols)
}

#[pyfunction]
#[pyo3(name = "_thermal_contact_builder")]
fn py_thermal_contact_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    tri_to_field_a: PyReadonlyArray2<i64>,
    tri_to_field_b: PyReadonlyArray2<i64>,
    tri_ids: PyReadonlyArray1<i64>,
    h_c: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    coo_f_py(
        py,
        _thermal_contact_builder(
            &nodes.as_array(),
            &tris.as_array(),
            &tri_to_field_a.as_array(),
            &tri_to_field_b.as_array(),
            &tri_ids.as_array(),
            h_c,
        ),
    )
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_thermal_contact_builder, m)?)?;
    Ok(())
}
