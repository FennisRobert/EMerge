//! Periodic boundary condition coupling matrix entries between linked triangle and edge DoFs.
use crate::compiled::ccbf::parse_dofcode;
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

type Periodic = (Array1<i64>, Array1<i64>, Array1<c64>, Array1<c64>);

/// COO (rows, cols, terms) linking target DoFs to source DoFs, plus the diagonal with zeros on linked targets.
pub fn _fill_periodic_matrix(
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    tri_to_field: &ArrayView2<i64>,
    edge_to_field: &ArrayView2<i64>,
    dofcodes: &ArrayView1<i64>,
    nfield: usize,
) -> Periodic {
    let (typearray, _) = parse_dofcode(&dofcodes.to_vec());
    let nedof = edge_to_field.shape()[0];
    let nt = tris.shape()[1];
    let ne = edges.shape()[1];
    let nfdof = typearray.iter().filter(|&&t| t == 1).count();
    let n_entries = nt * nfdof + ne * nedof;
    let mut diagonal = Array1::<c64>::from_elem(nfield, c64::new(1.0, 0.0));
    let mut rows = Array1::<i64>::zeros(n_entries);
    let mut cols = Array1::<i64>::zeros(n_entries);
    let mut terms = Array1::<c64>::zeros(n_entries);
    let mut i = 0;
    if nfdof > 0 {
        for it in 0..nt {
            let (t1, t2) = (tris[[0, it]] as usize, tris[[1, it]] as usize);
            for (k, &t) in typearray.iter().enumerate() {
                if t == 1 {
                    let (f1, f2) = (tri_to_field[[k, t1]], tri_to_field[[k, t2]]);
                    diagonal[f2 as usize] = c64::new(0.0, 0.0);
                    rows[i] = f2;
                    cols[i] = f1;
                    terms[i] = c64::new(1.0, 0.0);
                    i += 1;
                }
            }
        }
    }
    if nedof > 0 {
        for ie in 0..ne {
            let (e1, e2) = (edges[[0, ie]] as usize, edges[[1, ie]] as usize);
            for k in 0..nedof {
                let (f1, f2) = (edge_to_field[[k, e1]], edge_to_field[[k, e2]]);
                diagonal[f2 as usize] = c64::new(0.0, 0.0);
                rows[i] = f2;
                cols[i] = f1;
                terms[i] = c64::new(1.0, 0.0);
                i += 1;
            }
        }
    }
    (rows, cols, terms, diagonal)
}

#[pyfunction]
#[pyo3(name = "_fill_periodic_matrix")]
#[allow(clippy::type_complexity)]
fn py_fill_periodic_matrix<'py>(
    py: Python<'py>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    tri_to_field: PyReadonlyArray2<i64>,
    edge_to_field: PyReadonlyArray2<i64>,
    dofcodes: PyReadonlyArray1<i64>,
    nfield: usize,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<c64>>,
    Bound<'py, PyArray1<c64>>,
) {
    let r = _fill_periodic_matrix(
        &tris.as_array(),
        &edges.as_array(),
        &tri_to_field.as_array(),
        &edge_to_field.as_array(),
        &dofcodes.as_array(),
        nfield,
    );
    (
        r.0.into_pyarray_bound(py),
        r.1.into_pyarray_bound(py),
        r.2.into_pyarray_bound(py),
        r.3.into_pyarray_bound(py),
    )
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_fill_periodic_matrix, m)?)?;
    Ok(())
}
