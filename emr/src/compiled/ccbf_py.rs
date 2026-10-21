//! Python bindings of the dofcode parsing.
use super::ccbf::parse_dofcode;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "parse_dofcode")]
fn py_parse_dofcode<'py>(py: Python<'py>, dofcodes: PyReadonlyArray1<i64>) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let (t, i) = parse_dofcode(&dofcodes.as_array().to_vec());
    (t.into_pyarray_bound(py), i.into_pyarray_bound(py))
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_parse_dofcode, m)?)?;
    Ok(())
}
