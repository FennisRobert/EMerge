//! Kernels that locate COO entries in a fixed CSC pattern and add values into its data array in place.
use ndarray::{ArrayView1, ArrayViewMut1};
use num_complex::Complex64 as c64;
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

const LINEAR_SCAN_THRESHOLD: i64 = 16;

/// Position of row in the sorted slice indices[lo..hi], or -1 if absent.
pub fn _find_row_in_column(indices: &ArrayView1<i64>, lo: i64, hi: i64, row: i64) -> i64 {
    let (mut lo, mut hi) = (lo, hi);
    if hi - lo <= LINEAR_SCAN_THRESHOLD {
        for k in lo..hi {
            if indices[k as usize] == row {
                return k;
            }
        }
        return -1;
    }
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let val = indices[mid as usize];
        if val == row {
            return mid;
        } else if val < row {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    -1
}

/// Writes the CSC data index of every (rows[i], cols[i]) into data_indices (-1 if missing).
pub fn _locate_entries_kernel(
    indices: &ArrayView1<i64>,
    indptr: &ArrayView1<i64>,
    rows: &ArrayView1<i64>,
    cols: &ArrayView1<i64>,
    data_indices: &mut ArrayViewMut1<i64>,
) {
    let found: Vec<i64> = (0..rows.len())
        .into_par_iter()
        .map(|i| {
            let c = cols[i] as usize;
            _find_row_in_column(indices, indptr[c], indptr[c + 1], rows[i])
        })
        .collect();
    for (d, f) in data_indices.iter_mut().zip(found) {
        *d = f;
    }
}

/// out[data_indices[k]] += tri_coeffs[k / values_per_tri] * unit_values[k], accumulating repeated indices.
pub fn _scatter_add_per_triangle(
    out: &mut ArrayViewMut1<c64>,
    data_indices: &ArrayView1<i64>,
    unit_values: &ArrayView1<c64>,
    tri_coeffs: &ArrayView1<c64>,
    values_per_tri: usize,
) {
    for i in 0..tri_coeffs.len() {
        let c = tri_coeffs[i];
        for k in i * values_per_tri..(i + 1) * values_per_tri {
            out[data_indices[k] as usize] += c * unit_values[k];
        }
    }
}

/// out[data_indices[k]] += scale * values[k], accumulating repeated indices.
pub fn _scatter_add(out: &mut ArrayViewMut1<c64>, data_indices: &ArrayView1<i64>, values: &ArrayView1<c64>, scale: c64) {
    for k in 0..values.len() {
        out[data_indices[k] as usize] += scale * values[k];
    }
}

/// out = a + alpha * b.
pub fn _write_a_plus_alpha_b(out: &mut ArrayViewMut1<c64>, a: &ArrayView1<c64>, b: &ArrayView1<c64>, alpha: c64) {
    out.as_slice_mut()
        .expect("out must be contiguous")
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, o)| *o = a[i] + alpha * b[i]);
}

#[pyfunction]
#[pyo3(name = "_locate_entries_kernel")]
fn py_locate_entries_kernel(
    py: Python<'_>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    rows: PyReadonlyArray1<i64>,
    cols: PyReadonlyArray1<i64>,
    mut data_indices: PyReadwriteArray1<i64>,
) {
    let (ind, ip, r, c) = (indices.as_array(), indptr.as_array(), rows.as_array(), cols.as_array());
    let mut d = data_indices.as_array_mut();
    py.allow_threads(|| _locate_entries_kernel(&ind, &ip, &r, &c, &mut d));
}

#[pyfunction]
#[pyo3(name = "_scatter_add_per_triangle")]
fn py_scatter_add_per_triangle(
    mut out: PyReadwriteArray1<c64>,
    data_indices: PyReadonlyArray1<i64>,
    unit_values: PyReadonlyArray1<c64>,
    tri_coeffs: PyReadonlyArray1<c64>,
    values_per_tri: usize,
) {
    _scatter_add_per_triangle(
        &mut out.as_array_mut(),
        &data_indices.as_array(),
        &unit_values.as_array(),
        &tri_coeffs.as_array(),
        values_per_tri,
    );
}

#[pyfunction]
#[pyo3(name = "_scatter_add")]
fn py_scatter_add(mut out: PyReadwriteArray1<c64>, data_indices: PyReadonlyArray1<i64>, values: PyReadonlyArray1<c64>, scale: c64) {
    _scatter_add(&mut out.as_array_mut(), &data_indices.as_array(), &values.as_array(), scale);
}

#[pyfunction]
#[pyo3(name = "_write_a_plus_alpha_b")]
fn py_write_a_plus_alpha_b(
    py: Python<'_>,
    mut out: PyReadwriteArray1<c64>,
    a: PyReadonlyArray1<c64>,
    b: PyReadonlyArray1<c64>,
    alpha: c64,
) {
    let (av, bv) = (a.as_array(), b.as_array());
    let mut o = out.as_array_mut();
    py.allow_threads(|| _write_a_plus_alpha_b(&mut o, &av, &bv, alpha));
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_locate_entries_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(py_scatter_add_per_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(py_scatter_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_write_a_plus_alpha_b, m)?)?;
    Ok(())
}
