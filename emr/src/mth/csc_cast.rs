//! COO to CSC pattern precomputation and parallel scatter of COO values into CSC data.
use crate::SyncPtr;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::AddAssign;

type Pattern = (Array1<i64>, Array1<i64>, Array1<i64>, Array1<i64>, Array1<i64>);

/// Computes (indptr, indices, target_csc, col_offsets, perm) mapping COO entries onto a sorted, deduplicated CSC pattern.
pub fn precompute_csc_pattern(rows: &ArrayView1<i64>, cols: &ArrayView1<i64>, n: usize) -> Pattern {
    let nnz_coo = cols.len();
    let mut col_counts = vec![0i64; n];
    for k in 0..nnz_coo {
        col_counts[cols[k] as usize] += 1;
    }
    let mut col_offsets = vec![0i64; n + 1];
    for i in 0..n {
        col_offsets[i + 1] = col_offsets[i] + col_counts[i];
    }
    let mut grouped_perm = vec![0i64; nnz_coo];
    let mut cursor: Vec<i64> = col_offsets[..n].to_vec();
    for k in 0..nnz_coo {
        let c = cols[k] as usize;
        grouped_perm[cursor[c] as usize] = k as i64;
        cursor[c] += 1;
    }

    let per_col: Vec<(Vec<i64>, i64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let start = col_offsets[i] as usize;
            let end = col_offsets[i + 1] as usize;
            if start == end {
                return (Vec::new(), 0);
            }
            let mut seg: Vec<i64> = grouped_perm[start..end].to_vec();
            seg.sort_by_key(|&k| rows[k as usize]);
            let mut uniques = 1;
            for j in 1..seg.len() {
                if rows[seg[j] as usize] != rows[seg[j - 1] as usize] {
                    uniques += 1;
                }
            }
            (seg, uniques)
        })
        .collect();

    let mut indptr = vec![0i64; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + per_col[i].1;
    }
    let mut fully_sorted_perm = Vec::with_capacity(nnz_coo);
    for (seg, _) in per_col.iter() {
        fully_sorted_perm.extend_from_slice(seg);
    }

    let mut indices = vec![0i64; indptr[n] as usize];
    let mut target_csc = vec![0i64; nnz_coo];
    let ind_ptr = SyncPtr(indices.as_mut_ptr());
    let tgt_ptr = SyncPtr(target_csc.as_mut_ptr());
    (0..n).into_par_iter().for_each(|i| {
        let start = col_offsets[i] as usize;
        let end = col_offsets[i + 1] as usize;
        if start == end {
            return;
        }
        let mut csc_pos = indptr[i];
        let mut prev_row = rows[fully_sorted_perm[start] as usize];
        unsafe {
            ind_ptr.write(csc_pos as usize, prev_row);
            tgt_ptr.write(start, csc_pos);
        }
        for j in (start + 1)..end {
            let curr_row = rows[fully_sorted_perm[j] as usize];
            if curr_row != prev_row {
                csc_pos += 1;
                unsafe { ind_ptr.write(csc_pos as usize, curr_row) };
                prev_row = curr_row;
            }
            unsafe { tgt_ptr.write(j, csc_pos) };
        }
    });
    (
        Array1::from(indptr),
        Array1::from(indices),
        Array1::from(target_csc),
        Array1::from(col_offsets),
        Array1::from(fully_sorted_perm),
    )
}

/// Gathers data_coo by perm into sorted_data and sums it into data_csc at target_csc, column parallel.
pub fn scatter_to_csc<T: Copy + Send + Sync + AddAssign>(
    data_coo: &ArrayView1<T>,
    perm: &ArrayView1<i64>,
    target_csc: &ArrayView1<i64>,
    col_offsets: &ArrayView1<i64>,
    sorted_data: &mut ArrayViewMut1<T>,
    data_csc: &mut ArrayViewMut1<T>,
) {
    let n = data_coo.len();
    let sd = sorted_data.as_slice_mut().expect("sorted_data must be contiguous");
    sd[..n]
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, v)| *v = data_coo[perm[k] as usize]);
    let ncol = col_offsets.len() - 1;
    let out = SyncPtr(data_csc.as_slice_mut().expect("data_csc must be contiguous").as_mut_ptr());
    let sd: &[T] = sd;
    (0..ncol).into_par_iter().for_each(|i| {
        for k in col_offsets[i] as usize..col_offsets[i + 1] as usize {
            unsafe { out.add_assign(target_csc[k] as usize, sd[k]) };
        }
    });
}

#[pyfunction]
#[pyo3(name = "precompute_csc_pattern")]
#[allow(clippy::type_complexity)]
fn py_precompute_csc_pattern<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<i64>,
    cols: PyReadonlyArray1<i64>,
    n: usize,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
) {
    let (rows, cols) = (rows.as_array(), cols.as_array());
    let r = py.allow_threads(|| precompute_csc_pattern(&rows, &cols, n));
    (
        r.0.into_pyarray_bound(py),
        r.1.into_pyarray_bound(py),
        r.2.into_pyarray_bound(py),
        r.3.into_pyarray_bound(py),
        r.4.into_pyarray_bound(py),
    )
}

#[pyfunction]
#[pyo3(name = "scatter_to_csc_c16")]
fn py_scatter_to_csc_c16(
    py: Python<'_>,
    data_coo: PyReadonlyArray1<c64>,
    perm: PyReadonlyArray1<i64>,
    target_csc: PyReadonlyArray1<i64>,
    col_offsets: PyReadonlyArray1<i64>,
    mut sorted_data: PyReadwriteArray1<c64>,
    mut data_csc: PyReadwriteArray1<c64>,
) {
    let (d, p, t, c) = (data_coo.as_array(), perm.as_array(), target_csc.as_array(), col_offsets.as_array());
    let (mut s, mut o) = (sorted_data.as_array_mut(), data_csc.as_array_mut());
    py.allow_threads(|| scatter_to_csc(&d, &p, &t, &c, &mut s, &mut o));
}

#[pyfunction]
#[pyo3(name = "scatter_to_csc_f8")]
fn py_scatter_to_csc_f8(
    py: Python<'_>,
    data_coo: PyReadonlyArray1<f64>,
    perm: PyReadonlyArray1<i64>,
    target_csc: PyReadonlyArray1<i64>,
    col_offsets: PyReadonlyArray1<i64>,
    mut sorted_data: PyReadwriteArray1<f64>,
    mut data_csc: PyReadwriteArray1<f64>,
) {
    let (d, p, t, c) = (data_coo.as_array(), perm.as_array(), target_csc.as_array(), col_offsets.as_array());
    let (mut s, mut o) = (sorted_data.as_array_mut(), data_csc.as_array_mut());
    py.allow_threads(|| scatter_to_csc(&d, &p, &t, &c, &mut s, &mut o));
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_precompute_csc_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(py_scatter_to_csc_c16, m)?)?;
    m.add_function(wrap_pyfunction!(py_scatter_to_csc_f8, m)?)?;
    Ok(())
}
