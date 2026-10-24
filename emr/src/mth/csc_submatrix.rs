// [Written by Claude] All code in this module was written by Claude (AI).
//! A[:, ids][ids, :] of a square CSC matrix in one parallel pass.
//!
//! scipy's fancy indexing builds the column slice and then the row slice, each on one core and with an
//! intermediate matrix. Here every kept column is filtered once into its final place: a count pass gives
//! the column pointers, a fill pass copies the kept entries with their new row indices.
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// (indptr, indices, data) of A[:, keep][keep, :] for a square CSC matrix with n rows and increasing keep.
pub fn csc_submatrix(indptr: &[i32], indices: &[i32], data: &[c64], n: usize, keep: &[i64]) -> (Vec<i32>, Vec<i32>, Vec<c64>) {
    let mut newid = vec![-1i32; n];
    for (j, &c) in keep.iter().enumerate() {
        newid[c as usize] = j as i32;
    }
    let counts: Vec<usize> = keep
        .par_iter()
        .map(|&c| {
            let (a, b) = (indptr[c as usize] as usize, indptr[c as usize + 1] as usize);
            indices[a..b].iter().filter(|&&r| newid[r as usize] >= 0).count()
        })
        .collect();
    let mut outptr = vec![0i32; keep.len() + 1];
    for (j, &m) in counts.iter().enumerate() {
        outptr[j + 1] = outptr[j] + m as i32;
    }
    let nnz = outptr[keep.len()] as usize;
    let mut outind = vec![0i32; nnz];
    let mut outdata = vec![c64::new(0.0, 0.0); nnz];
    // one disjoint output slice per kept column
    let mut cols = Vec::with_capacity(keep.len());
    let (mut ri, mut rd) = (outind.as_mut_slice(), outdata.as_mut_slice());
    for &m in &counts {
        let (ci, ti) = ri.split_at_mut(m);
        let (cd, td) = rd.split_at_mut(m);
        cols.push((ci, cd));
        ri = ti;
        rd = td;
    }
    cols.into_par_iter().zip(keep.par_iter()).for_each(|((ci, cd), &c)| {
        let (a, b) = (indptr[c as usize] as usize, indptr[c as usize + 1] as usize);
        let mut p = 0;
        for k in a..b {
            let r = newid[indices[k] as usize];
            if r >= 0 {
                ci[p] = r;
                cd[p] = data[k];
                p += 1;
            }
        }
    });
    (outptr, outind, outdata)
}

// [Written by Claude]
#[pyfunction]
#[pyo3(name = "csc_submatrix_c16")]
#[allow(clippy::type_complexity)]
fn py_csc_submatrix_c16<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i32>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<c64>,
    n: usize,
    keep: PyReadonlyArray1<i64>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<c64>>)> {
    let (p, i, d, k) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, keep.as_slice()?);
    let r = py.allow_threads(|| csc_submatrix(p, i, d, n, k));
    Ok((r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py), r.2.into_pyarray_bound(py)))
}

// [Written by Claude]
/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_csc_submatrix_c16, m)?)?;
    Ok(())
}
