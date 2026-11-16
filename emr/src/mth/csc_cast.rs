//! COO to CSC pattern precomputation and parallel scatter of COO values into CSC data.
//!
//! [Written by Claude] Index arrays are i32 or i64. The COO row/col input may be either, and the pattern arrays
//! (perm, target_csc, col_offsets, indptr, indices) are i32 whenever the number of COO entries fits, which halves
//! the largest arrays of the assembly. Above 2^31 COO entries (about 2.4M tets with 30 DoFs per tet) they are i64.
use crate::SyncPtr;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use num_complex::Complex64 as c64;
use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::AddAssign;

/// An integer index type of the pattern arrays (i32 or i64).
pub trait Idx: Copy + Send + Sync + Ord + Default + AddAssign + Element + 'static {
    fn to_usize(self) -> usize;
    fn from_usize(v: usize) -> Self;
}

impl Idx for i32 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as i32
    }
}

impl Idx for i64 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as i64
    }
}

/// (indptr, indices, target_csc, col_offsets, perm)
type Pattern<I> = (Array1<I>, Array1<I>, Array1<I>, Array1<I>, Array1<I>);

/// Computes (indptr, indices, target_csc, col_offsets, perm) mapping COO entries onto a sorted, deduplicated CSC pattern.
///
/// perm lists the COO entries column by column, rows ascending (stable, so duplicates keep their COO order);
/// target_csc[k] is the CSC data index that perm[k] is summed into; col_offsets are the column starts in perm.
pub fn precompute_csc_pattern<R: Idx, I: Idx>(rows: &ArrayView1<R>, cols: &ArrayView1<R>, n: usize) -> Pattern<I> {
    crate::pool::install(|| {
        let nnz_coo = cols.len();
        let mut col_counts = vec![0usize; n];
        for k in 0..nnz_coo {
            col_counts[cols[k].to_usize()] += 1;
        }
        let mut col_offsets = vec![0usize; n + 1];
        for i in 0..n {
            col_offsets[i + 1] = col_offsets[i] + col_counts[i];
        }
        drop(col_counts);
        let mut perm = vec![I::default(); nnz_coo];
        let mut cursor: Vec<usize> = col_offsets[..n].to_vec();
        for k in 0..nnz_coo {
            let c = cols[k].to_usize();
            perm[cursor[c]] = I::from_usize(k);
            cursor[c] += 1;
        }
        drop(cursor);

        // Sort every column segment of perm by row, in place, and count its unique rows.
        let mut segments: Vec<&mut [I]> = Vec::with_capacity(n);
        let mut rest: &mut [I] = &mut perm;
        for i in 0..n {
            let (seg, tail) = rest.split_at_mut(col_offsets[i + 1] - col_offsets[i]);
            segments.push(seg);
            rest = tail;
        }
        let uniques: Vec<usize> = segments
            .into_par_iter()
            .map(|seg| {
                if seg.is_empty() {
                    return 0;
                }
                seg.sort_by_key(|&k| rows[k.to_usize()]);
                1 + seg.windows(2).filter(|w| rows[w[0].to_usize()] != rows[w[1].to_usize()]).count()
            })
            .collect();

        let mut indptr = vec![0usize; n + 1];
        for i in 0..n {
            indptr[i + 1] = indptr[i] + uniques[i];
        }
        drop(uniques);

        let mut indices = vec![I::default(); indptr[n]];
        let mut target_csc = vec![I::default(); nnz_coo];
        let ind_ptr = SyncPtr(indices.as_mut_ptr());
        let tgt_ptr = SyncPtr(target_csc.as_mut_ptr());
        let perm_ref: &[I] = &perm;
        (0..n).into_par_iter().for_each(|i| {
            let (start, end) = (col_offsets[i], col_offsets[i + 1]);
            if start == end {
                return;
            }
            let mut csc_pos = indptr[i];
            let mut prev_row = rows[perm_ref[start].to_usize()];
            unsafe {
                ind_ptr.write(csc_pos, I::from_usize(prev_row.to_usize()));
                tgt_ptr.write(start, I::from_usize(csc_pos));
            }
            for j in (start + 1)..end {
                let curr_row = rows[perm_ref[j].to_usize()];
                if curr_row != prev_row {
                    csc_pos += 1;
                    unsafe { ind_ptr.write(csc_pos, I::from_usize(curr_row.to_usize())) };
                    prev_row = curr_row;
                }
                unsafe { tgt_ptr.write(j, I::from_usize(csc_pos)) };
            }
        });
        let to_idx = |v: Vec<usize>| -> Array1<I> { v.into_iter().map(I::from_usize).collect() };
        (to_idx(indptr), Array1::from(indices), Array1::from(target_csc), to_idx(col_offsets), Array1::from(perm))
    })
}

/// Gathers data_coo by perm into sorted_data and sums it into data_csc at target_csc, column parallel.
pub fn scatter_to_csc<T: Copy + Send + Sync + AddAssign, I: Idx>(
    data_coo: &ArrayView1<T>,
    perm: &ArrayView1<I>,
    target_csc: &ArrayView1<I>,
    col_offsets: &ArrayView1<I>,
    sorted_data: &mut ArrayViewMut1<T>,
    data_csc: &mut ArrayViewMut1<T>,
) {
    crate::pool::install(|| {
        let n = data_coo.len();
        let sd = sorted_data.as_slice_mut().expect("sorted_data must be contiguous");
        sd[..n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(k, v)| *v = data_coo[perm[k].to_usize()]);
        let ncol = col_offsets.len() - 1;
        let out = SyncPtr(data_csc.as_slice_mut().expect("data_csc must be contiguous").as_mut_ptr());
        let sd: &[T] = sd;
        (0..ncol).into_par_iter().for_each(|i| {
            for k in col_offsets[i].to_usize()..col_offsets[i + 1].to_usize() {
                unsafe { out.add_assign(target_csc[k].to_usize(), sd[k]) };
            }
        });
    })
}

/// Whether a Python object is a 1-D numpy array of element type I.
fn is_array_of<I: Idx>(obj: &Bound<'_, PyAny>) -> bool {
    obj.downcast::<PyArray1<I>>().is_ok()
}

fn precompute_typed<'py, R: Idx, I: Idx>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
    cols: &Bound<'py, PyAny>,
    n: usize,
) -> PyResult<PyObject> {
    let rows: PyReadonlyArray1<R> = rows.extract()?;
    let cols: PyReadonlyArray1<R> = cols.extract()?;
    let (r, c) = (rows.as_array(), cols.as_array());
    let p = py.allow_threads(|| precompute_csc_pattern::<R, I>(&r, &c, n));
    Ok((
        p.0.into_pyarray_bound(py),
        p.1.into_pyarray_bound(py),
        p.2.into_pyarray_bound(py),
        p.3.into_pyarray_bound(py),
        p.4.into_pyarray_bound(py),
    )
        .into_py(py))
}

/// precompute_csc_pattern(rows, cols, n) -> (indptr, indices, target_csc, col_offsets, perm).
///
/// rows and cols are int32 or int64 arrays of the same type. The returned arrays are int32 when the number of
/// COO entries is below 2^31, otherwise int64.
#[pyfunction]
#[pyo3(name = "precompute_csc_pattern")]
fn py_precompute_csc_pattern<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
    cols: &Bound<'py, PyAny>,
    n: usize,
) -> PyResult<PyObject> {
    let nnz_coo = rows.downcast::<numpy::PyUntypedArray>()?.len();
    let small = nnz_coo < i32::MAX as usize && n < i32::MAX as usize;
    match (is_array_of::<i32>(rows) && is_array_of::<i32>(cols), is_array_of::<i64>(rows) && is_array_of::<i64>(cols), small) {
        (true, _, true) => precompute_typed::<i32, i32>(py, rows, cols, n),
        (true, _, false) => precompute_typed::<i32, i64>(py, rows, cols, n),
        (_, true, true) => precompute_typed::<i64, i32>(py, rows, cols, n),
        (_, true, false) => precompute_typed::<i64, i64>(py, rows, cols, n),
        _ => Err(PyTypeError::new_err("rows and cols must both be int32 or both be int64 arrays")),
    }
}

fn scatter_typed<'py, T: Copy + Send + Sync + AddAssign + Element, I: Idx>(
    py: Python<'py>,
    data_coo: PyReadonlyArray1<'py, T>,
    perm: &Bound<'py, PyAny>,
    target_csc: &Bound<'py, PyAny>,
    col_offsets: &Bound<'py, PyAny>,
    mut sorted_data: PyReadwriteArray1<'py, T>,
    mut data_csc: PyReadwriteArray1<'py, T>,
) -> PyResult<()> {
    let perm: PyReadonlyArray1<I> = perm.extract()?;
    let target_csc: PyReadonlyArray1<I> = target_csc.extract()?;
    let col_offsets: PyReadonlyArray1<I> = col_offsets.extract()?;
    let (d, p, t, c) = (data_coo.as_array(), perm.as_array(), target_csc.as_array(), col_offsets.as_array());
    let (mut s, mut o) = (sorted_data.as_array_mut(), data_csc.as_array_mut());
    py.allow_threads(|| scatter_to_csc(&d, &p, &t, &c, &mut s, &mut o));
    Ok(())
}

fn scatter_dispatch<'py, T: Copy + Send + Sync + AddAssign + Element>(
    py: Python<'py>,
    data_coo: PyReadonlyArray1<'py, T>,
    perm: &Bound<'py, PyAny>,
    target_csc: &Bound<'py, PyAny>,
    col_offsets: &Bound<'py, PyAny>,
    sorted_data: PyReadwriteArray1<'py, T>,
    data_csc: PyReadwriteArray1<'py, T>,
) -> PyResult<()> {
    if is_array_of::<i32>(perm) {
        scatter_typed::<T, i32>(py, data_coo, perm, target_csc, col_offsets, sorted_data, data_csc)
    } else {
        scatter_typed::<T, i64>(py, data_coo, perm, target_csc, col_offsets, sorted_data, data_csc)
    }
}

#[pyfunction]
#[pyo3(name = "scatter_to_csc_c16")]
fn py_scatter_to_csc_c16<'py>(
    py: Python<'py>,
    data_coo: PyReadonlyArray1<'py, c64>,
    perm: &Bound<'py, PyAny>,
    target_csc: &Bound<'py, PyAny>,
    col_offsets: &Bound<'py, PyAny>,
    sorted_data: PyReadwriteArray1<'py, c64>,
    data_csc: PyReadwriteArray1<'py, c64>,
) -> PyResult<()> {
    scatter_dispatch(py, data_coo, perm, target_csc, col_offsets, sorted_data, data_csc)
}

#[pyfunction]
#[pyo3(name = "scatter_to_csc_f8")]
fn py_scatter_to_csc_f8<'py>(
    py: Python<'py>,
    data_coo: PyReadonlyArray1<'py, f64>,
    perm: &Bound<'py, PyAny>,
    target_csc: &Bound<'py, PyAny>,
    col_offsets: &Bound<'py, PyAny>,
    sorted_data: PyReadwriteArray1<'py, f64>,
    data_csc: PyReadwriteArray1<'py, f64>,
) -> PyResult<()> {
    scatter_dispatch(py, data_coo, perm, target_csc, col_offsets, sorted_data, data_csc)
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_precompute_csc_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(py_scatter_to_csc_c16, m)?)?;
    m.add_function(wrap_pyfunction!(py_scatter_to_csc_f8, m)?)?;
    Ok(())
}
