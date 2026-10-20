//! Mesh topology construction: unique edges and triangles of a tet mesh and lookups of entities by vertex tuple.
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const MISSING_ID: i64 = -1234;
const IDSET1: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 1), (2, 3)];
const IDSET2: [(usize, usize, usize); 4] = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3)];

/// Sorted vertex ids of tet column itet.
fn sorted_tet(tets: &ArrayView2<i64>, itet: usize) -> [i64; 4] {
    let mut t = [tets[[0, itet]], tets[[1, itet]], tets[[2, itet]], tets[[3, itet]]];
    t.sort_unstable();
    t
}

/// Unique edges (2, nE) and triangles (3, nR) of a tet mesh, each with ascending vertex ids, sorted lexicographically.
pub fn extract_edges_tris(tets: &ArrayView2<i64>) -> (Array2<i64>, Array2<i64>) {
    let nt = tets.shape()[1];
    let (mut edges, mut tris): (Vec<[i64; 2]>, Vec<[i64; 3]>) = (0..nt)
        .into_par_iter()
        .map(|itet| {
            let [i1, i2, i3, i4] = sorted_tet(tets, itet);
            (
                vec![[i1, i2], [i1, i3], [i1, i4], [i2, i3], [i2, i4], [i3, i4]],
                vec![[i1, i2, i3], [i1, i2, i4], [i1, i3, i4], [i2, i3, i4]],
            )
        })
        .reduce(
            || (Vec::new(), Vec::new()),
            |(mut ea, mut ta), (eb, tb)| {
                ea.extend(eb);
                ta.extend(tb);
                (ea, ta)
            },
        );
    edges.par_sort_unstable();
    edges.dedup();
    tris.par_sort_unstable();
    tris.dedup();
    (
        Array2::from_shape_fn((2, edges.len()), |(r, c)| edges[c][r]),
        Array2::from_shape_fn((3, tris.len()), |(r, c)| tris[c][r]),
    )
}

/// Index of the lexicographically sorted key in sorted_keys, or MISSING_ID.
fn bsearch<const K: usize>(sorted_keys: &[([i64; K], i64)], key: &[i64; K]) -> i64 {
    match sorted_keys.binary_search_by(|(k, _)| k.cmp(key)) {
        Ok(p) => sorted_keys[p].1,
        Err(_) => MISSING_ID,
    }
}

/// Sorted (key, column index) table of the columns of a (K, N) array, each column sorted ascending first.
fn key_table<const K: usize>(table: &ArrayView2<i64>) -> Vec<([i64; K], i64)> {
    let mut keys: Vec<([i64; K], i64)> = (0..table.shape()[1])
        .into_par_iter()
        .map(|c| {
            let mut k = [0i64; K];
            for (r, v) in k.iter_mut().enumerate() {
                *v = table[[r, c]];
            }
            k.sort_unstable();
            (k, c as i64)
        })
        .collect();
    keys.par_sort_unstable();
    keys
}

/// Edge (6, nT) and triangle (4, nT) indices of every tet in the EMerge local ordering.
pub fn tet_to_edge_tri(tets: &ArrayView2<i64>, edges: &ArrayView2<i64>, tris: &ArrayView2<i64>) -> (Array2<i64>, Array2<i64>) {
    let nt = tets.shape()[1];
    let ekeys = key_table::<2>(edges);
    let tkeys = key_table::<3>(tris);
    let per_tet: Vec<([i64; 6], [i64; 4])> = (0..nt)
        .into_par_iter()
        .map(|itet| {
            let t = [tets[[0, itet]], tets[[1, itet]], tets[[2, itet]], tets[[3, itet]]];
            let e = IDSET1.map(|(i, j)| {
                let mut k = [t[i], t[j]];
                k.sort_unstable();
                bsearch(&ekeys, &k)
            });
            let f = IDSET2.map(|(i, j, l)| {
                let mut k = [t[i], t[j], t[l]];
                k.sort_unstable();
                bsearch(&tkeys, &k)
            });
            (e, f)
        })
        .collect();
    (
        Array2::from_shape_fn((6, nt), |(r, c)| per_tet[c].0[r]),
        Array2::from_shape_fn((4, nt), |(r, c)| per_tet[c].1[r]),
    )
}

/// For every query column (K, M) the column index in table (K, N) with the same vertex set, or MISSING_ID.
pub fn find_columns(table: &ArrayView2<i64>, queries: &ArrayView2<i64>) -> Array1<i64> {
    fn run<const K: usize>(table: &ArrayView2<i64>, queries: &ArrayView2<i64>) -> Array1<i64> {
        let keys = key_table::<K>(table);
        let out: Vec<i64> = (0..queries.shape()[1])
            .into_par_iter()
            .map(|c| {
                let mut k = [0i64; K];
                for (r, v) in k.iter_mut().enumerate() {
                    *v = queries[[r, c]];
                }
                k.sort_unstable();
                bsearch(&keys, &k)
            })
            .collect();
        Array1::from(out)
    }
    match table.shape()[0] {
        1 => run::<1>(table, queries),
        2 => run::<2>(table, queries),
        3 => run::<3>(table, queries),
        4 => run::<4>(table, queries),
        k => panic!("find_columns supports tuples of 1 to 4 vertices, got {k}"),
    }
}

#[pyfunction]
#[pyo3(name = "extract_edges_tris")]
fn py_extract_edges_tris<'py>(py: Python<'py>, tets: PyReadonlyArray2<i64>) -> (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<i64>>) {
    let t = tets.as_array();
    let (e, r) = py.allow_threads(|| extract_edges_tris(&t));
    (e.into_pyarray_bound(py), r.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "tet_to_edge_tri")]
fn py_tet_to_edge_tri<'py>(
    py: Python<'py>,
    tets: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
) -> (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<i64>>) {
    let (t, e, r) = (tets.as_array(), edges.as_array(), tris.as_array());
    let (a, b) = py.allow_threads(|| tet_to_edge_tri(&t, &e, &r));
    (a.into_pyarray_bound(py), b.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "find_columns")]
fn py_find_columns<'py>(
    py: Python<'py>,
    table: PyReadonlyArray2<i64>,
    queries: PyReadonlyArray2<i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (t, q) = (table.as_array(), queries.as_array());
    if t.shape()[0] != q.shape()[0] || t.shape()[0] == 0 || t.shape()[0] > 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "table and queries must both have 1 to 4 rows",
        ));
    }
    Ok(py.allow_threads(|| find_columns(&t, &q)).into_pyarray_bound(py))
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_extract_edges_tris, m)?)?;
    m.add_function(wrap_pyfunction!(py_tet_to_edge_tri, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_columns, m)?)?;
    Ok(())
}
