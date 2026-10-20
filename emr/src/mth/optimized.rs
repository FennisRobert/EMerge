//! Integration point generation, curve subsampling and small matrix helpers exposed to Python.
use super::complex::{matinv, CM3};
use super::float::{calc_area, calc_volume, col3, compute_distances, matinv_r, M3};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

type Flat4 = (Array1<f64>, Array1<f64>, Array1<f64>, Array1<i64>);
type Flat6 = (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<i64>);

/// Quadrature points of every triangle, flattened from shape (nPTS, nTri).
pub fn generate_int_points_tri(nodes: &ArrayView2<f64>, triangles: &ArrayView2<i64>, pts: &ArrayView2<f64>) -> Flat4 {
    let (x, y, z, _, _, shape) = generate_int_data_simplex(nodes, triangles, pts, false);
    (x, y, z, shape)
}

/// Quadrature points of every tetrahedron, flattened from shape (nPTS, nTet).
pub fn generate_int_points_tet(nodes: &ArrayView2<f64>, tets: &ArrayView2<i64>, pts: &ArrayView2<f64>) -> Flat4 {
    let (x, y, z, _, _, shape) = generate_int_data_simplex(nodes, tets, pts, false);
    (x, y, z, shape)
}

/// Quadrature points, weights and triangle areas, flattened from shape (nPTS, nTri).
pub fn generate_int_data_tri(nodes: &ArrayView2<f64>, triangles: &ArrayView2<i64>, pts: &ArrayView2<f64>) -> Flat6 {
    generate_int_data_simplex(nodes, triangles, pts, true)
}

/// Quadrature points, weights and tetrahedron volumes, flattened from shape (nPTS, nTet).
pub fn generate_int_data_tet(nodes: &ArrayView2<f64>, tetrahedra: &ArrayView2<i64>, pts: &ArrayView2<f64>) -> Flat6 {
    generate_int_data_simplex(nodes, tetrahedra, pts, true)
}

/// Shared implementation for triangle (3 vertex) and tetrahedron (4 vertex) quadrature data.
fn generate_int_data_simplex(nodes: &ArrayView2<f64>, elems: &ArrayView2<i64>, pts: &ArrayView2<f64>, with_measure: bool) -> Flat6 {
    let nv = elems.shape()[0];
    let ne = elems.shape()[1];
    let ndpt = pts.shape()[1];
    let mut xall = Array1::<f64>::zeros(ndpt * ne);
    let mut yall = Array1::<f64>::zeros(ndpt * ne);
    let mut zall = Array1::<f64>::zeros(ndpt * ne);
    let mut wall = Array1::<f64>::zeros(if with_measure { ndpt * ne } else { 0 });
    let mut aall = Array1::<f64>::zeros(if with_measure { ndpt * ne } else { 0 });
    for it in 0..ne {
        let vids: Vec<usize> = (0..nv).map(|v| elems[[v, it]] as usize).collect();
        let measure = if !with_measure {
            0.0
        } else if nv == 3 {
            calc_area(&col3(nodes, vids[0]), &col3(nodes, vids[1]), &col3(nodes, vids[2]))
        } else {
            calc_volume(
                &col3(nodes, vids[0]),
                &col3(nodes, vids[1]),
                &col3(nodes, vids[2]),
                &col3(nodes, vids[3]),
            )
        };
        for j in 0..ndpt {
            let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
            for (v, &vid) in vids.iter().enumerate() {
                x += nodes[[0, vid]] * pts[[v + 1, j]];
                y += nodes[[1, vid]] * pts[[v + 1, j]];
                z += nodes[[2, vid]] * pts[[v + 1, j]];
            }
            let idx = j * ne + it;
            xall[idx] = x;
            yall[idx] = y;
            zall[idx] = z;
            if with_measure {
                wall[idx] = pts[[0, j]];
                aall[idx] = measure;
            }
        }
    }
    (xall, yall, zall, wall, aall, Array1::from(vec![ndpt as i64, ne as i64]))
}

/// linspace as start + i*step with the last value set exactly to stop.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let mut arr = vec![0.0; num];
    if num == 0 {
        return arr;
    }
    let div = num - 1;
    if div > 0 {
        let step = (stop - start) / div as f64;
        for (i, v) in arr.iter_mut().enumerate() {
            *v = start + i as f64 * step;
        }
    } else {
        arr[0] = start;
    }
    if num > 1 {
        arr[num - 1] = stop;
    }
    arr
}

/// Reduces a finely sampled curve to a subset of points that traces it within tolerance.
pub fn _subsample_coordinates(xs: &ArrayView1<f64>, ys: &ArrayView1<f64>, tolerance: f64, xmin: f64) -> (Array1<f64>, Array1<f64>) {
    let n = xs.len();
    let mut ids = vec![0usize; n];
    let mut store_index = 1usize;
    let mut start_index = 0usize;
    let mut final_index = 0usize;
    for _iteration in 0..n {
        let i1 = start_index;
        let mut done = false;
        for i2 in (i1 + 1)..n {
            let ds = ((xs[i2] - xs[i1]).powi(2) + (ys[i2] - ys[i1]).powi(2)).sqrt();
            if i2 == n - 1 {
                ids[store_index] = i2 - 1;
                final_index = store_index + 1;
                done = true;
                break;
            }
            if ds < xmin {
                continue;
            }
            if ((ys[n - 1] - ys[i2]).powi(2) + (xs[n - 1] - xs[i2]).powi(2)).sqrt() < xmin {
                let imid = i1 + (n - 1 - i1) / 2;
                ids[store_index] = imid;
                ids[store_index + 1] = n - 1;
                final_index = store_index + 2;
                done = true;
                break;
            }
            let x_f = linspace(xs[i1], xs[i2], i2 - i1 + 1);
            let y_f = linspace(ys[i1], ys[i2], i2 - i1 + 1);
            let mut error = f64::NEG_INFINITY;
            for q in 0..(i2 - i1 + 1) {
                error = error.max(((x_f[q] - xs[i1 + q]).powi(2) + (y_f[q] - ys[i1 + q]).powi(2)).sqrt());
            }
            if error < tolerance {
                continue;
            }
            ids[store_index] = i2 - 1;
            start_index = i2;
            store_index += 1;
            break;
        }
        if done {
            break;
        }
    }
    let sel = &ids[0..final_index];
    (
        Array1::from_iter(sel.iter().map(|&i| xs[i])),
        Array1::from_iter(sel.iter().map(|&i| ys[i])),
    )
}

type PyFlat4<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
);
type PyFlat6<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
);

fn to_py4(py: Python<'_>, r: Flat4) -> PyFlat4<'_> {
    (
        r.0.into_pyarray_bound(py),
        r.1.into_pyarray_bound(py),
        r.2.into_pyarray_bound(py),
        r.3.into_pyarray_bound(py),
    )
}

fn to_py6(py: Python<'_>, r: Flat6) -> PyFlat6<'_> {
    (
        r.0.into_pyarray_bound(py),
        r.1.into_pyarray_bound(py),
        r.2.into_pyarray_bound(py),
        r.3.into_pyarray_bound(py),
        r.4.into_pyarray_bound(py),
        r.5.into_pyarray_bound(py),
    )
}

#[pyfunction]
#[pyo3(name = "generate_int_points_tri")]
fn py_generate_int_points_tri<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    pts: PyReadonlyArray2<f64>,
) -> PyFlat4<'py> {
    to_py4(
        py,
        generate_int_points_tri(&nodes.as_array(), &triangles.as_array(), &pts.as_array()),
    )
}

#[pyfunction]
#[pyo3(name = "generate_int_points_tet")]
fn py_generate_int_points_tet<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    pts: PyReadonlyArray2<f64>,
) -> PyFlat4<'py> {
    to_py4(py, generate_int_points_tet(&nodes.as_array(), &tets.as_array(), &pts.as_array()))
}

#[pyfunction]
#[pyo3(name = "generate_int_data_tri")]
fn py_generate_int_data_tri<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    pts: PyReadonlyArray2<f64>,
) -> PyFlat6<'py> {
    to_py6(py, generate_int_data_tri(&nodes.as_array(), &triangles.as_array(), &pts.as_array()))
}

#[pyfunction]
#[pyo3(name = "generate_int_data_tet")]
fn py_generate_int_data_tet<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tetrahedra: PyReadonlyArray2<i64>,
    pts: PyReadonlyArray2<f64>,
) -> PyFlat6<'py> {
    to_py6(
        py,
        generate_int_data_tet(&nodes.as_array(), &tetrahedra.as_array(), &pts.as_array()),
    )
}

#[pyfunction]
#[pyo3(name = "_subsample_coordinates")]
fn py_subsample_coordinates<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<f64>,
    ys: PyReadonlyArray1<f64>,
    tolerance: f64,
    xmin: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (x, y) = _subsample_coordinates(&xs.as_array(), &ys.as_array(), tolerance, xmin);
    (x.into_pyarray_bound(py), y.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "compute_distances")]
fn py_compute_distances<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<f64>,
    ys: PyReadonlyArray1<f64>,
    zs: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    compute_distances(&xs.as_array(), &ys.as_array(), &zs.as_array()).into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(name = "matinv")]
fn py_matinv<'py>(py: Python<'py>, m: PyReadonlyArray2<c64>) -> Bound<'py, PyArray2<c64>> {
    let m = m.as_array();
    let mut mm: CM3 = [[c64::new(0.0, 0.0); 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            mm[r][c] = m[[r, c]];
        }
    }
    let inv = matinv(&mm);
    Array2::from_shape_fn((3, 3), |(r, c)| inv[r][c]).into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(name = "matinv_r")]
fn py_matinv_r<'py>(py: Python<'py>, m: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
    let m = m.as_array();
    let mut mm: M3 = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            mm[r][c] = m[[r, c]];
        }
    }
    let inv = matinv_r(&mm);
    Array2::from_shape_fn((3, 3), |(r, c)| inv[r][c]).into_pyarray_bound(py)
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_generate_int_points_tri, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_int_points_tet, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_int_data_tri, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_int_data_tet, m)?)?;
    m.add_function(wrap_pyfunction!(py_subsample_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_distances, m)?)?;
    m.add_function(wrap_pyfunction!(py_matinv, m)?)?;
    m.add_function(wrap_pyfunction!(py_matinv_r, m)?)?;
    Ok(())
}
