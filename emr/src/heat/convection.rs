//! Convection (Robin) and linearised black body radiation boundary matrices for Lagrange-2 heat conduction.
use crate::heat::grad::coo_f_py;
use crate::heat::heatflux::{tri_area, TRI_EDGE_MAP};
use crate::mth::quadrature::strang;
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const KC: f64 = 5.670374419e-8;

/// The six Lagrange-2 triangle basis values at reference barycentric coordinates ls.
fn tri_basis(ls: &[f64; 3]) -> [f64; 6] {
    let mut n = [0.0; 6];
    for iv in 0..3 {
        n[iv] = ls[iv] * (2.0 * ls[iv] - 1.0);
    }
    for ie in 0..3 {
        n[3 + ie] = 4.0 * ls[TRI_EDGE_MAP[0][ie]] * ls[TRI_EDGE_MAP[1][ie]];
    }
    n
}

/// Reference triangle mass matrix Σ w N_i N_j over the quadrature points.
pub fn _tri_mass_stencil() -> [[f64; 6]; 6] {
    let tri_dpts = strang(3);
    let mut m = [[0.0; 6]; 6];
    for iq in 0..tri_dpts.ncols() {
        let w = tri_dpts[[0, iq]];
        let n = tri_basis(&[tri_dpts[[1, iq]], tri_dpts[[2, iq]], tri_dpts[[3, iq]]]);
        for i in 0..6 {
            for j in 0..6 {
                m[i][j] += w * n[i] * n[j];
            }
        }
    }
    m
}

/// Basis values N[i, iq] (6, 10) at the reference quadrature points (first four columns used).
pub fn _tri_mass_nonly() -> [[f64; 10]; 6] {
    let tri_dpts = strang(3);
    let mut out = [[0.0; 10]; 6];
    for iq in 0..tri_dpts.ncols() {
        let n = tri_basis(&[tri_dpts[[1, iq]], tri_dpts[[2, iq]], tri_dpts[[3, iq]]]);
        for i in 0..6 {
            out[i][iq] = n[i];
        }
    }
    out
}

type CooF = (Array1<f64>, Array1<i64>, Array1<i64>);

/// COO convection stiffness h ∫ N_i N_j dA of the selected triangles.
pub fn _robin_stiffness_builder(
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    tri_to_field: &ArrayView2<i64>,
    tri_ids_2d: &ArrayView2<i64>,
    h_coeff: f64,
) -> CooF {
    let n_tri = tri_ids_2d.shape()[1];
    let m = _tri_mass_stencil();
    let mut values = vec![0.0; n_tri * 36];
    let mut rows = vec![0i64; n_tri * 36];
    let mut cols = vec![0i64; n_tri * 36];
    values
        .par_chunks_mut(36)
        .zip(rows.par_chunks_mut(36))
        .zip(cols.par_chunks_mut(36))
        .enumerate()
        .for_each(|(idx, ((v, r), c))| {
            let itri = tri_ids_2d[[0, idx]] as usize;
            let scale = h_coeff * tri_area(nodes, tris, itri);
            for i in 0..6 {
                for j in 0..6 {
                    r[6 * i + j] = tri_to_field[[i, itri]];
                    c[6 * i + j] = tri_to_field[[j, itri]];
                    v[6 * i + j] = m[i][j] * scale;
                }
            }
        });
    (Array1::from(values), Array1::from(rows), Array1::from(cols))
}

/// COO radiation stiffness and load vector linearised around the temperature solution T_dofs.
#[allow(clippy::too_many_arguments)]
pub fn _radiation_builder(
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    tri_to_field: &ArrayView2<i64>,
    tri_ids_2d: &ArrayView2<i64>,
    t_dofs: &ArrayView1<f64>,
    emissivity: f64,
    t_amb: f64,
    n_field: usize,
) -> (Array1<f64>, Array1<i64>, Array1<i64>, Array1<f64>) {
    let tri_dpts = strang(3);
    let n_tri = tri_ids_2d.shape()[1];
    let mut k_values = Array1::<f64>::zeros(n_tri * 36);
    let mut k_rows = Array1::<i64>::zeros(n_tri * 36);
    let mut k_cols = Array1::<i64>::zeros(n_tri * 36);
    let mut f_global = Array1::<f64>::zeros(n_field);
    let t_amb2 = t_amb * t_amb;
    let n_pre = _tri_mass_nonly();
    for it in 0..n_tri {
        let p = it * 36;
        let itri = tri_ids_2d[[0, it]] as usize;
        let a = tri_area(nodes, tris, itri);
        let fids: Vec<usize> = (0..6).map(|i| tri_to_field[[i, itri]] as usize).collect();
        let t_local: Vec<f64> = fids.iter().map(|&f| t_dofs[f]).collect();
        let mut k_local = [[0.0; 6]; 6];
        let mut f_local = [0.0; 6];
        for iq in 0..tri_dpts.ncols() {
            let w = tri_dpts[[0, iq]];
            let mut t_q = 0.0;
            for i in 0..6 {
                t_q += n_pre[i][iq] * t_local[i];
            }
            let wh = w * emissivity * KC * (t_q * t_q + t_amb2) * (t_q + t_amb);
            let wht = wh * t_amb;
            for i in 0..6 {
                f_local[i] += wht * n_pre[i][iq];
                for j in 0..6 {
                    k_local[i][j] += wh * n_pre[i][iq] * n_pre[j][iq];
                }
            }
        }
        let scale = a;
        for i in 0..6 {
            f_global[fids[i]] += f_local[i] * scale;
            for j in 0..6 {
                let k = p + 6 * i + j;
                k_rows[k] = fids[i] as i64;
                k_cols[k] = fids[j] as i64;
                k_values[k] = k_local[i][j] * scale;
            }
        }
    }
    (k_values, k_rows, k_cols, f_global)
}

#[pyfunction]
#[pyo3(name = "_robin_stiffness_builder")]
fn py_robin_stiffness_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    tri_to_field: PyReadonlyArray2<i64>,
    tri_ids_2d: PyReadonlyArray2<i64>,
    h_coeff: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    coo_f_py(
        py,
        _robin_stiffness_builder(
            &nodes.as_array(),
            &tris.as_array(),
            &tri_to_field.as_array(),
            &tri_ids_2d.as_array(),
            h_coeff,
        ),
    )
}

#[pyfunction]
#[pyo3(name = "_radiation_builder")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_radiation_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    tri_to_field: PyReadonlyArray2<i64>,
    tri_ids_2d: PyReadonlyArray2<i64>,
    t_dofs: PyReadonlyArray1<f64>,
    emissivity: f64,
    t_amb: f64,
    n_field: usize,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let r = _radiation_builder(
        &nodes.as_array(),
        &tris.as_array(),
        &tri_to_field.as_array(),
        &tri_ids_2d.as_array(),
        &t_dofs.as_array(),
        emissivity,
        t_amb,
        n_field,
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
    m.add_function(wrap_pyfunction!(py_robin_stiffness_builder, m)?)?;
    m.add_function(wrap_pyfunction!(py_radiation_builder, m)?)?;
    Ok(())
}
