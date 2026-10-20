//! Surface heat flux and volumetric heat source load vectors for Lagrange-2 heat conduction.
use crate::heat::grad::LOCAL_EDGE_MAP;
use crate::mth::float::{calc_area, col3, tet_coefficients};
use crate::mth::quadrature::{keast, strang};
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

pub const TRI_EDGE_MAP: [[usize; 3]; 2] = [[0, 1, 0], [1, 2, 2]];

/// Area of triangle itri of the mesh.
pub fn tri_area(nodes: &ArrayView2<f64>, tris: &ArrayView2<i64>, itri: usize) -> f64 {
    calc_area(
        &col3(nodes, tris[[0, itri]] as usize),
        &col3(nodes, tris[[1, itri]] as usize),
        &col3(nodes, tris[[2, itri]] as usize),
    )
}

/// Reference triangle load vector ∫ N_i dA / A of the six Lagrange-2 basis functions.
pub fn _assemble_force_vector() -> [f64; 6] {
    let tri_dpts = strang(3);
    let mut f_local = [0.0; 6];
    for iq in 0..tri_dpts.ncols() {
        let w = tri_dpts[[0, iq]];
        let ls = [tri_dpts[[1, iq]], tri_dpts[[2, iq]], tri_dpts[[3, iq]]];
        for iv in 0..3 {
            f_local[iv] += w * ls[iv] * (2.0 * ls[iv] - 1.0);
        }
        for ie in 0..3 {
            f_local[3 + ie] += w * 4.0 * ls[TRI_EDGE_MAP[0][ie]] * ls[TRI_EDGE_MAP[1][ie]];
        }
    }
    f_local
}

/// Global load vector of a constant surface flux q_flux over the selected triangles.
pub fn _surface_flux_builder(
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    tri_to_field: &ArrayView2<i64>,
    tri_ids_2d: &ArrayView2<i64>,
    q_flux: f64,
    n_field: usize,
) -> Array1<f64> {
    let mut f_global = Array1::<f64>::zeros(n_field);
    for idx in 0..tri_ids_2d.shape()[1] {
        let itri = tri_ids_2d[[0, idx]] as usize;
        let a = tri_area(nodes, tris, itri);
        let f_local = _assemble_force_vector();
        let scale = q_flux * a;
        for i in 0..6 {
            f_global[tri_to_field[[i, itri]] as usize] += f_local[i] * scale;
        }
    }
    f_global
}

/// Global load vector of a volumetric heat source sampled at the quadrature points of each selected tet.
pub fn _volume_source_builder(
    nodes: &ArrayView2<f64>,
    tet_to_field: &ArrayView2<i64>,
    q_at_quad: &ArrayView2<f64>,
    n_field: usize,
) -> Array1<f64> {
    let dpts = keast(4);
    let mut f_global = Array1::<f64>::zeros(n_field);
    for idx in 0..q_at_quad.shape()[0] {
        let verts = Array2::from_shape_fn((3, 4), |(r, v)| nodes[[r, tet_to_field[[v, idx]] as usize]]);
        let xs = [0, 1, 2, 3].map(|k| verts[[0, k]]);
        let ys = [0, 1, 2, 3].map(|k| verts[[1, k]]);
        let zs = [0, 1, 2, 3].map(|k| verts[[2, k]]);
        let (aas, bbs, ccs, dds, v) = tet_coefficients(&xs, &ys, &zs);
        let inv6v = 1.0 / (6.0 * v);
        let cols = [0, 1, 2, 3].map(|m| [aas[m] * inv6v, bbs[m] * inv6v, ccs[m] * inv6v, dds[m] * inv6v]);
        let mut f_local = [0.0; 10];
        for iq in 0..dpts.ncols() {
            let [x, y, z] = [0, 1, 2].map(|r| {
                verts[[r, 0]] * dpts[[1, iq]]
                    + verts[[r, 1]] * dpts[[2, iq]]
                    + verts[[r, 2]] * dpts[[3, iq]]
                    + verts[[r, 3]] * dpts[[4, iq]]
            });
            let ls = [0, 1, 2, 3].map(|m| cols[m][0] + cols[m][1] * x + cols[m][2] * y + cols[m][3] * z);
            let wq = dpts[[0, iq]] * q_at_quad[[idx, iq]];
            for iv in 0..4 {
                f_local[iv] += wq * ls[iv] * (2.0 * ls[iv] - 1.0);
            }
            for ie in 0..6 {
                f_local[4 + ie] += wq * 4.0 * ls[LOCAL_EDGE_MAP[0][ie]] * ls[LOCAL_EDGE_MAP[1][ie]];
            }
        }
        for i in 0..10 {
            f_global[tet_to_field[[i, idx]] as usize] += f_local[i] * v;
        }
    }
    f_global
}

#[pyfunction]
#[pyo3(name = "_surface_flux_builder")]
fn py_surface_flux_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    tri_to_field: PyReadonlyArray2<i64>,
    tri_ids_2d: PyReadonlyArray2<i64>,
    q_flux: f64,
    n_field: usize,
) -> Bound<'py, PyArray1<f64>> {
    _surface_flux_builder(
        &nodes.as_array(),
        &tris.as_array(),
        &tri_to_field.as_array(),
        &tri_ids_2d.as_array(),
        q_flux,
        n_field,
    )
    .into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(name = "_volume_source_builder")]
fn py_volume_source_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tet_to_field: PyReadonlyArray2<i64>,
    q_at_quad: PyReadonlyArray2<f64>,
    n_field: usize,
) -> Bound<'py, PyArray1<f64>> {
    _volume_source_builder(&nodes.as_array(), &tet_to_field.as_array(), &q_at_quad.as_array(), n_field).into_pyarray_bound(py)
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_surface_flux_builder, m)?)?;
    m.add_function(wrap_pyfunction!(py_volume_source_builder, m)?)?;
    Ok(())
}
