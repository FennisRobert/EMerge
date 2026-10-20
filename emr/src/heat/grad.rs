//! Lagrange-2 thermal conduction stiffness and capacity matrix assembly over tetrahedra.
use crate::microwave::curlcurl::{quad_coords, tet_vertices, write_block_rowcol};
use crate::mth::float::{optim_matmul, tet_coefficients, M3};
use crate::mth::quadrature::keast;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

pub const LOCAL_EDGE_MAP: [[usize; 6]; 2] = [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]];

/// Gradient (3, N) of the vertex basis with coefficients coeff = (a, b, c, d).
pub fn grad_n1(coeff: &[f64; 4], coords: &ArrayView2<f64>) -> Array2<f64> {
    let [a, b, c, d] = *coeff;
    let n = coords.shape()[1];
    let mut out = Array2::<f64>::zeros((3, n));
    for p in 0..n {
        let q = 4.0 * (b * coords[[0, p]] + c * coords[[1, p]] + d * coords[[2, p]]) + (4.0 * a - 1.0);
        out[[0, p]] = q * b;
        out[[1, p]] = q * c;
        out[[2, p]] = q * d;
    }
    out
}

/// Gradient (3, N) of the edge basis spanned by vertex coefficients c1 and c2.
pub fn grad_n2(c1: &[f64; 4], c2: &[f64; 4], coords: &ArrayView2<f64>) -> Array2<f64> {
    let n = coords.shape()[1];
    let mut out = Array2::<f64>::zeros((3, n));
    for p in 0..n {
        let (x, y, z) = (coords[[0, p]], coords[[1, p]], coords[[2, p]]);
        let l1 = 4.0 * (c1[0] + c1[1] * x + c1[2] * y + c1[3] * z);
        let l2 = 4.0 * (c2[0] + c2[1] * x + c2[2] * y + c2[3] * z);
        out[[0, p]] = l2 * c1[1] + l1 * c2[1];
        out[[1, p]] = l2 * c1[2] + l1 * c2[2];
        out[[2, p]] = l2 * c1[3] + l1 * c2[3];
    }
    out
}

/// Normalised per vertex coefficient columns (a, b, c, d) and the volume of a tet.
pub fn vertex_coeffs(tet_vertices: &ArrayView2<f64>) -> ([[f64; 4]; 4], f64) {
    let xs = [0, 1, 2, 3].map(|v| tet_vertices[[0, v]]);
    let ys = [0, 1, 2, 3].map(|v| tet_vertices[[1, v]]);
    let zs = [0, 1, 2, 3].map(|v| tet_vertices[[2, v]]);
    let (a, b, c, d, v) = tet_coefficients(&xs, &ys, &zs);
    let s = 6.0 * v;
    ([0, 1, 2, 3].map(|m| [a[m] / s, b[m] / s, c[m] / s, d[m] / s]), v)
}

fn basis_grad(cols: &[[f64; 4]; 4], iv: usize, cs: &ArrayView2<f64>) -> Array2<f64> {
    if iv < 4 {
        grad_n1(&cols[iv], cs)
    } else {
        grad_n2(&cols[LOCAL_EDGE_MAP[0][iv - 4]], &cols[LOCAL_EDGE_MAP[1][iv - 4]], cs)
    }
}

/// 10x10 thermal stiffness matrix k ∫ ∇N_i · ∇N_j dV of one tet with conductivity tensor cond_tet.
pub fn leg2_tet_stiff(tet_vertices: &ArrayView2<f64>, cond_tet: &M3) -> Array2<f64> {
    let dpts = keast(4);
    let (cols, v) = vertex_coeffs(tet_vertices);
    let cs = quad_coords(tet_vertices, &dpts);
    let mut amat = Array2::<f64>::zeros((10, 10));
    for iv1 in 0..10 {
        let val_gn1 = optim_matmul(cond_tet, &basis_grad(&cols, iv1, &cs.view()).view());
        for iv2 in 0..10 {
            let val_gn2 = basis_grad(&cols, iv2, &cs.view());
            let mut acc = 0.0;
            for q in 0..dpts.ncols() {
                acc += (val_gn1[[0, q]] * val_gn2[[0, q]] + val_gn1[[1, q]] * val_gn2[[1, q]] + val_gn1[[2, q]] * val_gn2[[2, q]])
                    * dpts[[0, q]];
            }
            amat[[iv1, iv2]] = acc;
        }
    }
    amat * v
}

/// 10x10 thermal mass matrix rho_cp ∫ N_i N_j dV of one tet.
pub fn leg2_tet_mass(tet_vertices: &ArrayView2<f64>, rho_cp: f64) -> Array2<f64> {
    let dpts = keast(4);
    let (cols, v) = vertex_coeffs(tet_vertices);
    let cs = quad_coords(tet_vertices, &dpts);
    let basis = |iv: usize, ls: &[f64; 4]| {
        if iv < 4 {
            ls[iv] * (2.0 * ls[iv] - 1.0)
        } else {
            4.0 * ls[LOCAL_EDGE_MAP[0][iv - 4]] * ls[LOCAL_EDGE_MAP[1][iv - 4]]
        }
    };
    let mut mmat = Array2::<f64>::zeros((10, 10));
    for iv1 in 0..10 {
        for iv2 in iv1..10 {
            let mut val = 0.0;
            for iq in 0..dpts.ncols() {
                let (x, y, z) = (cs[[0, iq]], cs[[1, iq]], cs[[2, iq]]);
                let ls = [0, 1, 2, 3].map(|m| cols[m][0] + cols[m][1] * x + cols[m][2] * y + cols[m][3] * z);
                val += dpts[[0, iq]] * basis(iv1, &ls) * basis(iv2, &ls);
            }
            mmat[[iv1, iv2]] = val;
            mmat[[iv2, iv1]] = val;
        }
    }
    mmat * rho_cp * v
}

type CooF = (Array1<f64>, Array1<i64>, Array1<i64>);

/// Assembles 10x10 element blocks of every tet into COO (values, rows, cols).
fn tet_block_builder<F: Fn(usize, &ArrayView2<f64>) -> Array2<f64> + Sync>(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tet_to_field: &ArrayView2<i64>,
    block: F,
) -> CooF {
    let nt = tets.shape()[1];
    let mut vals = vec![0.0; nt * 100];
    let mut rows = vec![0i64; nt * 100];
    let mut cols = vec![0i64; nt * 100];
    vals.par_chunks_mut(100)
        .zip(rows.par_chunks_mut(100))
        .zip(cols.par_chunks_mut(100))
        .enumerate()
        .for_each(|(itet, ((v, r), c))| {
            let verts = tet_vertices(nodes, tets, itet);
            let sub = block(itet, &verts.view());
            let indices: Vec<i64> = (0..10).map(|f| tet_to_field[[f, itet]]).collect();
            write_block_rowcol(&indices, r, c);
            for (k, x) in sub.iter().enumerate() {
                v[k] = *x;
            }
        });
    (Array1::from(vals), Array1::from(rows), Array1::from(cols))
}

/// COO stiffness entries of all tets with per tet 3x3 conductivity tensors.
pub fn _stiff_matrix_builder(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tet_to_field: &ArrayView2<i64>,
    cond_termal: &ArrayView3<f64>,
) -> CooF {
    tet_block_builder(nodes, tets, tet_to_field, |itet, verts| {
        let mut cond = [[0.0; 3]; 3];
        for (r, row) in cond.iter_mut().enumerate() {
            for (c, x) in row.iter_mut().enumerate() {
                *x = cond_termal[[r, c, itet]];
            }
        }
        leg2_tet_stiff(verts, &cond)
    })
}

/// COO mass (capacity) entries of all tets with per tet density times specific heat.
pub fn _mass_matrix_builder(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tet_to_field: &ArrayView2<i64>,
    rho_cp_per_tet: &ArrayView1<f64>,
) -> CooF {
    tet_block_builder(nodes, tets, tet_to_field, |itet, verts| leg2_tet_mass(verts, rho_cp_per_tet[itet]))
}

type PyCooF<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

pub fn coo_f_py(py: Python<'_>, r: CooF) -> PyCooF<'_> {
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py), r.2.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "_stiff_matrix_builder")]
fn py_stiff_matrix_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    tet_to_field: PyReadonlyArray2<i64>,
    cond_termal: PyReadonlyArray3<f64>,
) -> PyCooF<'py> {
    let (n, t, f, c) = (nodes.as_array(), tets.as_array(), tet_to_field.as_array(), cond_termal.as_array());
    coo_f_py(py, py.allow_threads(|| _stiff_matrix_builder(&n, &t, &f, &c)))
}

#[pyfunction]
#[pyo3(name = "_mass_matrix_builder")]
fn py_mass_matrix_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    tet_to_field: PyReadonlyArray2<i64>,
    rho_cp_per_tet: PyReadonlyArray1<f64>,
) -> PyCooF<'py> {
    let (n, t, f, r) = (
        nodes.as_array(),
        tets.as_array(),
        tet_to_field.as_array(),
        rho_cp_per_tet.as_array(),
    );
    coo_f_py(py, py.allow_threads(|| _mass_matrix_builder(&n, &t, &f, &r)))
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_stiff_matrix_builder, m)?)?;
    m.add_function(wrap_pyfunction!(py_mass_matrix_builder, m)?)?;
    Ok(())
}
