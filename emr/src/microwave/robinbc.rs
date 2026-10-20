//! Robin (third kind) boundary condition surface matrices and forcing vectors on Nedelec-2 triangles.
use crate::compiled::ccbf::{_eval_f_2d, parse_dofcode};
use crate::mth::complex::optim_matmul_c;
use crate::mth::float::{construct_local_vertices, optim_matmul_vec, sign, tri_coefficients};
use crate::mth::quadrature::strang;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

const IVEC: [usize; 3] = [0, 1, 0];
const JVEC: [usize; 3] = [1, 2, 2];
const N_THREADS: usize = 20;

/// Local (i, j, k) of a triangle Nedelec DoF of the given type and index.
pub fn tri_dof_ijk(i_type: i64, i_index: i64) -> (usize, usize, usize) {
    if i_type == 0 {
        (IVEC[i_index as usize], JVEC[i_index as usize], 0)
    } else {
        (0, 1, 2)
    }
}

/// Quadrature points (2D local coordinates) of the selected triangles, flattened from (nDPT, nTri).
pub fn generate_points(
    vertices_local: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    dpts: &ArrayView2<f64>,
    surf_triangle_indices: &ArrayView1<i64>,
) -> (Array1<f64>, Array1<f64>) {
    let ns = surf_triangle_indices.len();
    let nd = dpts.shape()[1];
    let mut xflat = Array1::<f64>::zeros(nd * ns);
    let mut yflat = Array1::<f64>::zeros(nd * ns);
    for i in 0..ns {
        let itri = surf_triangle_indices[i] as usize;
        let v = [0, 1, 2].map(|k| tris[[k, itri]] as usize);
        for j in 0..nd {
            xflat[j * ns + i] = vertices_local[[0, v[0]]] * dpts[[1, j]]
                + vertices_local[[0, v[1]]] * dpts[[2, j]]
                + vertices_local[[0, v[2]]] * dpts[[3, j]];
            yflat[j * ns + i] = vertices_local[[1, v[0]]] * dpts[[1, j]]
                + vertices_local[[1, v[1]]] * dpts[[2, j]]
                + vertices_local[[1, v[2]]] * dpts[[3, j]];
        }
    }
    (xflat, yflat)
}

/// Global quadrature points of the selected triangles, flattened from (nDPT, nTri).
pub fn generate_points_3d(
    vertices: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    dpts: &ArrayView2<f64>,
    surf_triangle_indices: &ArrayView1<i64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let ns = surf_triangle_indices.len();
    let nd = dpts.shape()[1];
    let pts: Vec<[f64; 3]> = (0..nd * ns)
        .into_par_iter()
        .map(|idx| {
            let (j, i) = (idx / ns, idx % ns);
            let itri = surf_triangle_indices[i] as usize;
            let v = [0, 1, 2].map(|k| tris[[k, itri]] as usize);
            let (d1, d2, d3) = (dpts[[1, j]], dpts[[2, j]], dpts[[3, j]]);
            [0, 1, 2].map(|r| vertices[[r, v[0]]] * d1 + vertices[[r, v[1]]] * d2 + vertices[[r, v[2]]] * d3)
        })
        .collect();
    (
        Array1::from_iter(pts.iter().map(|p| p[0])),
        Array1::from_iter(pts.iter().map(|p| p[1])),
        Array1::from_iter(pts.iter().map(|p| p[2])),
    )
}

/// Local triangle frame data: basis, barycentric coefficients, area and local quadrature coordinates.
pub fn local_tri_frame(glob_vertices: &ArrayView2<f64>) -> ([[f64; 3]; 3], Array2<f64>, f64, Array2<f64>) {
    let dpts = strang(6);
    let (basis, local_vertices) = construct_local_vertices(glob_vertices);
    let txs = [local_vertices[[0, 0]], local_vertices[[0, 1]], local_vertices[[0, 2]]];
    let tys = [local_vertices[[1, 0]], local_vertices[[1, 1]], local_vertices[[1, 2]]];
    let (aas, bbs, ccs, a) = tri_coefficients(&txs, &tys);
    let rows = [aas, bbs, ccs];
    let coeff = Array2::from_shape_fn((3, 3), |(r, m)| rows[r][m]);
    let coords = Array2::from_shape_fn((2, dpts.ncols()), |(r, q)| {
        let t = if r == 0 { &txs } else { &tys };
        t[0] * dpts[[1, q]] + t[1] * dpts[[2, q]] + t[2] * dpts[[3, q]]
    });
    (basis, coeff, a, coords)
}

/// Projects the local in-plane excitation (Ux, Uy) on every DoF: b_i = -A Σ w (N_i · U).
fn tri_force_from_u(coeff: &Array2<f64>, coords: &Array2<f64>, a: f64, ux: &[c64], uy: &[c64], dofcodes: &[i64]) -> Array1<c64> {
    let dpts = strang(6);
    let (typearry, indexarry) = parse_dofcode(dofcodes);
    let ndof = dofcodes.len();
    let mut bvec = Array1::<c64>::zeros(ndof);
    let mut fdof = Array2::<c64>::zeros((2, coords.shape()[1]));
    for idof in 0..ndof {
        let (i1, j1, k1) = tri_dof_ijk(typearry[idof], indexarry[idof]);
        _eval_f_2d(&coeff.view(), &coords.view(), i1, j1, k1, dofcodes[idof], &mut fdof.view_mut());
        let mut acc = c64::new(0.0, 0.0);
        for q in 0..coords.shape()[1] {
            acc += (fdof[[0, q]] * ux[q] + fdof[[1, q]] * uy[q]) * dpts[[0, q]];
        }
        bvec[idof] = -a * acc;
    }
    bvec
}

/// Nedelec-2 triangle forcing vector of an incident field glob_Uinc (3, nDPT) for a Robin BC.
pub fn ned2_tri_force(glob_vertices: &ArrayView2<f64>, glob_uinc: &ArrayView2<c64>, dofcodes: &[i64]) -> Array1<c64> {
    let (basis, coeff, a, coords) = local_tri_frame(glob_vertices);
    let lcs_uinc = optim_matmul_c(&basis, glob_uinc);
    let ux: Vec<c64> = lcs_uinc.row(0).to_vec();
    let uy: Vec<c64> = lcs_uinc.row(1).to_vec();
    tri_force_from_u(&coeff, &coords, a, &ux, &uy, dofcodes)
}

/// Nedelec-2 triangle forcing vector of the scattered field formulation (field and curl excitation).
pub fn ned2_tri_force_scat(
    glob_vertices: &ArrayView2<f64>,
    glob_uinc: &ArrayView2<c64>,
    glob_uinc_curl: &ArrayView2<c64>,
    nhat: &[f64; 3],
    dofcodes: &[i64],
) -> Array1<c64> {
    let (basis, coeff, a, coords) = local_tri_frame(glob_vertices);
    let lcs_uinc = optim_matmul_c(&basis, glob_uinc);
    let lcs_uinc_curl = optim_matmul_c(&basis, glob_uinc_curl);
    let sgn = sign(optim_matmul_vec(&basis, nhat)[2]);
    let n = coords.shape()[1];
    let ux: Vec<c64> = (0..n).map(|q| lcs_uinc[[0, q]] + lcs_uinc_curl[[1, q]] * sgn).collect();
    let uy: Vec<c64> = (0..n).map(|q| lcs_uinc[[1, q]] - lcs_uinc_curl[[0, q]] * sgn).collect();
    tri_force_from_u(&coeff, &coords, a, &ux, &uy, dofcodes)
}

/// Sums per triangle vectors into Bvec using 20 contiguous chunks with private buffers (fixed summation order).
fn chunked_force_sum<F: Fn(usize) -> (Vec<i64>, Array1<c64>) + Sync>(bvec: &mut ArrayViewMut1<c64>, niter: usize, tri_vec: F) {
    let chunk_size = niter.div_ceil(N_THREADS);
    let n = bvec.len();
    let private: Vec<Vec<c64>> = (0..N_THREADS)
        .into_par_iter()
        .map(|t| {
            let mut buf = vec![c64::new(0.0, 0.0); n];
            let start = t * chunk_size;
            let end = (start + chunk_size).min(niter);
            for i in start..end {
                let (indices, b) = tri_vec(i);
                for (k, &idx) in indices.iter().enumerate() {
                    buf[idx as usize] += b[k];
                }
            }
            buf
        })
        .collect();
    for idx in 0..n {
        for buf in private.iter() {
            bvec[idx] += buf[idx];
        }
    }
}

/// Adds the Robin forcing vectors of all selected triangles into Bvec.
#[allow(clippy::too_many_arguments)]
pub fn compute_force_entries(
    vertices_global: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    bvec: &mut ArrayViewMut1<c64>,
    surf_triangle_indices: &ArrayView1<i64>,
    uglobal_all: &ArrayView3<c64>,
    tri_to_field: &ArrayView2<i64>,
    dofcodes: &ArrayView1<i64>,
) {
    let codes = dofcodes.to_vec();
    chunked_force_sum(bvec, surf_triangle_indices.len(), |i| {
        let itri = surf_triangle_indices[i] as usize;
        let verts = tri_global_vertices(vertices_global, tris, itri);
        let ulocal = uglobal_all.index_axis(ndarray::Axis(2), i);
        let b = ned2_tri_force(&verts.view(), &ulocal, &codes);
        (tri_to_field.column(itri).to_vec(), b)
    });
}

/// Adds the scattered field Robin forcing vectors of all selected triangles into Bvec.
#[allow(clippy::too_many_arguments)]
pub fn compute_force_entries_scat(
    vertices_global: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    bvec: &mut ArrayViewMut1<c64>,
    surf_triangle_indices: &ArrayView1<i64>,
    uglobal_all: &ArrayView3<c64>,
    uglobal_all_curl: &ArrayView3<c64>,
    tri_to_field: &ArrayView2<i64>,
    normals: &ArrayView2<f64>,
    dofcodes: &ArrayView1<i64>,
) {
    let codes = dofcodes.to_vec();
    chunked_force_sum(bvec, surf_triangle_indices.len(), |i| {
        let itri = surf_triangle_indices[i] as usize;
        let verts = tri_global_vertices(vertices_global, tris, itri);
        let u = uglobal_all.index_axis(ndarray::Axis(2), i);
        let uc = uglobal_all_curl.index_axis(ndarray::Axis(2), i);
        let nhat = [normals[[0, i]], normals[[1, i]], normals[[2, i]]];
        let b = ned2_tri_force_scat(&verts.view(), &u, &uc, &nhat, &codes);
        (tri_to_field.column(itri).to_vec(), b)
    });
}

/// Vertex coordinates (3, 3) of triangle itri.
pub fn tri_global_vertices(vertices: &ArrayView2<f64>, tris: &ArrayView2<i64>, itri: usize) -> Array2<f64> {
    Array2::from_shape_fn((3, 3), |(r, v)| vertices[[r, tris[[v, itri]] as usize]])
}

/// Nedelec-2 triangle Robin matrix gamma ∫ N_i · N_j dS.
pub fn ned2_tri_stiff(glob_vertices: &ArrayView2<f64>, gamma: c64, dofcodes: &[i64]) -> Array2<c64> {
    let dpts = strang(6);
    let (typearry, indexarry) = parse_dofcode(dofcodes);
    let ndof = dofcodes.len();
    let (_basis, coeff, a, coords) = local_tri_frame(glob_vertices);
    let a = a.abs();
    let n = coords.shape()[1];
    let mut bmat = Array2::<c64>::zeros((ndof, ndof));
    let mut fdof1 = Array2::<c64>::zeros((2, n));
    let mut fdof2 = Array2::<c64>::zeros((2, n));
    for idof1 in 0..ndof {
        let (i1, j1, k1) = tri_dof_ijk(typearry[idof1], indexarry[idof1]);
        _eval_f_2d(&coeff.view(), &coords.view(), i1, j1, k1, dofcodes[idof1], &mut fdof1.view_mut());
        for idof2 in 0..ndof {
            let (i2, j2, k2) = tri_dof_ijk(typearry[idof2], indexarry[idof2]);
            _eval_f_2d(&coeff.view(), &coords.view(), i2, j2, k2, dofcodes[idof2], &mut fdof2.view_mut());
            let mut acc = c64::new(0.0, 0.0);
            for q in 0..n {
                acc += (fdof1[[0, q]] * fdof2[[0, q]] + fdof1[[1, q]] * fdof2[[1, q]]) * dpts[[0, q]];
            }
            bmat[[idof1, idof2]] = gamma * acc;
        }
    }
    bmat * a
}

/// Adds the Robin matrices of the selected triangles into the compact COO data array Bmat (ndof² per triangle).
pub fn compute_bc_entries(
    vertices: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    bmat: &mut [c64],
    surf_triangle_indices: &ArrayView1<i64>,
    gamma: &ArrayView1<c64>,
    dofcodes: &ArrayView1<i64>,
) {
    let codes = dofcodes.to_vec();
    let n = codes.len() * codes.len();
    bmat.par_chunks_mut(n.max(1))
        .take(surf_triangle_indices.len())
        .enumerate()
        .for_each(|(i, chunk)| {
            let itri = surf_triangle_indices[i] as usize;
            let verts = tri_global_vertices(vertices, tris, itri);
            let bsub = ned2_tri_stiff(&verts.view(), gamma[i], &codes);
            for (k, v) in bsub.iter().enumerate() {
                chunk[k] += *v;
            }
        });
}

#[pyfunction]
#[pyo3(name = "generate_points")]
fn py_generate_points<'py>(
    py: Python<'py>,
    vertices_local: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    dpts: PyReadonlyArray2<f64>,
    surf_triangle_indices: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let r = generate_points(
        &vertices_local.as_array(),
        &tris.as_array(),
        &dpts.as_array(),
        &surf_triangle_indices.as_array(),
    );
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "generate_points_3d")]
fn py_generate_points_3d<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    dpts: PyReadonlyArray2<f64>,
    surf_triangle_indices: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (v, t, d, s) = (
        vertices.as_array(),
        tris.as_array(),
        dpts.as_array(),
        surf_triangle_indices.as_array(),
    );
    let r = py.allow_threads(|| generate_points_3d(&v, &t, &d, &s));
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py), r.2.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "compute_force_entries")]
#[allow(clippy::too_many_arguments)]
fn py_compute_force_entries<'py>(
    py: Python<'py>,
    vertices_global: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    bvec: Bound<'py, PyArray1<c64>>,
    surf_triangle_indices: PyReadonlyArray1<i64>,
    uglobal_all: PyReadonlyArray3<c64>,
    tri_to_field: PyReadonlyArray2<i64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray1<c64>> {
    let (v, t, s, u, ttf, dc) = (
        vertices_global.as_array(),
        tris.as_array(),
        surf_triangle_indices.as_array(),
        uglobal_all.as_array(),
        tri_to_field.as_array(),
        dofcodes.as_array(),
    );
    {
        let mut rw = bvec.readwrite();
        let mut b = rw.as_array_mut();
        py.allow_threads(|| compute_force_entries(&v, &t, &mut b, &s, &u, &ttf, &dc));
    }
    bvec
}

#[pyfunction]
#[pyo3(name = "compute_force_entries_scat")]
#[allow(clippy::too_many_arguments)]
fn py_compute_force_entries_scat<'py>(
    py: Python<'py>,
    vertices_global: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    bvec: Bound<'py, PyArray1<c64>>,
    surf_triangle_indices: PyReadonlyArray1<i64>,
    uglobal_all: PyReadonlyArray3<c64>,
    uglobal_all_curl: PyReadonlyArray3<c64>,
    tri_to_field: PyReadonlyArray2<i64>,
    normals: PyReadonlyArray2<f64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray1<c64>> {
    let (v, t, s, u, uc) = (
        vertices_global.as_array(),
        tris.as_array(),
        surf_triangle_indices.as_array(),
        uglobal_all.as_array(),
        uglobal_all_curl.as_array(),
    );
    let (ttf, nr, dc) = (tri_to_field.as_array(), normals.as_array(), dofcodes.as_array());
    {
        let mut rw = bvec.readwrite();
        let mut b = rw.as_array_mut();
        py.allow_threads(|| compute_force_entries_scat(&v, &t, &mut b, &s, &u, &uc, &ttf, &nr, &dc));
    }
    bvec
}

#[pyfunction]
#[pyo3(name = "compute_bc_entries")]
fn py_compute_bc_entries<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    bmat: Bound<'py, PyArray1<c64>>,
    surf_triangle_indices: PyReadonlyArray1<i64>,
    gamma: PyReadonlyArray1<c64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray1<c64>>> {
    let (v, t, s, g, dc) = (
        vertices.as_array(),
        tris.as_array(),
        surf_triangle_indices.as_array(),
        gamma.as_array(),
        dofcodes.as_array(),
    );
    {
        let mut rw = bmat.readwrite();
        let slice = rw
            .as_slice_mut()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Bmat must be contiguous"))?;
        py.allow_threads(|| compute_bc_entries(&v, &t, slice, &s, &g, &dc));
    }
    Ok(bmat)
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_generate_points, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_points_3d, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_force_entries, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_force_entries_scat, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_bc_entries, m)?)?;
    Ok(())
}
