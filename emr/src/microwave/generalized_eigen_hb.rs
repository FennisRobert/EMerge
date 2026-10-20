//! Assembly of the Nedelec-Lagrange triangle matrices of the generalized port mode eigenvalue problem.
use crate::compiled::ccbf::{_eval_curl_f_2d, _eval_f_2d, parse_dofcode};
use crate::compiled::legrange::{_ne_grad_tri, _ne_tri, _nv_grad_tri, _nv_tri};
use crate::microwave::curlcurl::write_block_rowcol;
use crate::mth::complex::{_gqi, _gqi2, mat3_at, matinv, matmul_2d, CM3};
use crate::mth::float::tri_coefficients;
use crate::mth::quadrature::strang;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

const IVEC: [usize; 3] = [0, 1, 0];
const JVEC: [usize; 3] = [1, 2, 2];

/// Upper-left 2x2 block of a complex 3x3 matrix.
fn block2(m: &CM3) -> [[c64; 2]; 2] {
    [[m[0][0], m[0][1]], [m[1][0], m[1][1]]]
}

/// Nedelec-2 / Lagrange-2 triangle matrices A and B of the generalized eigenproblem for one triangle.
pub fn generalized_matrix_gq(
    tri_vertices: &ArrayView2<f64>,
    urinv: &CM3,
    er: &CM3,
    k0: f64,
    dofcodes: &[i64],
) -> (Array2<c64>, Array2<c64>) {
    let dpts = strang(7);
    let (typearry, indexarry) = parse_dofcode(dofcodes);
    let ndof = dofcodes.len();
    let n = ndof + 6;
    let weights = dpts.row(0).to_slice().unwrap();
    let txs = [tri_vertices[[0, 0]], tri_vertices[[0, 1]], tri_vertices[[0, 2]]];
    let tys = [tri_vertices[[1, 0]], tri_vertices[[1, 1]], tri_vertices[[1, 2]]];
    let cs = Array2::from_shape_fn((2, dpts.ncols()), |(r, q)| {
        let t = if r == 0 { &txs } else { &tys };
        t[0] * dpts[[1, q]] + t[1] * dpts[[2, q]] + t[2] * dpts[[3, q]]
    });
    let (aas, bbs, ccs, area) = tri_coefficients(&txs, &tys);
    let rows = [aas, bbs, ccs];
    let coeff = Array2::from_shape_fn((3, 3), |(r, m)| rows[r][m]);
    let (cv, csv) = (coeff.view(), cs.view());

    let urinv_z = urinv[2][2];
    let er_z = er[2][2];
    let urinv2 = block2(urinv);
    let er2 = block2(er);

    let mut all_fdof = Array3::<c64>::zeros((ndof, 2, dpts.ncols()));
    let mut all_fdof_curl = Array2::<c64>::zeros((ndof, dpts.ncols()));
    let mut all_fdof_grad = Array3::<c64>::zeros((6, 2, dpts.ncols()));
    let mut all_fdof_z = Array2::<c64>::zeros((6, dpts.ncols()));
    for idof in 0..ndof {
        let (i1, j1, k1) = if typearry[idof] == 0 {
            (IVEC[indexarry[idof] as usize], JVEC[indexarry[idof] as usize], 0)
        } else {
            (0, 1, 2)
        };
        _eval_f_2d(&cv, &csv, i1, j1, k1, dofcodes[idof], &mut all_fdof.index_axis_mut(Axis(0), idof));
        _eval_curl_f_2d(
            &cv,
            &csv,
            i1,
            j1,
            k1,
            dofcodes[idof],
            &mut all_fdof_curl.index_axis_mut(Axis(0), idof),
        );
    }
    for idof in 0..6 {
        if idof < 3 {
            all_fdof_grad
                .index_axis_mut(Axis(0), idof)
                .assign(&_nv_grad_tri(&cv, &csv, idof, 0, 0));
            all_fdof_z.index_axis_mut(Axis(0), idof).assign(&_nv_tri(&cv, &csv, idof, 0, 0));
        } else {
            let (i1, j1) = (IVEC[idof - 3], JVEC[idof - 3]);
            all_fdof_grad
                .index_axis_mut(Axis(0), idof)
                .assign(&_ne_grad_tri(&cv, &csv, i1, j1, 0));
            all_fdof_z.index_axis_mut(Axis(0), idof).assign(&_ne_tri(&cv, &csv, i1, j1, 0));
        }
    }

    let zero = c64::new(0.0, 0.0);
    let mut att_stiff = Array2::<c64>::zeros((ndof, ndof));
    let mut att_mass = Array2::<c64>::zeros((ndof, ndof));
    let mut btt = Array2::<c64>::zeros((ndof, ndof));
    let mut btz = Array2::<c64>::zeros((ndof, 6));
    let mut bzz_stiff = Array2::<c64>::zeros((6, 6));
    let mut bzz_mass = Array2::<c64>::zeros((6, 6));
    for idof1 in 0..ndof {
        let f1 = all_fdof.index_axis(Axis(0), idof1);
        let fcurl1 = all_fdof_curl.index_axis(Axis(0), idof1).map(|v| urinv_z * v);
        let er_f1 = matmul_2d(&er2, &f1);
        let ur_f1 = matmul_2d(&urinv2, &f1);
        for idof2 in 0..ndof {
            let f2 = all_fdof.index_axis(Axis(0), idof2);
            att_stiff[[idof1, idof2]] = _gqi(&fcurl1.view(), &all_fdof_curl.index_axis(Axis(0), idof2), weights);
            att_mass[[idof1, idof2]] = _gqi2(&er_f1.view(), &f2, weights);
            btt[[idof1, idof2]] = _gqi2(&ur_f1.view(), &f2, weights);
        }
        for idof2 in 0..6 {
            btz[[idof1, idof2]] = _gqi2(&ur_f1.view(), &all_fdof_grad.index_axis(Axis(0), idof2), weights);
        }
    }
    for idof1 in 0..6 {
        let ur_g1 = matmul_2d(&urinv2, &all_fdof_grad.index_axis(Axis(0), idof1));
        let er_f1z = all_fdof_z.index_axis(Axis(0), idof1).map(|v| er_z * v);
        for idof2 in 0..6 {
            bzz_stiff[[idof1, idof2]] = _gqi2(&ur_g1.view(), &all_fdof_grad.index_axis(Axis(0), idof2), weights);
            bzz_mass[[idof1, idof2]] = _gqi(&er_f1z.view(), &all_fdof_z.index_axis(Axis(0), idof2), weights);
        }
    }
    let k02 = k0.powi(2);
    let mut a = Array2::<c64>::from_elem((n, n), zero);
    let mut b = Array2::<c64>::from_elem((n, n), zero);
    a.slice_mut(s![..ndof, ..ndof]).assign(&(&att_stiff - &att_mass.map(|v| v * k02)));
    b.slice_mut(s![..ndof, ..ndof]).assign(&btt);
    b.slice_mut(s![ndof.., ..ndof]).assign(&btz.t());
    b.slice_mut(s![..ndof, ndof..]).assign(&btz);
    b.slice_mut(s![ndof.., ndof..]).assign(&(&bzz_stiff - &bzz_mass.map(|v| v * k02)));
    let area = area.abs();
    (a * area, b * area)
}

type CooC = (Array1<c64>, Array1<c64>, Array1<i64>, Array1<i64>);

/// Assembles the COO (dataE, dataB, rows, cols) entries of the generalized eigenproblem over all triangles.
#[allow(clippy::too_many_arguments)]
pub fn _matrix_builder(
    nodes: &ArrayView2<f64>,
    tris: &ArrayView2<i64>,
    _edges: &ArrayView2<i64>,
    _tri_to_edge: &ArrayView2<i64>,
    tri_to_field: &ArrayView2<i64>,
    ur: &ArrayView3<c64>,
    er: &ArrayView3<c64>,
    k0: f64,
    dofcodes: &ArrayView1<i64>,
) -> CooC {
    let codes = dofcodes.to_vec();
    let n = codes.len() + 6;
    let nsq = n * n;
    let nnz = tris.shape()[1] * nsq;
    let mut rows = vec![0i64; nnz];
    let mut cols = vec![0i64; nnz];
    let mut data_e = vec![c64::new(0.0, 0.0); nnz];
    let mut data_b = vec![c64::new(0.0, 0.0); nnz];
    rows.par_chunks_mut(nsq)
        .zip(cols.par_chunks_mut(nsq))
        .zip(data_e.par_chunks_mut(nsq))
        .zip(data_b.par_chunks_mut(nsq))
        .enumerate()
        .for_each(|(itri, (((r, c), de), db))| {
            let urt = mat3_at(ur, itri);
            let ert = mat3_at(er, itri);
            let tri_nodes = Array2::from_shape_fn((nodes.shape()[0], 3), |(rr, v)| nodes[[rr, tris[[v, itri]] as usize]]);
            let (esub, bsub) = generalized_matrix_gq(&tri_nodes.view(), &matinv(&urt), &ert, k0, &codes);
            let indices: Vec<i64> = (0..n).map(|f| tri_to_field[[f, itri]]).collect();
            write_block_rowcol(&indices, r, c);
            for (k, (e, b)) in esub.iter().zip(bsub.iter()).enumerate() {
                de[k] = *e;
                db[k] = *b;
            }
        });
    (Array1::from(data_e), Array1::from(data_b), Array1::from(rows), Array1::from(cols))
}

#[pyfunction]
#[pyo3(name = "_matrix_builder")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_matrix_builder<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    tri_to_edge: PyReadonlyArray2<i64>,
    tri_to_field: PyReadonlyArray2<i64>,
    ur: PyReadonlyArray3<c64>,
    er: PyReadonlyArray3<c64>,
    k0: f64,
    dofcodes: PyReadonlyArray1<i64>,
) -> (
    Bound<'py, PyArray1<c64>>,
    Bound<'py, PyArray1<c64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
) {
    let (no, tr, ed, tte, ttf) = (
        nodes.as_array(),
        tris.as_array(),
        edges.as_array(),
        tri_to_edge.as_array(),
        tri_to_field.as_array(),
    );
    let (u, e, dc) = (ur.as_array(), er.as_array(), dofcodes.as_array());
    let r = py.allow_threads(|| _matrix_builder(&no, &tr, &ed, &tte, &ttf, &u, &e, k0, &dc));
    (
        r.0.into_pyarray_bound(py),
        r.1.into_pyarray_bound(py),
        r.2.into_pyarray_bound(py),
        r.3.into_pyarray_bound(py),
    )
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_matrix_builder, m)?)?;
    Ok(())
}
