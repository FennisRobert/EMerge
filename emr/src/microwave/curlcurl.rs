//! Nedelec-2 curl-curl stiffness and mass matrix assembly over tetrahedra.
use crate::compiled::ccbf::{_eval_curl_f_3d, _eval_f_3d, parse_dofcode};
use crate::mth::complex::{mat3_at, matinv, matmul_inplace, CM3};
use crate::mth::float::tet_coefficients;
use crate::mth::index::{local_tet_to_edgeid, local_tet_to_triid};
use crate::mth::quadrature::keast;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Tet vertex coordinates (3, 4) of tet itet.
pub fn tet_vertices(nodes: &ArrayView2<f64>, tets: &ArrayView2<i64>, itet: usize) -> Array2<f64> {
    Array2::from_shape_fn((3, 4), |(r, v)| nodes[[r, tets[[v, itet]] as usize]])
}

/// Normalised barycentric coefficient matrix (4, 4) and volume of a tet given its (3, 4) vertices.
pub fn bary_coeff(tet_vertices: &ArrayView2<f64>) -> (Array2<f64>, f64) {
    let txs = [0, 1, 2, 3].map(|v| tet_vertices[[0, v]]);
    let tys = [0, 1, 2, 3].map(|v| tet_vertices[[1, v]]);
    let tzs = [0, 1, 2, 3].map(|v| tet_vertices[[2, v]]);
    let (a, b, c, d, v) = tet_coefficients(&txs, &tys, &tzs);
    let rows = [a, b, c, d];
    (Array2::from_shape_fn((4, 4), |(r, m)| rows[r][m] / (6.0 * v)), v)
}

/// Global coordinates (3, N) of the barycentric quadrature points dpts (rows 1..5) in the tet.
pub fn quad_coords(tet_vertices: &ArrayView2<f64>, dpts: &ArrayView2<f64>) -> Array2<f64> {
    Array2::from_shape_fn((3, dpts.ncols()), |(r, q)| {
        tet_vertices[[r, 0]] * dpts[[1, q]]
            + tet_vertices[[r, 1]] * dpts[[2, q]]
            + tet_vertices[[r, 2]] * dpts[[3, q]]
            + tet_vertices[[r, 3]] * dpts[[4, q]]
    })
}

/// Local (i, j, k) vertex indices of a tet DoF of the given type and index.
pub fn tet_dof_ijk(i_type: i64, i_index: i64, local_edge_map: &Array2<i64>, local_tri_map: &Array2<i64>) -> (usize, usize, usize) {
    let ii = i_index as usize;
    if i_type == 0 {
        (local_edge_map[[0, ii]] as usize, local_edge_map[[1, ii]] as usize, 0)
    } else {
        (
            local_tri_map[[0, ii]] as usize,
            local_tri_map[[1, ii]] as usize,
            local_tri_map[[2, ii]] as usize,
        )
    }
}

/// Nedelec-2 element stiffness ∫(Ms ∇×N_i)·∇×N_j dV and mass ∫(Mm N_i)·N_j dV matrices of one tet.
#[allow(clippy::too_many_arguments)]
pub fn ned2_tet_stiff_mass(
    tet_vertices: &ArrayView2<f64>,
    local_edge_map: &Array2<i64>,
    local_tri_map: &Array2<i64>,
    ms: &CM3,
    mm: &CM3,
    dofcodes: &[i64],
    typearry: &[i64],
    indexarry: &[i64],
) -> (Array2<c64>, Array2<c64>) {
    let ndof = dofcodes.len();
    let (coeff, v) = bary_coeff(tet_vertices);
    let dpts = keast(5);
    let coords = quad_coords(tet_vertices, &dpts);
    let weights = dpts.row(0).to_slice().unwrap();
    let n = coords.shape()[1];

    let mut all_fdof = Array3::<c64>::zeros((ndof, 3, n));
    let mut all_fdof_curl = Array3::<c64>::zeros((ndof, 3, n));
    let mut all_ms_fdof_curl_w = Array3::<c64>::zeros((ndof, 3, n));
    let mut all_mm_fdof_w = Array3::<c64>::zeros((ndof, 3, n));
    for idof in 0..ndof {
        let (i1, j1, k1) = tet_dof_ijk(typearry[idof], indexarry[idof], local_edge_map, local_tri_map);
        _eval_f_3d(
            &coeff.view(),
            &coords.view(),
            i1,
            j1,
            k1,
            dofcodes[idof],
            &mut all_fdof.index_axis_mut(ndarray::Axis(0), idof),
        );
        _eval_curl_f_3d(
            &coeff.view(),
            &coords.view(),
            i1,
            j1,
            k1,
            dofcodes[idof],
            &mut all_fdof_curl.index_axis_mut(ndarray::Axis(0), idof),
        );
        matmul_inplace(
            ms,
            &all_fdof_curl.index_axis(ndarray::Axis(0), idof),
            &mut all_ms_fdof_curl_w.index_axis_mut(ndarray::Axis(0), idof),
        );
        matmul_inplace(
            mm,
            &all_fdof.index_axis(ndarray::Axis(0), idof),
            &mut all_mm_fdof_w.index_axis_mut(ndarray::Axis(0), idof),
        );
        for c in 0..3 {
            for q in 0..n {
                all_ms_fdof_curl_w[[idof, c, q]] *= weights[q];
                all_mm_fdof_w[[idof, c, q]] *= weights[q];
            }
        }
    }

    let mut mat_stiff = Array2::<c64>::zeros((ndof, ndof));
    let mut mat_mass = Array2::<c64>::zeros((ndof, ndof));
    for idof1 in 0..ndof {
        for idof2 in idof1..ndof {
            let mut acc_s = c64::new(0.0, 0.0);
            let mut acc_m = c64::new(0.0, 0.0);
            for q in 0..n {
                acc_s += all_ms_fdof_curl_w[[idof1, 0, q]] * all_fdof_curl[[idof2, 0, q]]
                    + all_ms_fdof_curl_w[[idof1, 1, q]] * all_fdof_curl[[idof2, 1, q]]
                    + all_ms_fdof_curl_w[[idof1, 2, q]] * all_fdof_curl[[idof2, 2, q]];
                acc_m += all_mm_fdof_w[[idof1, 0, q]] * all_fdof[[idof2, 0, q]]
                    + all_mm_fdof_w[[idof1, 1, q]] * all_fdof[[idof2, 1, q]]
                    + all_mm_fdof_w[[idof1, 2, q]] * all_fdof[[idof2, 2, q]];
            }
            mat_stiff[[idof1, idof2]] = acc_s;
            mat_mass[[idof1, idof2]] = acc_m;
            mat_stiff[[idof2, idof1]] = acc_s;
            mat_mass[[idof2, idof1]] = acc_m;
        }
    }
    (mat_stiff * v, mat_mass * v)
}

/// Writes the COO row/col indices of a dense (n, n) element block with global indices into row-major slots.
pub fn write_block_rowcol(indices: &[i64], rows: &mut [i64], cols: &mut [i64]) {
    let n = indices.len();
    for a in 0..n {
        for b in 0..n {
            rows[a * n + b] = indices[a];
            cols[a * n + b] = indices[b];
        }
    }
}

type CooC = (Array1<c64>, Array1<c64>, Array1<i64>, Array1<i64>);

/// Assembles the COO (dataE, dataB, rows, cols) curl-curl and mass entries of all tets in tetids.
#[allow(clippy::too_many_arguments)]
pub fn _matrix_builder(
    nodes: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    tris: &ArrayView2<i64>,
    edges: &ArrayView2<i64>,
    tet_to_field: &ArrayView2<i64>,
    tet_to_edge: &ArrayView2<i64>,
    tet_to_tri: &ArrayView2<i64>,
    ur: &ArrayView3<c64>,
    er: &ArrayView3<c64>,
    tetids: &ArrayView1<i64>,
    dofcodes: &ArrayView1<i64>,
) -> CooC {
    let codes = dofcodes.to_vec();
    let ndof = codes.len();
    let nsq = ndof * ndof;
    let nnz = tetids.len() * nsq;
    let (typearry, indexarry) = parse_dofcode(&codes);
    let mut rows = vec![0i64; nnz];
    let mut cols = vec![0i64; nnz];
    let mut data_e = vec![c64::new(0.0, 0.0); nnz];
    let mut data_b = vec![c64::new(0.0, 0.0); nnz];
    rows.par_chunks_mut(nsq.max(1))
        .zip(cols.par_chunks_mut(nsq.max(1)))
        .zip(data_e.par_chunks_mut(nsq.max(1)))
        .zip(data_b.par_chunks_mut(nsq.max(1)))
        .enumerate()
        .for_each(|(iassy, (((r, c), de), db))| {
            let itet = tetids[iassy] as usize;
            let urt = mat3_at(ur, itet);
            let ert = mat3_at(er, itet);
            let local_tri_map = local_tet_to_triid(tet_to_tri, tets, tris, itet);
            let local_edge_map = local_tet_to_edgeid(tet_to_edge, tets, edges, itet);
            let verts = tet_vertices(nodes, tets, itet);
            let (esub, bsub) = ned2_tet_stiff_mass(
                &verts.view(),
                &local_edge_map,
                &local_tri_map,
                &matinv(&urt),
                &ert,
                &codes,
                &typearry,
                &indexarry,
            );
            let indices: Vec<i64> = (0..ndof).map(|f| tet_to_field[[f, itet]]).collect();
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
    tets: PyReadonlyArray2<i64>,
    tris: PyReadonlyArray2<i64>,
    edges: PyReadonlyArray2<i64>,
    tet_to_field: PyReadonlyArray2<i64>,
    tet_to_edge: PyReadonlyArray2<i64>,
    tet_to_tri: PyReadonlyArray2<i64>,
    ur: PyReadonlyArray3<c64>,
    er: PyReadonlyArray3<c64>,
    tetids: PyReadonlyArray1<i64>,
    dofcodes: PyReadonlyArray1<i64>,
) -> (
    Bound<'py, PyArray1<c64>>,
    Bound<'py, PyArray1<c64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
) {
    let (no, te, tr, ed) = (nodes.as_array(), tets.as_array(), tris.as_array(), edges.as_array());
    let (tf, tte, ttt) = (tet_to_field.as_array(), tet_to_edge.as_array(), tet_to_tri.as_array());
    let (u, e, ti, dc) = (ur.as_array(), er.as_array(), tetids.as_array(), dofcodes.as_array());
    let r = py.allow_threads(|| _matrix_builder(&no, &te, &tr, &ed, &tf, &tte, &ttt, &u, &e, &ti, &dc));
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
