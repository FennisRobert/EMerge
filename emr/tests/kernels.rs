use emr::compiled::ccbf::{_eval_curl_f_3d, _eval_f_3d, parse_dofcode};
use emr::compiled::index_interp::index_interp;
use emr::compiled::legrange::{_ne, _nv};
use emr::mesh3d::{extract_edges_tris, find_columns, tet_to_edge_tri};
use emr::mth::complex::matinv;
use emr::mth::csc_cast::{precompute_csc_pattern, scatter_to_csc};
use emr::mth::float::{calc_volume, matinv_r, tet_coefficients};
use ndarray::{array, Array1, Array2};
use num_complex::Complex64 as c64;

const TET_X: [f64; 4] = [0.1, 1.3, 0.2, 0.4];
const TET_Y: [f64; 4] = [0.0, 0.2, 1.1, 0.3];
const TET_Z: [f64; 4] = [0.05, 0.1, 0.0, 0.9];

fn bary_coeff() -> Array2<f64> {
    let (a, b, c, d, v) = tet_coefficients(&TET_X, &TET_Y, &TET_Z);
    let rows = [a, b, c, d];
    Array2::from_shape_fn((4, 4), |(r, m)| rows[r][m] / (6.0 * v))
}

#[test]
fn parse_dofcode_counts_per_code() {
    let (t, i) = parse_dofcode(&[64, 64, 66, 128, 128, 129]);
    assert_eq!(t, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(i, vec![0, 1, 0, 0, 1, 0]);
}

#[test]
fn barycentric_coefficients_are_kronecker_at_vertices() {
    let coeff = bary_coeff();
    for v in 0..4 {
        for m in 0..4 {
            let l = coeff[[0, m]] + coeff[[1, m]] * TET_X[v] + coeff[[2, m]] * TET_Y[v] + coeff[[3, m]] * TET_Z[v];
            assert!((l - if v == m { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
    let v = calc_volume(
        &[TET_X[0], TET_Y[0], TET_Z[0]],
        &[TET_X[1], TET_Y[1], TET_Z[1]],
        &[TET_X[2], TET_Y[2], TET_Z[2]],
        &[TET_X[3], TET_Y[3], TET_Z[3]],
    );
    assert!((v - tet_coefficients(&TET_X, &TET_Y, &TET_Z).4).abs() < 1e-14);
}

#[test]
fn lagrange2_partition_of_unity() {
    let coeff = bary_coeff();
    let coords = array![[0.3, 0.5, 0.2], [0.2, 0.3, 0.4], [0.1, 0.2, 0.3]];
    let mut total = Array1::<c64>::zeros(3);
    for v in 0..4 {
        total = total + _nv(&coeff.view(), &coords.view(), v, v, 0);
    }
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        total = total + _ne(&coeff.view(), &coords.view(), i, j, 0);
    }
    assert!(total.iter().all(|s| (s.re - 1.0).abs() < 1e-12));
}

#[test]
fn nedelec_edge_function_has_constant_curl() {
    let coeff = bary_coeff();
    let coords = array![[0.3, 0.5], [0.2, 0.3], [0.1, 0.2]];
    let mut f = Array2::<c64>::zeros((3, 2));
    _eval_f_3d(&coeff.view(), &coords.view(), 0, 1, 0, 64, &mut f.view_mut());
    assert!(f.iter().any(|v| v.norm() > 0.0));
    _eval_curl_f_3d(&coeff.view(), &coords.view(), 0, 1, 0, 64, &mut f.view_mut());
    for r in 0..3 {
        assert!((f[[r, 0]] - f[[r, 1]]).norm() < 1e-12);
    }
}

#[test]
fn matrix_inverses() {
    let m = [[2.0, 0.5, 0.1], [0.3, 1.5, 0.2], [0.0, 0.4, 3.0]];
    let inv = matinv_r(&m);
    for r in 0..3 {
        for c in 0..3 {
            let s: f64 = (0..3).map(|k| m[r][k] * inv[k][c]).sum();
            assert!((s - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
    let mc = m.map(|row| row.map(|v| c64::new(v, 0.3 * v)));
    let invc = matinv(&mc);
    for r in 0..3 {
        let s: c64 = (0..3).map(|k| mc[r][k] * invc[k][r]).sum();
        assert!((s - c64::new(1.0, 0.0)).norm() < 1e-12);
    }
}

#[test]
fn two_tet_topology() {
    let tets = array![[0i64, 1], [1, 2], [2, 3], [3, 4]];
    let (edges, tris) = extract_edges_tris(&tets.view());
    assert_eq!(edges.shape(), &[2, 9]);
    assert_eq!(tris.shape(), &[3, 7]);
    let (t2e, t2t) = tet_to_edge_tri(&tets.view(), &edges.view(), &tris.view());
    assert!(t2e.iter().all(|&e| e >= 0) && t2t.iter().all(|&t| t >= 0));
    let shared = find_columns(&tris.view(), &array![[3i64], [1], [2]].view());
    assert_eq!(t2t.column(0).iter().filter(|&&t| t == shared[0]).count(), 1);
    assert_eq!(t2t.column(1).iter().filter(|&&t| t == shared[0]).count(), 1);
    assert_eq!(find_columns(&tris.view(), &array![[0i64], [1], [4]].view())[0], -1234);
}

#[test]
fn csc_pattern_sums_duplicates() {
    let rows = array![0i64, 2, 0, 1, 2, 0];
    let cols = array![0i64, 0, 0, 1, 1, 1];
    let (indptr, indices, target, col_offsets, perm) = precompute_csc_pattern(&rows.view(), &cols.view(), 2);
    assert_eq!(indptr.to_vec(), vec![0, 2, 5]);
    assert_eq!(indices.to_vec(), vec![0, 2, 0, 1, 2]);
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut sorted = Array1::<f64>::zeros(6);
    let mut out = Array1::<f64>::zeros(5);
    scatter_to_csc(
        &data.view(),
        &perm.view(),
        &target.view(),
        &col_offsets.view(),
        &mut sorted.view_mut(),
        &mut out.view_mut(),
    );
    assert_eq!(out.to_vec(), vec![4.0, 2.0, 6.0, 4.0, 5.0]);
}

#[test]
fn index_interp_locates_points() {
    let nodes = array![[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let tets = array![[0i64], [1], [2], [3]];
    let coords = array![[0.1, 0.9, 0.2], [0.1, 0.9, 0.2], [0.1, 0.9, 0.2]];
    let found = index_interp(&coords.view(), &tets.view(), &nodes.view(), &array![0i64].view(), 20);
    assert_eq!(found.to_vec(), vec![0, -1, 0]);
}

#[test]
fn quadrature_rules_are_normalised() {
    use emr::mth::quadrature::{keast, strang};
    for p in 1..=10 {
        let r = strang(p);
        assert_eq!(r.nrows(), 4);
        assert!((r.row(0).sum() - 1.0).abs() < 1e-12, "strang({p})");
    }
    for p in 1..=9 {
        let r = keast(p);
        assert_eq!(r.nrows(), 5);
        assert!((r.row(0).sum() - 1.0).abs() < 1e-12, "keast({p})");
    }
}
