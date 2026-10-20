//! Real valued vector, matrix and simplex geometry functions.
use ndarray::{Array2, ArrayView1, ArrayView2};

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

/// Returns column `i` of a (3, N) array as a 3-vector.
pub fn col3(a: &ArrayView2<f64>, i: usize) -> V3 {
    [a[[0, i]], a[[1, i]], a[[2, i]]]
}

/// Element-wise difference a - b.
pub fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two 3D vectors.
pub fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3D vectors.
pub fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Euclidean norm of a 3D vector.
pub fn norm(a: &V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Returns a / |a|.
pub fn normalize(a: &V3) -> V3 {
    let n = (a[0].powi(2) + a[1].powi(2) + a[2].powi(2)).powf(0.5);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Unit normal of triangle (n1, n2, n3) oriented along inward_normal.
pub fn calc_inward_normal(n1: &V3, n2: &V3, n3: &V3, inward_normal: &V3) -> V3 {
    let n = cross(&sub(n2, n1), &sub(n3, n1));
    let l = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
    let n = [n[0] / l, n[1] / l, n[2] / l];
    let sgn = if dot(&n, inward_normal) < 0.0 { -1.0 } else { 1.0 };
    [n[0] * sgn, n[1] * sgn, n[2] * sgn]
}

/// Area of the triangle spanned by three nodes.
pub fn calc_area(x1: &V3, x2: &V3, x3: &V3) -> f64 {
    let av = cross(&sub(x2, x1), &sub(x3, x1));
    (av[0].powi(2) + av[1].powi(2) + av[2].powi(2)).sqrt() / 2.0
}

/// Volume of the tetrahedron spanned by four nodes.
pub fn calc_volume(x1: &V3, x2: &V3, x3: &V3, x4: &V3) -> f64 {
    let (a_x, a_y, a_z) = (x1[0] - x4[0], x1[1] - x4[1], x1[2] - x4[2]);
    let (b_x, b_y, b_z) = (x2[0] - x4[0], x2[1] - x4[1], x2[2] - x4[2]);
    let (c_x, c_y, c_z) = (x3[0] - x4[0], x3[1] - x4[1], x3[2] - x4[2]);
    let vol = a_x * (b_y * c_z - b_z * c_y) - a_y * (b_x * c_z - b_z * c_x) + a_z * (b_x * c_y - b_y * c_x);
    vol.abs() / 6.0
}

/// Inverse of a real 3x3 matrix with a shortcut for diagonal matrices.
pub fn matinv_r(m: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    if m[0][1] == 0.0 && m[0][2] == 0.0 && m[1][0] == 0.0 && m[1][2] == 0.0 && m[2][0] == 0.0 && m[2][1] == 0.0 {
        out[0][0] = 1.0 / m[0][0];
        out[1][1] = 1.0 / m[1][1];
        out[2][2] = 1.0 / m[2][2];
        return out;
    }
    let det = m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] - m[0][1] * m[1][0] * m[2][2]
        + m[0][1] * m[1][2] * m[2][0]
        + m[0][2] * m[1][0] * m[2][1]
        - m[0][2] * m[1][1] * m[2][0];
    out[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    out[0][1] = -m[0][1] * m[2][2] + m[0][2] * m[2][1];
    out[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    out[1][0] = -m[1][0] * m[2][2] + m[1][2] * m[2][0];
    out[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    out[1][2] = -m[0][0] * m[1][2] + m[0][2] * m[1][0];
    out[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    out[2][1] = -m[0][0] * m[2][1] + m[0][1] * m[2][0];
    out[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    for row in out.iter_mut() {
        for v in row.iter_mut() {
            *v /= det;
        }
    }
    out
}

/// Determinant of a real 3x3 matrix.
pub fn det3(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Applies the 3x3 basis matrix B to every column of a (3, N) array.
pub fn optim_matmul(b: &M3, data: &ArrayView2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(data.raw_dim());
    for n in 0..data.shape()[1] {
        for r in 0..3 {
            out[[r, n]] = b[r][0] * data[[0, n]] + b[r][1] * data[[1, n]] + b[r][2] * data[[2, n]];
        }
    }
    out
}

/// Applies the 3x3 basis matrix B to a single 3-vector.
pub fn optim_matmul_vec(b: &M3, data: &V3) -> V3 {
    [
        b[0][0] * data[0] + b[0][1] * data[1] + b[0][2] * data[2],
        b[1][0] * data[0] + b[1][1] * data[1] + b[1][2] * data[2],
        b[2][0] * data[0] + b[2][1] * data[1] + b[2][2] * data[2],
    ]
}

/// Pairwise 3D distance matrix between points (xs, ys, zs).
pub fn compute_distances(xs: &ArrayView1<f64>, ys: &ArrayView1<f64>, zs: &ArrayView1<f64>) -> Array2<f64> {
    let n = xs.len();
    let mut ds = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let d = ((xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2) + (zs[i] - zs[j]).powi(2)).sqrt();
            ds[[i, j]] = d;
            ds[[j, i]] = d;
        }
    }
    ds
}

/// Pairwise 2D distance matrix between points (xs, ys).
pub fn compute_distances_2d(xs: &ArrayView1<f64>, ys: &ArrayView1<f64>) -> Array2<f64> {
    let n = xs.len();
    let mut ds = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let d = ((xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2)).sqrt();
            ds[[i, j]] = d;
            ds[[j, i]] = d;
        }
    }
    ds
}

/// Barycentric coefficients (a, b, c, d) of the four tet vertices (unnormalised) and the tet volume.
pub fn tet_coefficients(xs: &[f64; 4], ys: &[f64; 4], zs: &[f64; 4]) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4], f64) {
    let [x1, x2, x3, x4] = *xs;
    let [y1, y2, y3, y4] = *ys;
    let [z1, z2, z3, z4] = *zs;
    let v = compute_volume(xs, ys, zs);
    let aas = [
        x2 * y3 * z4 - x2 * y4 * z3 - x3 * y2 * z4 + x3 * y4 * z2 + x4 * y2 * z3 - x4 * y3 * z2,
        -x1 * y3 * z4 + x1 * y4 * z3 + x3 * y1 * z4 - x3 * y4 * z1 - x4 * y1 * z3 + x4 * y3 * z1,
        x1 * y2 * z4 - x1 * y4 * z2 - x2 * y1 * z4 + x2 * y4 * z1 + x4 * y1 * z2 - x4 * y2 * z1,
        -x1 * y2 * z3 + x1 * y3 * z2 + x2 * y1 * z3 - x2 * y3 * z1 - x3 * y1 * z2 + x3 * y2 * z1,
    ];
    let bbs = [
        -y2 * z3 + y2 * z4 + y3 * z2 - y3 * z4 - y4 * z2 + y4 * z3,
        y1 * z3 - y1 * z4 - y3 * z1 + y3 * z4 + y4 * z1 - y4 * z3,
        -y1 * z2 + y1 * z4 + y2 * z1 - y2 * z4 - y4 * z1 + y4 * z2,
        y1 * z2 - y1 * z3 - y2 * z1 + y2 * z3 + y3 * z1 - y3 * z2,
    ];
    let ccs = [
        x2 * z3 - x2 * z4 - x3 * z2 + x3 * z4 + x4 * z2 - x4 * z3,
        -x1 * z3 + x1 * z4 + x3 * z1 - x3 * z4 - x4 * z1 + x4 * z3,
        x1 * z2 - x1 * z4 - x2 * z1 + x2 * z4 + x4 * z1 - x4 * z2,
        -x1 * z2 + x1 * z3 + x2 * z1 - x2 * z3 - x3 * z1 + x3 * z2,
    ];
    let dds = [
        -x2 * y3 + x2 * y4 + x3 * y2 - x3 * y4 - x4 * y2 + x4 * y3,
        x1 * y3 - x1 * y4 - x3 * y1 + x3 * y4 + x4 * y1 - x4 * y3,
        -x1 * y2 + x1 * y4 + x2 * y1 - x2 * y4 - x4 * y1 + x4 * y2,
        x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2,
    ];
    (aas, bbs, ccs, dds, v)
}

/// Volume of a tetrahedron from its vertex coordinate arrays.
pub fn compute_volume(xs: &[f64; 4], ys: &[f64; 4], zs: &[f64; 4]) -> f64 {
    let [x1, x2, x3, x4] = *xs;
    let [y1, y2, y3, y4] = *ys;
    let [z1, z2, z3, z4] = *zs;
    (-x1 * y2 * z3 / 6.0 + x1 * y2 * z4 / 6.0 + x1 * y3 * z2 / 6.0 - x1 * y3 * z4 / 6.0 - x1 * y4 * z2 / 6.0
        + x1 * y4 * z3 / 6.0
        + x2 * y1 * z3 / 6.0
        - x2 * y1 * z4 / 6.0
        - x2 * y3 * z1 / 6.0
        + x2 * y3 * z4 / 6.0
        + x2 * y4 * z1 / 6.0
        - x2 * y4 * z3 / 6.0
        - x3 * y1 * z2 / 6.0
        + x3 * y1 * z4 / 6.0
        + x3 * y2 * z1 / 6.0
        - x3 * y2 * z4 / 6.0
        - x3 * y4 * z1 / 6.0
        + x3 * y4 * z2 / 6.0
        + x4 * y1 * z2 / 6.0
        - x4 * y1 * z3 / 6.0
        - x4 * y2 * z1 / 6.0
        + x4 * y2 * z3 / 6.0
        + x4 * y3 * z1 / 6.0
        - x4 * y3 * z2 / 6.0)
        .abs()
}

/// Barycentric coefficients of a 2D triangle normalised by twice the signed area, plus the area.
pub fn tri_coefficients(vxs: &[f64; 3], vys: &[f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
    let [x1, x2, x3] = *vxs;
    let [y1, y2, y3] = *vys;
    let (a1, a2, a3) = (x2 * y3 - y2 * x3, x3 * y1 - y3 * x1, x1 * y2 - y1 * x2);
    let (b1, b2, b3) = (y2 - y3, y3 - y1, y1 - y2);
    let (c1, c2, c3) = (x3 - x2, x1 - x3, x2 - x1);
    let sa = 0.5 * (b1 * c2 - b2 * c1);
    let s = 2.0 * sa;
    (
        [a1 / s, a2 / s, a3 / s],
        [b1 / s, b2 / s, b3 / s],
        [c1 / s, c2 / s, c3 / s],
        sa.abs(),
    )
}

/// Unnormalised barycentric coefficients of a 2D triangle multiplied by the orientation sign, plus the area.
pub fn tri_coefficients_signed(vxs: &[f64; 3], vys: &[f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
    let [x1, x2, x3] = *vxs;
    let [y1, y2, y3] = *vys;
    let (a1, a2, a3) = (x2 * y3 - y2 * x3, x3 * y1 - y3 * x1, x1 * y2 - y1 * x2);
    let (b1, b2, b3) = (y2 - y3, y3 - y1, y1 - y2);
    let (c1, c2, c3) = (x3 - x2, x1 - x3, x2 - x1);
    let sa = 0.5 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1));
    let sign = sign(sa);
    (
        [a1 * sign, a2 * sign, a3 * sign],
        [b1 * sign, b2 * sign, b3 * sign],
        [c1 * sign, c2 * sign, c3 * sign],
        sa.abs(),
    )
}

/// numpy style sign (0 for 0).
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Local orthonormal triangle basis (rows xhat, yhat, zhat) and the vertices expressed in it.
pub fn construct_local_vertices(glob_vertices: &ArrayView2<f64>) -> (M3, Array2<f64>) {
    let origin = col3(glob_vertices, 0);
    let edge_1 = sub(&col3(glob_vertices, 1), &origin);
    let edge_2 = sub(&col3(glob_vertices, 2), &origin);
    let zhat = normalize(&cross(&edge_1, &edge_2));
    let xhat = normalize(&edge_1);
    let yhat = normalize(&cross(&zhat, &xhat));
    let basis = [xhat, yhat, zhat];
    let mut shifted = glob_vertices.to_owned();
    for n in 0..shifted.shape()[1] {
        for r in 0..3 {
            shifted[[r, n]] -= origin[r];
        }
    }
    let local = optim_matmul(&basis, &shifted.view());
    (basis, local)
}

/// Inverse of a 2x2 matrix [[a, b], [c, d]].
pub fn inv2(m: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]
}

/// Inverse of a general (non-diagonal shortcut) 3x3 matrix.
pub fn inv3(m: &M3) -> M3 {
    let det = det3(m);
    let mut out = [[0.0; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
    out
}
