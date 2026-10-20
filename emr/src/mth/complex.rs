//! Complex valued vector and matrix functions.
use super::float::M3;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use num_complex::Complex64 as c64;

pub type C3 = [c64; 3];
pub type CM3 = [[c64; 3]; 3];

/// Complex dot product (no conjugation) of two 3D vectors.
pub fn dot_c(a: &C3, b: &C3) -> c64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Complex cross product of two 3D vectors.
pub fn cross_c(a: &C3, b: &C3) -> C3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Reads a 3x3 complex matrix from the leading two axes of a (3, 3, N) array at slice n.
pub fn mat3_at(a: &ndarray::ArrayView3<c64>, n: usize) -> CM3 {
    let mut m = [[c64::new(0.0, 0.0); 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = a[[r, c, n]];
        }
    }
    m
}

/// Inverse of a complex 3x3 matrix with a shortcut for diagonal matrices.
pub fn matinv(m: &CM3) -> CM3 {
    let zero = c64::new(0.0, 0.0);
    let one = c64::new(1.0, 0.0);
    let mut out = [[zero; 3]; 3];
    if m[0][1] == zero && m[0][2] == zero && m[1][0] == zero && m[1][2] == zero && m[2][0] == zero && m[2][1] == zero {
        out[0][0] = one / m[0][0];
        out[1][1] = one / m[1][1];
        out[2][2] = one / m[2][2];
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

/// Writes Mat @ Vec for a complex 3x3 matrix and a (3, N) array into Vout.
pub fn matmul_inplace(mat: &CM3, vec: &ArrayView2<c64>, vout: &mut ArrayViewMut2<c64>) {
    for n in 0..vec.shape()[1] {
        let (v0, v1, v2) = (vec[[0, n]], vec[[1, n]], vec[[2, n]]);
        vout[[0, n]] = mat[0][0] * v0 + mat[0][1] * v1 + mat[0][2] * v2;
        vout[[1, n]] = mat[1][0] * v0 + mat[1][1] * v1 + mat[1][2] * v2;
        vout[[2, n]] = mat[2][0] * v0 + mat[2][1] * v1 + mat[2][2] * v2;
    }
}

/// Complex 2x2 matrix times a (2, N) array.
pub fn matmul_2d(a: &[[c64; 2]; 2], b: &ArrayView2<c64>) -> Array2<c64> {
    let n = b.shape()[1];
    let mut out = Array2::<c64>::zeros((2, n));
    for p in 0..n {
        out[[0, p]] = a[0][0] * b[[0, p]] + a[0][1] * b[[1, p]];
        out[[1, p]] = a[1][0] * b[[0, p]] + a[1][1] * b[[1, p]];
    }
    out
}

/// Real 3x3 basis matrix applied to every column of a complex (3, N) array.
pub fn optim_matmul_c(b: &M3, data: &ArrayView2<c64>) -> Array2<c64> {
    let mut out = Array2::<c64>::zeros(data.raw_dim());
    for n in 0..data.shape()[1] {
        for r in 0..3 {
            out[[r, n]] = data[[0, n]] * b[r][0] + data[[1, n]] * b[r][1] + data[[2, n]] * b[r][2];
        }
    }
    out
}

/// Column-wise cross product of a 3-vector a with every column of a (3, N) array b.
pub fn cross_c_arry(a: &C3, b: &ArrayView2<c64>) -> Array2<c64> {
    let n = b.shape()[1];
    let mut out = Array2::<c64>::zeros((3, n));
    for p in 0..n {
        out[[0, p]] = a[1] * b[[2, p]] - a[2] * b[[1, p]];
        out[[1, p]] = a[2] * b[[0, p]] - a[0] * b[[2, p]];
        out[[2, p]] = a[0] * b[[1, p]] - a[1] * b[[0, p]];
    }
    out
}

/// Column-wise dot product of a 3-vector a with every column of a (3, N) array b.
pub fn dot_c_arry(a: &C3, b: &ArrayView2<c64>) -> Array1<c64> {
    Array1::from_iter((0..b.shape()[1]).map(|p| a[0] * b[[0, p]] + a[1] * b[[1, p]] + a[2] * b[[2, p]]))
}

/// Sum over i of v1[i] * v2[i] * w[i].
pub fn _gqi(v1: &ArrayView1<c64>, v2: &ArrayView1<c64>, w: &[f64]) -> c64 {
    let mut acc = c64::new(0.0, 0.0);
    for i in 0..w.len() {
        acc += v1[i] * v2[i] * w[i];
    }
    acc
}

/// Sum over points of w[p] * (v1[:, p] . v2[:, p]).
pub fn _gqi2(v1: &ArrayView2<c64>, v2: &ArrayView2<c64>, w: &[f64]) -> c64 {
    let mut acc = c64::new(0.0, 0.0);
    for p in 0..w.len() {
        let mut s = c64::new(0.0, 0.0);
        for r in 0..v1.shape()[0] {
            s += v1[[r, p]] * v2[[r, p]];
        }
        acc += s * w[p];
    }
    acc
}
