//! Second order Lagrange basis functions (vertex and edge) and their gradients/curls in 2D and 3D.
#![allow(unused_variables)]
use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64 as c64;

fn l3(coeff: &ArrayView2<f64>, i: usize, x: f64, y: f64, z: f64) -> f64 {
    coeff[[0, i]] + coeff[[1, i]] * x + coeff[[2, i]] * y + coeff[[3, i]] * z
}

fn l2(coeff: &ArrayView2<f64>, i: usize, x: f64, y: f64) -> f64 {
    coeff[[0, i]] + coeff[[1, i]] * x + coeff[[2, i]] * y
}

/// Vertex basis L_i(2L_i - 1) on a tetrahedron.
pub fn _nv(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array1<c64> {
    let (ai, bi, ci, di) = (coeff[[0, i]], coeff[[1, i]], coeff[[2, i]], coeff[[3, i]]);
    Array1::from_iter((0..coords.shape()[1]).map(|p| {
        let (xs, ys, zs) = (coords[[0, p]], coords[[1, p]], coords[[2, p]]);
        c64::new(
            (ai + bi * xs + ci * ys + di * zs) * (2.0 * ai + 2.0 * bi * xs + 2.0 * ci * ys + 2.0 * di * zs - 1.0),
            0.0,
        )
    }))
}

/// Edge basis 4 L_i L_j on a tetrahedron.
pub fn _ne(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array1<c64> {
    Array1::from_iter((0..coords.shape()[1]).map(|p| {
        let (xs, ys, zs) = (coords[[0, p]], coords[[1, p]], coords[[2, p]]);
        c64::new(4.0 * l3(coeff, i, xs, ys, zs) * l3(coeff, j, xs, ys, zs), 0.0)
    }))
}

/// Vertex basis L_i(2L_i - 1) on a triangle.
pub fn _nv_tri(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array1<c64> {
    let (ai, bi, ci) = (coeff[[0, i]], coeff[[1, i]], coeff[[2, i]]);
    Array1::from_iter((0..coords.shape()[1]).map(|p| {
        let (xs, ys) = (coords[[0, p]], coords[[1, p]]);
        c64::new((ai + bi * xs + ci * ys) * (2.0 * ai + 2.0 * bi * xs + 2.0 * ci * ys - 1.0), 0.0)
    }))
}

/// Edge basis 4 L_i L_j on a triangle.
pub fn _ne_tri(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array1<c64> {
    Array1::from_iter((0..coords.shape()[1]).map(|p| {
        let (xs, ys) = (coords[[0, p]], coords[[1, p]]);
        c64::new(4.0 * l2(coeff, i, xs, ys) * l2(coeff, j, xs, ys), 0.0)
    }))
}

/// Gradient of the tetrahedral vertex basis.
pub fn _nv_grad(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array2<c64> {
    let (ai, bi, ci, di) = (coeff[[0, i]], coeff[[1, i]], coeff[[2, i]], coeff[[3, i]]);
    let n = coords.shape()[1];
    let mut out = Array2::<c64>::zeros((3, n));
    for p in 0..n {
        let (xs, ys, zs) = (coords[[0, p]], coords[[1, p]], coords[[2, p]]);
        let q = 4.0 * ai + 4.0 * bi * xs + 4.0 * ci * ys + 4.0 * di * zs - 1.0;
        out[[0, p]] = c64::new(bi * q, 0.0);
        out[[1, p]] = c64::new(ci * q, 0.0);
        out[[2, p]] = c64::new(di * q, 0.0);
    }
    out
}

/// Gradient of the tetrahedral edge basis.
pub fn _ne_grad(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array2<c64> {
    let n = coords.shape()[1];
    let mut out = Array2::<c64>::zeros((3, n));
    for p in 0..n {
        let (xs, ys, zs) = (coords[[0, p]], coords[[1, p]], coords[[2, p]]);
        let (li, lj) = (l3(coeff, i, xs, ys, zs), l3(coeff, j, xs, ys, zs));
        for r in 0..3 {
            out[[r, p]] = c64::new(4.0 * coeff[[r + 1, i]] * lj + 4.0 * coeff[[r + 1, j]] * li, 0.0);
        }
    }
    out
}

/// Gradient of the triangular vertex basis.
pub fn _nv_grad_tri(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array2<c64> {
    let (ai, bi, ci) = (coeff[[0, i]], coeff[[1, i]], coeff[[2, i]]);
    let n = coords.shape()[1];
    let mut out = Array2::<c64>::zeros((2, n));
    for p in 0..n {
        let (xs, ys) = (coords[[0, p]], coords[[1, p]]);
        let q = 4.0 * ai + 4.0 * bi * xs + 4.0 * ci * ys - 1.0;
        out[[0, p]] = c64::new(bi * q, 0.0);
        out[[1, p]] = c64::new(ci * q, 0.0);
    }
    out
}

/// Gradient of the triangular edge basis.
pub fn _ne_grad_tri(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array2<c64> {
    let n = coords.shape()[1];
    let mut out = Array2::<c64>::zeros((2, n));
    for p in 0..n {
        let (xs, ys) = (coords[[0, p]], coords[[1, p]]);
        let (li, lj) = (l2(coeff, i, xs, ys), l2(coeff, j, xs, ys));
        out[[0, p]] = c64::new(4.0 * coeff[[1, i]] * lj + 4.0 * coeff[[1, j]] * li, 0.0);
        out[[1, p]] = c64::new(4.0 * coeff[[2, i]] * lj + 4.0 * coeff[[2, j]] * li, 0.0);
    }
    out
}

/// In-plane curl (z cross gradient) of the triangular vertex basis.
pub fn _nv_curl_tri(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array2<c64> {
    let (ai, bi, ci) = (coeff[[0, i]], coeff[[1, i]], coeff[[2, i]]);
    let n = coords.shape()[1];
    let mut out = Array2::<c64>::zeros((2, n));
    for p in 0..n {
        let (xs, ys) = (coords[[0, p]], coords[[1, p]]);
        out[[0, p]] = c64::new(ci * (4.0 * ai + 4.0 * bi * xs + 4.0 * ci * ys - 1.0), 0.0);
        out[[1, p]] = c64::new(bi * (-4.0 * ai - 4.0 * bi * xs - 4.0 * ci * ys + 1.0), 0.0);
    }
    out
}

/// In-plane curl (z cross gradient) of the triangular edge basis.
pub fn _ne_curl_tri(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize) -> Array2<c64> {
    let n = coords.shape()[1];
    let mut out = Array2::<c64>::zeros((2, n));
    for p in 0..n {
        let (xs, ys) = (coords[[0, p]], coords[[1, p]]);
        let (li, lj) = (l2(coeff, i, xs, ys), l2(coeff, j, xs, ys));
        out[[0, p]] = c64::new(4.0 * coeff[[2, i]] * lj + 4.0 * coeff[[2, j]] * li, 0.0);
        out[[1, p]] = c64::new(-4.0 * coeff[[1, i]] * lj - 4.0 * coeff[[1, j]] * li, 0.0);
    }
    out
}
