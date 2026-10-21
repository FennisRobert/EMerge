//! Nedelec-2 basis functions (field, curl, divergence, curl-curl) in 2D and 3D, dispatched by dofcode.
#![allow(
    unused_variables,
    unused_parens,
    clippy::too_many_arguments,
    clippy::neg_multiply,
    clippy::double_parens
)]
use ndarray::{ArrayView2, ArrayViewMut1, ArrayViewMut2};
use num_complex::Complex64 as c64;

pub const MASK_TYPE: i64 = 0b11000000;
pub const MASK_INDEX: i64 = 0b00111111;

/// Splits dofcodes into a type array (0=edge, 1=face) and the running index per basis function code.
pub fn parse_dofcode(dofcodes: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut typearray = vec![0i64; dofcodes.len()];
    let mut indexarray = vec![0i64; dofcodes.len()];
    let mut ne = [0i64; 64];
    let mut nf = [0i64; 64];
    for (i, &code) in dofcodes.iter().enumerate() {
        let idofcode = (code & 0b00111111) as usize;
        if (code & 0b11000000) == 64 {
            typearray[i] = 0;
            indexarray[i] = ne[idofcode];
            ne[idofcode] += 1;
        } else {
            typearray[i] = 1;
            indexarray[i] = nf[idofcode];
            nf[idofcode] += 1;
        }
    }
    (typearray, indexarray)
}

/// Returns the (basis function type, index) pair encoded in a dofcode.
pub fn get_type_index(number: i64) -> (i64, i64) {
    (number & MASK_TYPE, number & MASK_INDEX)
}

/// Basis function ne0_2d evaluated at coords.
pub fn _ne0_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        let bx = -_t0 * bi + _t1 * bj;
        let by = -_t0 * ci + _t1 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function ne1_2d evaluated at coords.
pub fn _ne1_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        let bx = _t0 * bi + _t1 * bj;
        let by = _t0 * ci + _t1 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function ne2_2d evaluated at coords.
pub fn _ne2_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = bj * xs;
        let _t1 = cj * ys;
        let _t2 = ai + bi * xs + ci * ys;
        let _t3 = -_t0 - _t1 + _t2 - aj;
        let _t4 = _t0 + _t1 + aj;
        let bx = -_t3 * (-_t2 * bj + _t4 * bi);
        let by = -_t3 * (-_t2 * cj + _t4 * ci);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function ne3_2d evaluated at coords.
pub fn _ne3_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = bj * xs;
        let _t1 = cj * ys;
        let _t2 = _t0 + _t1 + aj;
        let _t3 = ai + bi * xs + ci * ys;
        let _t4 = _t2 * _t3;
        let _t5 = -_t0 - _t1 + _t3 - aj;
        let _t6 = _t2 * _t5;
        let _t7 = _t3 * _t5;
        let bx = _t4 * (bi - bj) + _t6 * bi + _t7 * bj;
        let by = _t4 * (ci - cj) + _t6 * ci + _t7 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function ne4_2d evaluated at coords.
pub fn _ne4_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ai + bi * xs + ci * ys;
        let _t1 = aj + bj * xs + cj * ys;
        let bx = -_t0 * (-_t0 * bj + _t1 * bi);
        let by = -_t0 * (-_t0 * cj + _t1 * ci);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function ne5_2d evaluated at coords.
pub fn _ne5_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        let bx = -_t0 * (_t0 * bi - _t1 * bj);
        let by = -_t0 * (_t0 * ci - _t1 * cj);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function curl_ne0_2d evaluated at coords.
pub fn _curl_ne0_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        out[p] = c64::new((2.0 * bi * cj - 2.0 * bj * ci), 0.0);
    }
}

/// Basis function curl_ne1_2d evaluated at coords.
pub fn _curl_ne1_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function curl_ne2_2d evaluated at coords.
pub fn _curl_ne2_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = bj * xs;
        let _t1 = cj * ys;
        let _t2 = _t0 + _t1 + aj;
        let _t3 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (-(bi - bj) * (_t2 * ci - _t3 * cj) + (ci - cj) * (_t2 * bi - _t3 * bj) + 2.0 * (bi * cj - bj * ci) * (-_t0 - _t1 + _t3 - aj)),
            0.0,
        );
    }
}

/// Basis function curl_ne3_2d evaluated at coords.
pub fn _curl_ne3_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function curl_ne4_2d evaluated at coords.
pub fn _curl_ne4_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ai + bi * xs + ci * ys;
        let _t1 = aj + bj * xs + cj * ys;
        out[p] = c64::new(
            (2.0 * _t0 * (bi * cj - bj * ci) - bi * (-_t0 * cj + _t1 * ci) + ci * (-_t0 * bj + _t1 * bi)),
            0.0,
        );
    }
}

/// Basis function curl_ne5_2d evaluated at coords.
pub fn _curl_ne5_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (2.0 * _t0 * (bi * cj - bj * ci) - bj * (_t0 * ci - _t1 * cj) + cj * (_t0 * bi - _t1 * bj)),
            0.0,
        );
    }
}

/// Basis function div_ne0_2d evaluated at coords.
pub fn _div_ne0_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function div_ne1_2d evaluated at coords.
pub fn _div_ne1_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        out[p] = c64::new((2.0 * bi * bj + 2.0 * ci * cj), 0.0);
    }
}

/// Basis function div_ne2_2d evaluated at coords.
pub fn _div_ne2_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new((-(bi - bj) * (_t0 * bi - _t1 * bj) - (ci - cj) * (_t0 * ci - _t1 * cj)), 0.0);
    }
}

/// Basis function div_ne3_2d evaluated at coords.
pub fn _div_ne3_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = bi - bj;
        let _t1 = bj * xs;
        let _t2 = cj * ys;
        let _t3 = _t1 + _t2 + aj;
        let _t4 = ai + bi * xs + ci * ys;
        let _t5 = ci - cj;
        let _t6 = -_t1 - _t2 + _t4 - aj;
        out[p] = c64::new(
            (2.0 * _t0 * _t3 * bi
                + 2.0 * _t0 * _t4 * bj
                + 2.0 * _t3 * _t5 * ci
                + 2.0 * _t4 * _t5 * cj
                + 2.0 * _t6 * bi * bj
                + 2.0 * _t6 * ci * cj),
            0.0,
        );
    }
}

/// Basis function div_ne4_2d evaluated at coords.
pub fn _div_ne4_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new((-bi * (_t0 * bi - _t1 * bj) - ci * (_t0 * ci - _t1 * cj)), 0.0);
    }
}

/// Basis function div_ne5_2d evaluated at coords.
pub fn _div_ne5_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new((-bj * (_t0 * bi - _t1 * bj) - cj * (_t0 * ci - _t1 * cj)), 0.0);
    }
}

/// Basis function nf0_2d evaluated at coords.
pub fn _nf0_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ai + bi * xs + ci * ys;
        let _t1 = ak + bk * xs + ck * ys;
        let _t2 = aj + bj * xs + cj * ys;
        let bx = -_t0 * (_t1 * bj - _t2 * bk);
        let by = -_t0 * (_t1 * cj - _t2 * ck);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function nf1_2d evaluated at coords.
pub fn _nf1_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ak + bk * xs + ck * ys;
        let _t2 = ai + bi * xs + ci * ys;
        let bx = -_t0 * (_t1 * bi - _t2 * bk);
        let by = -_t0 * (_t1 * ci - _t2 * ck);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function nf2_2d evaluated at coords.
pub fn _nf2_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = aj + bj * xs + cj * ys;
        let _t2 = ai + bi * xs + ci * ys;
        let bx = _t0 * (_t1 * bi - _t2 * bj);
        let by = _t0 * (_t1 * ci - _t2 * cj);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function nf3_2d evaluated at coords.
pub fn _nf3_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ak + bk * xs + ck * ys;
        let _t2 = _t0 * _t1;
        let _t3 = ai + bi * xs + ci * ys;
        let _t4 = _t1 * _t3;
        let _t5 = 2.0 * _t0 * _t3;
        let bx = _t2 * bi + _t4 * bj - _t5 * bk;
        let by = _t2 * ci + _t4 * cj - _t5 * ck;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function nf4_2d evaluated at coords.
pub fn _nf4_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ai + bi * xs + ci * ys;
        let _t1 = ak + bk * xs + ck * ys;
        let _t2 = _t0 * _t1;
        let _t3 = aj + bj * xs + cj * ys;
        let _t4 = _t0 * _t3;
        let _t5 = 2.0 * _t1 * _t3;
        let bx = _t2 * bj + _t4 * bk - _t5 * bi;
        let by = _t2 * cj + _t4 * ck - _t5 * ci;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function nf5_2d evaluated at coords.
pub fn _nf5_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ak + bk * xs + ck * ys;
        let _t2 = _t0 * _t1;
        let _t3 = ai + bi * xs + ci * ys;
        let _t4 = _t0 * _t3;
        let _t5 = 2.0 * _t1 * _t3;
        let bx = _t2 * bi + _t4 * bk - _t5 * bj;
        let by = _t2 * ci + _t4 * ck - _t5 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function nf6_2d evaluated at coords.
pub fn _nf6_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ak + bk * xs + ck * ys;
        let _t2 = _t0 * _t1;
        let _t3 = ai + bi * xs + ci * ys;
        let _t4 = _t1 * _t3;
        let _t5 = _t0 * _t3;
        let bx = _t2 * bi + _t4 * bj + _t5 * bk;
        let by = _t2 * ci + _t4 * cj + _t5 * ck;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
    }
}

/// Basis function curl_nf0_2d evaluated at coords.
pub fn _curl_nf0_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = aj + bj * xs + cj * ys;
        out[p] = c64::new(
            (-bi * (_t0 * cj - _t1 * ck) + ci * (_t0 * bj - _t1 * bk) + 2.0 * (bj * ck - bk * cj) * (ai + bi * xs + ci * ys)),
            0.0,
        );
    }
}

/// Basis function curl_nf1_2d evaluated at coords.
pub fn _curl_nf1_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (-bj * (_t0 * ci - _t1 * ck) + cj * (_t0 * bi - _t1 * bk) + 2.0 * (bi * ck - bk * ci) * (aj + bj * xs + cj * ys)),
            0.0,
        );
    }
}

/// Basis function curl_nf2_2d evaluated at coords.
pub fn _curl_nf2_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (bk * (_t0 * ci - _t1 * cj) - ck * (_t0 * bi - _t1 * bj) - 2.0 * (bi * cj - bj * ci) * (ak + bk * xs + ck * ys)),
            0.0,
        );
    }
}

/// Basis function curl_nf3_2d evaluated at coords.
pub fn _curl_nf3_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (-3.0 * _t0 * bi * ck + 3.0 * _t0 * bk * ci - 3.0 * _t1 * bj * ck + 3.0 * _t1 * bk * cj),
            0.0,
        );
    }
}

/// Basis function curl_nf4_2d evaluated at coords.
pub fn _curl_nf4_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = aj + bj * xs + cj * ys;
        out[p] = c64::new(
            (3.0 * _t0 * bi * cj - 3.0 * _t0 * bj * ci + 3.0 * _t1 * bi * ck - 3.0 * _t1 * bk * ci),
            0.0,
        );
    }
}

/// Basis function curl_nf5_2d evaluated at coords.
pub fn _curl_nf5_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (-3.0 * _t0 * bi * cj + 3.0 * _t0 * bj * ci + 3.0 * _t1 * bj * ck - 3.0 * _t1 * bk * cj),
            0.0,
        );
    }
}

/// Basis function curl_nf6_2d evaluated at coords.
pub fn _curl_nf6_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function div_nf0_2d evaluated at coords.
pub fn _div_nf0_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = aj + bj * xs + cj * ys;
        out[p] = c64::new((-bi * (_t0 * bj - _t1 * bk) - ci * (_t0 * cj - _t1 * ck)), 0.0);
    }
}

/// Basis function div_nf1_2d evaluated at coords.
pub fn _div_nf1_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new((-bj * (_t0 * bi - _t1 * bk) - cj * (_t0 * ci - _t1 * ck)), 0.0);
    }
}

/// Basis function div_nf2_2d evaluated at coords.
pub fn _div_nf2_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        out[p] = c64::new((bk * (_t0 * bi - _t1 * bj) + ck * (_t0 * ci - _t1 * cj)), 0.0);
    }
}

/// Basis function div_nf3_2d evaluated at coords.
pub fn _div_nf3_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = aj + bj * xs + cj * ys;
        let _t1 = ai + bi * xs + ci * ys;
        let _t2 = ak + bk * xs + ck * ys;
        out[p] = c64::new(
            (-_t0 * bi * bk - _t0 * ci * ck - _t1 * bj * bk - _t1 * cj * ck + 2.0 * _t2 * bi * bj + 2.0 * _t2 * ci * cj),
            0.0,
        );
    }
}

/// Basis function div_nf4_2d evaluated at coords.
pub fn _div_nf4_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = aj + bj * xs + cj * ys;
        let _t2 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (-_t0 * bi * bj - _t0 * ci * cj - _t1 * bi * bk - _t1 * ci * ck + 2.0 * _t2 * bj * bk + 2.0 * _t2 * cj * ck),
            0.0,
        );
    }
}

/// Basis function div_nf5_2d evaluated at coords.
pub fn _div_nf5_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = ai + bi * xs + ci * ys;
        let _t2 = aj + bj * xs + cj * ys;
        out[p] = c64::new(
            (-_t0 * bi * bj - _t0 * ci * cj - _t1 * bj * bk - _t1 * cj * ck + 2.0 * _t2 * bi * bk + 2.0 * _t2 * ci * ck),
            0.0,
        );
    }
}

/// Basis function div_nf6_2d evaluated at coords.
pub fn _div_nf6_2d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let _t0 = ak + bk * xs + ck * ys;
        let _t1 = aj + bj * xs + cj * ys;
        let _t2 = ai + bi * xs + ci * ys;
        out[p] = c64::new(
            (2.0 * _t0 * bi * bj
                + 2.0 * _t0 * ci * cj
                + 2.0 * _t1 * bi * bk
                + 2.0 * _t1 * ci * ck
                + 2.0 * _t2 * bj * bk
                + 2.0 * _t2 * cj * ck),
            0.0,
        );
    }
}

/// Basis function ne0_3d evaluated at coords.
pub fn _ne0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        let bx = -_t0 * bi + _t1 * bj;
        let by = -_t0 * ci + _t1 * cj;
        let bz = -_t0 * di + _t1 * dj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function ne1_3d evaluated at coords.
pub fn _ne1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        let bx = _t0 * bi + _t1 * bj;
        let by = _t0 * ci + _t1 * cj;
        let bz = _t0 * di + _t1 * dj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function ne2_3d evaluated at coords.
pub fn _ne2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = bj * xs;
        let _t1 = cj * ys;
        let _t2 = dj * zs;
        let _t3 = ai + bi * xs + ci * ys + di * zs;
        let _t4 = -_t0 - _t1 - _t2 + _t3 - aj;
        let _t5 = _t0 + _t1 + _t2 + aj;
        let bx = -_t4 * (-_t3 * bj + _t5 * bi);
        let by = -_t4 * (-_t3 * cj + _t5 * ci);
        let bz = -_t4 * (-_t3 * dj + _t5 * di);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function ne3_3d evaluated at coords.
pub fn _ne3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = bj * xs;
        let _t1 = cj * ys;
        let _t2 = dj * zs;
        let _t3 = _t0 + _t1 + _t2 + aj;
        let _t4 = ai + bi * xs + ci * ys + di * zs;
        let _t5 = _t3 * _t4;
        let _t6 = -_t0 - _t1 - _t2 + _t4 - aj;
        let _t7 = _t3 * _t6;
        let _t8 = _t4 * _t6;
        let bx = _t5 * (bi - bj) + _t7 * bi + _t8 * bj;
        let by = _t5 * (ci - cj) + _t7 * ci + _t8 * cj;
        let bz = _t5 * (di - dj) + _t7 * di + _t8 * dj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function ne4_3d evaluated at coords.
pub fn _ne4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ai + bi * xs + ci * ys + di * zs;
        let _t1 = aj + bj * xs + cj * ys + dj * zs;
        let bx = -_t0 * (-_t0 * bj + _t1 * bi);
        let by = -_t0 * (-_t0 * cj + _t1 * ci);
        let bz = -_t0 * (-_t0 * dj + _t1 * di);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function ne5_3d evaluated at coords.
pub fn _ne5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        let bx = -_t0 * (_t0 * bi - _t1 * bj);
        let by = -_t0 * (_t0 * ci - _t1 * cj);
        let bz = -_t0 * (_t0 * di - _t1 * dj);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_ne0_3d evaluated at coords.
pub fn _curl_ne0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let bx = 2.0 * ci * dj - 2.0 * cj * di;
        let by = -2.0 * bi * dj + 2.0 * bj * di;
        let bz = 2.0 * bi * cj - 2.0 * bj * ci;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_ne1_3d evaluated at coords.
pub fn _curl_ne1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function curl_ne2_3d evaluated at coords.
pub fn _curl_ne2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = di - dj;
        let _t1 = bj * xs;
        let _t2 = cj * ys;
        let _t3 = dj * zs;
        let _t4 = _t1 + _t2 + _t3 + aj;
        let _t5 = ai + bi * xs + ci * ys + di * zs;
        let _t6 = _t4 * ci - _t5 * cj;
        let _t7 = ci - cj;
        let _t8 = _t4 * di - _t5 * dj;
        let _t9 = -2.0 * _t1 - 2.0 * _t2 - 2.0 * _t3 + 2.0 * _t5 - 2.0 * aj;
        let _t10 = _t4 * bi - _t5 * bj;
        let _t11 = bi - bj;
        let bx = _t0 * _t6 - _t7 * _t8 + _t9 * (ci * dj - cj * di);
        let by = -_t0 * _t10 + _t11 * _t8 - _t9 * (bi * dj - bj * di);
        let bz = _t10 * _t7 - _t11 * _t6 + _t9 * (bi * cj - bj * ci);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_ne3_3d evaluated at coords.
pub fn _curl_ne3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function curl_ne4_3d evaluated at coords.
pub fn _curl_ne4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ai + bi * xs + ci * ys + di * zs;
        let _t1 = 2.0 * _t0;
        let _t2 = aj + bj * xs + cj * ys + dj * zs;
        let _t3 = -_t0 * cj + _t2 * ci;
        let _t4 = -_t0 * dj + _t2 * di;
        let _t5 = -_t0 * bj + _t2 * bi;
        let bx = _t1 * (ci * dj - cj * di) + _t3 * di - _t4 * ci;
        let by = -_t1 * (bi * dj - bj * di) + _t4 * bi - _t5 * di;
        let bz = _t1 * (bi * cj - bj * ci) - _t3 * bi + _t5 * ci;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_ne5_3d evaluated at coords.
pub fn _curl_ne5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = 2.0 * _t0;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        let _t3 = _t0 * ci - _t2 * cj;
        let _t4 = _t0 * di - _t2 * dj;
        let _t5 = _t0 * bi - _t2 * bj;
        let bx = _t1 * (ci * dj - cj * di) + _t3 * dj - _t4 * cj;
        let by = -_t1 * (bi * dj - bj * di) + _t4 * bj - _t5 * dj;
        let bz = _t1 * (bi * cj - bj * ci) - _t3 * bj + _t5 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function div_ne0_3d evaluated at coords.
pub fn _div_ne0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function div_ne1_3d evaluated at coords.
pub fn _div_ne1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        out[p] = c64::new((2.0 * bi * bj + 2.0 * ci * cj + 2.0 * di * dj), 0.0);
    }
}

/// Basis function div_ne2_3d evaluated at coords.
pub fn _div_ne2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        out[p] = c64::new(
            (-(bi - bj) * (_t0 * bi - _t1 * bj) - (ci - cj) * (_t0 * ci - _t1 * cj) - (di - dj) * (_t0 * di - _t1 * dj)),
            0.0,
        );
    }
}

/// Basis function div_ne3_3d evaluated at coords.
pub fn _div_ne3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = bi - bj;
        let _t1 = bj * xs;
        let _t2 = cj * ys;
        let _t3 = dj * zs;
        let _t4 = _t1 + _t2 + _t3 + aj;
        let _t5 = ai + bi * xs + ci * ys + di * zs;
        let _t6 = ci - cj;
        let _t7 = di - dj;
        let _t8 = -_t1 - _t2 - _t3 + _t5 - aj;
        out[p] = c64::new(
            (2.0 * _t0 * _t4 * bi
                + 2.0 * _t0 * _t5 * bj
                + 2.0 * _t4 * _t6 * ci
                + 2.0 * _t4 * _t7 * di
                + 2.0 * _t5 * _t6 * cj
                + 2.0 * _t5 * _t7 * dj
                + 2.0 * _t8 * bi * bj
                + 2.0 * _t8 * ci * cj
                + 2.0 * _t8 * di * dj),
            0.0,
        );
    }
}

/// Basis function div_ne4_3d evaluated at coords.
pub fn _div_ne4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        out[p] = c64::new(
            (-bi * (_t0 * bi - _t1 * bj) - ci * (_t0 * ci - _t1 * cj) - di * (_t0 * di - _t1 * dj)),
            0.0,
        );
    }
}

/// Basis function div_ne5_3d evaluated at coords.
pub fn _div_ne5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        out[p] = c64::new(
            (-bj * (_t0 * bi - _t1 * bj) - cj * (_t0 * ci - _t1 * cj) - dj * (_t0 * di - _t1 * dj)),
            0.0,
        );
    }
}

/// Basis function nf0_3d evaluated at coords.
pub fn _nf0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ai + bi * xs + ci * ys + di * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = aj + bj * xs + cj * ys + dj * zs;
        let bx = -_t0 * (_t1 * bj - _t2 * bk);
        let by = -_t0 * (_t1 * cj - _t2 * ck);
        let bz = -_t0 * (_t1 * dj - _t2 * dk);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function nf1_3d evaluated at coords.
pub fn _nf1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        let bx = -_t0 * (_t1 * bi - _t2 * bk);
        let by = -_t0 * (_t1 * ci - _t2 * ck);
        let bz = -_t0 * (_t1 * di - _t2 * dk);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function nf2_3d evaluated at coords.
pub fn _nf2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = aj + bj * xs + cj * ys + dj * zs;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        let bx = _t0 * (_t1 * bi - _t2 * bj);
        let by = _t0 * (_t1 * ci - _t2 * cj);
        let bz = _t0 * (_t1 * di - _t2 * dj);
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function nf3_3d evaluated at coords.
pub fn _nf3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = _t0 * _t1;
        let _t3 = ai + bi * xs + ci * ys + di * zs;
        let _t4 = _t1 * _t3;
        let _t5 = 2.0 * _t0 * _t3;
        let bx = _t2 * bi + _t4 * bj - _t5 * bk;
        let by = _t2 * ci + _t4 * cj - _t5 * ck;
        let bz = _t2 * di + _t4 * dj - _t5 * dk;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function nf4_3d evaluated at coords.
pub fn _nf4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ai + bi * xs + ci * ys + di * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = _t0 * _t1;
        let _t3 = aj + bj * xs + cj * ys + dj * zs;
        let _t4 = _t0 * _t3;
        let _t5 = 2.0 * _t1 * _t3;
        let bx = _t2 * bj + _t4 * bk - _t5 * bi;
        let by = _t2 * cj + _t4 * ck - _t5 * ci;
        let bz = _t2 * dj + _t4 * dk - _t5 * di;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function nf5_3d evaluated at coords.
pub fn _nf5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = _t0 * _t1;
        let _t3 = ai + bi * xs + ci * ys + di * zs;
        let _t4 = _t0 * _t3;
        let _t5 = 2.0 * _t1 * _t3;
        let bx = _t2 * bi + _t4 * bk - _t5 * bj;
        let by = _t2 * ci + _t4 * ck - _t5 * cj;
        let bz = _t2 * di + _t4 * dk - _t5 * dj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function nf6_3d evaluated at coords.
pub fn _nf6_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = _t0 * _t1;
        let _t3 = ai + bi * xs + ci * ys + di * zs;
        let _t4 = _t1 * _t3;
        let _t5 = _t0 * _t3;
        let bx = _t2 * bi + _t4 * bj + _t5 * bk;
        let by = _t2 * ci + _t4 * cj + _t5 * ck;
        let bz = _t2 * di + _t4 * dj + _t5 * dk;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_nf0_3d evaluated at coords.
pub fn _curl_nf0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = 2.0 * ai + 2.0 * bi * xs + 2.0 * ci * ys + 2.0 * di * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = aj + bj * xs + cj * ys + dj * zs;
        let _t3 = _t1 * cj - _t2 * ck;
        let _t4 = _t1 * dj - _t2 * dk;
        let _t5 = _t1 * bj - _t2 * bk;
        let bx = _t0 * (cj * dk - ck * dj) + _t3 * di - _t4 * ci;
        let by = -_t0 * (bj * dk - bk * dj) + _t4 * bi - _t5 * di;
        let bz = _t0 * (bj * ck - bk * cj) - _t3 * bi + _t5 * ci;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_nf1_3d evaluated at coords.
pub fn _curl_nf1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = 2.0 * aj + 2.0 * bj * xs + 2.0 * cj * ys + 2.0 * dj * zs;
        let _t1 = ak + bk * xs + ck * ys + dk * zs;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        let _t3 = _t1 * ci - _t2 * ck;
        let _t4 = _t1 * di - _t2 * dk;
        let _t5 = _t1 * bi - _t2 * bk;
        let bx = _t0 * (ci * dk - ck * di) + _t3 * dj - _t4 * cj;
        let by = -_t0 * (bi * dk - bk * di) + _t4 * bj - _t5 * dj;
        let bz = _t0 * (bi * ck - bk * ci) - _t3 * bj + _t5 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_nf2_3d evaluated at coords.
pub fn _curl_nf2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = 2.0 * ak + 2.0 * bk * xs + 2.0 * ck * ys + 2.0 * dk * zs;
        let _t1 = aj + bj * xs + cj * ys + dj * zs;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        let _t3 = _t1 * ci - _t2 * cj;
        let _t4 = _t1 * di - _t2 * dj;
        let _t5 = _t1 * bi - _t2 * bj;
        let bx = -_t0 * (ci * dj - cj * di) - _t3 * dk + _t4 * ck;
        let by = _t0 * (bi * dj - bj * di) - _t4 * bk + _t5 * dk;
        let bz = -_t0 * (bi * cj - bj * ci) + _t3 * bk - _t5 * ck;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_nf3_3d evaluated at coords.
pub fn _curl_nf3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = _t0 * dk;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        let _t3 = _t2 * dk;
        let bx = 3.0 * _t0 * ck * di - 3.0 * _t1 * ci + 3.0 * _t2 * ck * dj - 3.0 * _t3 * cj;
        let by = -3.0 * _t0 * bk * di + 3.0 * _t1 * bi - 3.0 * _t2 * bk * dj + 3.0 * _t3 * bj;
        let bz = -3.0 * _t0 * bi * ck + 3.0 * _t0 * bk * ci - 3.0 * _t2 * bj * ck + 3.0 * _t2 * bk * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_nf4_3d evaluated at coords.
pub fn _curl_nf4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = _t0 * dj;
        let _t2 = aj + bj * xs + cj * ys + dj * zs;
        let _t3 = _t2 * dk;
        let bx = -3.0 * _t0 * cj * di + 3.0 * _t1 * ci - 3.0 * _t2 * ck * di + 3.0 * _t3 * ci;
        let by = 3.0 * _t0 * bj * di - 3.0 * _t1 * bi + 3.0 * _t2 * bk * di - 3.0 * _t3 * bi;
        let bz = 3.0 * _t0 * bi * cj - 3.0 * _t0 * bj * ci + 3.0 * _t2 * bi * ck - 3.0 * _t2 * bk * ci;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_nf5_3d evaluated at coords.
pub fn _curl_nf5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = _t0 * dj;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        let _t3 = _t2 * dj;
        let bx = 3.0 * _t0 * cj * di - 3.0 * _t1 * ci + 3.0 * _t2 * cj * dk - 3.0 * _t3 * ck;
        let by = -3.0 * _t0 * bj * di + 3.0 * _t1 * bi - 3.0 * _t2 * bj * dk + 3.0 * _t3 * bk;
        let bz = -3.0 * _t0 * bi * cj + 3.0 * _t0 * bj * ci + 3.0 * _t2 * bj * ck - 3.0 * _t2 * bk * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curl_nf6_3d evaluated at coords.
pub fn _curl_nf6_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function div_nf0_3d evaluated at coords.
pub fn _div_nf0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = aj + bj * xs + cj * ys + dj * zs;
        out[p] = c64::new(
            (-bi * (_t0 * bj - _t1 * bk) - ci * (_t0 * cj - _t1 * ck) - di * (_t0 * dj - _t1 * dk)),
            0.0,
        );
    }
}

/// Basis function div_nf1_3d evaluated at coords.
pub fn _div_nf1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        out[p] = c64::new(
            (-bj * (_t0 * bi - _t1 * bk) - cj * (_t0 * ci - _t1 * ck) - dj * (_t0 * di - _t1 * dk)),
            0.0,
        );
    }
}

/// Basis function div_nf2_3d evaluated at coords.
pub fn _div_nf2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        out[p] = c64::new(
            (bk * (_t0 * bi - _t1 * bj) + ck * (_t0 * ci - _t1 * cj) + dk * (_t0 * di - _t1 * dj)),
            0.0,
        );
    }
}

/// Basis function div_nf3_3d evaluated at coords.
pub fn _div_nf3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = aj + bj * xs + cj * ys + dj * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        let _t2 = ak + bk * xs + ck * ys + dk * zs;
        out[p] = c64::new(
            (-_t0 * bi * bk - _t0 * ci * ck - _t0 * di * dk - _t1 * bj * bk - _t1 * cj * ck - _t1 * dj * dk
                + 2.0 * _t2 * bi * bj
                + 2.0 * _t2 * ci * cj
                + 2.0 * _t2 * di * dj),
            0.0,
        );
    }
}

/// Basis function div_nf4_3d evaluated at coords.
pub fn _div_nf4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = aj + bj * xs + cj * ys + dj * zs;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        out[p] = c64::new(
            (-_t0 * bi * bj - _t0 * ci * cj - _t0 * di * dj - _t1 * bi * bk - _t1 * ci * ck - _t1 * di * dk
                + 2.0 * _t2 * bj * bk
                + 2.0 * _t2 * cj * ck
                + 2.0 * _t2 * dj * dk),
            0.0,
        );
    }
}

/// Basis function div_nf5_3d evaluated at coords.
pub fn _div_nf5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = ai + bi * xs + ci * ys + di * zs;
        let _t2 = aj + bj * xs + cj * ys + dj * zs;
        out[p] = c64::new(
            (-_t0 * bi * bj - _t0 * ci * cj - _t0 * di * dj - _t1 * bj * bk - _t1 * cj * ck - _t1 * dj * dk
                + 2.0 * _t2 * bi * bk
                + 2.0 * _t2 * ci * ck
                + 2.0 * _t2 * di * dk),
            0.0,
        );
    }
}

/// Basis function div_nf6_3d evaluated at coords.
pub fn _div_nf6_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut1<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let xs = coords[[0, p]];
        let ys = coords[[1, p]];
        let zs = coords[[2, p]];
        let _t0 = ak + bk * xs + ck * ys + dk * zs;
        let _t1 = aj + bj * xs + cj * ys + dj * zs;
        let _t2 = ai + bi * xs + ci * ys + di * zs;
        out[p] = c64::new(
            (2.0 * _t0 * bi * bj
                + 2.0 * _t0 * ci * cj
                + 2.0 * _t0 * di * dj
                + 2.0 * _t1 * bi * bk
                + 2.0 * _t1 * ci * ck
                + 2.0 * _t1 * di * dk
                + 2.0 * _t2 * bj * bk
                + 2.0 * _t2 * cj * ck
                + 2.0 * _t2 * dj * dk),
            0.0,
        );
    }
}

/// Basis function curlcurl_ne0_3d evaluated at coords.
pub fn _curlcurl_ne0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function curlcurl_ne1_3d evaluated at coords.
pub fn _curlcurl_ne1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function curlcurl_ne2_3d evaluated at coords.
pub fn _curlcurl_ne2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let _t0 = ci - cj;
        let _t1 = bi * cj - bj * ci;
        let _t2 = di - dj;
        let _t3 = bi * dj - bj * di;
        let _t4 = bi - bj;
        let _t5 = ci * dj - cj * di;
        let bx = 3.0 * _t0 * _t1 + 3.0 * _t2 * _t3;
        let by = -3.0 * _t1 * _t4 + 3.0 * _t2 * _t5;
        let bz = -3.0 * _t0 * _t5 - 3.0 * _t3 * _t4;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_ne3_3d evaluated at coords.
pub fn _curlcurl_ne3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Basis function curlcurl_ne4_3d evaluated at coords.
pub fn _curlcurl_ne4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let _t0 = bi * cj - bj * ci;
        let _t1 = bi * dj - bj * di;
        let _t2 = ci * dj - cj * di;
        let bx = 3.0 * _t0 * ci + 3.0 * _t1 * di;
        let by = -3.0 * _t0 * bi + 3.0 * _t2 * di;
        let bz = -3.0 * _t1 * bi - 3.0 * _t2 * ci;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_ne5_3d evaluated at coords.
pub fn _curlcurl_ne5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    for p in 0..coords.shape()[1] {
        let _t0 = bi * cj - bj * ci;
        let _t1 = bi * dj - bj * di;
        let _t2 = ci * dj - cj * di;
        let bx = 3.0 * _t0 * cj + 3.0 * _t1 * dj;
        let by = -3.0 * _t0 * bj + 3.0 * _t2 * dj;
        let bz = -3.0 * _t1 * bj - 3.0 * _t2 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_nf0_3d evaluated at coords.
pub fn _curlcurl_nf0_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let _t0 = bj * ck - bk * cj;
        let _t1 = bj * dk - bk * dj;
        let _t2 = cj * dk - ck * dj;
        let bx = 3.0 * _t0 * ci + 3.0 * _t1 * di;
        let by = -3.0 * _t0 * bi + 3.0 * _t2 * di;
        let bz = -3.0 * _t1 * bi - 3.0 * _t2 * ci;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_nf1_3d evaluated at coords.
pub fn _curlcurl_nf1_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let _t0 = bi * ck - bk * ci;
        let _t1 = bi * dk - bk * di;
        let _t2 = ci * dk - ck * di;
        let bx = 3.0 * _t0 * cj + 3.0 * _t1 * dj;
        let by = -3.0 * _t0 * bj + 3.0 * _t2 * dj;
        let bz = -3.0 * _t1 * bj - 3.0 * _t2 * cj;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_nf2_3d evaluated at coords.
pub fn _curlcurl_nf2_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let _t0 = bi * cj - bj * ci;
        let _t1 = bi * dj - bj * di;
        let _t2 = ci * dj - cj * di;
        let bx = -3.0 * _t0 * ck - 3.0 * _t1 * dk;
        let by = 3.0 * _t0 * bk - 3.0 * _t2 * dk;
        let bz = 3.0 * _t1 * bk + 3.0 * _t2 * ck;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_nf3_3d evaluated at coords.
pub fn _curlcurl_nf3_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let bx =
            -3.0 * bi * cj * ck - 3.0 * bi * dj * dk - 3.0 * bj * ci * ck - 3.0 * bj * di * dk + 6.0 * bk * ci * cj + 6.0 * bk * di * dj;
        let by =
            6.0 * bi * bj * ck - 3.0 * bi * bk * cj - 3.0 * bj * bk * ci - 3.0 * ci * dj * dk - 3.0 * cj * di * dk + 6.0 * ck * di * dj;
        let bz =
            6.0 * bi * bj * dk - 3.0 * bi * bk * dj - 3.0 * bj * bk * di + 6.0 * ci * cj * dk - 3.0 * ci * ck * dj - 3.0 * cj * ck * di;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_nf4_3d evaluated at coords.
pub fn _curlcurl_nf4_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let bx =
            6.0 * bi * cj * ck + 6.0 * bi * dj * dk - 3.0 * bj * ci * ck - 3.0 * bj * di * dk - 3.0 * bk * ci * cj - 3.0 * bk * di * dj;
        let by =
            -3.0 * bi * bj * ck - 3.0 * bi * bk * cj + 6.0 * bj * bk * ci + 6.0 * ci * dj * dk - 3.0 * cj * di * dk - 3.0 * ck * di * dj;
        let bz =
            -3.0 * bi * bj * dk - 3.0 * bi * bk * dj + 6.0 * bj * bk * di - 3.0 * ci * cj * dk - 3.0 * ci * ck * dj + 6.0 * cj * ck * di;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_nf5_3d evaluated at coords.
pub fn _curlcurl_nf5_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    let ai = coeff[[0, i]];
    let bi = coeff[[1, i]];
    let ci = coeff[[2, i]];
    let di = coeff[[3, i]];
    let aj = coeff[[0, j]];
    let bj = coeff[[1, j]];
    let cj = coeff[[2, j]];
    let dj = coeff[[3, j]];
    let ak = coeff[[0, k]];
    let bk = coeff[[1, k]];
    let ck = coeff[[2, k]];
    let dk = coeff[[3, k]];
    for p in 0..coords.shape()[1] {
        let bx =
            -3.0 * bi * cj * ck - 3.0 * bi * dj * dk + 6.0 * bj * ci * ck + 6.0 * bj * di * dk - 3.0 * bk * ci * cj - 3.0 * bk * di * dj;
        let by =
            -3.0 * bi * bj * ck + 6.0 * bi * bk * cj - 3.0 * bj * bk * ci - 3.0 * ci * dj * dk + 6.0 * cj * di * dk - 3.0 * ck * di * dj;
        let bz =
            -3.0 * bi * bj * dk + 6.0 * bi * bk * dj - 3.0 * bj * bk * di - 3.0 * ci * cj * dk + 6.0 * ci * ck * dj - 3.0 * cj * ck * di;
        out[[0, p]] = c64::new(bx, 0.0);
        out[[1, p]] = c64::new(by, 0.0);
        out[[2, p]] = c64::new(bz, 0.0);
    }
}

/// Basis function curlcurl_nf6_3d evaluated at coords.
pub fn _curlcurl_nf6_3d(coeff: &ArrayView2<f64>, coords: &ArrayView2<f64>, i: usize, j: usize, k: usize, out: &mut ArrayViewMut2<c64>) {
    out.fill(c64::new(0.0, 0.0));
}

/// Evaluates the f of the basis function identified by code into out.
pub fn _eval_f_2d(
    coeff: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    i: usize,
    j: usize,
    k: usize,
    code: i64,
    out: &mut ArrayViewMut2<c64>,
) {
    let (bftype, index) = get_type_index(code);
    match (bftype, index) {
        (64, 0) => _ne0_2d(coeff, coords, i, j, k, out),
        (64, 1) => _ne1_2d(coeff, coords, i, j, k, out),
        (64, 2) => _ne2_2d(coeff, coords, i, j, k, out),
        (64, 3) => _ne3_2d(coeff, coords, i, j, k, out),
        (64, 4) => _ne4_2d(coeff, coords, i, j, k, out),
        (64, 5) => _ne5_2d(coeff, coords, i, j, k, out),
        (128, 0) => _nf0_2d(coeff, coords, i, j, k, out),
        (128, 1) => _nf1_2d(coeff, coords, i, j, k, out),
        (128, 2) => _nf2_2d(coeff, coords, i, j, k, out),
        (128, 3) => _nf3_2d(coeff, coords, i, j, k, out),
        (128, 4) => _nf4_2d(coeff, coords, i, j, k, out),
        (128, 5) => _nf5_2d(coeff, coords, i, j, k, out),
        (128, 6) => _nf6_2d(coeff, coords, i, j, k, out),
        _ => panic!("Unrecognized basis function type."),
    }
}

/// Evaluates the curl of the basis function identified by code into out.
pub fn _eval_curl_f_2d(
    coeff: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    i: usize,
    j: usize,
    k: usize,
    code: i64,
    out: &mut ArrayViewMut1<c64>,
) {
    let (bftype, index) = get_type_index(code);
    match (bftype, index) {
        (64, 0) => _curl_ne0_2d(coeff, coords, i, j, k, out),
        (64, 1) => _curl_ne1_2d(coeff, coords, i, j, k, out),
        (64, 2) => _curl_ne2_2d(coeff, coords, i, j, k, out),
        (64, 3) => _curl_ne3_2d(coeff, coords, i, j, k, out),
        (64, 4) => _curl_ne4_2d(coeff, coords, i, j, k, out),
        (64, 5) => _curl_ne5_2d(coeff, coords, i, j, k, out),
        (128, 0) => _curl_nf0_2d(coeff, coords, i, j, k, out),
        (128, 1) => _curl_nf1_2d(coeff, coords, i, j, k, out),
        (128, 2) => _curl_nf2_2d(coeff, coords, i, j, k, out),
        (128, 3) => _curl_nf3_2d(coeff, coords, i, j, k, out),
        (128, 4) => _curl_nf4_2d(coeff, coords, i, j, k, out),
        (128, 5) => _curl_nf5_2d(coeff, coords, i, j, k, out),
        (128, 6) => _curl_nf6_2d(coeff, coords, i, j, k, out),
        _ => panic!("Unrecognized basis function type."),
    }
}

/// Evaluates the div of the basis function identified by code into out.
pub fn _eval_div_f_2d(
    coeff: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    i: usize,
    j: usize,
    k: usize,
    code: i64,
    out: &mut ArrayViewMut1<c64>,
) {
    let (bftype, index) = get_type_index(code);
    match (bftype, index) {
        (64, 0) => _div_ne0_2d(coeff, coords, i, j, k, out),
        (64, 1) => _div_ne1_2d(coeff, coords, i, j, k, out),
        (64, 2) => _div_ne2_2d(coeff, coords, i, j, k, out),
        (64, 3) => _div_ne3_2d(coeff, coords, i, j, k, out),
        (64, 4) => _div_ne4_2d(coeff, coords, i, j, k, out),
        (64, 5) => _div_ne5_2d(coeff, coords, i, j, k, out),
        (128, 0) => _div_nf0_2d(coeff, coords, i, j, k, out),
        (128, 1) => _div_nf1_2d(coeff, coords, i, j, k, out),
        (128, 2) => _div_nf2_2d(coeff, coords, i, j, k, out),
        (128, 3) => _div_nf3_2d(coeff, coords, i, j, k, out),
        (128, 4) => _div_nf4_2d(coeff, coords, i, j, k, out),
        (128, 5) => _div_nf5_2d(coeff, coords, i, j, k, out),
        (128, 6) => _div_nf6_2d(coeff, coords, i, j, k, out),
        _ => panic!("Unrecognized basis function type."),
    }
}

/// Evaluates the f of the basis function identified by code into out.
pub fn _eval_f_3d(
    coeff: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    i: usize,
    j: usize,
    k: usize,
    code: i64,
    out: &mut ArrayViewMut2<c64>,
) {
    let (bftype, index) = get_type_index(code);
    match (bftype, index) {
        (64, 0) => _ne0_3d(coeff, coords, i, j, k, out),
        (64, 1) => _ne1_3d(coeff, coords, i, j, k, out),
        (64, 2) => _ne2_3d(coeff, coords, i, j, k, out),
        (64, 3) => _ne3_3d(coeff, coords, i, j, k, out),
        (64, 4) => _ne4_3d(coeff, coords, i, j, k, out),
        (64, 5) => _ne5_3d(coeff, coords, i, j, k, out),
        (128, 0) => _nf0_3d(coeff, coords, i, j, k, out),
        (128, 1) => _nf1_3d(coeff, coords, i, j, k, out),
        (128, 2) => _nf2_3d(coeff, coords, i, j, k, out),
        (128, 3) => _nf3_3d(coeff, coords, i, j, k, out),
        (128, 4) => _nf4_3d(coeff, coords, i, j, k, out),
        (128, 5) => _nf5_3d(coeff, coords, i, j, k, out),
        (128, 6) => _nf6_3d(coeff, coords, i, j, k, out),
        _ => panic!("Unrecognized basis function type."),
    }
}

/// Evaluates the curl of the basis function identified by code into out.
pub fn _eval_curl_f_3d(
    coeff: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    i: usize,
    j: usize,
    k: usize,
    code: i64,
    out: &mut ArrayViewMut2<c64>,
) {
    let (bftype, index) = get_type_index(code);
    match (bftype, index) {
        (64, 0) => _curl_ne0_3d(coeff, coords, i, j, k, out),
        (64, 1) => _curl_ne1_3d(coeff, coords, i, j, k, out),
        (64, 2) => _curl_ne2_3d(coeff, coords, i, j, k, out),
        (64, 3) => _curl_ne3_3d(coeff, coords, i, j, k, out),
        (64, 4) => _curl_ne4_3d(coeff, coords, i, j, k, out),
        (64, 5) => _curl_ne5_3d(coeff, coords, i, j, k, out),
        (128, 0) => _curl_nf0_3d(coeff, coords, i, j, k, out),
        (128, 1) => _curl_nf1_3d(coeff, coords, i, j, k, out),
        (128, 2) => _curl_nf2_3d(coeff, coords, i, j, k, out),
        (128, 3) => _curl_nf3_3d(coeff, coords, i, j, k, out),
        (128, 4) => _curl_nf4_3d(coeff, coords, i, j, k, out),
        (128, 5) => _curl_nf5_3d(coeff, coords, i, j, k, out),
        (128, 6) => _curl_nf6_3d(coeff, coords, i, j, k, out),
        _ => panic!("Unrecognized basis function type."),
    }
}

/// Evaluates the div of the basis function identified by code into out.
pub fn _eval_div_f_3d(
    coeff: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    i: usize,
    j: usize,
    k: usize,
    code: i64,
    out: &mut ArrayViewMut1<c64>,
) {
    let (bftype, index) = get_type_index(code);
    match (bftype, index) {
        (64, 0) => _div_ne0_3d(coeff, coords, i, j, k, out),
        (64, 1) => _div_ne1_3d(coeff, coords, i, j, k, out),
        (64, 2) => _div_ne2_3d(coeff, coords, i, j, k, out),
        (64, 3) => _div_ne3_3d(coeff, coords, i, j, k, out),
        (64, 4) => _div_ne4_3d(coeff, coords, i, j, k, out),
        (64, 5) => _div_ne5_3d(coeff, coords, i, j, k, out),
        (128, 0) => _div_nf0_3d(coeff, coords, i, j, k, out),
        (128, 1) => _div_nf1_3d(coeff, coords, i, j, k, out),
        (128, 2) => _div_nf2_3d(coeff, coords, i, j, k, out),
        (128, 3) => _div_nf3_3d(coeff, coords, i, j, k, out),
        (128, 4) => _div_nf4_3d(coeff, coords, i, j, k, out),
        (128, 5) => _div_nf5_3d(coeff, coords, i, j, k, out),
        (128, 6) => _div_nf6_3d(coeff, coords, i, j, k, out),
        _ => panic!("Unrecognized basis function type."),
    }
}

/// Evaluates the curlcurl of the basis function identified by code into out.
pub fn _eval_curlcurl_f_3d(
    coeff: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    i: usize,
    j: usize,
    k: usize,
    code: i64,
    out: &mut ArrayViewMut2<c64>,
) {
    let (bftype, index) = get_type_index(code);
    match (bftype, index) {
        (64, 0) => _curlcurl_ne0_3d(coeff, coords, i, j, k, out),
        (64, 1) => _curlcurl_ne1_3d(coeff, coords, i, j, k, out),
        (64, 2) => _curlcurl_ne2_3d(coeff, coords, i, j, k, out),
        (64, 3) => _curlcurl_ne3_3d(coeff, coords, i, j, k, out),
        (64, 4) => _curlcurl_ne4_3d(coeff, coords, i, j, k, out),
        (64, 5) => _curlcurl_ne5_3d(coeff, coords, i, j, k, out),
        (128, 0) => _curlcurl_nf0_3d(coeff, coords, i, j, k, out),
        (128, 1) => _curlcurl_nf1_3d(coeff, coords, i, j, k, out),
        (128, 2) => _curlcurl_nf2_3d(coeff, coords, i, j, k, out),
        (128, 3) => _curlcurl_nf3_3d(coeff, coords, i, j, k, out),
        (128, 4) => _curlcurl_nf4_3d(coeff, coords, i, j, k, out),
        (128, 5) => _curlcurl_nf5_3d(coeff, coords, i, j, k, out),
        (128, 6) => _curlcurl_nf6_3d(coeff, coords, i, j, k, out),
        _ => panic!("Unrecognized basis function type."),
    }
}
