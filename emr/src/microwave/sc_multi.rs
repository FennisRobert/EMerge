// [Written by Claude] All code in this module was written by Claude (AI).
//! Far field of several excitations on the same surface in one pass (Stratton-Chu).
//!
//! The phase factor exp(j k0 r.x') of a direction r and a surface point x' is the same for every excitation
//! at one frequency, so it is computed once and applied to the n x E and n x H of all excitations. The result
//! is the same as calling `stratton_chu_ff` once per excitation.
use crate::constants::{PI, Z0};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const LR: f64 = 0.001;

/// Far field E and H (3 K, N) in directions tpout = (theta, phi) for K excitations.
///
/// ein/hin (3 K, Ns): the surface E and H of excitation k in rows 3k..3k+3, at the points vis (3, Ns) with
/// the weighted normals wns (3, Ns). A surface point is used when its |E| exceeds LR times the largest |E|
/// of any excitation's own maximum (the union of the points `stratton_chu_ff` uses per excitation).
pub fn stratton_chu_ff_multi(
    ein: &ArrayView2<c64>,
    hin: &ArrayView2<c64>,
    vis: &ArrayView2<f64>,
    wns: &ArrayView2<f64>,
    tpout: &ArrayView2<f64>,
    k0: f64,
) -> (Array2<c64>, Array2<c64>) {
    let k = ein.shape()[0] / 3;
    let ns = ein.shape()[1];

    // surface points that matter for at least one excitation
    let mut keep = vec![false; ns];
    for e in 0..k {
        let mag: Vec<f64> = (0..ns)
            .map(|i| (0..3).map(|c| ein[[3 * e + c, i]].norm_sqr()).sum::<f64>().sqrt())
            .collect();
        let level = mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * LR;
        for i in 0..ns {
            keep[i] |= mag[i] > level;
        }
    }
    let ids: Vec<usize> = (0..ns).filter(|&i| keep[i]).collect();
    let m = ids.len();

    // n x E and n x H of every kept point and excitation: src[j * k + e] = [nxE, nxH]
    let mut pts = Vec::with_capacity(m);
    let mut src = Vec::with_capacity(m * k);
    for &j in &ids {
        pts.push([k0 * vis[[0, j]], k0 * vis[[1, j]], k0 * vis[[2, j]]]);
        let (nx, ny, nz) = (wns[[0, j]], wns[[1, j]], wns[[2, j]]);
        for e in 0..k {
            let (ex, ey, ez) = (ein[[3 * e, j]], ein[[3 * e + 1, j]], ein[[3 * e + 2, j]]);
            let (hx, hy, hz) = (hin[[3 * e, j]], hin[[3 * e + 1, j]], hin[[3 * e + 2, j]]);
            src.push([
                ez * ny - ey * nz,
                ex * nz - ez * nx,
                ey * nx - ex * ny,
                hz * ny - hy * nz,
                hx * nz - hz * nx,
                hy * nx - hx * ny,
            ]);
        }
    }

    let n = tpout.shape()[1];
    let q = c64::new(0.0, -k0 / (4.0 * PI));
    let zero = c64::new(0.0, 0.0);
    // per direction: E of every excitation, [e][xyz]
    let out: Vec<Vec<[c64; 3]>> = (0..n)
        .into_par_iter()
        .map(|p| {
            let (st, ct) = tpout[[0, p]].sin_cos();
            let (sp, cp) = tpout[[1, p]].sin_cos();
            let (rx, ry, rz) = (st * cp, st * sp, ct);
            let mut acc = vec![[zero; 6]; k];
            for (j, x) in pts.iter().enumerate() {
                let g = c64::from_polar(1.0, rx * x[0] + ry * x[1] + rz * x[2]);
                let s = &src[j * k..(j + 1) * k];
                for e in 0..k {
                    let a = &mut acc[e];
                    for c in 0..6 {
                        a[c] += s[e][c] * g;
                    }
                }
            }
            acc.iter()
                .map(|a| {
                    // r x (A - Z0 r x B) with A = sum g n x E, B = sum g n x H
                    let (bx, by, bz) = (a[3], a[4], a[5]);
                    let (rbx, rby, rbz) = (bz * ry - by * rz, bx * rz - bz * rx, by * rx - bx * ry);
                    let (ix, iy, iz) = (a[0] - rbx * Z0, a[1] - rby * Z0, a[2] - rbz * Z0);
                    [q * (iz * ry - iy * rz), q * (ix * rz - iz * rx), q * (iy * rx - ix * ry)]
                })
                .collect()
        })
        .collect();

    let mut eout = Array2::<c64>::zeros((3 * k, n));
    let mut hout = Array2::<c64>::zeros((3 * k, n));
    for (p, per) in out.iter().enumerate() {
        let (st, ct) = tpout[[0, p]].sin_cos();
        let (sp, cp) = tpout[[1, p]].sin_cos();
        let (rx, ry, rz) = (st * cp, st * sp, ct);
        for (e, v) in per.iter().enumerate() {
            eout[[3 * e, p]] = v[0];
            eout[[3 * e + 1, p]] = v[1];
            eout[[3 * e + 2, p]] = v[2];
            hout[[3 * e, p]] = (v[2] * ry - v[1] * rz) / Z0;
            hout[[3 * e + 1, p]] = (v[0] * rz - v[2] * rx) / Z0;
            hout[[3 * e + 2, p]] = (v[1] * rx - v[0] * ry) / Z0;
        }
    }
    (eout, hout)
}

// [Written by Claude]
#[pyfunction]
#[pyo3(name = "stratton_chu_ff_multi")]
fn py_stratton_chu_ff_multi<'py>(
    py: Python<'py>,
    ein: PyReadonlyArray2<c64>,
    hin: PyReadonlyArray2<c64>,
    vis: PyReadonlyArray2<f64>,
    wns: PyReadonlyArray2<f64>,
    tpout: PyReadonlyArray2<f64>,
    k0: f64,
) -> (Bound<'py, PyArray2<c64>>, Bound<'py, PyArray2<c64>>) {
    let (e, h, v, w, t) = (ein.as_array(), hin.as_array(), vis.as_array(), wns.as_array(), tpout.as_array());
    let r = py.allow_threads(|| stratton_chu_ff_multi(&e, &h, &v, &w, &t, k0));
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py))
}

// [Written by Claude]
/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_stratton_chu_ff_multi, m)?)?;
    Ok(())
}
