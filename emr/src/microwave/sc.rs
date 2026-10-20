//! Stratton-Chu near to far field transformation of surface E and H fields.
use crate::constants::{PI, Z0};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64 as c64;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const LR: f64 = 0.001;

/// Far field E and H (3, N) in directions tpout = (theta, phi) from surface fields Ein/Hin at points vis with weighted normals wns.
pub fn stratton_chu_ff(
    ein: &ArrayView2<c64>,
    hin: &ArrayView2<c64>,
    vis: &ArrayView2<f64>,
    wns: &ArrayView2<f64>,
    tpout: &ArrayView2<f64>,
    k0: f64,
) -> (Array2<c64>, Array2<c64>) {
    let ns = ein.shape()[1];
    let emag: Vec<f64> = (0..ns)
        .map(|i| (ein[[0, i]].norm().powi(2) + ein[[1, i]].norm().powi(2) + ein[[2, i]].norm().powi(2)).sqrt())
        .collect();
    let elevel = emag.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * LR;
    let ids: Vec<usize> = (0..ns).filter(|&i| emag[i] > elevel).collect();

    let n = tpout.shape()[1];
    let rx: Vec<f64> = (0..n).map(|p| tpout[[0, p]].sin() * tpout[[1, p]].cos()).collect();
    let ry: Vec<f64> = (0..n).map(|p| tpout[[0, p]].sin() * tpout[[1, p]].sin()).collect();
    let rz: Vec<f64> = (0..n).map(|p| tpout[[0, p]].cos()).collect();
    let q = c64::new(0.0, -k0 / (4.0 * PI));
    let ii = c64::new(0.0, 1.0);

    let zero = || vec![c64::new(0.0, 0.0); n];
    let (eoutx, eouty, eoutz) = ids
        .par_iter()
        .fold(
            || (zero(), zero(), zero()),
            |(mut ox, mut oy, mut oz), &j| {
                let (ex, ey, ez) = (ein[[0, j]], ein[[1, j]], ein[[2, j]]);
                let (hx, hy, hz) = (hin[[0, j]], hin[[1, j]], hin[[2, j]]);
                let (nx, ny, nz) = (wns[[0, j]], wns[[1, j]], wns[[2, j]]);
                let (nxhx, nxhy, nxhz) = (hz * ny - hy * nz, hx * nz - hz * nx, hy * nx - hx * ny);
                let (nxex, nxey, nxez) = (ez * ny - ey * nz, ex * nz - ez * nx, ey * nx - ex * ny);
                let (xi, yi, zi) = (vis[[0, j]], vis[[1, j]], vis[[2, j]]);
                for p in 0..n {
                    let g = (ii * (k0 * rx[p] * xi + k0 * ry[p] * yi + k0 * rz[p] * zi)).exp();
                    let rxnxhx = nxhz * ry[p] - nxhy * rz[p];
                    let rxnxhy = nxhx * rz[p] - nxhz * rx[p];
                    let rxnxhz = nxhy * rx[p] - nxhx * ry[p];
                    let ie1x = (nxex - rxnxhx * Z0) * g;
                    let ie1y = (nxey - rxnxhy * Z0) * g;
                    let ie1z = (nxez - rxnxhz * Z0) * g;
                    ox[p] += q * (ie1z * ry[p] - ie1y * rz[p]);
                    oy[p] += q * (ie1x * rz[p] - ie1z * rx[p]);
                    oz[p] += q * (ie1y * rx[p] - ie1x * ry[p]);
                }
                (ox, oy, oz)
            },
        )
        .reduce(
            || (zero(), zero(), zero()),
            |(mut ax, mut ay, mut az), (bx, by, bz)| {
                for p in 0..n {
                    ax[p] += bx[p];
                    ay[p] += by[p];
                    az[p] += bz[p];
                }
                (ax, ay, az)
            },
        );
    let mut eout = Array2::<c64>::zeros((3, n));
    let mut hout = Array2::<c64>::zeros((3, n));
    for p in 0..n {
        eout[[0, p]] = eoutx[p];
        eout[[1, p]] = eouty[p];
        eout[[2, p]] = eoutz[p];
        hout[[0, p]] = (eoutz[p] * ry[p] - eouty[p] * rz[p]) / Z0;
        hout[[1, p]] = (eoutx[p] * rz[p] - eoutz[p] * rx[p]) / Z0;
        hout[[2, p]] = (eouty[p] * rx[p] - eoutx[p] * ry[p]) / Z0;
    }
    (eout, hout)
}

#[pyfunction]
#[pyo3(name = "stratton_chu_ff")]
fn py_stratton_chu_ff<'py>(
    py: Python<'py>,
    ein: PyReadonlyArray2<c64>,
    hin: PyReadonlyArray2<c64>,
    vis: PyReadonlyArray2<f64>,
    wns: PyReadonlyArray2<f64>,
    tpout: PyReadonlyArray2<f64>,
    k0: f64,
) -> (Bound<'py, PyArray2<c64>>, Bound<'py, PyArray2<c64>>) {
    let (e, h, v, w, t) = (ein.as_array(), hin.as_array(), vis.as_array(), wns.as_array(), tpout.as_array());
    let r = py.allow_threads(|| stratton_chu_ff(&e, &h, &v, &w, &t, k0));
    (r.0.into_pyarray_bound(py), r.1.into_pyarray_bound(py))
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_stratton_chu_ff, m)?)?;
    Ok(())
}
