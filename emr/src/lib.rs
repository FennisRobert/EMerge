//! EMR (EMerge Rust): compiled kernels of the EMerge FEM solver.
#![allow(
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::useless_conversion
)]
use pyo3::prelude::*;

pub mod aura;
pub mod compiled;
pub mod constants;
pub mod heat;
pub mod mesh3d;
pub mod microwave;
pub mod mth;

/// Raw pointer that may be shared between rayon workers that write provably disjoint indices.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T: Copy + std::ops::AddAssign> SyncPtr<T> {
    /// Writes value at offset i. Caller guarantees no other thread touches i.
    ///
    /// # Safety
    /// `i` must be in bounds and not be accessed concurrently.
    pub unsafe fn write(&self, i: usize, value: T) {
        *self.0.add(i) = value;
    }

    /// Adds value at offset i. Caller guarantees no other thread touches i.
    ///
    /// # Safety
    /// `i` must be in bounds and not be accessed concurrently.
    pub unsafe fn add_assign(&self, i: usize, value: T) {
        *self.0.add(i) += value;
    }
}

fn add_submodule(parent: &Bound<'_, PyModule>, name: &str, register: fn(&Bound<'_, PyModule>) -> PyResult<()>) -> PyResult<()> {
    let sub = PyModule::new_bound(parent.py(), name)?;
    register(&sub)?;
    parent.add_submodule(&sub)?;
    let qualified = format!("emr.{name}");
    sub.setattr("__name__", &qualified)?;
    parent.py().import_bound("sys")?.getattr("modules")?.set_item(qualified, &sub)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "_emr")]
fn emr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_submodule(m, "mth", |s| {
        mth::optimized::register(s)?;
        mth::integrals::register(s)?;
        mth::csc_cast::register(s)
    })?;
    add_submodule(m, "interp", compiled::interp::register)?;
    add_submodule(m, "index_interp", compiled::index_interp::register)?;
    add_submodule(m, "ccbf", compiled::ccbf_py::register)?;
    add_submodule(m, "curlcurl", microwave::curlcurl::register)?;
    add_submodule(m, "curlcurl_iso", microwave::curlcurl_iso::register)?; // [Written by Claude]
    add_submodule(m, "generalized_eigen_hb", microwave::generalized_eigen_hb::register)?;
    add_submodule(m, "robinbc", microwave::robinbc::register)?;
    add_submodule(m, "robin_abc_order2", microwave::robin_abc_order2::register)?;
    add_submodule(m, "periodicbc", microwave::periodicbc::register)?;
    add_submodule(m, "system_pattern", microwave::system_pattern::register)?;
    add_submodule(m, "ams_kernels", microwave::ams_kernels::register)?;
    add_submodule(m, "adaptive_mesh", microwave::adaptive_mesh::register)?;
    add_submodule(m, "sc", microwave::sc::register)?;
    add_submodule(m, "heat_grad", heat::grad::register)?;
    add_submodule(m, "heat_convection", heat::convection::register)?;
    add_submodule(m, "heat_heatflux", heat::heatflux::register)?;
    add_submodule(m, "heat_thermalcontact", heat::thermalcontact::register)?;
    add_submodule(m, "heat_thinconductor", heat::thinconductor::register)?;
    add_submodule(m, "mesh3d", mesh3d::register)?;
    add_submodule(m, "aura", aura::py::register)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
