//! [Written by Claude] The thread pool of the EMR kernels.
//!
//! Every kernel that uses rayon runs inside `install`, so all of them use one pool. Without a call to
//! `set_num_threads` that is rayon's global pool (sized by RAYON_NUM_THREADS, otherwise all logical cores).
//! Unlike the global pool it can be resized at any time.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::ThreadPool;
use std::sync::{Arc, RwLock};

static POOL: RwLock<Option<Arc<ThreadPool>>> = RwLock::new(None);

fn current() -> Option<Arc<ThreadPool>> {
    POOL.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Runs f in the EMR thread pool, so the rayon calls inside it use that pool.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match current() {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// Sets the number of threads of the EMR kernels. None returns to rayon's global pool.
#[pyfunction]
#[pyo3(signature = (nthreads=None))]
fn set_num_threads(nthreads: Option<usize>) -> PyResult<()> {
    let pool = match nthreads {
        None => None,
        Some(0) => return Err(PyValueError::new_err("The number of threads must be at least 1.")),
        Some(n) => Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("emr-{i}"))
                .build()
                .map_err(|e| PyValueError::new_err(format!("Could not create a pool of {n} threads: {e}")))?,
        )),
    };
    *POOL.write().unwrap_or_else(|e| e.into_inner()) = pool;
    Ok(())
}

/// The number of threads the EMR kernels use.
#[pyfunction]
fn get_num_threads() -> usize {
    match current() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn kernels_run_in_the_set_pool() {
        set_num_threads(Some(3)).unwrap();
        assert_eq!(install(rayon::current_num_threads), 3);
        let s: i64 = install(|| (0..1000i64).into_par_iter().sum());
        assert_eq!(s, 499500);
        set_num_threads(None).unwrap();
        assert_eq!(install(rayon::current_num_threads), rayon::current_num_threads());
    }
}
