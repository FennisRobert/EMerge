import os

class Config:
    """Configure EMERGE environment variables."""

    def set_threads(
        self,
        *,
        omp: int | None = None,
        mkl: int | None = None,
        openblas: int | None = None,
        veclib: int | None = None,
        veclib_max: int | None = None,
        numexpr: int | None = None,
        numba: int | None = None,
        numba_layer: str | None = None,
        rayon: str | None = None,
        rslab: str | None = None,
    ) -> None:
        """Set individual threading variables by friendly name."""
        _map = {
            "OMP_NUM_THREADS": omp,
            "MKL_NUM_THREADS": mkl,
            "OPENBLAS_NUM_THREADS": openblas,
            "VECLIB_NUM_THREADS": veclib,
            "VECLIB_MAXIMUM_THREADS": veclib_max,
            "NUMEXPR_NUM_THREADS": numexpr,
            "NUMBA_NUM_THREADS": numba,
            "NUMBA_THREADING_LAYER": numba_layer,
            "RAYON_NUM_THREADS": rayon,
            "RSLAB_NUM_THREADS": rslab,
        }
        for key, val in _map.items():
            if val is not None:
                os.environ[key] = str(val)

    def set_pardiso_threads(self, n: int) -> None:
        """Tuned for PARDISO solver (MKL-heavy)."""
        self.set_threads(mkl=n, omp=n)

    def set_acc_threads(self, n: int) -> None:
        """Tuned for ACC assembly (OpenMP-heavy)."""
        self.set_threads(veclib_max=n, veclib=1)

    def set_rslab_threads(self, n: int) -> None:
        """ Tuned for RS-lab solver"""
        self.set_threads(rslab=n)
        
    def set_mumps_threads(self, n: int) -> None:
        """Tuned for MUMPS solver (balanced OMP + MKL)."""
        self.set_threads(omp=n)

    def set_emerge_threads(self, n: int) -> None:
        """Tuned for EMerge assemblers and processors"""
        self.set_threads(numba=n, rayon=n)

    def set_logging(
        self,
        *,
        console: str = "INFO",
        file: str = "DEBUG",
    ) -> None:
        """Set EMERGE log levels."""
        os.environ["EMERGE_STD_LOGLEVEL"] = console
        os.environ["EMERGE_FILE_LOGLEVEL"] = file



config = Config()