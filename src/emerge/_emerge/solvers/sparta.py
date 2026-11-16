# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""The SPARTA sparse direct solver (Metal GPU and CPU)."""
from __future__ import annotations
import inspect
import os
from typing import Literal
import numpy as np
from scipy.sparse import csc_matrix  # type: ignore
from loguru import logger
from packaging.version import Version
from .solver_base import Solver, SolverType, SolveReport, _pfx

# [Written by Claude] EMerge must import and run without SPARTA, also when an installed SPARTA is broken (its native
# library does not load). EMERGE_DISABLE_SPARTA=1 skips importing it at all, e.g. to rule it out as the cause of a
# problem without uninstalling it; the other solvers are used instead.
AVAILABLE = False
if os.environ.get("EMERGE_DISABLE_SPARTA", "").strip().lower() in ("1", "true", "yes", "on"):
    logger.info("SPARTA disabled by EMERGE_DISABLE_SPARTA.")
else:
    try:
        import sparta  # type: ignore
        AVAILABLE = True
        if Version(sparta.__version__) < Version("0.2.0"):
            logger.error('Sparta version is outdated, install the latest release (at least 0.2.0): https://github.com/FennisRobert/SPARTA/releases/latest')
            AVAILABLE = False
    except ModuleNotFoundError:
        logger.trace("SPARTA Interface not found. (not harmful)")
    except Exception as e:
        logger.warning(
            f"SPARTA is installed but could not be loaded ({type(e).__name__}: {e}). EMerge continues with the other "
            "solvers. Reinstall it (pip install --force-reinstall sparta-solver) or set EMERGE_DISABLE_SPARTA=1 to "
            "silence this."
        )

# [Written by Claude] If SPARTA can use a Metal GPU (Apple silicon). Without one only its CPU path works.
GPU = False
if AVAILABLE:
    try:
        GPU = bool(sparta.gpu_available())
    except Exception:
        GPU = False


# [Written by Claude] SPARTA settings that EMerge sets itself, with what to use instead.
_MANAGED = {
    "force_cpu": "use set_options('CPU') or model.set_device(...)",
    "hybrid": "use set_options('HYBRID') or model.set_device(...)",
    "mtype": "EMerge declares the matrix type",
    "threads": "use set_nthreads(...)",
    "matrix": "the matrix is EMerge's system matrix",
}


# [Written by Claude]
def _analyze_params() -> set[str]:
    """The keyword arguments of the installed sparta.analyze()."""
    return set(inspect.signature(sparta.analyze).parameters) - {"matrix"}


# [Written by Claude]
def _factorize_params() -> set[str]:
    """The keyword arguments of the installed sparta Solver.factorize(); the analysis-only settings are not."""
    return set(inspect.signature(sparta.Solver.factorize).parameters) - {"self", "matrix"}


class SolverSparta(Solver):
    """Implements the SPARTA Sparse Metal based GPU solver.

    Requires SPARTA 0.2.0 or newer, installed from the latest release: https://github.com/FennisRobert/SPARTA/releases/latest
    """

    available = AVAILABLE
    gpu = GPU
    requires = "SPARTA 0.2.0 or newer, installed from the latest release: https://github.com/FennisRobert/SPARTA/releases/latest"

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP
    name = "Sparta"

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating Sparta solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None
        self._csym: bool = True
        self.factor = None
        self.sym = None
        self.initalized = False
        self.mode: Literal['CPU','GPU','HYBRID'] = 'GPU'
        self.nthreads: int | None = None
        self.settings: dict = dict(
            ordering=sparta.Ordering.SCALPEL,
            refinement_steps=0,
        )

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + "Initializing SPARTA Solver")
        self.initalized = True

    def reset(self) -> None:
        logger.trace(self.pre + "Resetting SPARTA solver state")
        if self.factor is not None:
            self.factor.destroy()
            self.factor = None
        if self.sym is not None:
            self.sym.destroy()
            self.sym = None
        self.fact_symb = False

    def set_options(self, mode: Literal['GPU','CPU','HYBRID']) -> SolverSparta:
        """Set the factorization mode for the factorization. Options are:
         - 'GPU': Use the GPU only
         - 'CPU': Use the CPU only
         - 'HYBRID': Use a Hybrid GPU + CPU solver (if available)
        """
        self.mode = mode.strip().upper()
        return self

    # [Written by Claude]
    def config(self, **settings) -> SolverSparta:
        """Set SPARTA's own settings directly; they are passed to sparta.analyze() and, where it takes them,
        to Solver.factorize(). Example: model.set_solver(em.solvers.Sparta).config(supernodal=True).

        Only settings of the installed SPARTA version are accepted; an unknown name raises a ValueError that
        names the installed version. A new setting discards the current analysis, so the next solve analyzes
        the matrix again. Calling config() again adds to (or overrides) the earlier settings.

        EMerge sets these itself and they cannot be set here: force_cpu and hybrid (use set_options or
        model.set_device), mtype (EMerge declares the matrix as complex symmetric or structurally symmetric)
        and threads (use set_nthreads). EMerge's defaults are ordering=SCALPEL and refinement_steps=0.

        Accuracy and refinement:
            refinement_steps (int, SPARTA default -1, EMerge default 0): Correction solves per right-hand side
                after the first solve. -1 takes as many as refinement_rtol needs, 0 switches refinement off, a
                positive value is a hard cap. The GPU factors are single precision: a bare GPU solve lands near
                1e-5 relative residual, and refinement brings it down in complex128.
            refinement_rtol (float, default 1e-6): Stop refining at this relative residual ||b - Ax|| / ||b||.
                This is a residual, not the solution error: an ill-conditioned matrix amplifies it, so ask for
                a tighter residual than the accuracy you need.
            refinement_minimize (bool, default True): Minimize the residual along each correction direction.
                One extra CPU matrix-vector product per step; about halves the steps on badly conditioned
                matrices.
            equilibrate (bool, default False): Balance row and column magnitudes before the float32 (GPU)
                conversion. LDLT keeps the symmetry. Extra CPU work and memory; not always better.

        Pivoting:
            static_pivot (float, default 1e-12, SPARTA 0.3.5+): LDLT static pivoting threshold relative to the
                largest matrix entry. A smaller pivot is raised to it (counted in perturbed_pivots), so a tiny
                pivot (e.g. a zero diagonal) does not ruin the factorization. 0 replaces exact zeros only. LU
                ignores it.
            lu_diag_threshold (float): LU only. 0 keeps strict partial pivoting. Above 0 a diagonal pivot is
                accepted when within this fraction of the column's best candidate: faster, less stable, and it
                changes the answers.

        Factorization method:
            supernodal (bool, default True, SPARTA 0.3.4+): Factor LDLT supernodally: faster and less memory.
                On the CPU no frontal matrix is built; on the GPU the fronts keep only the lower half of their
                trailing blocks. Same pivoting, same solve. False is the multifrontal route. LU ignores it.
            cpu_supernodal (bool, SPARTA 0.3.3 only): The 0.3.3 name of supernodal, for the CPU only.
            path (sparta.Path or str, default AUTO): Force the factorization, 'lu' or 'ldlt'. LDLT needs
                A == A.T. AUTO follows the matrix type EMerge declares.
            max_front (int, default None): Largest frontal matrix, in rows; a larger front fails the analysis.

        Ordering (the fill-reducing nested dissection, SCALPEL):
            ordering (sparta.Ordering or str, default SCALPEL): SCALPEL is the only implemented ordering.
            imbalance (float, 0.5 to 0.9): Largest share of the remaining weight in either half of a split.
                More imbalance gives smaller separators and less fill.
            wide_cycles (int): Separator refinement cycles per split. 0 orders faster and fills more; the
                default costs about 70 ms for about 4 % fewer factor entries on a 109k system.
            fm_passes (int): Fiduccia-Mattheyses refinement passes. 0 is much faster and much worse.
            separator_trials (int): Starting separators tried per split, the best is kept. Analysis time grows
                about linearly.
            nemin (int): Fewest eliminated columns before two supernodes may merge. Larger gives fewer, wider
                fronts and more factor storage. 16 suits unstructured meshes, 32 often structured grids.

        CPU kernel tuning:
            cpu_panel (int): CPU panel width in pivots (2x2 pivots may extend it by one column).
            cpu_block_rows (int): Rows per tile for small CPU updates below the Accelerate threshold.
            cpu_blas_min_rows (int): Fewest trailing rows before CPU updates use Apple Accelerate.

        GPU (Metal) kernel tuning. The last six are compiled into the Metal kernels: a combination not used
        before in this process costs about 300 ms of shader compilation once. Leave them unless you tune for
        other hardware:
            small_front_rows (int): Front height where the GPU switches from the serial to the cooperative
                kernel. Best near 32.
            schur_batch (int): Pivot panels between two Schur drains. 0 drains once per level.
            panel_threads (int): Threads per threadgroup in the panel factorization.
            solve_threads (int): Threads per front in the triangular solves. 0 picks per front width.
            panel (int): Pivots per panel before the trailing update. Only 16 is supported.
            schur_block (int, 1 to 8): Per-thread register block edge in the Schur update.
            strip_block (int, 1 to 8): Per-thread register block height in the strip update.
            narrow_cols (int): Fully summed width above which a front takes the wide kernel path.
            pivot_threads (int): Threads on one pivot search, a multiple of 32.
            schur_chunk (int): Panel columns staged per Schur pass.

        Other:
            config (sparta.Config or str): A SPARTA preset or a profile saved by sparta.optimize(); the other
                settings given here still override it.
            safe_mode (bool, default False): Check every new matrix for infinities, NaNs and its storage.
            verbose (bool, default False): Print a short summary of every solve to stderr.

        Args:
            **settings: SPARTA settings by name, see above.

        Returns:
            SolverSparta: This solver.
        """
        accepted = _analyze_params()
        for name, value in settings.items():
            if name in _MANAGED:
                raise ValueError(f"SPARTA setting '{name}' is set by EMerge: {_MANAGED[name]}")
            if name not in accepted:
                raise ValueError(
                    f"SPARTA {sparta.__version__} has no setting '{name}' (newer settings need a newer SPARTA: "
                    "https://github.com/FennisRobert/SPARTA/releases/latest)"
                )
            if name == "ordering" and isinstance(value, str):
                value = sparta.Ordering(value.lower())
            elif name == "path" and isinstance(value, str):
                value = sparta.Path(value.lower())
            self.settings[name] = value
        if settings and self.sym is not None:
            logger.trace(f"{self.pre}SPARTA settings changed, analyzing again.")
            self.reset()
        return self
    
    # [Written by Claude]
    def set_nthreads(self, nthreads: int | None) -> SolverSparta:
        """Install the number of CPU threads SPARTA uses for its analysis, CPU factorization, solves and the
        hybrid's CPU share. None leaves it to SPARTA (the process-wide pool sized by RAYON_NUM_THREADS).

        SPARTA sizes its thread pool in analyze(), so a different count discards the current analysis.

        Args:
            nthreads (int | None): The number of threads.
        """
        if Version(sparta.__version__) < Version("0.3.0"):
            return super().set_nthreads(nthreads)
        if nthreads != self.nthreads and self.sym is not None:
            logger.trace(f"{self.pre}SPARTA thread count changed to {nthreads}, analyzing again.")
            self.reset()
        self.nthreads = nthreads
        return self
    
    def set_symmetry(self, complex_symmetric: bool) -> None:
        self._csym = complex_symmetric

    def duplicate(self) -> Solver:
        # [Written by Claude] Worker copies keep the mode and thread count
        new = self.__class__(self.pre).set_options(self.mode).set_nthreads(self.nthreads)
        new.settings = dict(self.settings)
        return new

    def solve(self, A: csc_matrix, b: np.ndarray, precon, id: int = -1) -> tuple[np.ndarray, SolveReport]:

        modemap = {
            'GPU': (False, False),
            'CPU': (True, False),
            'HYBRID': (False, True),
        }
        cpu, hybrid = modemap.get(self.mode, (False, False))
        
        logger.info(f"{_pfx(self.pre, id)} Calling SPARTA Solver ({self.mode}, {sparta.__version__}).")
        logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
        
        mtype = sparta.MatrixType.COMPLEX_STRUCTURALLY_SYMMETRIC

        if self._csym:
            mtype = sparta.MatrixType.COMPLEX_SYMMETRIC
            
        if self.sym is None:
            # threads is an analysis setting only; factorize() does not accept it
            threads = {} if self.nthreads is None else dict(threads=self.nthreads)
            self.sym = sparta.analyze(A, force_cpu=cpu, hybrid=hybrid, mtype=mtype, **threads, **self.settings)
        
        # [Written by Claude] factorize() only takes part of the settings; the analysis-only ones (e.g. nemin)
        # were applied by analyze().
        fparams = _factorize_params()
        fsettings = {k: v for k, v in self.settings.items() if k in fparams}
        self.factor = self.sym.factorize(A, force_cpu=cpu, hybrid=hybrid, mtype=mtype, **fsettings)
        if not self.factor.quality.usable:
            logger.warning('Matrix is so singular that it is broken on single precision. Use a double-precision CPU solver instead.')

        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = self.factor.solve_many(b)
        
        return x, SolveReport(solver=str(self), exit_code=0, aux=None)
