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

# [Written by Claude]
"""The number of CPU threads EMerge uses, set with em.set_threads()."""
from __future__ import annotations
from loguru import logger
from emr import pool as _emr_pool

_NTHREADS: int | None = None


def set_threads(nthreads: int | None) -> None:
    """Set the number of CPU threads EMerge uses, e.g. em.set_threads(6).

    The count applies to the compiled kernels (matrix assembly, interpolation, far fields, mesh processing) and
    is the default thread count of the solvers that support one. An nthreads argument of a run method, e.g.
    run_sweep(nthreads=4), still takes precedence for the solver. It can be called at any time.

    run_sweep_distributed() solves n_workers frequency points at the same time, each with this many solver
    threads.

    Args:
        nthreads (int | None): The number of threads. None returns to the defaults: all logical cores for the
            kernels (or RAYON_NUM_THREADS) and each solver's own setting.
    """
    global _NTHREADS
    if nthreads is not None and (not isinstance(nthreads, int) or nthreads < 1):
        raise ValueError(f"The number of threads must be a positive integer or None, not {nthreads!r}.")
    _emr_pool.set_num_threads(nthreads)
    _NTHREADS = nthreads
    logger.debug(f"EMerge uses {get_threads()} CPU threads.")


def get_threads() -> int:
    """The number of CPU threads EMerge uses (set with em.set_threads, otherwise the kernels' default)."""
    return _emr_pool.get_num_threads()


def solver_threads(nthreads: int | None) -> int | None:
    """The thread count for a solve: nthreads if given, otherwise the one set with em.set_threads (or None)."""
    return nthreads if nthreads is not None else _NTHREADS
