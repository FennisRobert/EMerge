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

"""The solve routine: picks a solver for each system and runs the solve and eigenvalue studies."""
from __future__ import annotations
import os
import time
from typing import Literal
import numpy as np
from scipy.sparse import csc_matrix, save_npz  # type: ignore
from loguru import logger
from .mth.csc_submatrix import csc_submatrix  # [Written by Claude]
from .precon_data import PreconData
from .threads import solver_threads
from .solvers import (
    Solver, EigSolver, Sorter, Preconditioner, SolveReport, MatrixType, SolverType, classify_matrix,
    complex_to_real_block, real_to_complex_block, _pfx, ReverseCuthillMckee, JacobiPrecon, ALL_SOLVERS,
    unavailable_message, reject_emsolver, SolverSuperLU, SolverUMFPACK, SolverPardiso, SolverMUMPS, SolverAASDS, SolverCHOLMOD,
    SolverRSLAB, SolverSparta, SolverCorvid, SolverBicgstab, SolverCG, SolverLAPACK, SolverARPACK, SolverARPACK_BMA,
    SolverCuDSS,
)

# [Written by Claude] The hardware a SolveRoutine solves on, set with set_device()
Device = Literal["CPU", "GPU", "HYBRID"]
_DEVICES: tuple[str, ...] = ("CPU", "GPU", "HYBRID")


############################################################
#                   ENVIRONMENT VARIABLES                  #
############################################################

# Environment variables are used to transfer settings from the
# main solver environment to parallel-processes that otherwise
# don't have access to these settings.

_FORCED_SOLVER = os.getenv("EMERGE_MP_SOLVER", default="")
_SYMMETRY_LIMIT = os.getenv("EMERGE_SYM_LIMIT", default="0.02")



############################################################
#                       SOLVE ROUTINE                      #
############################################################


class SolveRoutine:
    """A generic class describing a solve routine.
    A solve routine contains all the relevant sorter preconditioner and solver objects
    and goes through a sequence of steps to solve a linear system or find eigenmodes.

    """

    def __init__(self, thread_nr: int = 0, proc_nr: int = 0):

        """Create the solve routine with all available solvers.

        Args:
            thread_nr (int, optional): The thread number used in the log prefix. Defaults to 0.
            proc_nr (int, optional): The process number used in the log prefix. Defaults to 0.
        """
        self.pre: str = ""
        self._set_name(thread_nr, proc_nr)
        self.device: Device = "CPU"

        self.sorter: Sorter = ReverseCuthillMckee()
        self.precon: Preconditioner = JacobiPrecon()
        self.solvers: dict[type[Solver] | type[EigSolver], Solver | EigSolver] = self._create_solvers()

        self.parallel: Literal["SI", "MT", "MP"] = "SI"
        self.smart_search: bool = False
        self.forced_solver: list[Solver | EigSolver] = []
        self.disabled_solver: list[type[Solver] | type[EigSolver]] = []

        self.symmetry_limit: float = float(_SYMMETRY_LIMIT)
        self.force_symmetric: bool = False
        self.use_sorter: bool = False
        self.use_preconditioner: bool = False
        self.use_direct: bool = True

        self.save_disk: bool = False

        for solverkey, solver in self.solvers.items():
            if solver.name in _FORCED_SOLVER:
                self.forced_solver.append(solver)

    # [Written by Claude]
    def _create_solvers(self) -> dict[type[Solver] | type[EigSolver], Solver | EigSolver]:
        """One instance of every solver that is available on this system, by class."""
        solvers = {cls: cls(self.pre) for cls in ALL_SOLVERS if cls.available}
        sparta = solvers.get(SolverSparta)
        if sparta is not None:
            sparta.set_options(self._sparta_mode())  # type: ignore
        return solvers

    # [Written by Claude]
    def _sparta_mode(self) -> Device:
        """The SPARTA mode for the device: its CPU path unless a GPU device is set and SPARTA has a GPU."""
        if self.device == "CPU" or not SolverSparta.gpu:
            return "CPU"
        return self.device

    # [Written by Claude]
    def set_device(self, device: Device) -> None:
        """Set the hardware the direct solvers of this routine use: "CPU" (default), "GPU" or "HYBRID".

         - "CPU": CPU solvers only. SPARTA, when installed, runs its CPU path.
         - "GPU": SPARTA on the Apple GPU, otherwise cuDSS on an NVIDIA GPU. Without either, the CPU solvers are
           used and a warning is given.
         - "HYBRID": SPARTA's combined GPU + CPU factorization (Apple silicon only). Without it, "GPU" is used.

        The device applies to the solver EMerge picks. Systems with fewer than 10,000 unknowns, eigenmode problems and
        thread-distributed sweeps keep using CPU solvers. A solver set with set_solver() is used as is, except that
        SPARTA's mode follows the device.

        Args:
            device (Literal["CPU", "GPU", "HYBRID"]): The device

        Raises:
            ValueError: If the device is not one of the options
        """
        self._set_device(device)
        if self.device == "CPU":
            return
        fallback = "the GPU with cuDSS" if SolverCuDSS.available else "the CPU"
        if self.device == "HYBRID" and not SolverSparta.gpu:
            logger.warning(
                f"The HYBRID device needs SPARTA on an Apple GPU, which is not available. Solving on {fallback} instead."
            )
        elif not (SolverSparta.gpu or SolverCuDSS.available):
            logger.warning(
                "No GPU solver is available (SPARTA on an Apple GPU, or cuDSS on an NVIDIA GPU: "
                "pip install emerge[cudss]). Solving on the CPU instead."
            )

    # [Written by Claude]
    def _set_device(self, device: str) -> None:
        """Set the device without the availability warnings (used for copies and worker processes)."""
        name = str(device).strip().upper()
        if name not in _DEVICES:
            raise ValueError(f"The device must be one of {', '.join(_DEVICES)}, not {device!r}.")
        self.device = name  # type: ignore
        sparta = self.solvers.get(SolverSparta) if hasattr(self, "solvers") else None
        if sparta is not None and sparta.mode != self._sparta_mode():  # type: ignore
            sparta.reset()  # the analysis depends on the mode
            sparta.set_options(self._sparta_mode())  # type: ignore

    # [Written by Claude]
    def _device_solver(self) -> Solver | None:
        """The GPU solver for the device and parallel mode, or None to pick a CPU solver."""
        if self.device == "CPU" or self.parallel == "MT":
            return None
        candidates: list[type[Solver]] = [SolverSparta] if SolverSparta.gpu else []
        if self.parallel == "SI":
            candidates.append(SolverCuDSS)  # one cuDSS solver per process and GPU
        for cls in candidates:
            solver = self.solvers.get(cls)
            if solver is not None and not isinstance(solver, tuple(self.disabled_solver)):
                return solver  # type: ignore
        return None

    def _set_name(self, thread_nr: int, proc_nr: int):
        self.pre = f"p{int(proc_nr):02d}/t{int(thread_nr):02d}"

    def _set_environ_variables(self) -> None:
        os.environ["EMERGE_MP_SOLVER"] = "_".join(
            [solver.name for solver in self.forced_solver]
        )
        os.environ["EMERGE_SYM_LIMIT"] = str(self.symmetry_limit)

    def __str__(self) -> str:
        return "SolveRoutine()"

    def _legal_solver(self, solver: Solver | EigSolver) -> bool:
        """Checks if a solver is a legal option.

        Args:
            solver (Solver): The solver to test against

        Returns:
            bool: If the solver is legal
        """
        if isinstance(solver, tuple(self.disabled_solver)):
            logger.warning(
                self.pre
                + f"The selected solver {solver} cannot be used as it is disabled."
            )
            return False
        if self.parallel == "MT" and not solver.stype == SolverType.SINGLE_MT:
            logger.warning(
                self.pre
                + f"The selected solver {solver} cannot be used in MultiThreading as it does not release the GIL"
            )
            return False
        return True

    @property
    def all_solvers(self) -> list[Solver]:
        """All available Ax=b solvers."""
        return [
                solver
                for solver in self.solvers.values()
                if not isinstance(solver, EigSolver)
            ]

    @property
    def all_eig_solvers(self) -> list[EigSolver]:
        """All available eigenvalue solvers."""
        return [
                solver
                for solver in self.solvers.values()
                if isinstance(solver, EigSolver)
            ]

    def _try_solver(self, solver_type: type[Solver]) -> Solver:
        """Try to use the selected solver or else find another one that is working.

        Args:
            solver_type (type[Solver]): The solver class to try

        Raises:
            RuntimeError: Error if no valid solver is found.

        Returns:
            Solver: The working solver.
        """
        solver = self.solvers.get(solver_type)
        if solver is not None and self._legal_solver(solver):
            return solver  # type: ignore
        for alternative in self.all_solvers:
            if self._legal_solver(alternative):
                logger.debug(self.pre + f"Falling back on legal solver: {alternative}")
                return alternative
        raise RuntimeError(
            self.pre
            + f"No legal solver could be found. The following are disabled: {self.disabled_solver}"
        )

    def duplicate(self) -> SolveRoutine:
        """Creates a copy of this SolveRoutine class object.

        Returns:
            SolveRoutine: The copied version
        """
        new_routine = self.__class__()
        new_routine._set_device(self.device)
        new_routine.parallel = self.parallel
        new_routine.smart_search = self.smart_search
        new_routine.force_symmetric = self.force_symmetric
        copies = {}
        for tpe, solver in self.solvers.items():
            copies[id(solver)] = new_routine.solvers[tpe] = solver.duplicate()
        # Forced solvers must point to the copies too, or threads share one solver (and its factorization)
        new_routine.forced_solver = [
            copies[id(s)] if id(s) in copies else s.duplicate() for s in self.forced_solver
        ]
        return new_routine

    def set_solver(self, *solvers: type[Solver] | type[EigSolver] | Solver | EigSolver) -> None:
        """Set the solver to use, as a class (e.g. em.solvers.Pardiso) or a solver instance.
        Solvers will be checked on validity for the given problem.

        Args:
            *solvers (type[Solver] | type[EigSolver] | Solver | EigSolver): The solver classes or objects

        Raises:
            RuntimeError: If the solver is not available on this system
        """
        for solver in solvers:
            reject_emsolver(solver)
            if isinstance(solver, type):
                if not solver.available:
                    raise RuntimeError(unavailable_message(solver))
                self.forced_solver = [self.solvers[solver]]
            else:
                self.forced_solver = [solver]

    def disable(self, *solvers: type[Solver] | type[EigSolver] | Solver | EigSolver) -> None:
        """Never use the given solvers.

        Args:
            *solvers (type[Solver] | type[EigSolver] | Solver | EigSolver): The solver classes (or objects) to disable
        """
        for solver in solvers:
            reject_emsolver(solver)
            self.disabled_solver.append(solver if isinstance(solver, type) else type(solver))

    def _configure_routine(
        self,
        parallel: Literal["SI", "MT", "MP"] = "SI",
        smart_search: bool = False,
        thread_nr: int = 0,
        proc_nr: int = 0,
    ) -> SolveRoutine:
        """Configure the solver with the given settings

        Args:
            parallel (Literal['SI','MT','MP'], optional):
                The solver parallism, Defaults to 'SI'.
                    - "SI" = Single threaded
                    - "MT" = Multi threaded
                    - "MP" = Multi-processing,
            smart_search (bool, optional): Wether to use smart-search solvers
            for eigenmode problems. Defaults to False.

        Returns:
            SolveRoutine: The same SolveRoutine object.
        """
        self.parallel = parallel
        self.smart_search = smart_search
        if thread_nr != 1 or proc_nr != 1:
            self._set_name(thread_nr, proc_nr)
            for solver in self.solvers.values():
                if not isinstance(solver, (Solver, EigSolver)):
                    continue
                solver.pre = self.pre
        return self

    def configure(self, pivoting_threshold: float | None = None) -> None:
        """Sets general user configurations for all solvers.

        Args:
            pivoting_threshold (float | None, optional):
                The diagonal pivoting threshold used in direct solvers. Standard values are 0.001.
                In simulations with a very low surface impedance (such as with copper walls) a much
                lower pivoting threshold is desired.
        """
        for solver in self.solvers.values():
            if isinstance(solver, Solver):
                solver.set_options(pivoting_threshold=pivoting_threshold)

    def reset(self, reset_solver_preference: bool = False, hard: bool = False) -> None:
        """Reset all solver states

        Args:
            reset_solver_preference (bool, optional): Also clear the forced and disabled solvers and return to the CPU
                device. Defaults to False.
            hard (bool, optional): Recreate all solver objects. Defaults to False.
        """
        for solver in self.solvers.values():
            solver.reset()
        self.sorter.reset()
        self.parallel = "SI"
        self.smart_search = False
        self.force_symmetric = False
        if reset_solver_preference:
            self.forced_solver = []
            self.disabled_solver = []
            self._set_device("CPU")
        if hard:
            self.solvers = self._create_solvers()

    def _get_solver(
        self, A: csc_matrix, b: np.ndarray, matrix_type: MatrixType, direct: bool = True
    ) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.

        Args:
            A (csc_matrix): The Matrix to solve for
            b (np.ndarray): the vector to solve for

        Returns:
            Solver: Returns the direct solver

        """
        for solver in self.forced_solver:
            if solver is None:
                continue
            if not self._legal_solver(solver):
                continue
            if isinstance(solver, Solver):
                return solver
        return self.pick_solver(A, b, matrix_type, direct)

    def pick_solver(
        self, A: csc_matrix, b: np.ndarray, matrix_type: MatrixType, direct: bool = True
    ) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.

        Args:
            A (csc_matrix): The Matrix to solve for
            b (np.ndarray): the vector to solve for
            matrix_type (MatrixType): Type of the matrix (e.g., SPD, HPD)
            direct (bool, optional): Whether to use a direct solver. Defaults to True.

        Returns:
            Solver: Returns the direct solver

        """
        if direct:
            if matrix_type is MatrixType.SPD and SolverCHOLMOD.available:
                return self._try_solver(SolverCHOLMOD)
            return self._try_solver(SolverSuperLU)

        else:
            if matrix_type in (MatrixType.SPD, MatrixType.HPD):
                return self._try_solver(SolverCG)
            else:
                return self._try_solver(SolverBicgstab)

    def _get_eig_solver(
        self, A: csc_matrix, b: csc_matrix, direct: bool | None = None
    ) -> EigSolver:
        """Returns the relevant eigenmode Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.

        Args:
            A (csc_matrix): The Matrix to solve for
            b (np.ndarray): the vector to solve for
            direct (bool): If the direct solver should be used.

        Returns:
            Solver: Returns the solver object

        """
        for solver in self.forced_solver:
            if isinstance(solver, EigSolver):
                return solver  # type: ignore
        if direct or A.shape[0] < 1000:
            return self.solvers[SolverLAPACK]  # type: ignore
        else:
            return self.solvers[SolverARPACK]  # type: ignore

    # [Written by Claude]
    def _solve_with(
        self,
        solver: Solver,
        Asel: csc_matrix,
        bsel: np.ndarray,
        matrix_type: MatrixType,
        id: int = -1,
        nthreads: int | None = None,
    ) -> tuple[np.ndarray, SolveReport, str, str, csc_matrix]:
        """Solve the reduced system with one solver, including its real conversion, sorting and preconditioner.

        Args:
            solver (Solver): The solver to use
            Asel (csc_matrix): The matrix on the solved DoFs
            bsel (np.ndarray): The right hand side(s) on the solved DoFs
            matrix_type (MatrixType): The matrix type
            id (int, optional): An identifier of this solve used in log messages. Defaults to -1.
            nthreads (int | None, optional): The preferred number of CPU threads of the solver. Defaults to None.

        Returns:
            tuple[np.ndarray, SolveReport, str, str, csc_matrix]: The solution on the solved DoFs, the solver report,
                the sorter name, the preconditioner name and the matrix as given to the solver.
        """
        solver.initialize()
        solver.set_symmetry(matrix_type.symmetric or self.force_symmetric)
        # None keeps what the solver has (e.g. a count set with solver.set_nthreads()), unless em.set_threads() was used
        nthreads = solver_threads(nthreads)
        if nthreads is not None:
            solver.set_nthreads(nthreads)

        if solver.real_only:
            logger.debug(f"{_pfx(self.pre, id)} Converting to real matrix")
            Asel, bsel = complex_to_real_block(Asel, bsel)

        # SORT
        sorter = "None"
        if solver.req_sorter and self.use_sorter:
            sorter = str(self.sorter)
            Asorted, bsorted = self.sorter.sort(Asel, bsel)
        else:
            Asorted, bsorted = Asel, bsel

        # Preconditioner
        precon = "None"
        if self.use_preconditioner:
            if not solver.own_preconditioner:
                self.precon.init(Asorted, bsorted)
                precon = str(self.precon)

        x_solved, report = solver.solve(Asorted, bsorted, self.precon, id=id)

        if self.use_sorter and solver.req_sorter:
            x = self.sorter.unsort(x_solved)
        else:
            x = x_solved

        if solver.real_only:
            logger.debug(f"{_pfx(self.pre, id)} Converting back to complex matrix")
            x = real_to_complex_block(x)

        return x, report, sorter, precon, Asorted

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
        solve_ids: np.ndarray,
        direct: bool = True,
        id: int = -1,
        matrix_type: MatrixType = MatrixType.UNKNOWN,
        precon_dataset: PreconData | None = None,
        k0: float | None = None,
        nthreads: int | None = None,
    ) -> tuple[np.ndarray, SolveReport]:
        """Solve the system of equations defined by Ax=b for x.

        Solve is the main function call to solve a linear system of equations defined by Ax=b.
        The solve routine will go through the required steps defined in the routine to tackle the problme.

        Args:
            A (csc_matrix): The (Sparse) matrix
            b (np.ndarray): The source vector
            solve_ids (np.ndarray): A vector of ids for which to solve the problem. For EM problems this
                implies all non-PEC degrees of freedom.
            direct (bool, optional): Use a direct solver. Defaults to True.
            id (int, optional): An identifier of this solve used in log messages and cache file names. Defaults to -1.
            matrix_type (MatrixType, optional): The matrix type. Classified from A if UNKNOWN. Defaults to MatrixType.UNKNOWN.
            precon_dataset (PreconData | None, optional): The mesh topology needed by preconditioned iterative
                solvers such as CORVID. Defaults to None.
            k0 (float | None, optional): The free space wave number of this system, used by CORVID. Defaults to None.
            nthreads (int | None, optional): The preferred number of CPU threads, passed to the solver's
                set_nthreads(). None keeps the solver's own setting. Defaults to None.

        Returns:
            tuple[np.ndarray, SolveReport]: The solution and the solve report.
        """
        if matrix_type is MatrixType.UNKNOWN:
            matrix_type = classify_matrix(A)

        if self.save_disk:
            name_A = f'SparseMatrix_{id}.npz'
            name_b = f'SolVector_{id}'
            name_solve_ids = f'SolveIds_{id}'

            save_npz(name_A, A)
            np.save(name_b, b)
            np.save(name_solve_ids, solve_ids)
            logger.info(f'Saved {name_A} and {name_b}')

        logger.debug(f"{_pfx(self.pre, id)} matrix type: {matrix_type}")

        if b.ndim == 1:
            b = b.reshape((b.shape[0], 1))
        solver: Solver = self._get_solver(A, b, matrix_type=matrix_type, direct=direct)

        NF = A.shape[0]
        NS = solve_ids.shape[0]
        NB = b.shape[1]

        Asel = csc_submatrix(A, solve_ids)  # [Written by Claude] same as A[:, solve_ids][solve_ids, :]
        bsel = b[solve_ids, :]
        nnz = Asel.nnz

        logger.debug(
            f"{_pfx(self.pre, id)} Removed {NF - NS} prescribed DOFs ({NS:,} left, {nnz:,}≠0)"
        )

        if solver.needs_precon_data:
            if precon_dataset is None:
                logger.warning(
                    f"{_pfx(self.pre, id)} {solver} needs mesh data that is not available for this system "
                    "(e.g. periodic boundaries or thin conductors). Using a direct solver instead."
                )
                solver = self.pick_solver(A, b, matrix_type, direct=True)
            else:
                solver.set_precon_data(precon_dataset, solve_ids, k0=k0)

        start = time.time()
        x, report, sorter, precon, Asorted = self._solve_with(solver, Asel, bsel, matrix_type, id, nthreads)

        if report.exit_code != 0 and solver.needs_precon_data:
            logger.warning(
                f"{_pfx(self.pre, id)} {solver} did not converge. Solving again with a direct solver."
            )
            solver = self.pick_solver(A, b, matrix_type, direct=True)
            x, report, sorter, precon, Asorted = self._solve_with(solver, Asel, bsel, matrix_type, id, nthreads)

        end = time.time()
        simtime = end - start
        logger.info(f"{_pfx(self.pre, id)} Elapsed time taken: {simtime:.3f} seconds")
        logger.debug(
            f"{_pfx(self.pre, id)} O(N^2) performance = {(NS ** (2)) / ((end - start + 1e-6) * 1e6):.1f} MDoF/s"
        )

        solution = np.zeros((NF, NB), dtype=A.dtype)

        solution[solve_ids, :] = x

        if self.save_disk:
            name_solution = f'Solution_{id}'
            np.save(name_solution, solution)
            logger.info(f'Saved {name_solution}')

        logger.debug(f"{_pfx(self.pre, id)} Solver complete!")
        report.jobid = id
        report.sorter = str(sorter)
        report.simtime = simtime
        report.nnz = A.nnz
        report.ndof = b.shape[0]
        report.nnz_solve = Asorted.nnz
        report.ndof_solve = Asorted.shape[0]
        report.precon = precon
        report.worker_name = self.pre

        return solution, report

    # [Written by Claude] One entry point for the port modes of ModalPort and WavePort.
    def eig_boundary(
        self,
        A: csc_matrix,
        B: csc_matrix,
        solve_ids: np.ndarray,
        nmodes: int,
        shifts: list[complex],
        kmax2: float,
        direct: bool | None = None,
        keep_evanescent: bool = True,
        descending: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, SolveReport]:
        """Find the boundary modes A x = λ B x (λ = -β²) of a port face closest to a list of shifts.

        Args:
            A (csc_matrix): The Stiffness matrix
            B (csc_matrix): The mass matrix
            solve_ids (np.ndarray): The free nodes (non PEC)
            nmodes (int): The number of modes to return
            shifts (list[complex]): The values of λ = -β² to search around
            kmax2 (float): The largest physical β² (k0² εr μr max), which sets the scale of the spurious cluster
            direct (bool | None, optional): Use the dense solver. Defaults to None.
            keep_evanescent (bool, optional): Keep modes with β² <= 0. Defaults to True.
            descending (bool, optional): Return the nmodes modes with the largest β² instead of those closest to
                the shifts. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, SolveReport]: The eigenvalues, the eigenvectors and the solve report
        """
        solver: SolverARPACK_BMA = self.solvers[SolverARPACK_BMA]  # type: ignore
        # A forced dense eigen solver selects the dense path
        direct = bool(direct) or any(isinstance(s, SolverLAPACK) for s in self.forced_solver)
        Asel = A[np.ix_(solve_ids, solve_ids)]
        Bsel = B[np.ix_(solve_ids, solve_ids)]
        logger.debug(self.pre + f" Removing {A.shape[0] - solve_ids.shape[0]} prescribed DOFs ({solve_ids.shape[0]} left)")

        start = time.time()
        eigen_values, eigen_modes = solver.eig(
            Asel, Bsel, nmodes, shifts, kmax2, direct=direct, keep_evanescent=keep_evanescent, descending=descending
        )
        return (
            eigen_values,
            eigen_modes,
            SolveReport(
                ndof=A.shape[0],
                nnz=A.nnz,
                ndof_solve=Asel.shape[0],
                nnz_solve=Asel.nnz,
                simtime=time.time() - start,
                solver=str(solver),
                sorter="None",
                precon="None",
            ),
        )

    def eig(
        self,
        A: csc_matrix,
        B: csc_matrix,
        solve_ids: np.ndarray,
        nmodes: int = 6,
        direct: bool | None = None,
        target_f0: float = 0.0,
        which: str = "LM",
    ) -> tuple[np.ndarray, np.ndarray, SolveReport]:
        """
        Find the eigenmodes for the system Ax = λBx for a boundary mode problem
        
        Args:
            A (csc_matrix): The Stiffness matrix
            B (csc_matrix): The mass matrix
            solve_ids (np.ndarray): The free nodes (non PEC)
            nmodes (int, optional): The number of modes to solve for. Defaults to 6.
            direct (bool | None, optional): If the direct solver should be used (always). Defaults to None.
            target_f0 (float, optional): The eigenvalue to search around. Defaults to 0.0.
            which (str, optional): The search method. Defaults to 'LM' (Largest Magnitude).
        Returns:
            tuple[np.ndarray, np.ndarray, SolveReport]: The eigenvalues, the eigenvectors and the solve report.
        """
        solver = self._get_eig_solver(A, B, direct=direct)

        NF = A.shape[0]
        NS = solve_ids.shape[0]

        logger.debug(self.pre + f" Removing {NF - NS} prescribed DOFs ({NS} left)")

        Asel = A[np.ix_(solve_ids, solve_ids)]
        Bsel = B[np.ix_(solve_ids, solve_ids)]

        start = time.time()
        eigen_values, eigen_modes = solver.eig(
            Asel, Bsel, nmodes, target_f0, which, sign=1.0
        )
        end = time.time()
        simtime = end - start

        Nsols = eigen_modes.shape[1]
        sols = np.zeros((NF, Nsols), dtype=np.complex128)
        for i in range(Nsols):
            sols[solve_ids, i] = eigen_modes[:, i]

        return (
            eigen_values,
            sols,
            SolveReport(
                ndof=A.shape[0],
                nnz=A.nnz,
                ndof_solve=Asel.shape[0],
                nnz_solve=Asel.nnz,
                simtime=simtime,
                solver=str(solver),
                sorter="None",
                precon="None",
            ),
        )

class AutomaticRoutine(SolveRoutine):
    """Defines the Automatic Routine for EMerge."""

    def pick_solver(
        self, A: csc_matrix, b: np.ndarray, matrix_type: MatrixType, direct: bool = True
    ) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        The current implementation only looks at matrix size to select the best solver. Matrices
        with a large size will use iterative solvers while smaller sizes will use either Pardiso
        for medium sized problems or SPSolve for small ones.

        Args:
            A (csc_matrix): The Matrix to solve for
            b (np.ndarray): the vector to solve for
            matrix_type (MatrixType): Type of the matrix (e.g., SPD, HPD)
            direct (bool, optional): Whether to use a direct solver. Defaults to True.

        Returns:
            Solver: A solver object appropriate for solving the problem.

        """

        N = A.shape[0]
        if not direct:
            if matrix_type in (MatrixType.COMPLEX_SYMMETRIC,):
                return self._try_solver(SolverCorvid)
            elif matrix_type in (MatrixType.SPD, MatrixType.HPD):
                return self._try_solver(SolverCG)
            else:
                return self._try_solver(SolverBicgstab)

        if N < 10_000:
            if matrix_type is MatrixType.SPD and SolverCHOLMOD.available:
                return self._try_solver(SolverCHOLMOD)
            return self._try_solver(SolverSuperLU)
        gpu = self._device_solver()  # [Written by Claude] set_device("GPU"/"HYBRID")
        if self.parallel == "SI":
            if matrix_type is MatrixType.SPD and SolverCHOLMOD.available:
                return self._try_solver(SolverCHOLMOD)
            if gpu is not None:
                return gpu
            if SolverPardiso.available:
                return self._try_solver(SolverPardiso)
            elif SolverSparta.available:
                return self._try_solver(SolverSparta)
            elif SolverAASDS.available:
                return self._try_solver(SolverAASDS)
            elif SolverRSLAB.available:
                return self._try_solver(SolverRSLAB)
            elif SolverMUMPS.available:
                return self._try_solver(SolverMUMPS)
            elif SolverUMFPACK.available:
                return self._try_solver(SolverUMFPACK)
            else:
                return self._try_solver(SolverSuperLU)
        elif self.parallel == "MP":
            if gpu is not None:
                return gpu
            if SolverSparta.available:
                return self._try_solver(SolverSparta)
            elif SolverAASDS.available:
                return self._try_solver(SolverAASDS)
            elif SolverRSLAB.available:
                return self._try_solver(SolverRSLAB)
            if SolverUMFPACK.available:
                return self._try_solver(SolverUMFPACK)
            else:
                return self._try_solver(SolverSuperLU)
        elif self.parallel == "MT":
            return self._try_solver(SolverSuperLU)

        return self._try_solver(SolverSuperLU)
