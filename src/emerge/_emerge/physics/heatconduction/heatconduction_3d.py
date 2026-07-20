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

# Last Cleanup: 2026-03-04
from ...mesher import Mesher
from ...mesh3d import Mesh3D
from ...elements.leg2 import Legrange2
from ...solver import SolveRoutine, MatrixType, SolveReport, AutomaticRoutine
from ...settings import Settings
from ...simstate import SimState

from .heatconduction_bc import HCBoundaryConditionSet
from .heatconduction_data import HCData
from .assembly.assembler import Assembler
from .simjob import SimJob

from loguru import logger
from typing import Any, Literal
import multiprocessing as mp
import numpy as np
import threading
import time

from ..physics_generic import SimulationError, GenericPhysics3D

############################################################
#                 MULTI PROCESSING FUNCTION                #
############################################################


def run_job_multi(job: SimJob) -> SimJob:
    """The job launcher for Multi-Processing environements

    Args:
        job (SimJob): The Simulation Job

    Returns:
        SimJob: The solved SimJob
    """
    nr = int(mp.current_process().name.split("-")[1])
    routine = AutomaticRoutine()._configure_routine("MP", proc_nr=nr)
    A, bmat, ids, aux = job.get_Ab()
    solution, report = routine.solve(A, bmat, ids, id=job.id)
    report.add(**aux)
    job.submit_solution(solution, report)
    return job


def _dimstring(data: list[float] | np.ndarray) -> str:
    """A String formatter for dimensions in millimeters

    Args:
        data (list[float]): The list of floating point dimensions

    Returns:
        str: The formatted string
    """
    return "(" + ", ".join([f"{x * 1000:.1f}mm" for x in data]) + ")"


class HeatConduction3D(GenericPhysics3D):
    def __init__(
        self, state: SimState, mesher: Mesher, settings: Settings, order: int = 2
    ):

        self._settings: Settings = settings

        self.T_initial_K: float = 293.15
        self.order: int = order  # Discretization order. Is always 2 (legacy)

        self.mesher: Mesher = mesher  # A reference to the Mesher object
        self._state: SimState = state  # A reference to the Simulation stat object

        self.assembler: Assembler = Assembler(self._settings)  # The assembler class
        self.bc: HCBoundaryConditionSet = HCBoundaryConditionSet(
            None
        )  # The boundary condition set class.
        self.basis: Legrange2 | None = None
        self.solveroutine: SolveRoutine = AutomaticRoutine()
        self.cache_matrices: bool = True

        ## States
        self._bc_initialized: bool = False
        self._simstart: float = 0.0
        self._simend: float = 0.0
        self._container: dict[str, Any] = dict()
        self._completed: bool = False

    @property
    def _params(self) -> dict[str, float]:
        return self._state.params

    @property
    def mesh(self) -> Mesh3D:
        return self._state.mesh

    @property
    def data(self) -> HCData:
        return self._state.data.hc

    def _check_meshed(self) -> None:
        """Checks if a mesh is generated"""
        if not self.mesh.defined:
            raise SimulationError("Mesh is not defined. Call generate_mesh() first!")

    def _initialize_field(self):
        """Initializes the physics basis to the correct FEMBasis object.

        Currently it defaults to Nedelec2. Mixed basis are used for modal analysis.
        This function does not have to be called by the user. Its automatically invoked.
        """
        if self.basis is not None:
            return
        if self.order == 1:
            raise NotImplementedError("Legrange order 1 is currently not supported")
        elif self.order == 2:
            self.basis = Legrange2(self.mesh)

    def _initialize_bc_data(self):
        """Initializes auxilliary required boundary condition information before running simulations."""
        logger.debug("(Heat conduction) Initializing boundary condition data")
        # Removes non-assigned boundary conditions.
        # This happens for example if the initial boundary PEC gets overwritten.
        self.bc.cleanup()
        self.bc._selections_post_boolean_fragment()

    def _check_physics(self) -> None:
        """Executes a physics check before a simulation can be run.

        Raises:
           BoundaryConditionError: If any boundary condition is not setup correctly

        """

        pass

    def set_initial_temperature(self, temp_K: float) -> None:
        """Define the initial temperature of the simulation.

        This is used by the black body radiation boundary condition

        Args:
            temp_K (float): The temperature in Kelvin
        """
        self.T_initial_K = temp_K

    def run_steady_state(
        self, direct_solver: bool = True, preconditioner: bool = False
    ) -> HCData:
        """Run a steady state thermal analysis (Linear).

        This function can be used to just solve linear steady state heat conduction problems.

        For non-linear problems such as with black-body radiation boundary conditions you can use: run_steady_state_nl

        Args:
            direct_solver (bool, optional): If a direct solver is to be used. Defaults to True.
            preconditioner (bool, optional): If a preconditioner is to be used in case of Iterative solvers. Defaults to False.

        Returns:
            HCData: The Heat Conduction Data object.
        """

        self._completed = False
        self._simstart = time.time()

        # --------------------------------------------------------------------
        # Local Variables
        # --------------------------------------------------------------------

        material_set: tuple[np.ndarray,] = []

        # --------------------------------------------------------------------
        # Checks
        # --------------------------------------------------------------------

        if self.bc._initialized_with_defaults is False:
            raise SimulationError(
                "Cannot run a modal analysis because no default boundary conditions have been assigned."
            )

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        # --------------------------------------------------------------------
        # Material Assignments
        # --------------------------------------------------------------------

        logger.debug("Resolving material assingments.")
        materials = self._get_material_assignment(self.mesher.volumes)

        # --------------------------------------------------------------------
        # Initializing solve functions
        # --------------------------------------------------------------------

        thread_local = None

        def get_routine() -> SolveRoutine:
            if not hasattr(thread_local, "routine"):
                worker_nr = int(threading.current_thread().name.split("_")[1]) + 1
                thread_local.routine = self.solveroutine.duplicate()._configure_routine(
                    "MT", thread_nr=worker_nr
                )
            return thread_local.routine

        # Single threaded does not need this routine as the class's own routine can be used.
        def run_job_single(job: SimJob) -> SimJob:
            A, bmat, ids, aux = job.get_Ab()
            solution, report = self.solveroutine.solve(
                A, bmat, ids, id=job.id, direct=direct_solver
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        # Assemble the FEM problem
        job, mats, T_current = self.assembler.assemble_hc_matrix(
            self.basis, materials, self.bc.boundary_conditions, self.T_initial_K
        )
        material_set = mats

        # Solver configuration
        if preconditioner:
            self.solveroutine.use_preconditioner = True

        # Finally solve the problems for each frequency individually.
        logger.info("Starting single threaded solve.")
        job = run_job_single(job)

        self.solveroutine.reset()

        logger.info("Solving complete")

        # --------------------------------------------------------------------
        # Writing solve reports
        # --------------------------------------------------------------------

        self.data.setreport(job.reports, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        # --------------------------------------------------------------------
        # Post Processing
        # --------------------------------------------------------------------

        self._post_process(job, material_set)
        self._completed = True
        return self.data

    def run_transient(
        self,
        run_time_s: float,
        dt_s: float,
        direct_solver: bool = True,
        preconditioner: bool = False,
        stepping_algorithm: Literal["BackwardEuler", "CrankNicolson"] = "CrankNicolson",
    ) -> HCData:
        """Run a transient simulation with a given total runtime and time step

        This function runs a transient simulation of the linear thermal problem. Non linear transient simulations are not yet supported.
        The transient solver currently supports two time stepping algorithms:
            Backward-Euler: "BackwardEuler"
            Crank-Nicolson: "CrankNicolson"

        Please bear in mind that the Crank-Nicolson time stepping algorithm can have oscillations but it is more accurate than backward Euler.

        Args:
            run_time_s (float): The total run time in seconds
            dt_s (float): The time step in seconds
            direct_solver (bool, optional): If a direct solver is to be used. Defaults to True.
            preconditioner (bool, optional): If a preconditioner is to be used in case of Iterative solvers. Defaults to False.
            stepping_algorithm (Literal['BackwardEuler','CrankNicolson']): The time stepping algorithm to use

        Returns:
            HCData: The Heat Conduction Data object.
        """
        self._completed = False
        self._simstart = time.time()

        # --------------------------------------------------------------------
        # Local Variables
        # --------------------------------------------------------------------

        material_set: tuple[np.ndarray,] = []

        # --------------------------------------------------------------------
        # Checks
        # --------------------------------------------------------------------

        if self.bc._initialized_with_defaults is False:
            raise SimulationError(
                "Cannot run a modal analysis because no default boundary conditions have been assigned."
            )

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        # --------------------------------------------------------------------
        # Material Assignments
        # --------------------------------------------------------------------

        logger.debug("Resolving material assingments.")
        materials = self._get_material_assignment(self.mesher.volumes)

        # --------------------------------------------------------------------
        # Initializing solve functions
        # --------------------------------------------------------------------

        # Assemble the FEM problem
        job, mats, T_current = self.assembler.assemble_hc_matrix(
            self.basis,
            materials,
            self.bc.boundary_conditions,
            self.T_initial_K,
            transient=True,
        )
        material_set = mats

        # Solver configuration
        if preconditioner:
            self.solveroutine.use_preconditioner = True

        # --- Time Iteration Scheme
        A_left, A_right, b, solve_ids, aux = job.get_ABb(dt_s, stepping_algorithm)
        b = b.ravel()
        jobs = []

        T_full = T_current.copy()
        T_free = T_full[job.i_free]
        t_values = np.arange(0, run_time_s, dt_s)

        for istep, tstep in enumerate(t_values):
            if istep == 0:
                jobs.append(
                    job.submit_copy(T_free.reshape((job.n_dof, 1)), SolveReport())
                )
                continue

            logger.info(f"Simulation time ({istep}) = {tstep:.2f} s")

            rhs = A_right @ T_free + b
            solution, report = self.solveroutine.solve(
                A_left,
                rhs,
                solve_ids,
                id=job.id,
                direct=direct_solver,
                matrix_type=MatrixType.SPD,
            )

            solution = solution.ravel()
            report.add(**aux)
            newjob = job.submit_copy(solution, report)
            jobs.append(newjob)
            T_free = solution
        self.solveroutine.reset()

        logger.info("Solving complete")

        # --------------------------------------------------------------------
        # Writing solve reports
        # --------------------------------------------------------------------
        for job in jobs:
            self.data.setreport(job.reports, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        # --------------------------------------------------------------------
        # Post Processing
        # --------------------------------------------------------------------

        self._post_process_transient(jobs, t_values, material_set)
        self._completed = True
        return self.data

    def run_steady_state_nl(
        self,
        direct_solver: bool = True,
        preconditioner: bool = False,
        max_nonlinear_iter: int = 50,
        nonlinear_tol: float = 1e-6,
        anderson_m: int = 5,
        anderson_beta: float = 1.0,
        anderson_reg: float = 0.0,
    ):
        """Run a non linear steady state thermal analysis.

        This function can be used to just solve non linear steady state of heat conduction problems.
        It uses Anderson Acceleration to converge faster to the steady state solution.

        Args:
            direct_solver (bool, optional): If a direct solver is to be used. Defaults to True.
            preconditioner (bool, optional): If a preconditioner is to be used in case of Iterative solvers. Defaults to False.
            max_nonlinear_iter (int, optional): The maximum number of iterations
            nonlinear_tol (float, optional): The tolerance before a solution is considered converged
            anderson_m (int, optional): The window size for the Anderson update calculation.

        Copyright clarification:
        The logic of the solve loop including the Anderson update is a mix of hand written code
        And code suggestions by Claude Code. The code in this function is mostly modified by Claude from
        the steady state solver to implement the non-linear solve loop. The code in this function is not entirely written by Claude code.

        Returns:
            HCData: The Heat Conduction Data object.
        """

        self._completed = False
        self._simstart = time.time()

        material_set: tuple[np.ndarray,] = []

        if self.bc._initialized_with_defaults is False:
            raise SimulationError(
                "Cannot run a modal analysis because no default boundary conditions have been assigned."
            )

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        logger.debug("Resolving material assignments.")
        materials = self._get_material_assignment(self.mesher.volumes)

        def run_job_single(job: SimJob) -> SimJob:
            A, bmat, ids, aux = job.get_Ab()
            solution, report = self.solveroutine.solve(
                A,
                bmat,
                ids,
                id=job.id,
                direct=direct_solver,
                matrix_type=MatrixType.SYMMETRIC,
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        if preconditioner:
            self.solveroutine.use_preconditioner = True

        # Initial solve without radiation
        T_current = self.T_initial_K

        T_history = []
        R_history = []

        for nl_iter in range(max_nonlinear_iter):
            logger.info(f"Nonlinear iteration {nl_iter + 1}/{max_nonlinear_iter}")

            # Assemble with current temperature for radiation linearization
            job, mats, T_current = self.assembler.assemble_hc_matrix(
                self.basis, materials, self.bc.boundary_conditions, T_current
            )
            material_set = mats

            job = run_job_single(job)

            T_new = job.solution.copy()

            # Fixed-point residual
            R = T_new - T_current
            res_norm = np.linalg.norm(R)
            sol_norm = np.linalg.norm(T_new)

            rel_res = res_norm / sol_norm if sol_norm > 0 else res_norm

            logger.info(
                f"  |dT| = {res_norm:.4e}, |T| = {sol_norm:.4e}, rel = {rel_res:.4e}"
            )

            if rel_res < nonlinear_tol:
                logger.info(
                    f"Nonlinear solver converged in {nl_iter + 1} iterations "
                    f"(rel residual = {rel_res:.2e})"
                )
                break

            # Store history for Anderson mixing
            T_history.append(T_current.copy())
            R_history.append(R.copy())

            if len(T_history) > anderson_m + 1:
                T_history.pop(0)
                R_history.pop(0)

            # Anderson acceleration or simple update
            if len(T_history) >= 2:
                T_current = self._anderson_update(
                    T_history, R_history, anderson_m, anderson_reg, anderson_beta
                )
            else:
                T_current = T_new

        else:
            logger.warning(
                f"Nonlinear solver did NOT converge in {max_nonlinear_iter} "
                f"iterations (rel residual = {rel_res:.2e})"
            )

        self.solveroutine.reset()
        logger.info("Solving complete")

        self.data.setreport(job.reports, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        self._post_process(job, material_set)
        self._completed = True
        return self.data

    @staticmethod
    def _anderson_update(
        T_history: list[np.ndarray],
        R_history: list[np.ndarray],
        m: int,
        reg: float = 0.0,
        beta: float = 1.0,
    ) -> np.ndarray:
        """Anderson acceleration for a good next guess estimate based on prior guesses.

        Algorithm based on the code by Josh Nguyen:

         - https://github.com/joshnguyen99/anderson_acceleration
        Based on the paper:
         - T. D. Nguyen, A. R. Balef, C. T. Dinh, N. H. Tran, D. T. Ngo, T. A. Le, and P. L. Vo. "Accelerating federated edge learning," in IEEE Communications Letters, 25(10):3282–3286, 2021.
        Blog:
         - https://joshnguyen.net/posts/anderson-acceleration

        Args:
            T_history (list[np.ndarray]): Temperature Histories Tn
            R_history (list[np.ndarray]): Residual history Tn-f(Tn-1)
            m (int): Window size
            reg (float, optional): Regularization parameter. Defaults to 0.0.
            beta (float, optional): Mixing parameter. Defaults to 0.5.

        Returns:
            np.ndarray: _description_
        """
        n_temp_solutions = len(T_history)
        avg_win_length = min(m, n_temp_solutions - 1)

        if avg_win_length == 0:
            return (1 - beta) * T_history[-1] + beta * (T_history[-1] + R_history[-1])

        Ft = np.stack(R_history[-avg_win_length - 1 :])

        RR = Ft @ Ft.T
        RR += reg * np.eye(RR.shape[0])

        try:
            RR_inv = np.linalg.inv(RR)
            alpha = np.sum(RR_inv, 1)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(RR, np.ones(Ft.shape[0]), -1)[0]

        alpha = alpha / alpha.sum() + 1e-12

        if len(T_history) <= 0:
            T_acc = T_history[-1]
        else:
            T_acc = 0
            for i in range(len(alpha)):
                alpha_i = alpha[i]
                x_i = T_history[-avg_win_length - 1 + i]
                Gx_i = x_i + Ft[i]

                T_acc += (1 - beta) * alpha_i * x_i
                T_acc += beta * alpha_i * Gx_i
        return T_acc

    def _post_process(self, job: SimJob, materialset: list[tuple[np.ndarray,]]):
        """Post Processes the heat transfer solution data

        Args:
            job (SimJob): _description_
            materialset (list[tuple[np.ndarray,]]): _description_
        """
        # Scalar data is currently not used for anything
        scalardata = self.data.scalar.new(**self._params)
        fielddata = self.data.field.new(**self._params)

        fielddata.T = job.solution
        fielddata._dcond_thermal = materialset[0]
        fielddata.basis = self.basis

    def _post_process_transient(
        self,
        jobs: list[SimJob],
        times: np.ndarray,
        materialset: list[tuple[np.ndarray,]],
    ):
        """Post Processes the transient heat transfer solution data

        Args:
            job (SimJob): _description_
            times: (np.ndarray): The simuliation time moments.
            materialset (list[tuple[np.ndarray,]]): _description_
        """

        # Scalar data is currently not used for anything
        for job, time in zip(jobs, times):
            scalardata = self.data.scalar.new(time=time, **self._params)
            fielddata = self.data.field.new(time=time, **self._params)

            fielddata.T = job.solution
            fielddata.time = time
            fielddata._dcond_thermal = materialset[0]
            fielddata.basis = self.basis
