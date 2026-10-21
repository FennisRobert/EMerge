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

from __future__ import annotations

import numpy as np
from typing import Literal
from scipy.sparse import csc_matrix  # type: ignore
from ...solver import SolveReport


class SimJob:
    def __init__(
        self,
        A: csc_matrix | None,
        B: csc_matrix | None,
        b_vec: np.ndarray,
        n_dof: int,
        i_prescribed: list[int],
        t_prescribed: list[float],
    ):

        self.A: csc_matrix = A
        self.B: csc_matrix | None = B
        self.b_vec: np.ndarray = b_vec
        self.i_prescribed: list[int] = i_prescribed

        mask = np.ones(n_dof, dtype=bool)
        mask[self.i_prescribed] = False
        self.i_free: np.ndarray = np.flatnonzero(mask)
        self.n_free: int = self.i_free.size
        self.n_dof: int = n_dof

        self.t_prescribed: np.ndarray = np.array(t_prescribed)
        self.solution: np.ndarray | None = None
        self.reports: list[SolveReport] = []
        self.id: int = -1

    def get_Ab(self) -> tuple[csc_matrix, np.ndarray, list[int], dict]:
        """Returns the Stiffness matrix and forcing vector b.

        Due to the expected data format it also returns a dict with auxilliary data that will be filled later.

        Returns:
            tuple[csc_matrix, np.ndarray, list[int], dict]: A, b, data-dict
        """
        Afd = self.A[self.i_free, :][:, self.i_prescribed]

        bvec = self.b_vec[self.i_free] - Afd @ self.t_prescribed
        solve_ids = np.array([i for i in range(self.n_free)])
        bvec = bvec.reshape((self.n_free, 1))
        return self.A[self.i_free, :][:, self.i_free], bvec, solve_ids, dict()

    def get_ABb(
        self, dt: float, stepping_algorithm: Literal["BackwardEuler", "CrankNicolson"]
    ):
        """Generalized theta-method with Dirichlet elimination.

        Algorithm proposed by Claude Code.

        Args:
            dt (float): The time step in seconds
            stepping_algorithm (str): The time stepping algorithm to use

        A_left  = M/dt + theta*K
        A_right = M/dt - (1-theta)*K
        """
        if stepping_algorithm == "BackwardEuler":
            theta = 1.0
        elif stepping_algorithm == "CrankNicolson":
            theta = 0.5

        F = self.i_free
        D = self.i_prescribed

        M = self.B
        K = self.A

        A_left_full = M / dt + theta * K
        A_right_full = M / dt - (1.0 - theta) * K

        A_left_ff = A_left_full[F, :][:, F]
        A_right_ff = A_right_full[F, :][:, F]

        A_left_fd = A_left_full[F, :][:, D]
        b_left_dirichlet = -A_left_fd @ self.t_prescribed

        A_right_fd = A_right_full[F, :][:, D]
        b_right_dirichlet = A_right_fd @ self.t_prescribed

        b_free = self.b_vec[F] + b_left_dirichlet + b_right_dirichlet
        b_free = b_free.reshape((self.n_free, 1))

        solve_ids = np.array([i for i in range(self.n_free)])

        return A_left_ff, A_right_ff, b_free, solve_ids, dict()

    def submit_solution(self, solution: np.ndarray, report: dict):
        """Submit a solution vector to the current job.

        Args:
            solution (np.ndarray): _description_
            report (dict): _description_
        """
        self.reports.append(report)
        self.A = None
        self.B = None
        self.solution = np.zeros((self.n_dof,), dtype=np.float64)
        self.solution[self.i_prescribed] = self.t_prescribed
        self.solution[self.i_free] = solution[:, 0]

    def submit_copy(self, solution: np.ndarray, report: dict) -> SimJob:
        """Submit a SimJob solution to a copy of the SimJob

        This method creates a new derived SimJob object with the given solution.
        It is used by the transient solver as no new matrices are ever generated but just
        new simulations with a different solution vector.

        Args:
            solution (np.ndarray): The solution vector
            report (dict): The Simreport object

        Returns:
            SimJob: _description_
        """
        newjob = SimJob(
            None,
            None,
            None,
            self.n_dof,
            self.i_prescribed,
            self.t_prescribed,
        )
        newjob.reports.append(report)
        newjob.solution = np.zeros((self.n_dof,), dtype=np.float64)
        newjob.solution[self.i_prescribed] = self.t_prescribed
        newjob.solution[self.i_free] = solution.ravel()
        return newjob
