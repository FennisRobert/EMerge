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

# Last Cleanup: 2025-01-01
import numpy as np
import os
from typing import Hashable
from scipy.sparse import csc_matrix, save_npz, load_npz, issparse  # type: ignore
from ...solver import SolveReport, MatrixType
from loguru import logger


class SimJob:

    def __init__(
        self,
        A: csc_matrix,
        b_vectors: dict[float | int, np.ndarray] | None,
        freq: float,
        B: csc_matrix | None = None,
        symmetric: bool = False,
    ):

        self.A: csc_matrix = A
        self.B: csc_matrix | None = B
        if symmetric:
            self.mtype: MatrixType = MatrixType.COMPLEX_SYMMETRIC
        else:
            self.mtype: MatrixType = MatrixType.UNSYMMETRIC
        if b_vectors is not None:
            self.nb: int = len(b_vectors)
            self.b_ind_map: dict[Hashable, np.ndarray] = {
                key: i for i, key in enumerate(b_vectors.keys())
            }
            self.b_vectors: np.ndarray = np.zeros(
                (A.shape[0], self.nb), dtype=np.complex128
            )
            for key, index in self.b_ind_map.items():
                self.b_vectors[:, index] = b_vectors[key]
        else:
            self.b_vectors = None
        self.P: csc_matrix | None = None
        self.has_periodic: bool = False
        self.solve_ids: np.ndarray | None = None
        self.freq: float = freq
        self.k0: float = 2 * np.pi * freq / 299792458
        self.is_solved: bool = False

        self._solutions: np.ndarray | None = None
        self._solutions_dict: dict[Hashable, np.ndarray] = None

        self.store_data: bool = False

        self.relative_path: str | None = None
        self._store_location: dict = {}
        self._stored: bool = False

        self._sorter: np.ndarray | None = None
        self._isorter: np.ndarray | None = None

        self._active_b_key: int = -1
        self.reports: list[SolveReport] = []
        self.id: int = -1

    def ensure_directory(self) -> None:
        """Ensures that the store directory exists."""
        os.makedirs(self.relative_path, exist_ok=True)

    def gen_filename(self, matrix: np.ndarray | csc_matrix, name: str) -> str:
        if issparse(matrix):
            return os.path.join(
                self.relative_path, f"csc_{str(self.freq).replace('.','_')}_{name}.npz"
            )
        elif isinstance(matrix, np.ndarray):
            return os.path.join(
                self.relative_path, f"np_{str(self.freq).replace('.','_')}_{name}.npy"
            )

    def maybe_store(self, matrix: csc_matrix | np.ndarray, name: str) -> None:
        if matrix is not None and self.store_data:
            # Create temp directory if needed
            self.ensure_directory()
            path = self.gen_filename(matrix, name)
            logger.trace(f'   Caching Matrix {name} as "{path}"')
            if issparse(matrix):
                save_npz(path, matrix, compressed=False)
            else:
                np.save(path, matrix)
            self._store_location[name] = path
            self._stored = True
            return None  # Unload from memory

        return matrix

    def set_sorter(self, order: np.ndarray):
        self._sorter = order
        self._isorter = np.argsort(order)

    def store_if_needed(self, path: str):
        self.store_data = True
        self.relative_path = path
        logger.debug(f'Caching matrices in "{path}".')
        self.A = self.maybe_store(self.A, "A")
        self.b_vectors = self.maybe_store(self.b_vectors, "b_vectors")
        if self.has_periodic:
            self.P = self.maybe_store(self.P, "P")
        if self.B is not None:
            self.B = self.maybe_store(self.B, "B")

    def load_if_needed(self, name: str):
        if name in self._store_location:
            filename = self._store_location[name]
            logger.trace(f'   loading {name} from "{filename}".')
            if ".npy" in filename:
                return np.load(filename)
            else:
                return load_npz(filename)
        return getattr(self, name)

    def get_Ab(self) -> tuple[csc_matrix, np.ndarray, np.ndarray, bool, dict]:
        # Set port as active and add the port mode to the forcing vector

        self.A = self.load_if_needed("A")
        self.b_vectors = self.load_if_needed("b_vectors")

        aux = dict()
        return self.A, self.b_vectors, self.solve_ids, aux

    def get_AC(self):
        self.A = self.load_if_needed("A")
        self.B = self.load_if_needed("B")
        return self.A, self.B, self.solve_ids

    def store_solution(self):
        self._solution = self.maybe_store(self._solutions, "_solutions")

    def load_solutions(self) -> None:
        solutions = self.load_if_needed("_solutions")
        self._solutions_dict = dict()
        for key, index in self.b_ind_map.items():
            self._solutions_dict[key] = solutions[:, index]
        self.cleanup_data()

    def clear_solutions(self) -> None:
        self._solutions = None
        self._solutions_dict = None

    def unpermute_solutions(self, solution: np.ndarray) -> np.ndarray:
        # Called after eigenmode
        if self.has_periodic:
            return (self.P @ solution.T).T
        else:
            return solution

    def submit_solution(self, solution: np.ndarray, report: SolveReport):
        # Solve the Ax=b problem
        if self.has_periodic:
            solution = self.P @ solution
        self._solutions = solution
        self.reports.append(report)
        self.routine = None
        self.is_solved = True
        self.store_solution()
        self.A = None
        self.B = None

    def cleanup_data(self):
        if not self._stored:
            return

        if not os.path.isdir(self.relative_path):
            return
        logger.debug("Cleaning up directory.")
        # Remove only the files we saved
        for path in self._store_location.values():
            if os.path.isfile(path):
                os.remove(path)
                logger.trace(f'   removing "{path}"')

        # If the directory is now empty, remove it
        if not os.listdir(self.relative_path):
            logger.trace(f'   removing directory "{self.relative_path}"')
            os.rmdir(self.relative_path)
