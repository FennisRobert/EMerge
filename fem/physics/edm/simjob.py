from .emdata import EMDataSet
import numpy as np
from loguru import logger
from ...solver import ParallelRoutine

class SimJob:

    def __init__(self, 
                 A: np.ndarray,
                 b: np.ndarray,
                 freq: float,
                 cache_factorization: bool = False,
                 ):

        self.A: np.ndarray = A
        self.b: np.ndarray = b

        self.freq: float = freq
        self.k0: float = 2*np.pi*freq/299792458
        self.cache_factorization: bool = cache_factorization
        self._fields: dict[int, np.ndarray] = dict()
        self.port_vectors: dict = None
        self.solve_ids = None

        self.ertri: np.ndarray = None
        self.urtri: np.ndarray = None

        self.routine = ParallelRoutine()

    def solve(self):

        reuse_factorization = False

        for key, mode in self.port_vectors.items():
            # Set port as active and add the port mode to the forcing vector
            b_active = self.b + mode

            # Solve the Ax=b problem
            solution = self.routine.solve(self.A, b_active, self.solve_ids, reuse=reuse_factorization)

            # From now reuse the factorization
            reuse_factorization = True

            self._fields[key] = solution

        self.routine = None