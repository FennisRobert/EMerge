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
from scipy.optimize import minimize
from scipy.optimize import direct as _direct
import numpy as np
from typing import Generator, Callable, Literal
from loguru import logger

class _StopMinimize(Exception):
    pass

class OptimizationError(Exception):
    pass

def _null_callback(*args, **kwargs):
    return

# All scipy.optimize.minimize methods are deterministic given identical
# starting points, bounds and cached function values, so they replay
# correctly under this optimizer's "raise-and-resume" trick.
# 'direct' (scipy.optimize.direct - the DIRECT global search algorithm) is
# also fully deterministic and is supported as a distinct code path below,
# since it does not take an x0 and is not a valid `method=` for minimize().
OptimizerMethod = Literal[
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'COBYQA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov',
    'direct',
]

class Optimizer:
    
    def __init__(self):
        self.clear_mesh: bool = True
        self.value_cache: dict[np.ndarray, float] = dict()
        self._param_data: list[tuple[str, tuple[float, float, float]]] = []
        self.last_iter: np.ndarray = None
        self.method: OptimizerMethod = 'Powell'
        self._stop: bool = False
        self.callback: Callable = _null_callback
        self._updated: bool = False
        self._maximize: bool = False
        self.tolerance: float = 0.0001
        
    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(p[1][1], p[1][2]) for p in self. _param_data]
    
    @property
    def x0(self) -> np.ndarray:
        return np.array([p[1][0] for p in self._param_data])
    
    @property
    def last(self) -> tuple[float, ...]:
        return tuple([x for x in self.last_iter])
    
    @property
    def params(self) -> dict[str, float]:
        return {p[0]: value for p, value in zip(self._param_data, self.last_iter)}
    
    
    @property
    def N(self):
        return len(self.value_cache)
    
    def maximize(self) -> None:
        """ Sets the optimizer to a maximization instead of minimization. """
        self._maximize = True

    def set_method(self, method: OptimizerMethod) -> None:
        """Set the optimization method used on the next call to .run().

        Accepts any scipy.optimize.minimize method string, or 'direct' for
        scipy.optimize.direct (the DIRECT global search algorithm).

        Note: 'direct' requires every parameter to have finite lower and
        upper bounds - it will not accept the open bounds that the local
        minimize() methods tolerate.

        Args:
            method: One of 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'COBYQA', 'SLSQP',
                'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
                'trust-krylov', or 'direct'.
        """
        logger.debug(f'Setting optimization method to {method}')
        self.method = method

    def reset(self):
        """Reset the optimizer state
        """
        logger.info('Resetting optimizer')
        self.value_cache = {}
        self._param_data = []
        self.last_iter = None
        self.method = 'Powell'
        self._stop = False
        self.callback: Callable = _null_callback
        self.clear_mesh = True
        self._updated: bool = False
        
    def add_param(self, name: str, x0: float, bounds: tuple[float, float] = (None, None)) -> None:
        """Add a new optimization parameter to the optimizer

        Args:
            name (str): _description_
            x0 (float): _description_
            bounds (tuple[float, float], optional): _description_. Defaults to (None, None).
        """
        logger.debug(f'Adding {name}={x0} ∈ ({bounds[0]},{bounds[1]})')
        self._param_data.append((name, (x0, bounds[0], bounds[1])))

    def _check_finite_bounds(self) -> None:
        """Raise a clear error if 'direct' is selected but a bound is open.

        scipy.optimize.direct requires a fully finite bounded box; unlike
        the local minimize() methods it will not accept None on either side.
        """
        for name, (_, lo, hi) in self._param_data:
            if lo is None or hi is None:
                raise OptimizationError(
                    f"Parameter '{name}' has an open bound ({lo}, {hi}). "
                    "The 'direct' method requires finite lower and upper "
                    "bounds on every parameter - please supply both when "
                    "calling add_param()."
                )
    
    def run(self, max_iter: int = 1_000, clear_mesh: bool = True) -> Generator[tuple[float,...], None, None]:
        """Run an optimization sweep

        Be careful that all results will be saved in RAM, so constrain the maximum number of iterations.
        Also make sure to call .update(value) with a metric that determines the quality of the latest solution.
        
        Args:
            max_iter (int, optional): The maximum number of iterations. Defaults to 1_000.
            clear_mesh (bool, optional): If the entire mesh should be cleared and rebuild each iteration. Defaults to True.

        Yields:
            Generator[tuple[float,...], None, None]: A tuple of the parameters of the latest iterations
        """
        i = 0
        logger.info('Starting optimization run!')
        
        self.clear_mesh = clear_mesh
        Q = 1.0
        if self._maximize:
            Q = -1.0

        if self.method == 'direct':
            self._check_finite_bounds()
            
        while not self._stop:
            
            i += 1
            logger.info(f'Optimization step {i}')
            
            if i>max_iter:
                break
            
            if i>1:
                if not self._updated:
                    raise OptimizationError('You must call .update() after each optimization step with the new optimization value.')
                self._updated = False
            
            success = True
            def f(x):
                if tuple(x) in self.value_cache:
                    return self.value_cache[tuple(x)]*Q
                
                self.last_iter = x
                raise _StopMinimize

            try:
                if self.method == 'direct':
                    # scipy.optimize.direct has no x0 - it deterministically
                    # partitions the full bounded box, so it is called
                    # identically on every restart, exactly like minimize()
                    # is below.
                    _direct(
                        f,
                        bounds=self.bounds,
                        maxiter=max_iter,
                    )
                else:
                    options = {'maxiter': max_iter,}
                    minimize(
                        f,
                        self.x0,
                        method=self.method,
                        bounds=self.bounds,
                        options=options,
                        tol=self.tolerance,
                    )
            except _StopMinimize:
                success = False
                pass
            
            
            
            if success:
                logger.info(f'Optimization succesfull! Best result: {self.best}')
                break
            
            self.callback()
            
            logger.info(f'New iter = {self.params}')
            yield self.last
            
            

    def update(self, value: float):
        """Call this function to inform the optimizer of the latest result

        Args:
            value (float): _description_
        """
        logger.info(f'Latest iteration metric: {value}')
        self.value_cache[tuple(self.last_iter)] = value
        self._updated = True

    def stop(self) -> None:
        self._stop = True
        
    @property
    def best(self) -> tuple[dict[str, float], float]:
        if not self._maximize:
            smallest_key = sorted(self.value_cache.keys(), key=lambda x: self.value_cache[x])[0]
        else:
            smallest_key = sorted(self.value_cache.keys(), key=lambda x: self.value_cache[x])[-1]
        return {p[0]: value for p, value in zip(self._param_data, smallest_key)}, self.value_cache[smallest_key]