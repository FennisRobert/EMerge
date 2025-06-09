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
from scipy.interpolate import interpn, RegularGridInterpolator
from loguru import logger

class Axis:

    def __init__(self, name: str,
                 values: np.ndarray,
                 units: str = 'SI'):
        values = np.array(values)
        self.name: str = name
        self.values: np.ndarray = values
        self.units: str = units
        self.N: int = self.values.shape[0]

    def __call__(self, value: float) -> int:
        """Returns the index of the value in the axis

        Args:
            value (float): _description_

        Returns:
            int: _description_
        """
        return np.argmin(np.abs(self.values - value))
    


class Dataset:
    """N-dimensional field data container class
    """

    def __init__(self, axes: list[Axis] = None):
        """Initializes the N-dimensional field data container

        Field data data is expected to be of shape (..., nfield_pts)


        Args:
            mdim_data (np.ndarray): _description_
            axes (list[np.ndarray]): _description_
        """        
        mdim_data = np.array([0,])
        if axes is None:
            axes = []

        self._data: np.ndarray = mdim_data
        self._axes: list[Axis] = axes
        self._shape: tuple[int] = self._data.shape
        self._parshape: tuple[int] = self._shape[:-1]
        self._extra_dims: int = len(self._shape) - 1
        self._axes_map: dict[str, int] = {ax.name: i for i, ax in enumerate(self._axes)}
        self._ndims: int = len(self._data.shape)

        self._globals: dict[str, np.ndarray] = dict()
        self._slice: tuple[int] = None

        self._selected_data: np.ndarray = self._data
        self._selected_source: str = 'field'

        self._return_data: np.ndarray = None

        self._constants: dict = dict()

    @property
    def array(self) -> np.ndarray:
        return self._return_data
    
    def add_axis(self, name: str, values: np.ndarray, unit: str = '') -> None:
        ''' Adds an outer variable axis to the dataset in which the data can be stored.'''
        self._axes.append(Axis(name, values, unit))

    def init(self, n_field: int) -> None:
        dimensions = [ax.N for ax in self._axes] + [n_field,]
        self._data = np.zeros(tuple(dimensions), dtype=np.complex128)
        self._shape: tuple[int] = self._data.shape
        self._parshape: tuple[int] = self._shape[:-1]
        self._extra_dims: int = len(self._shape) - 1
        self._axes_map: dict[str, int] = {ax.name: i for i, ax in enumerate(self._axes)}
        self._ndims: int = len(self._data.shape)

        for key in self._globals:
            self._globals[key] = np.zeros(self._parshape, dtype=self._globals[key].dtype)

        self._current_iter: tuple[int] = ()

        self._reset_sel()

    @property
    def field(self) -> Dataset:
        self._selected_data = self._data
        self._selected_source = 'field'
        return self
    
    def ax(self, name: str) -> np.ndarray:
        return self._axes[self._axes_map[name]].values
    
    def glob(self, var: str) -> Dataset:
        if var not in self._globals:
            raise ValueError(f'Global variable {var} not found in dataset')
        self._selected_data = self._globals[var]
        self._selected_source = var
        return self
    
    def add_global(self, name: str, dtype = np.float64) -> None:
        self._globals[name] = np.zeros(self._parshape, dtype=dtype)

    @property
    def _s_ndims(self) -> int:
        return len(self._selected_data.shape)
    
    @property
    def shape(self) -> tuple[int]:
        return self._shape
    
    @property
    def _outer_axes(self) -> tuple:
        return tuple([ax for ax in self._axes[:self._extra_dims]])
    
    @property
    def _outer_slice(self) -> tuple:
        return tuple([slice(None) for _ in range(self._extra_dims)])
    
    def _reset_sel(self) -> None:
        self._selected_data = self._data
        self._selected_source = 'field'

    def item(self, index: int) -> Dataset:
        ''' Return the field data as a numpy array for the given specified axis.
        
        Example:
        >>        ds = Dataset()'''
        slc = np.unravel_index(index, self._parshape)
        self._return_data = self[slc]
        
        self._slice = tuple(slc)
        # for key in self._globals:
        #     self._globals[key] = self._globals[key][slc]
        return self
    
    def __call__(self, **kwargs: float) -> Dataset:
        ''' Return the field data as a numpy array for the given specified axis.
        
        Example:
        >>        ds = Dataset()'''
        slice_obj = [slice(None) for _ in range(self._extra_dims)]
        for key, value in kwargs.items():
            if key not in self._axes_map:
                raise ValueError(f'Axis {key} not found in dataset axes')
            ax_id = self._axes_map[key]

            slice_obj[ax_id] = self._axes[ax_id](value)
        
        self._slice = tuple(slice_obj)
        return_val = self._selected_data[self._slice]
        
        self._reset_sel()

        self._return_data = return_val
        return self

    def write(self, writevalue: np.ndarray, **kwargs) -> None:

        slice_obj = [slice(None) for _ in range(self._extra_dims)]

        for key, value in kwargs.items():
            if key not in self._axes_map:
                raise ValueError(f'Axis {key} not found in dataset axes')
            ax_id = self._axes_map[key]
            slice_obj[ax_id] = self._axes[ax_id](value)
        
        if self._selected_source == 'field':
            self._data[tuple(slice_obj)] = writevalue
        else:
            self._globals[self._selected_source][tuple(slice_obj)] = writevalue

        self._reset_sel()
        #self._selected_data[tuple(slice_obj)] = value
    
    def __setitem__(self, key, value) -> None:
        key = key + (slice(None),)*(len(key) - self._s_ndims)
        self._selected_data[key] = value
        self._reset_sel()

    def __getitem__(self, key) ->  np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(key) - self._s_ndims)

        rv = self._selected_data[key]
        self._reset_sel()
        self._return_data = rv
        return rv
    
    def iterate(self):
        ''' Iterates over the complete permutation of all axes and returns the axes keys and valus'''
        matrices = np.meshgrid(*[ax.values for ax in self._axes], indexing='ij')
        vararrays = [mat.flatten() for mat in matrices]
        for i, variables in enumerate(zip(*vararrays)):
            dict_variables = {ax.name: variables[i] for i, ax in enumerate(self._axes)}
            yield dict_variables
    

if __name__ == '__main__':

    ds = Dataset()

    ax1 = np.linspace(1,10,6)
    ax2 = np.linspace(10,110,6)

    mainax = np.linspace(1,100,100)

    A1,A2,M = np.meshgrid(ax1, ax2, mainax, indexing='ij')

    ds.add_axis('freq', ax1, 'GHz')
    ds.add_axis('Length', ax2, 'm')
    ds.add_global('S21')
    ds.init(100)

    print(ds.shape)
    print(ds.field(freq=1.5, Length=10))

    ds.glob('S21')[0,0] = 5
    print(ds.glob('S21')[1,:])

    ds.field.write(5*np.ones((100,)), freq=10)

    print(ds._globals)

    vals = 0
    for keys in ds.iterate():
        #print(keys)
        ds.glob('S21').write(vals, **keys)
        vals += 1
    
    print(ds._globals['S21'])