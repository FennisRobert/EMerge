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
from __future__ import annotations
import numpy as np
from loguru import logger
from typing import TypeVar, Generic, Any, List, Union, Dict, Generator
from collections import defaultdict
from .file import Saveable
from .solver import SolveReport

T = TypeVar("T")
M = TypeVar("M")


def assemble_nd_data(
    data_in: List[Union[float, np.ndarray]],
    vars_in: List[Dict[str, float]],
    axes: Dict[str, List[float]],
) -> np.ndarray:
    """
    Assemble a flat list of data entries into an N-dimensional array
    based on provided axis definitions.

    Parameters
    ----------
    data_in : List[float or np.ndarray]
        Flat list of data entries. Scalars are treated as shape ().
    vars_in : List[Dict[str, float]]
        List of dictionaries mapping variable names to their value for each data entry.
        Length must match data_in.
    axes : Dict[str, List[float]]
        Mapping from variable name to list of allowed values along that axis.
        If 'freq' is a key, its axis will be placed last.

    Returns
    -------
    np.ndarray
        An array of shape (*axis_lengths, *shp_out), where axis_lengths correspond
        to lengths of each axis in `axes` (with 'freq' last if present) and
        shp_out is the shape of the first ndarray in data_in or () if all scalars.
    """
    # Determine axis ordering
    axis_names = sorted(list(axes.keys()))
    if "freq" in axis_names:
        axis_names.append(axis_names.pop(axis_names.index("freq")))
    axis_lengths = [len(axes[name]) for name in axis_names]

    # Determine output shape suffix
    # Assume all array entries share the same shape
    first = next((d for d in data_in if isinstance(d, np.ndarray)), None)
    shp_out = first.shape if isinstance(first, np.ndarray) else ()

    # Initialize output array
    out_shape = tuple(axis_lengths) + shp_out
    out = np.empty(
        out_shape, dtype=first.dtype if isinstance(first, np.ndarray) else float
    )

    # Fill in data
    for entry, var_map in zip(data_in, vars_in):
        # build index for each axis
        idx = tuple(axes[name].index(var_map[name]) for name in axis_names)
        if isinstance(entry, np.ndarray):
            out[idx + tuple(slice(None) for _ in shp_out)] = entry
        else:
            out[idx] = entry

    return out


def generate_ndim(
    outer_data: dict[str, list[float]],
    inner_data: list[float],
    outer_labels: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    """
    Generates an N-dimensional grid of values from flattened data, and returns each axis array plus the grid.

    Parameters
    ----------
    outer_data : dict of {label: flat list of coordinates}
        Each key corresponds to one axis label, and the list contains coordinate values for each point.
    inner_data : list of float
        Flattened list of data values corresponding to each set of coordinates.
    outer_labels : tuple of str
        Order of axes (keys of outer_data) which defines the dimension order in the output array.

    Returns
    -------
    *axes : np.ndarray
        One 1D array for each axis, containing the sorted unique coordinates for that dimension,
        in the order specified by outer_labels.
    grid : np.ndarray
        N-dimensional array of shape (n1, n2, ..., nN), where ni is the number of unique
        values along the i-th axis. Missing points are filled with np.nan.
    """
    # Convert inner data to numpy array
    values = np.asarray(inner_data)

    # Determine unique sorted coordinates for each axis
    axes = [np.unique(np.asarray(outer_data[label])) for label in outer_labels]
    grid_shape = tuple(axis.size for axis in axes)

    # Initialize grid with NaNs
    grid = np.full(grid_shape, np.nan, dtype=values.dtype)

    # Build coordinate arrays for each axis
    coords = [np.asarray(outer_data[label]) for label in outer_labels]

    # Map coordinates to indices in the grid for each axis
    idxs = [np.searchsorted(axes[i], coords[i]) for i in range(len(axes))]

    # Assign values into the grid
    grid[tuple(idxs)] = values

    # Return each axis array followed by the grid
    return tuple(axes) + (grid,)


def drop_constant_variables(listvars: list[dict[str, float]]) -> list[dict[str, float]]:
    if not listvars:
        return []

    # Collect all keys
    keys = listvars[0].keys()

    # Determine which keys actually vary
    varying_keys = {k for k in keys if len({d[k] for d in listvars}) > 1}

    # Filter dictionaries to only keep varying keys
    return [{k: d[k] for k in varying_keys} for d in listvars]


class DataEntry(Saveable):
    def __init__(self, variables: dict[str, float]):
        self.vars: dict[str, float] = variables
        self.data: dict[str, Any] = dict()
        self.reports: list[SolveReport] = []

    def is_non_empty(self) -> bool:
        return len(self.data) > 0

    # ALLOWED PRINT
    def print(self) -> None:
        """Print the content of the DataEntry object"""
        for key, value in self.data.items():
            # ALLOWED PRINT
            print(f"    {key} = {value}")

    def values(self) -> list[Any]:
        """Return all values stored in the DataEntry"""
        return list(self.data.values())

    def keys(self) -> list[str]:
        """Return all names of data stored in the DataEntry"""
        return list(self.data.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Returns a list of all key: value pairs of the DataEntry."""
        return list(self.data.items())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, dict):
            return False
        allkeys = set(list(self.vars.keys()) + list(other.keys()))
        return all(self.vars[key] == other[key] for key in allkeys)

    def _dist(self, other: dict[str, float]) -> float:
        return sum(
            [
                (abs(self.vars.get(key, 1e20) - other[key]) / (other[key] + 1e-12))
                for key in other.keys()
            ]
        )

    def __getitem__(self, key) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def set_reports(self, reports: list[SolveReport]) -> None:
        self.reports = reports

class DataContainer(Saveable):
    """The DataContainer class is a generalized class to store data for a set of parameter sweeps"""

    def __init__(self):
        self.entries: list[DataEntry] = []

    @staticmethod
    def combine_sets(containers: list[DataContainer]) -> DataContainer:
        """Generate one big container from a list of DataContainers

        Args:
            containers (list[DataContainer]): _description_

        Returns:
            DataContainer: _description_
        """
        return containers[0].merge_with(*containers[1:])

    def merge_with(self, *others: DataContainer) -> DataContainer:
        """Combines this DataContainer with another container

        Args:
            other (DataContainer): The container to merge with

        Returns:
            DataContainer: _description_
        """
        # Copy Step
        for container in others:
            for other in container.entries:
                merged = False
                for entry in self.entries:
                    if entry.vars == other.vars:
                        entry.data.update(other.data)
                        merged = True
                        break
                if merged:
                    continue
                self.entries.append(other)

        return self

    # ALLOWED PRINT
    def print(self) -> None:
        """Print an overview of all data in the DataContainer"""
        for entry in self.entries:
            # ALLOWED PRINT
            entry.print()

    def new(self, **variables: float) -> DataEntry:
        """Create a new entry in the DataContainer for the given value setting"""
        entry = DataEntry(variables)
        self.entries.append(entry)
        return entry

    def iterate(self) -> Generator[DataEntry, None, None]:
        yield from self.entries

    @property
    def first(self) -> DataEntry:
        """Returns the first added entry"""
        return self.entries[0]

    @property
    def last(self) -> DataEntry:
        """Returns the last added entry"""
        return self.entries[-1]

    def index(self, index: int) -> DataEntry:
        """Returns the last added entry"""
        return self.entries[index]

    def select(self, **vars: float) -> DataEntry | None:
        """Returns the data entry corresponding to the provided parametric sweep set"""
        for entry in self.entries:
            if entry == vars:
                return entry
        return None

    def find(self, **vars: float) -> DataEntry:
        """Returns the DataEntry closest to the provided parametric sweep setting."""
        return sorted(
            [(entry, entry._dist(vars)) for entry in self.entries], key=lambda x: x[1]
        )[0][0]

    def __getitem__(self, key: str) -> DataEntry:
        """Returns the requested item from the default DataEntry"""
        return self.last[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Writes a value to the requested default DataEntry"""
        self.last[key] = value

    def remove_empty_datasets(self) -> None:
        self.entries = [entry for entry in self.entries if entry.is_non_empty()]


class BaseDataset(Generic[T, M], Saveable):
    # skip_fields = ('_datatype','_matrixtype')

    def __init__(self, datatype: type[T], matrixtype: type[M], scalar: bool):
        self._datatype: type[T] = datatype
        self._matrixtype: type[M] = matrixtype
        self._variables: list[dict[str, float]] = []
        self._data_entries: list[T] = []
        self._scalar: bool = scalar

        self._gritted: bool | None = None
        self._axes: dict[str, np.ndarray] | None = None
        self._ax_ids: dict[str, int] | None = None
        self._ids: np.ndarray | None = None
        self._gridobj: M | None = None

        self._data: dict[str, Any] = dict()

    def __getitem__(self, index: int) -> T:
        return self._data_entries[index]

    @property
    def _fields(self) -> list[str]:
        return self._datatype._fields  # type: ignore

    @property
    def n(self) -> int:
        return len(self._data_entries)

    def iter(self) -> Generator[T, None, None]:
        for item in self._data_entries:
            yield item

    @property
    def _copy(self) -> list[str]:
        return self._datatype._copy  # type: ignore

    def merge_with(self, *others: BaseDataset) -> BaseDataset[T, M]:
        for other in others:
            if self._datatype is not other._datatype:
                raise TypeError(
                    f"Dissimilar datatypes {self._datatype} and {other._datatype}. Aborting merge."
                )
            if self._matrixtype is not other._matrixtype:
                raise TypeError(
                    f"Dissimilar matrix types {self._matrixtype} and {other._matrixtype}. Aborting merge."
                )

            self._variables.extend(other._variables)
            self._data_entries.extend(other._data_entries)
            self._data.update(other._data)

        self._gritted: bool | None = None
        self._axes: dict[str, np.ndarray] | None = None
        self._ax_ids: dict[str, int] | None = None
        self._ids: np.ndarray | None = None
        self._gridobj: M | None = None

        return self

    def store(self, key: str, value: Any) -> None:
        """Stores a variable with some value in the provided key.
        Make sure that all values passed are picklable.

        Args:
            key (str): The name for the data entry
            value (Any): The value of the data entry
        """
        self._data[key] = value

    def load(self, key: str) -> Any | None:
        """Returns the data entry for a given key

        Args:
            key (str): The name of the data entry

        Returns:
            Any: The value of the data entry
        """
        return self._data.get(key, None)

    def get_entry(self, index: int) -> T:
        """Returns the physics dataset for  the given index

        Args:
            index (int): The index of the solution

        Returns:
            T: The Physics dataset
        """
        return self._data_entries[index]

    def select(self, **variables: float) -> T:
        """Returns the first physics dataset that satisfies the variable assignemnt.

        Returns:
            T: The physics dataset
        """
        for i, var_map in enumerate(self._variables):
            if all(var_map.get(k) == v for k, v in variables.items()):
                return self.get_entry(i)

    def filter(self, **variables: float) -> list[T]:
        """Returns a list of all physics datasets that are valid for the given variable assignment

        Returns:
            list[T]: A list of all matching datasets
        """
        output = []
        for i, var_map in enumerate(self._variables):
            if all(var_map.get(k) == v for k, v in variables.items()):
                output.append(self.get_entry(i))
        return output

    def slice_set(
        self, i1: int, i2: int | None = None, sort_by: str | None = None
    ) -> BaseDataset[T, M]:
        """Reduces the DataSet object to a slice with the given range

        Args:
            i1 (int): Start index
            i2 (int | None, optional): final index. Defaults to None.

        Returns:
            BaseDataset[T,M]: _description_
        """
        new_set = BaseDataset(self._datatype, self._matrixtype, self._scalar)
        new_set._data_entries = self._data_entries[slice(i1, i2)]
        new_set._variables = drop_constant_variables(self._variables[slice(i1, i2)])
        if sort_by is not None:
            ids = range(self.n)
            ids = sorted(ids, key=lambda x: getattr(self._data_entries[x], sort_by))
            self._data_entries = [self._data_entries[i] for i in ids]
            self._variables = [self._variables[i] for i in ids]
        return new_set

    def find(self, **variables: float) -> T:
        """Returns the physics dataset that is closest to the constraint given by the variables.

        Returns:
            T: The physics dataset.
        """
        output = []
        for i, var_map in enumerate(self._variables):
            error = sum([abs(var_map.get(k, 1e30) - v) for k, v in variables.items()])
            output.append((i, error))
        selection_id = sorted(output, key=lambda x: x[1])[0][0]
        entry = self.get_entry(selection_id)
        variables = ", ".join(
            [f"{key}={value}" for key, value in self._variables[selection_id].items()]
        )
        logger.info(f"Selected entry: {variables}")
        return entry

    def axis(self, name: str) -> np.ndarray:
        """Returns a sorted list of all variables for the given name

        Args:
            name (str): The name of the variable axis

        Returns:
            np.ndarray: A sorted list of all unique values.
        """
        return np.sort(np.unique(np.array([var[name] for var in self._variables])))

    def new(self, **vars: float) -> T:
        """Creates a new dataset

        Returns:
            T: The physics dataset object
        """
        self._variables.append(vars)
        new_entry = self._datatype()
        self._data_entries.append(new_entry)
        return new_entry

    def _grid_axes(self) -> bool:
        """This method attepmts to create a gritted version of the scalar dataset. It may fail
        if the data in the dataset cannot be cast into a gridded structure.

        Returns:
            None
        """
        logger.debug("Attempting to grid simulation data")
        variables = defaultdict(set)
        for var in self._variables:
            for key, value in var.items():
                variables[key].add(value)

        N_entries = len(self._variables)
        N_prod = 1
        N_dim = len(variables)

        for key, val_list in variables.items():
            N_prod *= len(val_list)

        if N_entries == N_prod:
            logger.debug("Multi-dimensional grid found!")
            self._gritted = True
        else:
            logger.debug("Multi-dimensional grid not found")
            self._gritted = False
            return False

        self._axes = dict()
        self._ax_ids = dict()

        revax = dict()
        i = 0

        for key, val_set in variables.items():
            self._axes[key] = np.sort(np.array(list(val_set)))
            self._ax_ids[key] = i
            revax[i] = key
            i += 1

        axlist = []

        for idim in range(N_dim):
            axlist.append(self._axes[revax[idim]])

        indices = np.arange(N_entries)
        Ndimlist = [len(dim) for dim in axlist]
        self._ids = indices.reshape(Ndimlist)

        obj = self._matrixtype()

        axes_list = {key: list(self._axes[key]) for key in self._axes.keys()}
        for field in self._fields:
            data_field = [
                self._data_entries[i].__dict__[field] for i in range(N_entries)
            ]
            data_set = assemble_nd_data(data_field, self._variables, axes_list)
            obj.__setattr__(field, data_set)
        for copyfield in self._copy:
            obj.__setattr__(copyfield, self._data_entries[0].__dict__[copyfield])
        self._gridobj = obj

        return True

    @property
    def grid(self) -> M:
        """Returns the gridded version of the scalar dataset.

        Raises:
            ValueError: _description_

        Returns:
            M: The gritted physics dataset
        """
        if self._gritted is None:
            self._grid_axes()

        if self._gritted is False:
            logger.error("The dataset cannot be cast to a structured grid.")
            raise ValueError("Data not in regular grid")

        return self._gridobj
