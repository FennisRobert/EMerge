import numpy as np
from typing import TypeVar, Generic, Type, Any

T = TypeVar('T', bound='DataSet')


class DataSet:

    def __init__(self, **vars):
        for key, value in vars.items():
            self.__dict__[key] = value
        
        self._vars: dict = vars

    @property
    def scalars(self) -> dict[str, float]:
        return {key: value for key,value in self._vars.items() if isinstance(value, (float, complex, int, str)) and value is not None}

    def __repr__(self):
        varstr = ', '.join([f'{key}={value}' for key,value in self.scalars.items()])
        return f'{self.__class__.__name__}({varstr})'
    
    def equals(self, **vars):
        for name, value in vars.items():
            if not self.__dict__[name] == value:
                return False
        return True
    
    def _getvalue(self, name: str) -> Any | None:
        return self.__dict__.get(name, None)

class DataAxis(Generic[T]):
    def __init__(self, data: list[T], axis_name: str):
        self._data = sorted(data, key=lambda d: getattr(d, axis_name))
        self._axis_name = axis_name

    def __getattr__(self, name: str):
        try:
            values = [getattr(d, name) for d in self._data]
            # Convert to numpy array if all are scalars or arrays
            if all(isinstance(v, (int, float, complex, np.ndarray)) for v in values):
                return np.array(values)
            return values
        except AttributeError:
            raise AttributeError(f"'DataAxis' object has no attribute '{name}'")

    def __repr__(self):
        return f"<DataAxis sorted by '{self._axis_name}' with {len(self._data)} entries>"
    
class SimData(Generic[T]):
    datatype: type = DataSet
    def __init__(self):
        self.datasets: list[T] = []
        self._injections: dict = {}
        self._axis: str = None

    def new(self, **vars: float) -> T:
        vars.update(self._injections)
        data = self.datatype(**vars)
        self.datasets.append(data)
        return data
    
    def item(self, id: int) -> T:
        return self.datasets[id]
    
    def __call__(self, **vars: float) -> T | list[T]:
        collect = []
        for data in self.datasets:
            if data.equals(**vars):
                collect.append(data)
        if len(collect)==1:
            return collect[0]
        elif len(collect)> 1:
            return collect
        return None
    
    def collect(
        self,
        axis: str,
        field_name: str,
        *,
        dropna: bool = True,
        sort: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gather two sequences of values from all stored datasets.

        Parameters
        ----------
        axis
            The name of the attribute to use as the x-axis.
        field_name
            The name of the attribute to use as the y-axis.
        dropna
            If True, exclude any pairs where x or y is None.
        sort
            If True, sort the results by the x-axis.

        Returns
        -------
        xs, ys
            Two 1-D numpy arrays of the collected values.
            If no valid pairs are found, both will be empty arrays.
        """
        if not self.datasets:
            return np.array([]), np.array([])

        # quick validation on first dataset
        sample = self.datasets[0]
        if not hasattr(sample, "_getvalue"):
            raise AttributeError(f"{type(sample).__name__} has no method `_getvalue`")
        # try a single call to catch typos early
        try:
            sample._getvalue(axis), sample._getvalue(field_name)
        except Exception as e:
            raise ValueError(f"Invalid axis/field_name: {e}")

        # pull out (x, y) pairs
        pairs = [
            (d._getvalue(axis), d._getvalue(field_name))
            for d in self.datasets
        ]

        # optionally drop missing data
        if dropna:
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]

        if not pairs:
            return np.array([]), np.array([])

        xs, ys = zip(*pairs)
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        # optionally sort by x
        if sort:
            idx = np.argsort(xs)
            xs = xs[idx]
            ys = ys[idx]

        return xs, ys
