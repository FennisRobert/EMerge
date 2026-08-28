from __future__ import annotations
from typing import Type, TypeVar, Optional, List, Generator, Literal, overload, Any, Callable

import copy
from emsutil import Material, PEC, AIR, Saveable

from .cs import Axis, _parse_axis

T = TypeVar('T', bound='PhysicalAttribute')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')
T5 = TypeVar('T5')

def format_length(meters: float) -> str:
    """Formats a float value in meters to a human-readable engineering string."""
    abs_val = abs(meters)
    
    if abs_val == 0:
        return "0.0m"
    elif abs_val >= 1.0:
        return f"{meters:.1f}m"
    elif abs_val >= 1e-3:
        return f"{meters * 1e3:.1f}mm"
    elif abs_val >= 1e-6:
        return f"{meters * 1e6:.1f}um"  # Use "µm" if you prefer the actual micro symbol
    else:
        return f"{meters * 1e9:.1f}nm"
    
class PhysicalAttribute(Saveable):
    name: str = "UnnamedAttribute"

    def copy(self, deep: bool = False) -> PhysicalAttribute:
        """
        Returns a fresh copy of this physical attribute.
        
        If deep=True, it recursively clones any nested mutable structures.
        If deep=False, it performs a standard shallow copy of the attribute's data fields.
        """
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)
    
    def __str__(self) -> str:
        return f'{self.name}[{self.__descr__()}]'
    
    def __descr__(self) -> str:
        return ''
    
    def __add__(self, other: PhysicalAttribute | Material | PhysicalAttributeSet) -> PhysicalAttributeSet:
        if isinstance(other, PhysicalAttribute):
            return PhysicalAttributeSet(self, other)
        elif isinstance(other, Material):
            return PhysicalAttributeSet(self, other)
        elif isinstance(other, PhysicalAttributeSet):
            return other.add_to(self)
        else:
            raise TypeError(f'Cannot combine a PhysicalAttribute with an object {other} of type {type(other)}')
    
    def __radd__(self, other) -> PhysicalAttributeSet:
        return self.__add__(other)
    
    def __and__(self, other) -> PhysicalAttributeSet:
        return self.__add__(other)
    
    def __rand__(self, other) -> PhysicalAttributeSet:
        return self.__add__(other)
    
    def __iand__(self, other)  -> PhysicalAttributeSet:
        return self.__and__(other)
    
    def to_set(self) -> PhysicalAttributeSet:
        return PhysicalAttributeSet(self)
    
    def _get_value(self) -> Any:
        return None
    
class PhysicalAttributeSet(Saveable):

    def __init__(self, *attributes: PhysicalAttribute | Material):
        self.attrs: list[PhysicalAttribute | Material] = list(attributes)

    def __str__(self) -> str:
        return f'AttributeSet[{", ".join([str(x) for x in self.attrs])}]'
    
    def __add__(self, other: PhysicalAttribute | Material | PhysicalAttributeSet) -> PhysicalAttributeSet:
        if isinstance(other, PhysicalAttribute):
            return self.add_to(other)
        elif isinstance(other, Material):
            return self.add_to(other)
        elif isinstance(other, PhysicalAttributeSet):
            return self.combine(other)
        else:
            raise TypeError(f'Cannot combine a PhysicalAttributeSet with an object {other} of type {type(other)}')
    
    def __radd__(self, other) -> PhysicalAttributeSet:
        return self.__add__(other)
    
    def __and__(self, other) -> PhysicalAttributeSet:
        return self.__add__(other)
    
    def __rand__(self, other) -> PhysicalAttributeSet:
        return self.__add__(other)
    
    def __iand__(self, other)  -> PhysicalAttributeSet:
        return self.__and__(other)
    
    def __iter__(self) -> Generator[PhysicalAttribute | Material, None, None]:
        for attr in self.attrs:
            yield attr

    def copy(self, deep: bool = True) -> PhysicalAttributeSet:
        if deep:
            new_attrs = []
            for attr in self.attrs:
                if isinstance(attr, Material):
                    new_attrs.append(attr)
                else:
                    new_attrs.append(copy.deepcopy(attr))
            
            return PhysicalAttributeSet(*new_attrs)
        else:
            return PhysicalAttributeSet(*self.attrs)

    def __copy__(self) -> PhysicalAttributeSet:
        return self.copy(deep=False)

    def __deepcopy__(self, memo: dict[int, Any]) -> PhysicalAttributeSet:
        """Hooks into standard copy.deepcopy(set_obj)"""
        # 'memo' is required by Python to track already-copied objects and prevent infinite loops
        return PhysicalAttributeSet(*(copy.deepcopy(attr, memo) for attr in self.attrs))
    
    def _clean_duplicates(self) -> None:
        """Removes duplicates by keeping the last ones
        """
        new = []
        types = []
        for attr in reversed(self.attrs):
            if type(attr) not in types:
                new.append(attr)
                types.append(type(attr))
        self.attrs = new

    def add_to(self, other: PhysicalAttribute | Material) -> PhysicalAttributeSet:
        self.attrs.append(other)
        self._clean_duplicates()
        return self
    
    def combine(self, other: PhysicalAttributeSet) -> PhysicalAttributeSet:
        self.attrs.extend(other.attrs)
        self._clean_duplicates()
        return self
    
    def get(self, attr_type: Type[T]) -> Optional[T]:
        for a in self.attrs:
            if isinstance(a, attr_type):
                return a
        return None
    
    def get_value(self, attr_type: PhysicalAttribute, default_value: Any) -> Any:
        if attr := self.get(attr_type):
            if value := attr._get_value():
                return value
            else:
                return default_value
        return default_value
    
    def get_attr(self, attr_type: PhysicalAttribute | Material, attribute: str, default = None) -> Any:
        if attr := self.get(attr_type):
            obj = attr
            for name in attribute.split('.'):
                obj = getattr(obj, name)
            return obj
        return default
    # 1 Property Requested
    @overload
    def get_all(self, __t1: Type[T1]) -> Optional[tuple[T1]]: ...
    
    # 2 Properties Requested
    @overload
    def get_all(self, __t1: Type[T1], __t2: Type[T2]) -> Optional[tuple[T1, T2]]: ...
    
    # 3 Properties Requested
    @overload
    def get_all(self, __t1: Type[T1], __t2: Type[T2], __t3: Type[T3]) -> Optional[tuple[T1, T2, T3]]: ...
    
    # 4 Properties Requested
    @overload
    def get_all(self, __t1: Type[T1], __t2: Type[T2], __t3: Type[T3], __t4: Type[T4]) -> Optional[tuple[T1, T2, T3, T4]]: ...
    
    # 5 Properties Requested
    @overload
    def get_all(self, __t1: Type[T1], __t2: Type[T2], __t3: Type[T3], __t4: Type[T4], __t5: Type[T5]) -> Optional[tuple[T1, T2, T3, T4, T5]]: ...


    def get_all(self, *attr_types: Type[T]) -> Optional[tuple[T,...]]:
        out = [self.get(at) for at in attr_types]
        if all(prop is not None for prop in out):
            return tuple(out)
        return None
    
    

class PhysicalAttributeDescriptor:
    """
    A descriptor that intercepts assignments to ensure the target property
    is always wrapped inside or maintained as a PhysicalAttributeSet.
    """
    def __set_name__(self, owner, name):
        # Automatically names the underlying private storage (e.g., '_properties')
        self.private_name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        # If the property hasn't been set yet, default to an empty set
        if self.private_name not in instance.__dict__:
            instance.__dict__[self.private_name] = PhysicalAttributeSet()
            
        return instance.__dict__[self.private_name]

    def __set__(self, instance, value):
        if value is None:
            instance.__dict__[self.private_name] = PhysicalAttributeSet()
            
        elif isinstance(value, PhysicalAttributeSet):
            instance.__dict__[self.private_name] = value
            
        elif isinstance(value, (PhysicalAttribute, Material)):
            # Intercept single attributes/materials and normalize them to a set
            instance.__dict__[self.private_name] = PhysicalAttributeSet(value)
            
        else:
            raise TypeError(
                f"Cannot assign type {type(value).__name__} to geometry properties. "
                f"Expected PhysicalAttribute, Material, or PhysicalAttributeSet."
            )
############################################################
#                   PREDEFINED ATTRIBUTES                  #
############################################################

class FiniteThickness(PhysicalAttribute):
    name: str = 'FiniteThickness'

    def __init__(self, thickness: float):
        self.thickness: float = float(thickness)

    def _get_value(self):
        return self.thickness
    
    def __descr__(self) -> str:
        return format_length(self.thickness)
    
class SurfaceRoughness(PhysicalAttribute):
    name: str = 'SurfaceRoughness'

    def __init__(self, rmsval_m: float):
        self.rmsval: float = float(rmsval_m)

    def _get_value(self):
        return self.rmsval
    
    def __descr__(self):
        return format_length(self.rmsval)

class WavePortAttribute(PhysicalAttribute):
    name: str = 'WavePort'

    def __init__(self, port_number: int, mode_type: Literal['TEM','TE','TM'] = 'TEM'):
        self.port_number: int = port_number
        self.mode_type: str = mode_type

    def __descr__(self):
        return f'{self.port_number}, {self.mode_type}'

class LumpedPortAttribute(PhysicalAttribute):
    name: str = 'LumpedPort'

    def __init__(self, port_number: int, 
                 direction: tuple[float, float, float] | Axis, 
                 width: float,
                 height: float,
                 impedance: float = 50.0):
        self.port_number: int = port_number
        self.width: float = width
        self.height: float = height
        self.direction: Axis = _parse_axis(direction)
        self.z0: float = impedance
    
    def __descr__(self):
        return f'{self.port_number},{format_length(self.width)}x{format_length(self.height)}, {self.direction}, {self.z0:.1f}Ω'

class MetalCoating(PhysicalAttribute):
    name: str = 'MetalCoating'

    def __init__(self, material: Material, thickness: float):
        self.material: Material = material
        self.thickness: float = float(thickness)

    def __descr__(self):
        return f'{format_length(self.thickness)}, {self.material}'

class LumpedElementAttribute(PhysicalAttribute):
    name: str = 'LumpedElement'
    skip_fields = ['impedance_function',]

    def __init__(self, impedance_function: Callable, width: float, height: float):
        self.impedance_function: Callable = impedance_function
        self.width: float = float(width)
        self.height: float = float(height)

    def __descr__(self):
        return f'{format_length(self.width)}x{format_length(self.height)}'

class VoidAttribute(PhysicalAttribute):
    name: str = 'VoidAttribute'

    def __init__(self):
        pass

    def __descr__(self):
        return ''
    
if __name__ == "__main__":
    pset = PhysicalAttributeSet()

    print(pset)
    pset += PEC
    print(pset)
    pset += AIR
    print(pset)