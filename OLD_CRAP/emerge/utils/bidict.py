from __future__ import annotations
from collections import UserDict
from itertools import product
import time

class BiJunction(UserDict):

    def __init__(self):
        super().__init__(self)
        
    def __setitem__(self, key, value) -> None:
        self.data[key] = value
        self.data[value] = key

    @staticmethod
    def from_dict(inputdict: dict) -> BiJunction:
        bi = BiJunction()
        for key, value in inputdict.items():
            bi[key] = value
        return bi


class TupIntMap(UserDict):

    def __init__(self):
        super().__init__(self)
        self._inverse = None

    def __getitem__(self, key):
        key = tuple(sorted(key))
        return self.data[key]
    
    def __setitem__(self, key, item):
        skey = tuple(sorted(key))
        self.data[skey] = item
        self._inverse.data[item] = skey

class IntTupMap(UserDict):

    def __init__(self):
        super().__init__(self)
        self._inverse = None

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, item):
        sitem = tuple(sorted(item))
        self.data[key] = sitem
        self._inverse.data[sitem] = key

class MeshMap:

    def __init__(self):
        self.tuple = TupIntMap()
        self.int = IntTupMap()
        self.tuple._inverse = self.int
        self.int._inverse = self.tuple

    def __len__(self) -> int:
        return len(self.tuple.data)
    
    def __repr__(self) -> str:
        return f'MeshMap[{repr(self.tuple)}]'
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.tuple[key]
        else:
            return self.int[key]
        
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.tuple[key] = value
        else:
            self.int[key] = value
