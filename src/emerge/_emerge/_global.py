from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from collections import deque
import gmsh

class _BaseManager(ABC):

    @abstractmethod
    def reset(self) -> None:
        pass

class _GlobalHandler(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance
    
    def __init__(self):
        self.generator: _BaseManager = None
        self.geomanager: _BaseManager = None
        self.pcbmanager: _BaseManager = None
        self.selector: _BaseManager = None
        self.generator: _BaseManager = None
        self.logcontroller: _BaseManager = None
        self.debugcollector: _BaseManager = None
        self.logbuffer: _BaseManager = deque()
        self.simstates: _BaseManager = None

    @classmethod
    def active(cls) -> Optional[_GlobalHandler]:
        return cls._instance
    
    def reset(self):
        self.geomanager.reset()
        self.pcbmanager.reset()
        self.selector.reset()
        self.generator.reset()
        self.logbuffer = deque()
        self.simstates.reset()
        self.logcontroller.reset()
        self.debugcollector.reset()
        if gmsh.isInitialized():
            gmsh.clear()
            gmsh.finalize()
            
def cleanup():
    _GlobalHandler.active().reset()