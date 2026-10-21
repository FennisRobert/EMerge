from __future__ import annotations

from emerge._emerge.geometry import GeoVolume, GeoSurface
from emerge import CoordinateSystem, Anchor
from emerge._emerge.geo import stick

from enum import Enum
from typing import NamedTuple
import emerge as em
from abc import ABC, abstractmethod

import numpy as np

class WaveguideDim(NamedTuple):
    a: float  # Width in mm
    b: float  # Height in mm


def up_dydx0(x: float):
    return 1 - np.cos(x)

def up_dydx1(x: float):
    return np.sin(x)

def down_dydx0(x: float):
    return np.cos(x)

def down_dydx1(x: float):
    return 1 - np.sin(x)

class WGStandard(Enum):
    WR_2300 = WaveguideDim(584.20, 292.10)
    WR_2100 = WaveguideDim(533.40, 266.70)
    WR_1800 = WaveguideDim(457.20, 228.60)
    WR_1500 = WaveguideDim(381.00, 190.50)
    WR_1150 = WaveguideDim(292.10, 146.05)
    WR_975  = WaveguideDim(247.65, 123.825)
    WR_770  = WaveguideDim(195.58, 97.79)
    WR_650  = WaveguideDim(165.10, 82.55)
    WR_510  = WaveguideDim(129.54, 64.77)
    WR_430  = WaveguideDim(109.22, 54.61)
    WR_340  = WaveguideDim(86.36, 43.18)
    WR_284  = WaveguideDim(72.14, 34.04)
    WR_229  = WaveguideDim(58.17, 29.08)
    WR_187  = WaveguideDim(47.55, 22.15)
    WR_159  = WaveguideDim(40.39, 20.19)
    WR_137  = WaveguideDim(34.85, 15.80)
    WR_112  = WaveguideDim(28.50, 12.62)
    WR_90   = WaveguideDim(22.86, 10.16)
    WR_75   = WaveguideDim(19.05, 9.525)
    WR_62   = WaveguideDim(15.80, 7.90)
    WR_51   = WaveguideDim(12.95, 6.48)
    WR_42   = WaveguideDim(10.67, 4.32)
    WR_34   = WaveguideDim(8.64, 4.32)
    WR_28   = WaveguideDim(7.11, 3.56)
    WR_22   = WaveguideDim(5.69, 2.84)
    WR_19   = WaveguideDim(4.78, 2.39)
    WR_15   = WaveguideDim(3.76, 1.88)
    WR_12   = WaveguideDim(3.10, 1.55)
    WR_10   = WaveguideDim(2.54, 1.27)
    WR_8    = WaveguideDim(2.03, 1.02)
    WR_6    = WaveguideDim(1.65, 0.825)
    WR_5    = WaveguideDim(1.30, 0.65)
    WR_4    = WaveguideDim(1.092, 0.546)
    WR_3    = WaveguideDim(0.864, 0.432)
    WR_2    = WaveguideDim(0.508, 0.254)
    WR_1_5  = WaveguideDim(0.381, 0.191)
    WR_1_2  = WaveguideDim(0.305, 0.152)
    WR_1    = WaveguideDim(0.254, 0.127)

    @property
    def a(self) -> float:
        return self.value.a

    @property
    def b(self) -> float:
        return self.value.b

    @property
    def ab(self) -> tuple[float, float]:
        return self.value

class WGConfig:
    
    def __init__(self, 
                 dims: tuple[float, float] | WGStandard,
                 model_walls: bool = False,
                 wall_thickenss_mm: float = 1.0,
                 material: em.Material = em.lib.PEC,
                 coating: em.Material | None = None,
                 coating_thickenss_um: float = 1.0):
        
        if isinstance(dims, WGStandard):
            a,b = dims.ab
        else:
            a,b = dims
        
        self.a: float = a * 0.001
        self.b: float = b * 0.001
        self.material: em.Material = material
        self.model_walls: bool = model_walls
        self.thmm: float = wall_thickenss_mm * 0.001
        self.coat_mat: em.Material | None = coating
        self.coat_th: float = coating_thickenss_um * 1e-6
        
    
    
class WGComponent:
    
    def __init__(self):
        self._active_anchor: Anchor | None = None
        
    @property
    def anchors(self) -> list[Anchor]:
        raise NotImplementedError(f'Comps is not defined for class {self.__class__.name}')
    
    @property
    def _comps(self) -> list[GeoVolume | None]:
        return []
    
    @property
    def comps(self) -> list[GeoVolume]:
        return [x for x in self._comps if x is not None]
    
    def build(self) -> None:
        raise NotImplementedError(f'Comps is not defined for class {self.__class__.name}')
    
    def _reset(self) -> None:
        self._active_anchor = None
        
    def place(self, 
                anchor: Anchor | WGComponent, 
                flipx: bool = False, 
                flipy: bool = False, 
                flipz: bool = False) -> WGComponent:
        
        if self._active_anchor is None:
            raise ValueError(f'You must specify an active achor for this component: {self}')
        
        anchor_self = self._active_anchor
        
        if isinstance(anchor, WGComponent):
            if anchor._active_anchor is None:
                raise ValueError(f'You must specify an active achor for the joined component: {anchor}')
            anchor = anchor._active_anchor
        
        if flipx:
            anchor = anchor.mx
        if flipz:
            anchor = anchor.mz
        if flipy:
            anchor = anchor.my
            
        for geo in self.comps:
            stick(geo, anchor_self, anchor)

        A = anchor_self.compute_affine(anchor)
        for anch in self.anchors:
            anch.affine_transform(A)
            
        self._reset()
        
        return self
    
    
    
class WGRect(WGComponent):
    
    def __init__(self, length: float, config: WGConfig):
        self.length: float = length
        self.config: WGConfig = config

        self.air: GeoVolume = None
        self.metal: GeoVolume = None
        
        self._a1: Anchor | None = None
        self._a2: Anchor | None = None

        self.build()
    
    @property
    def anchors(self) -> list[Anchor]:
        return (self._a1, self._a2)
    
    @property
    def front(self) -> WGRect:
        self._active_anchor = self._a1
        return self
    
    @property
    def back(self) -> WGRect:
        self._active_anchor = self._a2
        return self
    
    @property
    def _comps(self):
        return [self.air, self.metal]
    
    def build(self):
        a,b,L = self.config.a, self.config.b, self.length

        self.air = em.geo.Box(a,b,L, (-a/2, -b/2, 0))
        
        if self.config.model_walls:
            th = self.config.thmm
            self.metal = em.geo.Box(a+2*th,b+2*th,L, (-a/2-th, -b/2-th, 0))
            self.metal = em.geo.subtract(self.metal, self.air, remove_tool=False).set_material(self.config.material)
            
        self._a1 = Anchor((0,0,0))
        self._a2 = Anchor((0,0,L))

class WGETurnRound(WGComponent):
    
    def __init__(self, 
                 config: WGConfig, 
                 ratio: float = 1.2, 
                 angle_deg: float = 90):
        
        self.ratio: float = ratio
        self.angle: float = angle_deg * np.pi/180
        self.config: WGConfig = config

        self.air: GeoVolume = None
        self.metal: GeoVolume = None
        
        self._a1: Anchor | None = None
        self._a2: Anchor | None = None

        self.build()
        
    
    def build(self) -> None:
        a,b = self.config.a, self.config.b
        
        zex_in = b*max(0.0, self.ratio - 1.0)
        zex_out = b*max(0.0, 1.0 - self.ratio)
        
        R = self.ratio*b
        if self.ratio < 1.0:
            poly = em.geo.XYPolygon([-b/2, -b/2],[0, zex_out])\
                .parametric(lambda t: -b/2 + R*(1 - np.cos(t*np.pi/2)), lambda t: zex_out + R*np.sin(t*np.pi/2))\
                .extend([zex_out, b/2, b/2], [b, b, 0])
            air = poly.extrude(a, em.YZPLANE.cs(-a/2,0,0))
            self._a1 = Anchor((0,0,0))
            self._a2 = Anchor((0,b/2,b/2), (1,0,0), (0,0,-1), (0,1,0))
            
        self.air = air

    @property
    def anchors(self) -> list[Anchor]:
        return (self._a1, self._a2)
    
    @property
    def front(self) -> WGRect:
        self._active_anchor = self._a1
        return self
    
    @property
    def back(self) -> WGRect:
        self._active_anchor = self._a2
        return self
    
    @property
    def _comps(self):
        return [self.air]