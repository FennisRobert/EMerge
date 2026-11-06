from __future__ import annotations

from emerge._emerge.geometry import GeoVolume, GeoSurface
from emerge._emerge.selection import FaceSelection
from emerge import CoordinateSystem, Anchor
from emerge._emerge.geo import stick

from enum import Enum
from typing import Literal, NamedTuple
import emerge as em
from abc import ABC, abstractmethod

import numpy as np

class WaveguideDim(NamedTuple):
    a: float  # Width in mm
    b: float  # Height in mm


def _up_dydx0(x: float):
    return 1 - np.cos(x)

def _up_dydx1(x: float):
    return np.sin(x)

def _down_dydx0(x: float):
    return np.cos(x)

def _down_dydx1(x: float):
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
        
    
    
class _WGComponent:
    
    def __init__(self):
        self._active_anchor: Anchor | None = None
        self.air: GeoVolume = None

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
                anchor: Anchor | _WGComponent, 
                flipx: bool = False, 
                flipy: bool = False, 
                flipz: bool = False) -> _WGComponent:
        
        if self._active_anchor is None:
            raise ValueError(f'You must specify an active achor for this component: {self}')
        
        anchor_self = self._active_anchor
        
        if isinstance(anchor, _WGComponent):
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
    
    def face(self, name: str) -> FaceSelection:
        anchor = getattr(self, name)._active_anchor
        return self.air.face_from_anchor(anchor)
    
class WGRect(_WGComponent):
    
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

def _turn_leg_length(w: float, angle: float, ratio: float, inside_radius: float, lead: float | None) -> float:
    """The centreline length Lc of both legs of a waveguide bend of width w, see _turn_polygon.

    The legs are just long enough for the outer corner fillet (tangent length ratio·w·tan(angle/2)) and the
    inner corner, plus the lead.
    """
    T = np.tan(abs(angle)/2)
    R = ratio*w
    if lead is None:
        lead = w/2 if ratio >= 1.0 else 0.0
    if ratio >= 1.0 and lead <= 0:
        # A fillet that consumes a whole edge leaves slivers in OCC
        raise ValueError(f'A waveguide bend needs a lead > 0 for ratio >= 1, got {lead}.')

    Lc = T*max(R - w/2, w/2) + lead
    Li = Lc - T*w/2
    if inside_radius > 0.0 and inside_radius*T >= Li:
        raise ValueError(f'The inside radius {inside_radius} does not fit on the inner leg of length {Li}. Increase the lead.')
    return Lc


def _turn_polygon(w: float, Lc: float, angle: float, R: float, inside_radius: float, miter: bool,
                  t: float = 0.0, e: float = 0.0) -> em.geo.XYPolygon:
    """The 2D profile of a waveguide bend of width w in the bend plane, offset outwards by t.

    With t=0 this is the air: an input leg along +y from y=0 with its centreline on x=0 and an output leg
    in the direction d=(sin(angle), cos(angle)), both with a centreline length of Lc. The outer corner
    (node 1) is filleted with radius R and the inner corner (node 4) with inside_radius. With miter=True the
    corners are cut straight between the same points, an inset of radius·tan(angle/2) from the corner along
    both edges. A negative angle mirrors the profile in x.

    With t>0 this is the outline of the walls: the offset of the air by t. Both legs are extended by e beyond
    the port planes, which can be cut off afterwards.
    """
    sign = np.sign(angle)
    angle = abs(angle)
    T = np.tan(angle/2)
    h = w/2 + t
    d = np.array([np.sin(angle), np.cos(angle)])
    n_out = np.array([-np.cos(angle), np.sin(angle)])   # The outer side of the output leg
    m = (np.array([-1.0, 0.0]) + n_out)/(1 + np.cos(angle))
    C = np.array([0.0, Lc])
    E = C + Lc*d
    up = np.array([0.0, 1.0])

    # The miter inset of an offset corner grows (outer) or shrinks (inner) by k·t
    k = 1/np.cos(angle/2)**2 - 1/np.cos(angle/2)
    if miter:
        Ro, ri = R*T + k*t, inside_radius*T - k*t
    else:
        Ro, ri = R + t, inside_radius - t

    P0 = np.array([-h, -e])
    P1 = C + h*m
    P2 = E + e*d + h*n_out
    P3 = E + e*d - h*n_out
    P4 = C - h*m
    P5 = np.array([h, -e])

    pts = [P0, P1, P2, P3]
    inner = Lc + e - h*T > 1e-12
    if inner:
        pts += [P4, P5]
    if inner and ri > 0.0:
        if miter:
            pts[4:5] = [P4 + ri*d, P4 - ri*up]
    if miter:
        pts[1:2] = [P1 - Ro*up, P1 + Ro*d]

    xs = np.array([pt[0] for pt in pts])*sign
    ys = np.array([pt[1] for pt in pts])
    poly = em.geo.XYPolygon(xs, ys)
    if not miter:
        poly.fillet(Ro, 1)
        if inner and ri > 0.0:
            poly.fillet(ri, 4)
    return poly


class WGTurnGeneric(_WGComponent):
    """The base class of the waveguide bends.

    A subclass sets the bend plane and the corner shape:
     - _plane = 'E': the bend is in the plane of the narrow side b, so the bend width w is b.
     - _plane = 'H': the bend is in the plane of the broad side a, so the bend width w is a.
     - _miter = False: the corners are rounded with fillets, _miter = True: they are cut straight.
    """
    _plane: Literal['E', 'H'] | None = None
    _miter: bool = False
    
    def __init__(self, 
                 config: WGConfig, 
                 ratio: float = 1.2,
                 angle_deg: float = 90,
                 inside_radius: float = 0.0,
                 lead: float | None = None):
        """A waveguide bend of width w (b for an E-plane and a for an H-plane bend).

        Args:
            config (WGConfig): The waveguide configuration.
            ratio (float, optional): The outer corner radius divided by w. A miter cuts the corner between the
                same points as the fillet. Defaults to 1.2.
            angle_deg (float, optional): The bend angle in degrees, between -90 and 90 and not 0. A positive
                angle bends towards +Y (E-plane) or +X (H-plane). Defaults to 90.
            inside_radius (float, optional): The inner corner radius in meters, 0 for a sharp corner. Defaults to 0.
            lead (float | None, optional): The length of a straight section in meters added to both legs
                before the bend. Must be > 0 for ratio >= 1. Defaults to w/2 for ratio >= 1 and 0 otherwise.
        """
        if self._plane is None:
            raise TypeError('WGTurnGeneric is a base class, use WGETurnRound, WGHTurnRound, WGETurnMiter or WGHTurnMiter.')
        if not (0 < abs(angle_deg) <= 90):
            raise ValueError(f'The bend angle must be between -90 and 90 degrees and not 0, got {angle_deg}.')
        self.ratio: float = ratio
        self.lead: float | None = lead
        self.angle: float = angle_deg * np.pi/180
        self.config: WGConfig = config
        self.inside_radius: float = inside_radius
        self.air: GeoVolume = None
        self.metal: GeoVolume = None
        
        self._a1: Anchor | None = None
        self._a2: Anchor | None = None

        self.build()
        
    
    def build(self) -> None:
        a,b = self.config.a, self.config.b
        th = self.config.thmm
        
        # w is the width in the bend plane, h the height of the extrusion
        w, h = (b, a) if self._plane == 'E' else (a, b)
        R = self.ratio*w
        Lc = _turn_leg_length(w, self.angle, self.ratio, self.inside_radius, self.lead)
        
        def extrude(poly: em.geo.XYPolygon, t: float) -> GeoVolume:
            if self._plane == 'E':
                return poly.extrude(h + 2*t, em.YZPLANE.cs(-h/2 - t, 0, 0))
            return poly.extrude(h + 2*t, em.XZPLANE.cs(0, h/2 + t, 0))
        
        self.air = extrude(_turn_polygon(w, Lc, self.angle, R, self.inside_radius, self._miter), 0.0)
        
        # The exit centre and the output direction in the bend plane
        s, c = np.sin(self.angle), np.cos(self.angle)
        Eu, Ev = Lc*s, Lc*(1 + c)
        
        if self.config.model_walls:
            # The walls are the air offset by th with legs extended beyond the port planes, which are cut off
            e = w + 2*th
            outer = extrude(_turn_polygon(w, Lc, self.angle, R, self.inside_radius, self._miter, th, e), th)
            B = 2*(Lc + e + w)
            d = np.array([s, c])
            n = np.array([-c, s])
            E = np.array([Eu, Ev])
            behind_input = em.geo.XYPolygon([-B, B, B, -B], [-B, -B, 0, 0])
            beyond_output = [E + B*n, E + B*n + B*d, E - B*n + B*d, E - B*n]
            beyond_output = em.geo.XYPolygon([pt[0] for pt in beyond_output], [pt[1] for pt in beyond_output])
            outer = em.geo.subtract(outer, extrude(behind_input, 2*th))
            outer = em.geo.subtract(outer, extrude(beyond_output, 2*th))
            self.metal = em.geo.subtract(outer, self.air, remove_tool=False).set_material(self.config.material)
        
        # The output anchor is the input anchor rotated by the bend angle
        self._a1 = Anchor((0,0,0))
        if self._plane == 'E':
            self._a2 = Anchor((0, Eu, Ev), (1,0,0), (0, c, -s), (0, s, c))
        else:
            self._a2 = Anchor((Eu, 0, Ev), (c, 0, -s), (0,1,0), (s, 0, c))

    @property
    def anchors(self) -> list[Anchor]:
        return (self._a1, self._a2)
    
    @property
    def front(self) -> WGTurnGeneric:
        self._active_anchor = self._a1
        return self
    
    @property
    def back(self) -> WGTurnGeneric:
        self._active_anchor = self._a2
        return self
    
    @property
    def _comps(self):
        return [self.air, self.metal]


class WGETurnRound(WGTurnGeneric):
    """An E-plane waveguide bend (in the plane of the narrow side b) with rounded corners.

    The outer corner has a radius of ratio·b and the inner corner a radius of inside_radius.
    """
    _plane = 'E'


class WGHTurnRound(WGTurnGeneric):
    """An H-plane waveguide bend (in the plane of the broad side a) with rounded corners.

    The outer corner has a radius of ratio·a and the inner corner a radius of inside_radius.
    """
    _plane = 'H'


class WGETurnMiter(WGTurnGeneric):
    """An E-plane waveguide bend (in the plane of the narrow side b) with straight (mitered) corners.

    The cuts start at the same inset distances from the corners as the fillets of WGETurnRound: ratio·b for
    the outer corner and inside_radius for the inner corner.
    """
    _plane = 'E'
    _miter = True


class WGHTurnMiter(WGTurnGeneric):
    """An H-plane waveguide bend (in the plane of the broad side a) with straight (mitered) corners.

    The cuts start at the same inset distances from the corners as the fillets of WGHTurnRound: ratio·a for
    the outer corner and inside_radius for the inner corner.
    """
    _plane = 'H'
    _miter = True

