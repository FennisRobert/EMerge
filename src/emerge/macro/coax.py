import emerge as em
from emerge.ext import GeoSurface
from typing import Literal

_END_OPTIONS = ('none', 'pin', 'port')


class CoaxAssy:

    def __init__(self,
                 rin: float,
                 length: float,
                 dielectric: em.Material,
                 cs: em.CoordinateSystem,
                 rout: float | None = None,
                 Z0: float | None = None,
                 conductor: em.Material = em.lib.PEC,
                 shield_thickness: float = 0.001,
                 Nsections: int | None = None,
                 start: Literal['none', 'pin', 'port'] = 'none',
                 end: Literal['none', 'pin', 'port'] = 'none',
                 pin_length: float | None = None,
                 void_length: float = 0.0005,
                 start_port_nr: int | None = None,
                 end_port_nr: int | None = None,
                 ):
        """A coaxial line along the z-axis of cs, from z = 0 to z = length.

        Each end of the line is one of:
            'none': the pin, dielectric and shield end flush.
            'pin':  the pin sticks out pin_length beyond the end, e.g. as a probe.
            'port': a wave port on the end face of the dielectric. Behind it is a void of void_length, closed by
                    the shield, which is extended by void_length + shield_thickness.

        Args:
            rin (float): The radius of the inner conductor (pin)
            length (float): The length of the dielectric
            dielectric (em.Material): The material between the conductors
            cs (em.CoordinateSystem): The coordinate system; the line starts in its origin, along its z-axis
            rout (float | None, optional): The inner radius of the shield. Give rout or Z0. Defaults to None.
            Z0 (float | None, optional): The characteristic impedance, from which rout is computed. Defaults to None.
            conductor (em.Material, optional): The material of the pin and shield. Defaults to em.lib.PEC.
            shield_thickness (float, optional): The wall thickness of the shield. Defaults to 0.001.
            Nsections (int | None, optional): Build the cylinders as prisms with this many sides. Defaults to None.
            start (Literal['none', 'pin', 'port'], optional): The start (z = 0) of the line. Defaults to 'none'.
            end (Literal['none', 'pin', 'port'], optional): The end (z = length) of the line. Defaults to 'none'.
            pin_length (float | None, optional): How far the pin sticks out at a 'pin' end. Defaults to None.
            void_length (float, optional): The length of the void behind a 'port' end. Defaults to 0.0005.
            start_port_nr (int | None, optional): The port number of a 'port' start. Defaults to None.
            end_port_nr (int | None, optional): The port number of a 'port' end. Defaults to None.
        """
        start = start.lower().strip()
        end = end.lower().strip()
        self._check_arguments(rin, length, rout, Z0, shield_thickness, start, end, pin_length, void_length,
                              start_port_nr, end_port_nr)

        if rout is None:
            rout = em.coax_rout(rin, dielectric.er.value, Z0)
        if rout <= rin:
            raise ValueError(f'CoaxAssy: rout = {rout} must be larger than rin = {rin}.')
        self.rin: float = rin
        self.rout: float = rout
        self.length: float = length

        lin = length
        lshield = length
        z0pin = 0.0
        z0shield = 0.0

        if start == 'pin':
            lin += pin_length
            z0pin -= pin_length
        elif start == 'port':
            lshield += void_length + shield_thickness
            z0shield -= void_length + shield_thickness

        if end == 'pin':
            lin += pin_length
        elif end == 'port':
            lshield += void_length + shield_thickness

        dzx, dzy, dzz = cs.zax.np

        def along(z: float) -> em.CoordinateSystem:
            return cs.displace(dzx*z, dzy*z, dzz*z)

        rshield = rout + shield_thickness

        self.pin: em.geo.Cylinder = em.geo.Cylinder(rin, lin, along(z0pin), Nsections).set_material(conductor)
        self.diel: em.geo.Cylinder = em.geo.Cylinder(rout, length, cs, Nsections).set_material(dielectric)
        self.shield: em.geo.Cylinder = em.geo.Cylinder(rshield, lshield, along(z0shield), Nsections).set_material(conductor)
        self.start_void: em.geo.Cylinder | None = None
        self.end_void: em.geo.Cylinder | None = None
        self.start_port: GeoSurface | None = None
        self.end_port: GeoSurface | None = None

        # The shield is cut with the full dielectric cylinder, then the pin is cut out of the dielectric
        self.shield = em.geo.subtract(self.shield, self.diel, remove_tool=False)
        self.diel = em.geo.subtract(self.diel, self.pin, remove_tool=False)

        # The port sits on the face shared by the void and the dielectric
        if start == 'port':
            self.start_void = em.geo.Cylinder(rout, void_length, along(-void_length), Nsections)
            self.start_void.properties += em.VoidAttribute()
            self.shield = em.geo.subtract(self.shield, self.start_void, remove_tool=False)
            self.start_port = em.geo.duplicate(self.start_void.back)
            self.start_port.properties += em.WavePortAttribute(start_port_nr, 'TEM', True)
        if end == 'port':
            self.end_void = em.geo.Cylinder(rout, void_length, along(length), Nsections)
            self.end_void.properties += em.VoidAttribute()
            self.shield = em.geo.subtract(self.shield, self.end_void, remove_tool=False)
            self.end_port = em.geo.duplicate(self.end_void.front)
            self.end_port.properties += em.WavePortAttribute(end_port_nr, 'TEM', True)

    @staticmethod
    def _check_arguments(rin, length, rout, Z0, shield_thickness, start, end, pin_length, void_length,
                         start_port_nr, end_port_nr) -> None:
        """Raises a ValueError if the arguments do not fit the chosen start and end options."""
        for name, option in (('start', start), ('end', end)):
            if option not in _END_OPTIONS:
                raise ValueError(f"CoaxAssy: {name} = '{option}' must be one of {_END_OPTIONS}.")
        if rin <= 0 or length <= 0:
            raise ValueError(f'CoaxAssy: rin and length must be positive, got rin = {rin}, length = {length}.')
        if (rout is None) == (Z0 is None):
            raise ValueError('CoaxAssy: give either rout or Z0 (not both).')
        if shield_thickness <= 0:
            raise ValueError(f'CoaxAssy: shield_thickness must be positive, got {shield_thickness}.')
        if 'pin' in (start, end) and (pin_length is None or pin_length <= 0):
            raise ValueError(f"CoaxAssy: a 'pin' start or end needs a positive pin_length, got {pin_length}.")
        if 'port' in (start, end) and void_length <= 0:
            raise ValueError(f"CoaxAssy: a 'port' start or end needs a positive void_length, got {void_length}.")
        if start == 'port' and start_port_nr is None:
            raise ValueError("CoaxAssy: start = 'port' needs a start_port_nr.")
        if end == 'port' and end_port_nr is None:
            raise ValueError("CoaxAssy: end = 'port' needs an end_port_nr.")
        if start == 'port' and end == 'port' and start_port_nr == end_port_nr:
            raise ValueError(f'CoaxAssy: start_port_nr and end_port_nr must differ, both are {start_port_nr}.')
