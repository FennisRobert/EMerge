import emerge as em
from emerge.ext import GeoVolume, GeoSurface

mm = 0.001


class SMAConnector:


    def __init__(self, thpcb: float,  cs: em.CoordinateSystem, Nsections: int = 12):
        """The 

        Args:
            thpcb (float): _description_
            cs (em.CoordinateSystem): _description_
            Nsections (int, optional): _description_. Defaults to 12.
        """
        self.th: float = thpcb 

        self.metal: GeoVolume | None = None
        self.teflon: GeoVolume | None = None
        self.port_face: GeoSurface | None = None
        self.void: GeoVolume | None = None
        self.center: GeoVolume | None = None

        self.cs: em.CoordinateSystem = cs
        self.ns: int = Nsections
        self.build()
        self.move()

    def move(self):
        em.geo.change_coordinate_system(self.metal, self.cs)
        em.geo.change_coordinate_system(self.teflon, self.cs)
        em.geo.change_coordinate_system(self.port_face, self.cs)
        em.geo.change_coordinate_system(self.void, self.cs)
        em.geo.change_coordinate_system(self.center, self.cs)

    def build(self):
        cri = 0.76*mm/2

        w = 9.53*mm 
        d = (cri+self.th+1.68*mm)*2
        th = 1.65*mm
        
        ro = 6.35*mm
        dy0 = 0.38*mm

        cro = em.coax_rout(0.76*mm/2, eps_r=2.1, Z0=50)
        

        hv = 1*mm
        # brass
        box = em.geo.Box(w, d, th, (-w/2, -d/2+dy0, 0))
        cylout = em.geo.Cylinder(ro/2, 9.53*mm, em.cs(origin=(0, dy0,0)), Nsections=self.ns)
        p1l = em.geo.Box(1.02*mm, 1.68*mm, 4.75*mm, (-w/2, -d/2+dy0, -4.75*mm))
        p2l = em.geo.Box(1.02*mm, 1.02*mm, 4.75*mm, (-w/2, 0, -4.75*mm))
        p1r = em.geo.mirror(p1l, (0,0,0), (1,0,0))
        p2r = em.geo.mirror(p2l, (0,0,0), (1,0,0))

        pc = em.geo.Cylinder(0.76*mm/2, 4.75*mm+9.53*mm-hv, em.cs(origin=(0,+dy0,-4.75*mm)), Nsections=self.ns)
        pinbox = em.geo.Box(cri*2, cri, 4.75*mm, (-cri, 0, -4.75*mm))

        pc = em.geo.unite(pc, pinbox)

        metal = em.geo.unite(box, cylout, p1l, p2l, p1r, p2r)
        teflon = em.geo.Cylinder(cro, 9.53*mm-hv, em.GCS.displace(0,dy0,0), Nsections=self.ns)
        port = em.geo.duplicate(teflon.back)
        teflon = em.geo.subtract(teflon, pc, remove_tool=False)
        void = em.geo.Cylinder(cro, hv, em.GCS.displace(0,dy0,9.53*mm-hv), Nsections=self.ns)

        metal = em.geo.subtract(metal, teflon, remove_tool=False)
        metal = em.geo.subtract(metal, void, remove_tool=False)

        teflon.set_material(em.lib.DIEL_TEFLON)
        metal.set_material(em.lib.PEC)
        pc.set_material(em.lib.PEC)

        self.metal = metal
        self.teflon = teflon
        self.port_face = port
        self.void = void
        self.center = pc
        self.void.properties += em.VoidAttribute()

        self.metal.prio_set(40)
        self.teflon.prio_set(30)
        self.center.prio_set(40)
