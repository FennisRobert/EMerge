import fem
from fem.modeling.extrude import XYPolygon, Extrusion

with fem.Simulation3D('extrusion') as m:

    circ = XYPolygon.circle(0.1, tolerance=0.05)

    extrusion = Extrusion(circ, fem.cs.GCS)

    vol = extrusion.extrude_z(0,0.5, N=2)

    m.define_geometry(vol)

    m.physics.set_frequency(1e9)

    m.generate_mesh()

    from fem.plotting.pyvista import PVDisplay

    d = PVDisplay(m.mesh)
    d.add_object(vol, opacity=0.5)
    d.add_object(vol.front_face, color='red')
    d.show()