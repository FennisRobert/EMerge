import emerge as em
import numpy as np
from emerge.plot import plot_sp

""" OPTIMIZATION

In this demo we will do a fairly simple optimization of a coax to waveguide transition.

For the purpose of this demonstration, the mesh is intentionally kept very simple. The results as presented here during
optimization and during the final simulation are not converged properly. However, fine meshes would add significant
computational time which is not desired for the purpose of this demonstration.
"""

# Value definitions
mm = 0.001
wga = 22.86 * mm
wgb = 10.16 * mm
Ltot = 50 * mm

rin = 1 * mm
rout = em.coax_rout(rin, Z0=50)
Lcoax_outside = 10 * mm

model = em.Simulation("Optimization")
model.check_version("3.0.0")

# Here will will set out optimization. We will tune two parameters
# Offset is the distance from the back wall
# Lin is the distance the centre pin protrudes into the waveguide
model.opt.add_param("offset", 10 * mm, (5 * mm, 20 * mm))
model.opt.add_param("Lin", 5 * mm, (1 * mm, 10 * mm))
# You can set your optimizer method of choice
model.opt.method = "COBYQA"

# We start our simulation optimizatino loop.
for offset, Lin in model.opt.run(max_iter=40):
    waveguide = em.geo.Box(wga, Ltot, wgb, (-wga / 2, 0, -wgb))
    coax_out = em.geo.Cylinder(
        rout, Lcoax_outside, em.cs(origin=(0, offset, 0)), Nsections=14
    )
    coax_in = em.geo.Cylinder(
        rin, Lcoax_outside + Lin, em.cs(origin=(0, offset, -Lin)), Nsections=8
    ).set_material(em.lib.PEC)

    model.commit_geometry()
    model.mw.set_frequency_range(8e9, 10e9, 5)
    model.set_resolution(0.2)

    model.mw.bc.ModalPort(coax_out.face("+z"), 1, modetype="TEM")
    model.mw.bc.RectangularWaveguide(waveguide.face("+y"), 2)

    # Ideally we would be using adaptive mesh refinement but to decrease the run time of this
    # example we will manually refine the mesh close to the probe tip to improve the solution quality.
    model.mesher.set_domain_size(coax_out, 1 * mm)
    model.mesher.set_boundary_size(coax_in.face("-z"), 1 * mm, growth_rate=1.5)

    model.generate_mesh()

    data = model.mw.run_sweep()

    # Due to some peculiarities, we have to constrain ourselves to the last 5 simulation data entries
    # Because those are the last 5 frequency points that we computed in this sweep.
    # This is a rather awekward feature which will be ironed out later.

    grid = data.scalar.slice_set(-5).grid

    # We want to minimize the greatest reflection coefficient inside our frequency band
    max_S11dB = np.max(20 * np.log10(np.abs(grid.S(1, 1))))

    # We pass this largest S11 in dB to the optimizer to give it the result of our optimization
    model.opt.update(max_S11dB)

    # Optional
    # You can whipe the simulation data after each run to prevent RAM clogging
    # sim.reset(data=True)

# To simulate our final result we will for convenience whipe our entire simulation object and start fresh.
# This is not a necessary step but ok for this application.
model.reset(all=True)

# Inside our optimizer we can retreive the best solution which contains the solution dictionary and the value at that instance
solution, value = model.opt.best

# We extract the parameters we need and execute a detailed simulation
offset = solution["offset"]
Lin = solution["Lin"]

waveguide = em.geo.Box(wga, Ltot, wgb, (-wga / 2, 0, -wgb))
coax_out = em.geo.Cylinder(
    rout, Lcoax_outside, em.cs(origin=(0, offset, 0)), Nsections=14
)
coax_in = em.geo.Cylinder(
    rin, Lcoax_outside + Lin, em.cs(origin=(0, offset, -Lin)), Nsections=8
).set_material(em.lib.PEC)

model.commit_geometry()
model.mw.set_frequency_range(8e9, 10e9, 21)
model.set_resolution(0.2)

model.mw.bc.ModalPort(coax_out.face("+z"), 1, modetype="TEM")
model.mw.bc.RectangularWaveguide(waveguide.face("+y"), 2)

model.mesher.set_domain_size(coax_out, 1 * mm)
model.mesher.set_boundary_size(coax_in.face("-z"), 1 * mm, growth_rate=1.5)

model.generate_mesh()
model.view(plot_mesh=True)

data = model.mw.run_sweep()

grid = data.scalar.grid

fd = grid.dense_f(1001)

plot_sp(fd, [grid.model_S(1, 1), grid.model_S(2, 1)], labels=["S11", "S21"])

# 3D view
field = data.field.find(freq=9e9)
model.display.add_objects(*model.all_geos())
model.display.animate().add_field(
    field.cutplane(1 * mm, x=0).scalar("Ez", "real"), symmetrize=True
)
model.display.show()
