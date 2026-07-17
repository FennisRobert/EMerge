"""Scattered field, RCS

In this demonstartion we will show how a radar cross section computation can be performed (RCS)
using the new Scattered Field boundary condition.


You can modify this script to recreate the results from the following paper:

" RCS Estimation of Singly Curved Dielectric Shell Structure with PMCHWT Method and Experimental Verification"
By Hyeong-Rae Im, Woobin Kim, Yeong-Hoon Noh, Ic-Pyo Hong and Jong-Gwan Yook

 https://www.mdpi.com/1424-8220/22/3/734


"""

import numpy as np
import emerge as em
from emerge.plot import plot

# The paper discusses three objects, a 0.5mm radius PEC sphere, 0.5m radius Dielectric sphere and Dielectric cylinder
OBJECT = "PEC SPHERE"
# OBJECT = 'DIEL SPHERE'
# OBJECT = 'DIEL CYLINDER'

air_radius = 1.5
# First we create our simulation object
model = em.Simulation("RCS")
model.check_version("3.0.0")  # Checks version compatibility

# We select the material of choice
if OBJECT == "PEC SPHERE":
    material = em.lib.PEC
else:
    material = em.Material(er=4, color="#17B258")

if OBJECT in ("PEC SPHERE", "DIEL SPHERE"):
    scatter_object = em.geo.Sphere(radius=0.5).set_material(material)
else:
    scatter_object = em.geo.Cylinder(
        radius=0.5, height=1.0, cs=em.XAX.construct_cs((-0.5, 0, 0))
    ).set_material(material)

# Then we create the sphere with radius of 0.5m and an air sphere of 1.0m radius
air = em.geo.Sphere(radius=air_radius)

# Next we finalize the geometry modelling by calling commit_geometry.
model.commit_geometry()

# We set the frequency to 300MHz and air resolution to 0.1 (wavelength ratio)
model.mw.set_frequency(300e6)

# We rather have a resolution of 0.15 or even 0.1 for accuracy but the RAM requirements are up to 20GB
model.mw.set_resolution(0.18)

# Now we can generate the mesh
model.generate_mesh()
model.view()
model.view(plot_mesh=True)

# To be able to model the radar cross-section we will use the ScatteredField boundary condition
# Currently only an azimuth-elevation style angle definition system is imple
scat = model.mw.bc.ScatteredField(air.boundary())

# +Y Polarization
scat.set_excitations(polarizations=90)


# We can improve the absorption of the waves by specifying a curvature radius for the absorbing boundary.
scat.radius = air_radius

# We have to run our simulation using run_scattered() instead of run_sweep(). No S-parameters will be calculated.
data = model.mw.run_scattered()

# The field solutions are in data.field. We take the first entry as there is only one.
field = data.field[0]

# For farfield calculations it is better to use the integration boundary of the scattering object in this case.
# This minimizes the total acculmulated phase error due to numerical dispersion.
# The Stratton-Chu integrals used are always valid as long as all current sources are contained inside the boundary.

ff3d = field.farfield_3d(scatter_object.boundary())

# We create a 3D view of the solution

# Notice that we call field.relative.grid() instead of field.grid().
# Calling .relative first makes sure that only the scattered field is shown WITHOUT the background field.
# For farfield integration this is not necessary because numerically the integral of external sources always yields 0.
display = model.display
display.populate(smooth_shading=True)  # Adds all geometries
display.add_field(field.relative.grid(N=5000).vector("E"))  # Adds a vector polot
display.add_farfield3d(
    ff3d, component="RCS", rmax=0.6, offset=(0, 0, 1.2)
)  # Adds the 3D Farfield plot
display.animate().add_field(
    field.relative.grid(N=10_000).scalar("Ey", "complex"), symmetrize=True
)  # Adds an animation of the plane wave
display.show()

# Finally we will also create a 2D plot
if OBJECT == "PEC SPHERE":
    ylim = (-10, 10)
elif OBJECT == "DIEL SPHERE":
    ylim = (-10, 15)
elif OBJECT == "DIEL CYLINDER":
    ylim = (-5, 20)

ff2d_Y = field.farfield_2d(
    (-1, 0, 0), em.YAX, scatter_object.boundary(), ang_range=(0, 180)
)
ff2d_Z = field.farfield_2d(
    (-1, 0, 0), em.ZAX, scatter_object.boundary(), ang_range=(0, 180)
)
plot(
    ff2d_Y.ang * 180 / np.pi,
    [10 * np.log10(np.abs(ff2d_Z.RCS)), 10 * np.log10(np.abs(ff2d_Y.RCS))],
    labels=("RCS Z-plane", "RCS Y-plane"),
    xlabel="Angle (deg)",
    ylabel="Bistatic RCS (dBsm)",
    xlim=[0, 180],
    linestyles=["-", "--"],
    ylim=ylim,
)
