# -----------------------------------------------------------------------------
# Parallel plate waveguides
#
# In this example we will look at a simple parallel plate transmisison line. The
# wave propagating inside is a simple constant E-field. This is a typical EM problem
# taught in electromagnetics that can be simulated in EMerge.
# -----------------------------------------------------------------------------

# The parallel plates will be places in the XZ plane as following
#
# Top plate @ y=10mm     -------------------------------------
#                          y
#                          ↑ Ey(x,y,z) = 1.0 * exp(-jk₀z)
#                          └─→ x
#
# Bottom plate @ y=-10mm -------------------------------------
#
# We start by importing EMerge and Numpy

import emerge as em
import numpy as np

# We define mm to be 0.001 meters as EMerge uses normal SI units
mm = 0.001

# Then we create our simulation object
model = em.Simulation("ParallelPlates")

# Parallel plate transmision lines are infinitely big. To model a finite volume we have to use the appropriate boundary conditions
# We will simulate a wave at 10GHz so we dimension it appropriately as a box of 20mm wide, 20mm deep and 50mm long.
box = em.geo.Box(20 * mm, 20 * mm, 50 * mm, position=(-10 * mm, -10 * mm, 0))

# As the box is all we need, we will say that we stop with our modelling so we can proceed
model.commit_geometry()
model.check_version("2.8.2")

# Before we can generate the mesh we have to set a simulation frequency (range)
model.mw.set_frequency(10e9)

# Then we can generate the mesh
model.generate_mesh()
# And view the output
model.view()


# We will use a User defined port mode. In our case the E-field of a prallel plate transmission line is easy because its just a constant Ey=1.0
def Ey_field(k0, x, y, z):
    # We have to return ones_like(x) and not 1.0 because we need an array of 1.0 values, not a single one.
    return np.ones_like(x)


# We create the ports at the bottom and top. We will call the bottom face port 1 and the top 2.
# EMerge also needs the out of plane propagation constant as a function of k0 but it assumes kz=k0 by default.

p1 = model.mw.bc.UserDefinedPort(box.bottom, 1, Ey=Ey_field)
p2 = model.mw.bc.UserDefinedPort(box.top, 2, Ey=Ey_field)

# We will set the left and right side of the environment as PMC to prevent the tangential E-field from being 0.
# We can simply add the left and right box side selections together.
model.mw.bc.PMC(box.left + box.right)

# Finally we can just run the simulation
data = model.mw.run_sweep()

# Here we create a 3D view. First we add the box as an object to be displayed.
model.display.add_object(box)
# To create a good animation plot we use method chianing to build our cutplane.
# The first .animate() call will toggle on an animation.
# Then we add a surface plot. We use the first field solution (and only one) and use the cutplane() method
# to easily compute the solution on a regular grid defined as a cutplane. From this solution we need the
# X, Y, and Z coordinates and the field amplitude. THis is what the .scalar() function returns.
# We return a complex valued field because then we can actually animate it.
# We set symmetrize true to make sure the color range goes from -val to +val
model.display.animate().add_field(
    data.field[0].cutplane(1 * mm, y=0).scalar("Ey", "complex"), symmetrize=True
)
# With these two plots we can show the field mode that we created.
model.display.add_portmode(p1, k0=data.field[0].k0, color="red")
model.display.add_portmode(p2, k0=data.field[0].k0, color="red")
# Finally we display our animation.
model.display.show()
