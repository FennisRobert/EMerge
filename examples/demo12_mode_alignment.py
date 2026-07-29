import emerge as em
from emerge.plot import plot_sp

""" DEMO 12: Mode Alignment

In some applications we wish to simulate transmission lines with multiple modes. In this
example we look at a circular waveguide with 4 ridges. The modal analysis performed to find
the modes are not guaranteed however to return each mode in order, nor in the right polarization
direction (+Z vs -Z). In EMerge we can alignm modes using the .align_mode() method.

"""

# First we define our simulation
model = em.Simulation("AlignmentDemo")
model.check_version("2.8.2")  # Checks version compatibility.

# We create a cyllindrical waveguide in the Y-axis.
cyl = em.geo.Cylinder(0.012, 0.05, em.YAX.construct_cs())
model.parameter_sweep()
# We remove 4 ridges from this with a width of 2mm.
wr = 0.002
gap = 0.015
ridge1 = em.geo.Box(wr, 0.05, 0.02, position=(-wr / 2, 0, gap / 2))
ridge2 = em.geo.Box(wr, 0.05, 0.02, position=(-wr / 2, 0, gap / 2))
ridge3 = em.geo.Box(wr, 0.05, 0.02, position=(-wr / 2, 0, gap / 2))
ridge4 = em.geo.Box(wr, 0.05, 0.02, position=(-wr / 2, 0, gap / 2))

ridge2 = em.geo.rotate(ridge2, (0, 0, 0), (0, 1, 0), 90)
ridge3 = em.geo.rotate(ridge3, (0, 0, 0), (0, 1, 0), 180)
ridge4 = em.geo.rotate(ridge4, (0, 0, 0), (0, 1, 0), 270)

wg = em.geo.subtract(cyl, ridge1)
wg = em.geo.subtract(wg, ridge2)
wg = em.geo.subtract(wg, ridge3)
wg = em.geo.subtract(wg, ridge4)


# We define a resolution and mesh our geometry
model.mw.set_resolution(0.1)
model.mw.set_frequency_range(8e9, 10e9, 7)

model.generate_mesh()

model.view()

# Next we define our modal ports.
# We will use the number_of_modes property to indicate how many modes that are solved for will be included in the
# port sweep. By default this is 1 mode.
# We also specify the mode type to ensure the proper computation of the port properties.

p1 = model.mw.bc.ModalPort(
    wg.face("front", tool=cyl), 1, modetype="TE", number_of_modes=2
)
p2 = model.mw.bc.ModalPort(
    wg.face("back", tool=cyl), 2, modetype="TE", number_of_modes=2
)

# To align our modes we use the .align_modes() method. This takes a series of arguments
# in the form of an Axis object, a tuple like (1,0,0) for +X or a numpy array (np.array([0,1,0])).ArithmeticError
# After solving for the modes, each mode will be sorted based on which mode has the largest inner product with these axis objects.
# The axes will also be used to align the phase (0 degrees vs 180 degrees).
# Because the waveguide supports two modes in the X and Z polarization direction we will use these axes to align them.ArithmeticError

p1.align_modes(em.ZAX, em.XAX)
p2.align_modes(em.ZAX, em.XAX)

data = model.mw.run_sweep()

# We can now visualize the mode fields in order. Please keep in mind that by default we look into the -X, -Y, -Z direction.

for i in range(1, 3):
    for j in range(1, 3):
        model.display.add_object(wg)
        model.display.add_portmode(p1, k0=data.field[0].k0, mode_number=i)
        model.display.add_portmode(p2, k0=data.field[0].k0, mode_number=j)
        model.display.show()

# We extract the gritted data if we want and verify that the coupling between the modes is low.

sgrid = data.scalar.grid

# In emerge we can use floating point notation to specify different modes.
# Due to the previously specified alignment vectors we know that:
# 1.1 = Port 1 Z-polarized
# 1.2 = Port 1 X-polarized
# 2.1 = Port 2 Z-polarized
# 2.2 = Port 2 X-polarized

S21_11 = sgrid.S(2.1, 1.1)
S21_12 = sgrid.S(2.1, 1.2)
S21_21 = sgrid.S(2.1, 2.1)
S21_22 = sgrid.S(2.1, 2.2)

plot_sp(
    sgrid.freq,
    [S21_11, S21_12, S21_21, S21_22],
    labels=["S2:1,1:1", "S2:1,1:2", "S2:1,2:1", "S2:1,2:2"],
)
