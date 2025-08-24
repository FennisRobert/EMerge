import emerge as em
from emerge.plot import plot_sp

""" DEMO 12: Mode Alignment

In some applications we wish to simulate transmission lines with multiple modes. In this
example we look at a circular waveguide with 4 ridges. The modal analysis performed to find
the modes are not guaranteed however to return each mode in order, nor in the right polarization
direction (+Z vs -Z). In EMerge we can alignm modes using the .align_mode() method.

"""

# First we define our simulation
m = em.Simulation('aligntest')
m.check_version("0.6.7") # Checks version compatibility.

# We create a cyllindrical waveguide in the Y-axis.
cyl = em.geo.Cylinder(0.012, 0.05, em.CS(em.XAX, em.ZAX, em.YAX))

# We remove 4 ridges from this with a width of 2mm. 
wr= 0.002
gap = 0.015
ridge1 = em.geo.Box(wr, 0.05, 0.02, position=(-wr/2, 0, gap/2))
ridge2 = em.geo.Box(wr, 0.05, 0.02, position=(-wr/2, 0, gap/2))
ridge3 = em.geo.Box(wr, 0.05, 0.02, position=(-wr/2, 0, gap/2))
ridge4 = em.geo.Box(wr, 0.05, 0.02, position=(-wr/2, 0, gap/2))

ridge2 = em.geo.rotate(ridge2, (0,0,0), (0,1,0), 90)
ridge3 = em.geo.rotate(ridge3, (0,0,0), (0,1,0), 180)
ridge4 = em.geo.rotate(ridge4, (0,0,0), (0,1,0), 270)

wg = em.geo.subtract(cyl, ridge1)
wg = em.geo.subtract(wg, ridge2)
wg = em.geo.subtract(wg, ridge3)
wg = em.geo.subtract(wg, ridge4)


# We define a resolution and mesh our geometry
m.mw.set_resolution(0.2)
m.mw.set_frequency_range(8e9,10e9, 7)

m.generate_mesh()

m.view()

# Next we define our modal ports.
p1 = m.mw.bc.ModalPort(wg.face('front', tool=cyl), 1)
p2 = m.mw.bc.ModalPort(wg.face('back', tool=cyl), 2)

# To align our modes we use the .align_modes() method. This takes a series of arguments
# in the form of an Axis object, a tuple like (1,0,0) for +X or a numpy array (np.array([0,1,0])).ArithmeticError
# After solving for the modes, each mode will be sorted based on which mode has the largest inner product with these axis objects.
# The axes will also be used to align the phase (0 degrees vs 180 degrees).
# Because the waveguide supports two modes in the X and Z polarization direction we will use these axes to align them.ArithmeticError

p1.align_modes(em.ZAX, em.XAX)
p2.align_modes(em.ZAX, em.XAX)

# We will now iterate through our modes and solve our system. The mode used for a simulation is defined by setting the
# mode_obj.selected_mode index. The first is 0, the second 1 etc. EMerge will automatically bundle our data.

for i1, i2 in m.parameter_sweep(False, p1=(0,1), p2=(0,1)):
    p1.selected_mode = i1
    p2.selected_mode = i2
    data = m.mw.run_sweep()

# We can now visualize the mode fields in order. Please keep in mind that by default we look into the -X, -Y, -Z direction.

for i in range(2):
    for j in range(2):
        m.display.add_object(wg)
        m.display.add_portmode(p1, 30, k0=data.field[0].k0, mode_number=i)
        m.display.add_portmode(p2, 30, k0=data.field[0].k0, mode_number=j)
        m.display.show()

# We extract the gritted data if we want and verify that the coupling between the modes is low.

sgrid = data.scalar.grid

S21 = sgrid.S(2,1)
S21_11 = S21[0,0,:]
S21_21 = S21[1,0,:]
S21_12 = S21[0,1,:]
S21_22 = S21[1,1,:]

plot_sp(sgrid.freq[0,0,:], [S21_11, S21_12, S21_21, S21_22], labels=['T11','T12','T21','T22'])