import emerge as em
import numpy as np
from emerge.plot import plot_sp

""" DEMO: COMBLINE FILTER

In this demo we will look at the design of a combline filter in EMerge. The filter design was taken from
the book "Modern RF and Microwave Filter Design" by Protap Pramanick and Prakash Bhartia.
Some of the dimensions where not clear.

In this demo we will look at the Modeler class and how it can help us quickly create complicated geometries.

"""

# First we define some quantities for our simulation.
mm = 0.001
mil = 0.0254*mm

a = 240*mil
b = 248*mil
d1 = 10*mil
d2 = 10*mil
dc = 8.5*mil
lr1 = b-d1
lr2 = b-d2
W = 84*mil
S1 = 117*mil
S2 = 136*mil
C1 = b-dc
h = 74*mil
wi = 84*mil
Lbox = 5*W + 2*(S1+S2+wi)

x1 = wi+W/2
x2 = x1 + W + S1
x3 = x2 + W + S2
x4 = x3 + W + S2
x5 = x4 + W + S1

rout = 40.5*mil
rin = 12.5*mil
lfeed = 100*mil

# A usual we start our simulation file
model = em.Simulation('Combline_DEMO')
model.check_version("0.6.7") # Checks version compatibility.

# The filter consists of quarter lamba cylindrical pins inside an airbox.
# First we create the airbox
box = em.geo.Box(Lbox, a, b, position=(0,-a/2,0))

# Next we create 5 cylinders using the Modeler class.
# The modeler class also implements a method chaining interface. In this example we stick to simpler features.
# The modeler class allows us to create a parameter series using the modeler.series() method. We provid it with quantities.
# We can do this for multiple at the same time (as you can also see with the position). The modeler class
# will recognize the multiple quantities and simply create 5 different cylinders, one for each parameter pair.
stubs = model.modeler.cylinder(W/2, model.modeler.series(C1, lr1, lr2, lr1, C1), position=(model.modeler.series(x1, x2, x3, x4, x5), 0, 0), NPoly=10)

# Next we create the in and output feed cylinders for the coaxial cable. We will use the Nsections feature in order to guarantee a better
# adherence to the boundary.
feed1out = em.geo.Cylinder(rout, lfeed, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([-lfeed, 0, h])), Nsections=12)
feed1in = em.geo.Cylinder(rin, lfeed+wi+W/2, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([-lfeed, 0, h])), Nsections=8)
feed2out = em.geo.Cylinder(rout, lfeed, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([Lbox, 0, h])), Nsections=12)
feed2in = em.geo.Cylinder(rin, lfeed+wi+W/2, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([Lbox-wi-W/2, 0, h])), Nsections=8)

# Next we subtract the stubs and the center conductor from the box and feedline.
for ro in stubs:
    box = em.geo.subtract(box, ro)
box = em.geo.subtract(box, feed1in, remove_tool=False)
box = em.geo.subtract(box, feed2in, remove_tool=False)
feed1out = em.geo.subtract(feed1out, feed1in)
feed2out = em.geo.subtract(feed2out, feed2in)

# Finally we may define our geometry
model.commit_geometry()

model.view()

# We define our frequency range and a fine sampling.
model.mw.set_frequency_range(6e9, 8e9, 21)

model.mw.resolution = 0.05

# To improve simulation quality we refine the faces at the top of the cylinders.
for stub in stubs:
    model.mesher.set_boundary_size(box.face('back', tool=stub), 0.0002)

# Finally we may create our mesh.
model.generate_mesh()

# We define our modal ports, assign the boundary condition and execute a modal analysis to solve for the
# coaxial field mode.
port1 = model.mw.bc.ModalPort(model.select.face.near(-lfeed, 0, h), 1, TEM=True)
port2 = model.mw.bc.ModalPort(model.select.face.near(Lbox+lfeed, 0, h), 2, TEM=True)

# At last we can compute the frequency domain study
data = model.mw.run_sweep(parallel=True)

# Next we will use the Vector Fitting algorithm to model our S-parameters with a Rational function

fdense = np.linspace(6e9, 9e9, 2001)

S11 = data.scalar.grid.model_S(1,1,fdense)
S21 = data.scalar.grid.model_S(2,1,fdense)

plot_sp(fdense, [S11, S21], labels=['S11','S21'])

# We can also plot the field inside. First we create a grid of sample point coordinates
xs = np.linspace(0, Lbox, 41)
ys = np.linspace(-a/2, a/2, 11)
zs = np.linspace(0, b, 15)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# The E-field can be interpolated by selecting a desired solution and then interpolating it.
Ex, Ey, Ez = data.field[3].interpolate(X,Y,Z).E

# We can add the objects we want and fields using the shown methods.
model.display.add_object(box, opacity=0.1, show_edges=True)
model.display.add_quiver(X,Y,Z, Ex.real, Ey.real, Ez.real)
model.display.add_object(feed1out, opacity=0.1)
model.display.add_portmode(port1, 21)
model.display.add_portmode(port2, 21)
model.display.add_surf(*data.field[3].cutplane(ds=0.5*mm, y=0).scalar('normE'))
model.display.show()