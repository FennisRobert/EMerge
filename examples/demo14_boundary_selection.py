import emerge as em
import numpy as np
from emerge.plot import plot_ff_polar, plot_sp

"""
In this demonstration we are going to show how one can make selections using the 
available tools in EMerge. 

To understand the need for the given selection methods it is important to understand
what the difficulty is.

Any CAD implementation will create objects/shapes that can be subtracted or added
from one another. The direct result of this is that any face that logically exists
or is definable (like the front) on an input geometry (of say a box) is not guaranteed 
to exist after manipulations of the shapes. Additionally, the face tags that GMSH keeps
track off will change after you make changes to the geometries.

To still facilitate face selection, EMerge implements a heuristic face
selection algorithm that defines faces of objects as  by two sets of data:
an origin and the face normal. These pairs of origins/normals will be transformed
with all transformations on objects such that at a later stage, these
orinal references can be accessed. Then when a selection is requested, the boundaries
of objects will be computed together with their origin and normal. If an origin is close
to an original face surface and the normals are aligned, it is treated as part of the selection.

On top of that, EMerge has a selection interface that allows the user to select
faces and objects based on coordinates, layers etc.

In this demonstration we will create a complicated waveguide structure ending
in an air box to demonstrate the various selection methods.
"""

# First lets define some dimensions
mm = 0.001
wga = 22.86*mm
wgb = 10.16*mm
L = 50*mm

model = em.Simulation('Test Mode')
model.check_version("0.6.7") # Checks version compatibility.

# first lets define a WR90 waveguide
wg_box = em.geo.Box(L, wga, wgb, position=(-L, -wga/2, -wgb/2))
# Then define a capacitive iris cutout
cutout = em.geo.Box(2*mm, wga, wgb/2, position=(-L/2, -wga/2, -wgb/2))

# remove the cutout from the box. Notice that we use a different name.
# Geometry properties are not persistent after boolean operations so we 
# need the information of previous boxes.
wg_box_new = em.geo.remove(wg_box, cutout)

# define an air-box to radiat in.
airbox = em.geo.Box(L/2, L, L, position=(0,-L/2, -L/2))

# Now define the geometry
model.commit_geometry()

# Lets define a frequency range for our simulation. This is needed
# If we want to mesh our model.
model.mw.set_frequency_range(8e9, 10e9, 11)

# Now lets mesh our geometry
model.generate_mesh()

## We can now select faces and show them using the .view() interface

# The box is defined in XYZ space. The sides left/right correspond to the
# X-axis, The sides top/down to the Z-axis and front/back to the Y-axis.
# We have to provide which original object we want to pick the left side from.
feed_port = wg_box_new.face('left', tool=wg_box)

# We can also select the outside and exclude a given face. Because our airbox
# is not modified, we don't have to work with tools.
radiation_boundary = airbox.boundary(exclude=('left',))

# Lets view our result
model.view(selections=[feed_port, radiation_boundary])

# As you can see, the appropriate faces have been selected.
# You can also see that the bottom side of the resultant box
# which has been split in two can still be selected because of the selection system.
model.view(selections=[wg_box_new.face('bottom', tool=wg_box),])

# You can also access faces of the original tool objects.
model.view(selections=[wg_box_new.face('left', tool=cutout),])

# Another way to select the radiation boundary on the right is by using the 
# selction interface.

# The interface works by a language like method-chaining philosophy.
# The .select attribute is a Selector class. The property .face returns
# The same selector class but with the 'face selection mode' turned on.
# Now we can call the 'inlayer' method which selects all faces of which the
# Center of mass is inside the layer ranging from the provided starting
# coordinate up to all coordinates that extend to the vector (L,0,0).
#
#            |                    |
#            |       ____\        |
#   (origin) + ---- vector ------>|
#            |                    |
#            |                    |
#            < inside is selected >
#

radiation_boundary_2 = model.select.face.inlayer(1*mm, 0,0, (L,0,0))
model.view(selections=[radiation_boundary_2,])

# Now lets define our simulation futher and do some farfield-computation!

port = model.mw.bc.ModalPort(feed_port, 1)
rad = model.mw.bc.AbsorbingBoundary(radiation_boundary)

model.mw.modal_analysis(port, 1)

# Run the simulation
data = model.mw.run_sweep()


# First the S11 plot
f = data.scalar.grid.freq
S11 = data.scalar.grid.S(1,1)

plot_sp(f, S11, labels=['S11'])

# First we need to create a boundary mesh
rad_surf = model.mesh.boundary_surface(radiation_boundary.tags, (0,0,0))
# Then we need to compute the E-field on the edges.  We will pick the first frequency.
Ein, Hin = data.field[0].interpolate(*rad_surf.exyz).EH

# Then we define some angles for our plot
theta = np.linspace(-np.pi/2, 1.5*np.pi, 201)
phi = 0*theta
k0 =  data.field[0].k0
E, H = em.stratton_chu(Ein, Hin, rad_surf, theta, phi, k0)

# Finally we create the plot
plot_ff_polar(theta, em.norm(E))