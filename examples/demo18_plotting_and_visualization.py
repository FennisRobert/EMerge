import emerge as em
from emerge.plot import plot_sp, plot_ff, plot_ff_polar, smith
import numpy as np

""" DEMO 18: Plotting and Visualization

In this demo we will show the various plotting and visualization features in EMerge. 
We will use a quite simple geometry with some basic features that can run quickly to showcase
all the features in EMerge. 

The model will be a simple WR90 waveguide with a small inductive iris radiating into a half sphere.
In front of the radiator we place a 5mm radius copper ball, just for visualization purposes, to make things interesting!

It is not supposed to do anything special nor be well matched. Its only about demonstrating plot features.

"""
# First define some constants

mm = 0.001
wga = 22.86 * mm
wgb = 10.16 * mm
L = 50 * mm
rad = 30 * mm
th = 1 * mm
opening = 0.8
distance = 2 * mm

# Setup the simulation

m = em.Simulation("PlottingDemo")
m.check_version("2.8.8")

box = em.geo.Box(wgb, wga, L, (-wgb / 2, -wga / 2, -L))
cut1 = em.geo.Box(
    wgb, wga * (1 - opening) / 2, th, (-wgb / 2, -wga / 2, -distance - th / 2)
)
cut2 = em.geo.Box(
    wgb, wga * (1 - opening) / 2, th, (-wgb / 2, wga * opening / 2, -distance - th / 2)
)

feed = em.geo.subtract(box, cut1)
feed = em.geo.subtract(feed, cut2)

halfsphere = em.geo.HalfSphere(rad, direction=(0, 0, 1))

sphere = em.geo.Sphere(5 * mm, (0, 0, 10 * mm)).set_material(
    em.lib.COPPER
)  # This sphere is just for instersesting visuals and scattering.
sphere.max_meshsize = 2 * mm

# The simplest way to view the geometry is to call the "view" method on our simulation object.
# The view method can use the GMSH viewer or the EMerge PyVista viewer.
# Since version 2.2 you can call view at any time. A quick coarse mesh will be made.

m.view()

# You can also use the traditional GMSH viewer

m.view(use_gmsh=True)

# We commit the geometry and generate the mesh.
m.commit_geometry()
m.mw.set_frequency_range(9e9, 10e9, 3)
m.mw.set_resolution(0.2)
m.generate_mesh()

# If we now call view, we will see the PyVista plot window. Just click it away if you are done to proceed to the next.
m.view()

# We can pass one or more selections to the viewer to highlight them in red.

m.view(selections=[feed.face("bottom", tool=box)])

# We can also view all object labels by turning labels on

m.view(labels=True)

# Whenever it is important to know face names for selections we can simply show all of them like this for a specific geometry.

m.view(selections=feed.all_faces())

# or this for all faces

m.view(face_labels=True)

# We can also view the mesh in various ways

m.view(plot_mesh=True)
m.view(plot_mesh=True, volume_mesh=False)  # Faces only!

############################################################
#                        SIMULATION                       #
############################################################

# Now lets run our simulation to view the data we produce.

abc = halfsphere.outside
my_port = m.mw.bc.RectangularWaveguide(feed.face("-z"), 1)
m.mw.bc.AbsorbingBoundary(abc)

# We can use the view method to also show our boundary conditions!
m.view(bc=True)

data = m.mw.run_sweep()

glob = data.scalar.grid

############################################################
#                         1D PLOTS                        #
############################################################

# We have now created our data. First lets look at some 1D plot options. First we can visualize S-parameters

plot_sp(
    glob.freq,
    glob.S(1, 1),
    labels=[
        "S11",
    ],
)

# Or a smith plot

smith(
    glob.S(1, 1), f=glob.freq
)  # The frequency is not necessary but sometimes useful to show!


############################################################
#                      FARIELD PLOTS                      #
############################################################

# There are also some useful farfield plot functions. First lets compute the farfield in 2 slices

field = data.field.find(freq=9.5e9)

ff_azi = field.farfield_2d((0, 0, 1), (0, 1, 0), abc, (-90, 90))
ff_ele = field.farfield_2d((0, 0, 1), (1, 0, 0), abc, (-90, 90))

plot_ff(
    ff_azi.ang * 180 / 3.1415,
    [ff_azi.normE / em.EISO, ff_ele.normE / em.EISO],
    dB=True,
    labels=["E-plane", "H-plane"],
    ylabel="Gain [dBi]",
)

# Or a polar plot

plot_ff_polar(
    ff_azi.ang,
    [ff_azi.normE / em.EISO, ff_ele.normE / em.EISO],
    dB=True,
    labels=["E-plane", "H-plane"],
)

############################################################
#                         3D PLOTS                        #
############################################################

# We will now look at some of the 3D plot functionality.
# To customize 3D plot windows, we use the display attribute of our simulation model.
# We can name it for simplicity

dp = m.display

# I will now go through some steps to demonstrate how it works.
# By default the Display is empty, we have to populate it.

dp.show()

# We can add objects one by one...

dp.add_object(feed)
dp.add_object(halfsphere)
dp.add_object(sphere)

dp.show()

# Or in sequence

dp.add_objects(feed, halfsphere, sphere)
dp.show()

# Or automatically

dp.add_objects(*m.all_geos())
dp.show()

# The default add object method has some extra features to change how geometries are visulaized like opacity.

dp.add_object(feed)
dp.add_object(halfsphere)
dp.add_object(sphere, opacity=0.2)

dp.show()

""" 
We often want to add fields to visualize. We will add those now.
The display class has no direct features to add cut-plane E-fields or stuff like that. Instead it just has a
surface plot function .add_surf(X,Y,Z,Scalar) that takes at least 4 positional arguments. 
    
    X: 2D array of X-coordinates
    Y: 2D array of Y-coordinates
    Z: 2D array of Z-coordinates
    Scalars: 2C array of scalar values
    
Thus we need to create those. To create that data we can use our dataset. By using the .find() method we create
a Microwave field object that we can use to interpolate our E-field. The slow way to create a good dataset for a cutplane
is as following.
"""
ys = np.arange(-rad, rad, 2 * mm)
zs = np.arange(-L, rad, 2 * mm)
Y, Z = np.meshgrid(ys, zs)
X = 0 * Y
field_data = field.interpolate(X, Y, Z)

dp.add_objects(*m.all_geos())
dp.add_surf(
    X, Y, Z, field_data.Ex.real, symmetrize=True
)  # Symmetrize automatically makes the color range symemtrical and selects a cool color map.
dp.show()

# Of course we don't want to go through this laboreous process every time. EMerge has some convenient functions in the MWData
# class that makes creating this data and the plot arguments easier.

dp.add_objects(*m.all_geos())
dp.add_field(field.cutplane(1 * mm, x=0).scalar("Ex", "real"), symmetrize=True)
dp.show()

# The .scalar() method automatrically returns a FieldPlotData object. This contains all the information needed to plot our result.
# We can plug this into the add_field() function which accepts the FieldPlotData objects.
# To animate this we simply add the animate call.

#                                                               v Don't forget to set this to complex!
dp.add_objects(*m.all_geos())
dp.animate().add_field(
    field.cutplane(1 * mm, x=0).scalar("Ex", "complex"), symmetrize=True
)
dp.show()

# Plotting fields on the surface of an object is also possible.
dp.add_objects(*m.all_geos())
dp.add_field(
    field.boundary(abc).scalar("Ex", "abs"), clim=(0, 600)
)  # we don't unpack here because of how this method is designed.
dp.show()

# To ensure that two color bars are shared we can use the .cbar function

dp.add_objects(*m.all_geos())
dp.cbar("normE [V/m]", clim=(0, 4000)).add_field(
    field.cutplane(1 * mm, x=0).scalar("normE"), symmetrize=False
)
dp.cbar("normE [V/m]").add_field(
    field.cutplane(1 * mm, y=0).scalar("normE"), symmetrize=False
)
dp.show()

# We can also add farfield patterns using .add_surf(). All we need to do is create a Farfield dataset using the farfield_3d method.
# This dataset has the .surfplot() method that generates the data needed to create a farfield bulp. In this method you can tune

# isotropic (bool): If the isotropic radiated power should be computed.
# dB (bool): If the data should be computed in dB
# dBfloor (float): Which dB value should be considered 0mm radius. (farfield plots can otherwise go to -infinity dB)
# rmax (float): The radius corresponding to the largest Farfield value
# offset (float, float, float): The x,y,z coordinate at which the blob is centered.

ff3d = field.farfield_3d(abc)

dp.add_objects(*m.all_geos())
dp.add_field(field.cutplane(1 * mm, x=0).scalar("Ex", "real"), symmetrize=True)
dp.add_field(ff3d.surfplot("normE", "abs", True, rmax=20 * mm, offset=(0, 0, 30 * mm)))
dp.show()

# Other useful methods are

griddata = field.grid(2.5 * mm)
dp.add_objects(*m.all_geos())
dp.add_title("Some text here!")
dp.add_text("Instruction here", "blue", "lower_right")
dp.add_portmode(
    my_port, k0=field.k0
)  # k0 is not needed here but it is often with modal-ports.
dp.cbar("H").add_field(griddata.vector("H", "real"))
dp.cbar("E").add_contour(
    *griddata.scalar("Ex", "real").xyzf, symmetrize=True, Nlevels=30
)
dp.cbar("E").add_field(
    field.cutplane(1 * mm, x=0).scalar("Ex", "real"), symmetrize=True
)
dp.cbar("Eff").add_field(
    ff3d.surfplot("normE", "abs", True, rmax=20 * mm, offset=(0, 0, 30 * mm))
)
dp.show()
