# -----------------------------------------------------------------------------
# STEP File import: Dielectric Rod Antenna
#
# This example demonstrates how to import a STEP geometry, define materials,
# assign ports and absorbing boundaries, and run a sweep with far-field output.
# Toggle the if-statements to inspect geometry and mesh behavior.
# -----------------------------------------------------------------------------

import emerge as em
from emerge.plot import plot_sp, plot_ff, plot_ff_polar
from pathlib import Path


# Unit helper: 1 mm in meters
mm = 0.001
Lrod = 70 * mm

# Create simulation container
sim = em.Simulation("StepImport")
sim.check_version("2.8.3")

# Set field resolution and frequency sweep
sim.mw.set_resolution(0.25)
sim.mw.set_frequency_range(8e9, 10e9, 21)

# Define custom dielectric material for the rod. We choose something similar to 3D printing resin.
RESIN = em.Material(er=2.5, tand=0.00, color="#777777", opacity=1.0)
# Load STEP file.
# STEPItems groups imported solids. Depending on the configuration of the STEP files, sometimes you have to provide unit=0.001 if the model
# is made in millimeters for example. This does not always go automatically yet.
step_path = Path(__file__).parent / "DielectricRod.step"
step = em.geo.STEPItems("DielRod", str(step_path), unit=1)

# Unpack labeled volumes.
# Order depends on STEP file: adjust if labels differ in your geometry.
flange, air_cutout, dielectric_rod = step.volumes

# Create an enclosing air region with margin around the geometry.
air = step.enclose(0.01)

# Optional:
# The order of geometries might not always be evident. To visualize them, create a mesh
# and set labels=True to see approximately which geometry is which which label. The same order
# of numbering is used as as contained in the volumes property.
if False:  # Set this to True if you want to view this step.
    sim.commit_geometry()
    sim.generate_mesh()
    sim.view(labels=True)

# You can show all face names as well. This becomes quite a mess
if False:
    sim.commit_geometry()
    sim.generate_mesh()
    sim.view(face_labels=True)

# You can also only show the faces of a single geometry
if False:
    sim.commit_geometry()
    sim.generate_mesh()
    sim.view(selections=air_cutout.all_faces())

# Assign materials to metal flange and dielectric rod.
flange.set_material(em.lib.PEC)
dielectric_rod.set_material(RESIN)

# Cut port and rod clearance out of the air region and mark result as background material.
# We need to have the port face be backed by an empty space, not air.
air_fin = em.geo.subtract(air, air_cutout).background()

# Commit final geometry and inspect labeled objects and mesh.
# This will crash if you have the previous if statement turned to True.
sim.commit_geometry()
sim.generate_mesh()
sim.view(labels=True)

# Identify port face. This is the top Z face of the cutout region.
# These faces are automatically named
portface = air_fin.face("+z", tool=air_cutout)

# Collect exterior air faces for absorbing boundary assignment.
abc = air_fin.exterior_faces(air)

# Visualize selected exterior boundary region with reduced opacity.
sim.view(selections=[abc], opacity=0.3)

# Define a rectangular waveguide port on the selected face.
port = sim.mw.bc.RectangularWaveguide(portface, 1)

# Apply absorbing boundary condition to outer air faces.
# The abctype 'D' is better at absorbing energy at oblique angles of incidence.
abc_bc = sim.mw.bc.AbsorbingBoundary(abc, abctype="D")

# Run frequency-domain sweep.
data = sim.mw.run_sweep()

# Access structured S-parameter grid.
glob = data.scalar.grid

# Plot reflection coefficient S11 for the input port.
plot_sp(glob.freq, glob.S(1, 1))

# Compute 2D far-field cuts at 9 GHz for two principal planes.
ff1 = data.field.find(freq=9e9).farfield_2d((0, 0, 1), (1, 0, 0), abc)
ff2 = data.field.find(freq=9e9).farfield_2d((0, 0, 1), (0, 1, 0), abc)

# Plot normalized gain patterns in dBi versus theta.
plot_ff(
    ff1.ang * 180 / 3.1415,
    [ff1.normE / em.lib.EISO, ff2.normE / em.lib.EISO],
    dB=True,
    ylabel="Gain [dBi]",
)

# Polar plot of the same far-field cuts with floor at -20 dB.
plot_ff_polar(
    ff1.ang, [ff1.normE / em.lib.EISO, ff2.normE / em.lib.EISO], dB=True, dBfloor=-20
)

# Compute full 3D far-field pattern on the absorbing boundary surface.
ffdata = data.field.find(freq=9e9).farfield_3d(abc)

# Add geometry for context in final visualization.
sim.display.add_objects(*sim.all_geos(), opacity=0.0)

# Overlay 3D far-field surface and internal field cut for inspection.
sim.display.add_farfield3d(ffdata, rmax=40 * mm, offset=(0, 0, Lrod + 10 * mm))
sim.display.animate().add_field(
    data.field.find(freq=9e9).cutplane(1 * mm, x=0).scalar("Ey", "complex"),
    symmetrize=True,
)

# Launch interactive visualization window.
sim.display.show()
