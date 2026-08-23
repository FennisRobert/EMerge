import emerge as em
from emerge.plot import plot_sp, plot_ff
import numpy as np

""" STANDARD GAIN HORN ANTENNA

This demo sets up and simulates a rectangular horn antenna in an
absorbing domain with PML layers. We compute return loss (S11) over a
90–110 GHz band and plot the far-field radiation pattern. 

The dimensions come from this paper:
https://pure.tue.nl/ws/portalfiles/portal/332971061/Uncertainties_in_the_Estimation_of_the_Gain_of_a_Standard_Gain_Horn_in_the_Frequency_Range_of_90_GHz_to_140_GHz.pdf

Adviced solvers/hardware:
 - This model runs best on UMFPACK, PARDISO, AASDS or MUMPS
 -
"""

# --- Units ---------------------------------------------------------------
mm = 0.001  # meters per millimeter

# --- Horn and feed dimensions -------------------------------------------
wga = 2.01 * mm  # waveguide width
wgb = 1.01 * mm  # waveguide height
WH = 10 * mm  # Aperture width
HH = 7 * mm  # Aperture height
Lhorn = 21 * mm  # Horn length

# --- Feed and simulation setup ------------------------------------------
Lfeed = 2 * mm  # length of feed waveguide
th = 1 * mm  # PML thickness
dx = 2 * mm  # distance from horn exit to PML start

# Create simulation object

model = em.Simulation("StandardGainHornAntenna")
model.check_version("2.8.8")  # Checks version compatibility.

# --- Coordinate system for horn geometry -------------------------------
hornCS = em.CS(em.YAX, em.ZAX, em.XAX)

# Feed waveguide as rectangular box (metal)
feed = em.geo.Box(
    Lfeed,  # length along X
    wga / 2,  # half-width along Y (centered)
    wgb / 2,  # half-height along Z
    position=(-Lfeed, 0, 0),
)
# --- Horn geometry ------------------------------------------------------
# Inner horn taper from (D,E) at throat to (B,C) at mouth over length F
horn_in = em.geo.Horn(
    (wga, wgb),
    (WH, HH),
    Lhorn,
    hornCS,
)
# Outer horn (including metal thickness) helps define PML subtraction
horn_out = em.geo.Horn(
    (wga + 2 * th, wgb + 2 * th), (WH + 2 * th, HH + 2 * th), Lhorn, hornCS
)

# --- Bounding objects and PML -------------------------------------------
# Define large intersection box to trim horn geometry
ibox = em.geo.Box(30 * mm, 30 * mm, 30 * mm)
horn_in = em.geo.intersect(horn_in, ibox, remove_tool=False)
horn_out = em.geo.intersect(horn_out, ibox)

# Create airbox with PML layers on +X, +Y, +Z faces
rat = 1.6  # PML extension ratio
air, *pmls = em.geo.pmlbox(
    4 * mm,  # air padding before PML
    rat * WH / 2,  # half-height in Y
    rat * HH / 2,  # half-width in Z
    (Lhorn - dx, 0, 0),  # PML origin offset along X
    thickness=4 * mm,
    N_mesh_layers=4,
    sides="TRA",  # [T]op, [R]ight, B[A]ck
)
# Subtract horn volume from airbox so PML does not cover metal
air2 = em.geo.subtract(air, horn_out)

# --- Solver parameters --------------------------------------------------
model.mw.set_frequency_range(90e9, 110e9, 11)  # 90–110 GHz sweep
model.mw.set_resolution(0.3)  # mesh resolution fraction

# --- Assemble geometry and mesh -----------------------------------------
model.generate_mesh()

# --- Boundary conditions ------------------------------------------------
p1 = model.mw.bc.ModalPort(feed.left, 1)  # excite TE10 in feed
PMC = model.mw.bc.PMC(
    model.select.face.inplane(0, 0, 0, plane=em.XZPLANE)
)  # perfect magnetic on symmetry
radiation_boundary = air2.faces("back", "top", "right", tool=air)  # open faces
abc = model.mw.bc.AbsorbingBoundary(
    model.select.face.inplane(Lhorn - dx, 0, 0, plane=em.YZPLANE)
)

# View mesh and BC selections
model.view(selections=[radiation_boundary], bc=True)

# --- Run frequency-domain solver ----------------------------------------
data = model.mw.run_sweep()

# --- Plot return loss ---------------------------------------------------
scal = data.scalar.grid
plot_sp(scal.freq, scal.S(1, 1), labels=["S11"])  # S11 vs frequency

# --- Far-field radiation pattern ----------------------------------------
# Compute E and H on 2D cut for phi=0 plane over -90° to 90°
ff_data = data.field[0].farfield_2d(
    (1, 0, 0),
    (0, 1, 0),
    radiation_boundary,
    (-90, 90),
    syms=["Ez", "Hy"],
    origin=(0, 0, 0),
)

plot_ff(ff_data.ang * 180 / np.pi, ff_data.gain.norm, dB=True, ylabel="Gain [dBi]")
# Normalize to free-space impedance and convert to dB

model.display.add_object(horn_in, opacity=0.1)
model.display.add_object(air2, opacity=0.1)
model.display.add_object(feed, opacity=0.1)

ff3d = data.field[0].farfield_3d(
    radiation_boundary, syms=["Ez", "Hy"], origin=(0, 0, 0)
)

model.display.add_farfield3d(
    ff3d, component="gain.norm", dB=True, dBfloor=-30, rmax=5 * mm, offset=(Lhorn, 0, 0)
)
model.display.animate().add_field(
    data.field[0].cutplane(0.1 * mm, z=0).scalar("Ez", "complex"), symmetrize=True
)
model.display.show()
