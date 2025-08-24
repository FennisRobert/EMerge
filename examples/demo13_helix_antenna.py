# -----------------------------------------------------------------------------
# HELIX ANTENNA DEMO — Simulation setup, mesh, sweep, and far-field plots
#
# This script builds a helical radiator with a short feed, places it in an
# airbox, applies a lumped port and absorbing boundary, runs a frequency sweep,
# and visualizes S-parameters and radiation patterns (2D cuts + 3D).
# -----------------------------------------------------------------------------

import emerge as em
from emerge.plot import plot_sp, plot_ff, smith
import numpy as np

# --- Units & basic constants -------------------------------------------------
mm = 0.001                      # meters per millimeter

f0 = 3e9                        # center frequency (Hz)
c0 = 299792458                  # speed of light (m/s)
wl = c0/f0                      # free-space wavelength at f0 (m)
rad0 = wl/(2*3.1415)            # reference radius ~= λ/(2π)
radw = 1*mm                     # wire/pipe radius for the helix

L = 4*rad0                      # helix axial length

porth = 2*mm                    # vertical height of the feed extrusion

# --- Simulation object -------------------------------------------------------
model = em.Simulation('helix')
model.check_version("0.6.7") # Checks version compatibility.

dfeed = 3*mm                    # straight feed length before the helix starts

# --- Geometry: helix curve and metal pipe -----------------------------------
# Helix curve from (0,0,porth+dfeed/2) to (0,0,porth+dfeed/2+L)
# with initial radius rad0, 13 degree pitch, tapered end radius r_end
# The 'startfeed' parameter is radius like parameter to turn the start of the helix downward.
#
# This is illustrated in the diagrom below. Normally a helix would start at x1, y1.ArithmeticError
# If some start startfeed distance is provided, the feed will curve downwards adding an extra distance
# y1-y0 equal to startfeed/2. You can access the coordinate x0,0 usinging curve_obj.p0
#
#             ______
#           /       `-- (x1,y1)  --> helix
#         /                
#       /                  
#      |                   
#   (x0,y1)               

# We add the porth height + half of the startfeed distance to the total height to put the helix in the right spot.
h_curve = em.geo.Curve.helix_lh((0,0,porth+dfeed/2), (0,0,porth+dfeed/2+L), rad0, 13, r_end=0.8*rad0)

cross_section = em.geo.XYPolygon.circle(radw, Nsections=6)

# Sweep a circular cross-section along the curve to make a metallic pipe
helix = h_curve.pipe(cross_section).set_material(em.lib.MET_COPPER)

# We add a block to make attachment of ports easier.
block = em.geo.Box(dfeed, dfeed, dfeed, position=h_curve.p0, alignment=em.geo.Alignment.CENTER).set_material(em.lib.MET_COPPER)
# Optional preview of current scene (geometry only at this point)
model.view()

# --- Meshing preferences on helix -------------------------------------------
helix.max_meshsize = 3*radw     # set a courser mesh size for the helix

# --- Feed geometry: short vertical extrusion at the base ---------------------
x0, y0, z0 = h_curve.p0         # helix start point x and y coordinate
feed_poly = em.geo.XYPolygon.circle(radw, Nsections=6)
feed = feed_poly.extrude(porth, em.GCS.displace(x0, y0, 0))  # vertical feed stub

# --- Background domain (airbox) & frequency sweep ---------------------------
# Airbox centered around the helix with some clearance in X/Y and height in Z
airbox = em.geo.Box(4*rad0, 4*rad0, 1.5*L, (-2*rad0, -2*rad0, 0)).background()

# Sweep across 2.8–3.4 GHz (11 points) to cover the operating band
model.mw.set_frequency_range(2.8e9, 3.4e9, 11)

# --- Mesh generation & preview ----------------------------------------------
model.generate_mesh()
model.view()

# --- Boundary selections for BCs & port -------------------------------------
abc_sel = airbox.boundary(exclude=('bottom',))     # absorbing boundary applied at bottom face of airbox
port_sel = feed.boundary(exclude=('front','back'))# port faces on the feed (two opposite faces)

# --- Boundary conditions -----------------------------------------------------
abc = model.mw.bc.AbsorbingBoundary(abc_sel)                     # open-space termination
port_sel = model.mw.bc.LumpedPort(port_sel, 1, feed_poly.length, porth, em.ZAX ,Z0=130)  # lumped port (Z-axis)

# --- Solve frequency sweep ---------------------------------------------------
data = model.mw.run_sweep()

# --- S-parameters: access and plots -----------------------------------------
glob = data.scalar.grid
plot_sp(glob.freq, glob.S(1,1))         # return loss vs frequency
smith(glob.S(1,1), f=glob.freq)         # Smith chart of S11

# --- Far-field 2D cuts (two principal planes) --------------------------------
# Cut 1: elevation sweep with phi defined by basis (0,0,1) and (0,1,0)
ff1 = data.field.find(freq=3.1e9).farfield_2d((0,0,1), (0,1,0), abc_sel, (-90, 90))
# Cut 2: elevation sweep with orthogonal basis (0,0,1) and (1,0,0)
ff2 = data.field.find(freq=3.1e9).farfield_2d((0,0,1), (1,0,0), abc_sel, (-90, 90))

# Normalize circular components to isotropic reference for dB patterns
Eiso = em.lib.EISO
plot_ff(
    ff1.ang*180/np.pi,
    [ff1.Elhcp/Eiso, ff1.Erhcp/Eiso, ff2.Elhcp/Eiso, ff2.Erhcp/Eiso],
    dB=True,
    labels=['LHCP 1','RHCP 1','LHCP 2','RHCP 2']
)

# --- 3D far-field visualization ---------------------------------------------
# Add geometry for context
model.display.add_object(helix)
model.display.add_object(airbox)

# Compute full 3D far-field (at the same frequency) and display |Erhcp|
model.display.add_surf(*data.field.find(freq=3.1e9).farfield_3d(abc_sel).surfplot('Elhcp','abs',rmax=L/2, isotropic=True, offset=(0,0,L)))

# Show interactive 3D scene
model.display.show()
