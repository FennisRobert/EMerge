import emerge as em
from emerge.plot import plot_sp
import numpy as np

""" COUPLED LINE FILTER DEMO

In this demo we construct a coupled-line bandpass filter using the
PCB Layouter interface in EMerge.
"""
# --- Unit definitions -----------------------------------------------------
mm = 0.001               # meter per millimeter
mil = 0.0254 * mm        # meter per mil (thousandth of an inch)

# --- Geometric parameters ------------------------------------------------
W = 20                   # reference trace width (mil)
D = 20                   # trace separation (mil)
w0 = 37                 # input/output line width (mil)
l0 = 100                # input/output line length (mil)
l1 = 314.22             # coupled section total length (mil)
l2 = 301.658            # inner coupled section length (mil)
l3 = 300.589            # outer coupled section length (mil)

# widths of each coupled segment (mil)
w1, w2, w3, w4, w5, w6 = 18.8, 43.484, 44.331, 44.331, 43.484, 18.8
# gaps between segments (mil)
g1, g2, g3, g4, g5, g6 = 9.63, 24.84, 41.499, 41.499, 24.84, 9.63

# substrate thickness (mil) and dielectric constant
th = 20
# dielectric constant of substrate (relative)
e = 0

# overall board dimensions/clearance (for bounding box)
Wtot = 2 * l1 + 5 * l2 + 7 * e + 2 * l0
WP = 200                # wide padding region (mil)
Dtot = 750              # total clearance (mil)
extra = 100             # extra margin (mil)

# --- Simulation setup ----------------------------------------------------
model = em.Simulation('Demo3')
model.check_version("0.6.7") # Checks version compatibility.
# --- Material and layouter -----------------------------------------------
mat = em.Material(er=3.55, color="#488343", opacity=0.1)
# Create PCB layouter with given substrate thickness and units
pcb = em.geo.PCB(th, unit=mil)

# --- Route coupled-line trace --------------------------------------------
# start at (0,140) with width w0
# label input port
pcb.new(0, 140, w0, (1, 0)) \
    .store('p1') \
    .straight(l0) \
    .turn(0) \
    .straight(l1 * 0.8) \
    .straight(l1, w1, dy=abs(w1 - w0) / 2) \
    .jump(gap=g1, side='left', reverse=l1 - e) \
    .straight(l1, w1) \
    .straight(l2, w2, dy=abs(w2 - w1) / 2) \
    .jump(gap=g2, side='left', reverse=l2 - e) \
    .straight(l2, w2) \
    .straight(l3, w3) \
    .jump(gap=g3, side='left', reverse=l2 - e) \
    .straight(l2, w3) \
    .straight(l3, w4) \
    .jump(gap=g4, side='left', reverse=l2 - e) \
    .straight(l2, w4) \
    .straight(l2, w5) \
    .jump(gap=g5, side='left', reverse=l2 - e) \
    .straight(l2, w5) \
    .straight(l1, w6, dy=abs(w2 - w1) / 2) \
    .jump(gap=g6, side='left', reverse=l1 - e) \
    .straight(l1, w6) \
    .turn(0) \
    .straight(l1 * 0.8, w0) \
    .straight(l0, w0, dy=abs(w1 - w0) / 2) \
    .store('p2')                        # label output port

# Compile the routed paths into a single GeoSurface
stripline = pcb.compile_paths(merge=True)

# --- Define simulation bounding box --------------------------------------
pcb.determine_bounds(topmargin=150, bottommargin=150)

# --- Generate dielectric and air blocks ----------------------------------
diel = pcb.generate_pcb()                     # substrate dielectric block
air = pcb.generate_air(4 * th)               # surrounding air box
# assign our custom material to the dielectric block
diel.material = mat

# --- Define ports for simulation ----------------------------------------
p1 = pcb.modal_port(pcb.load('p1'), width_multiplier=5, height=4 * th)
p2 = pcb.modal_port(pcb.load('p2'), width_multiplier=5, height=4 * th)

# --- Solver settings -----------------------------------------------------
model.mw.set_resolution(0.2)            # mesh density: fraction of wavelength
model.mw.set_frequency_range(5.2e9, 6.2e9, 31)  # 5.2–6.2 GHz, 31 points

# --- Assemble geometry into simulation -----------------------------------
model.commit_geometry(stripline, diel, p1, p2, air)

# --- Mesh refinement -----------------------------------------------------
model.mesher.set_boundary_size(stripline, 0.5 * mm)

# --- Mesh generation and view --------------------------------------------
model.generate_mesh()                    # build mesh
model.view(use_gmsh=True)                # visualize with Gmsh viewer

# --- Boundary conditions ------------------------------------------------
port1 = model.mw.bc.ModalPort(p1, 1, TEM=True)
port2 = model.mw.bc.ModalPort(p2, 2, TEM=True)
pec = model.mw.bc.PEC(stripline)

# --- Display setup -------------------------------------------------------
d = model.display
# show dielectric block translucent green and stripline in red
d.add_object(diel, color='green', opacity=0.5)
d.add_object(stripline)
d.show()

# --- Run frequency-domain solver ----------------------------------------
data = model.mw.run_sweep(parallel=True, njobs=4, frequency_groups=8)

# --- Extract and plot S-parameters ---------------------------------------
f = data.scalar.grid.freq                  # frequency axis
S11 = data.scalar.grid.S(1, 1)             # return loss
S21 = data.scalar.grid.S(2, 1)             # insertion loss
plot_sp(f, [S11, S21], labels=['S11', 'S21'])

# --- Vector fitting and supersampled plot -------------------------------
f_fit = np.linspace(5.2e9, 6.2e9, 1001)
S11_fit = data.scalar.grid.model_S(1, 1, f_fit)
S21_fit = data.scalar.grid.model_S(2, 1, f_fit)
plot_sp(f_fit, [S11_fit, S21_fit], labels=['S11', 'S21'])