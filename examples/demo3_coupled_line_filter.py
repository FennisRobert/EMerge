import emerge as em
from emerge.plot import plot_sp
import numpy as np


""" COUPLED LINE FILTER DEMO

In this demo we construct a coupled-line bandpass filter using the
PCB Layouter interface in EMerge.
"""
# --- Unit definitions -----------------------------------------------------
mm = 0.001  # meter per millimeter
mil = 0.0254 * mm  # meter per mil (thousandth of an inch)

# --- Geometric parameters ------------------------------------------------
W = 20.0  # reference trace width (mil)
D = 20.0  # trace separation (mil)
w0 = 37.0  # input/output line width (mil)
l0 = 100.0  # input/output line length (mil)
l1 = 314.22  # coupled section total length (mil)
l2 = 301.658  # inner coupled section length (mil)
l3 = 300.589  # outer coupled section length (mil)

# widths of each coupled segment (mil)
w1, w2, w3, w4, w5, w6 = 18.8, 43.484, 44.331, 44.331, 43.484, 18.8
# gaps between segments (mil)
g1, g2, g3, g4, g5, g6 = 9.63, 24.84, 41.499, 41.499, 24.84, 9.63

# substrate thickness (mil) and dielectric constant
th = 20.0

# Overlap margin (optinal in mil)
e = 0.0

# overall board dimensions/clearance (for bounding box)
Wtot = 2 * l1 + 5 * l2 + 7 * e + 2 * l0
WP = 200  # wide padding region (mil)
Dtot = 750  # total clearance (mil)
extra = 100  # extra margin (mil)

# --- Simulation setup ----------------------------------------------------
model = em.Simulation("CoupledLineFilter")
model.check_version("2.8.7")  # Checks version compatibility.

# --- Material and layouter -----------------------------------------------
mat = em.Material(er=3.55, color="#488343", opacity=0.4)

# Create PCB layouter with given substrate thickness and units
pcb = em.geo.PCBNew(th, unit=mil, material=mat)

# --- Route coupled-line trace --------------------------------------------
# start at (0,140) with width w0
# label input port
pcb.new(0, 140, w0, (1, 0))["p1"].straight(l0).straight(l1 * 0.8).straight(
    l1, w1, dy=abs(w1 - w0) / 2
).jump(gap=g1, side="left", reverse=l1 - e).straight(l1, w1).straight(
    l2, w2, dy=abs(w2 - w1) / 2
).jump(gap=g2, side="left", reverse=l2 - e).straight(l2, w2).straight(l3, w3).jump(
    gap=g3, side="left", reverse=l2 - e
).straight(l2, w3).straight(l3, w4).jump(gap=g4, side="left", reverse=l2 - e).straight(
    l2, w4
).straight(l2, w5).jump(gap=g5, side="left", reverse=l2 - e).straight(l2, w5).straight(
    l1, w6, dy=abs(w2 - w1) / 2
).jump(gap=g6, side="left", reverse=l1 - e).straight(l1, w6).straight(
    l1 * 0.8, w0, dy=abs(w1 - w0) / 2
).straight(l0, w0)["p2"]  # label output port

# Compile the routed paths into a single GeoSurface
stripline = pcb.compile_paths(merge=True)

# --- Define simulation bounding box --------------------------------------
pcb.determine_bounds(topmargin=150, bottommargin=150)

# --- Generate dielectric and air blocks ----------------------------------
diel = pcb.generate_pcb()  # substrate dielectric block
air = pcb.generate_air(4 * th)  # surrounding air box
# --- Define ports for simulation ----------------------------------------
p1 = pcb.modal_port(pcb["p1"], width_multiplier=5, height=4 * th)
p2 = pcb.modal_port(pcb["p2"], width_multiplier=5, height=4 * th)

# --- Solver settings -----------------------------------------------------
model.mw.set_resolution(0.2)  # mesh density: fraction of wavelength
model.mw.set_frequency_range(5.2e9, 6.2e9, 30)  # 5.2–6.2 GHz, 31 points

# --- Assemble geometry into simulation -----------------------------------
model.commit_geometry()
# --- Mesh refinement -----------------------------------------------------
# High growth rates are generally not adviced but used here to save some memory.
model.mesher.set_boundary_size(stripline, 2 * mm, growth_rate=10)

model.mesher.set_face_size(p1, 1 * mm)
model.mesher.set_face_size(p2, 1 * mm)

# --- Mesh generation and view --------------------------------------------
model.generate_mesh()  # build mesh
model.view()

# --- Boundary conditions ------------------------------------------------
port1 = model.mw.bc.ModalPort(p1, 1, modetype="TEM")
port2 = model.mw.bc.ModalPort(p2, 2, modetype="TEM")

# --- Run frequency-domain solver ----------------------------------------
data = model.mw.run_sweep()

# --- Extract and plot S-parameters ---------------------------------------
f = data.scalar.grid.freq  # frequency axis
S11 = data.scalar.grid.S(1, 1)  # return loss
S21 = data.scalar.grid.S(2, 1)  # insertion loss
plot_sp(f, [S11, S21], labels=["S11", "S21"])

# --- Vector fitting and supersampled plot -------------------------------
f_fit = np.linspace(5.2e9, 6.2e9, 1001)
S11_fit = data.scalar.grid.model_S(1, 1, f_fit)
S21_fit = data.scalar.grid.model_S(2, 1, f_fit)
plot_sp(f_fit, [S11_fit, S21_fit], labels=["S11", "S21"])

# --- 3D field visualization ------------------------------------------------

field = data.field.find(freq=5.7e9)
model.display.add_portmode(port1, k0=field.k0)
model.display.add_portmode(port2, k0=field.k0)
model.display.add_object(diel)
model.display.add_object(stripline)
model.display.animate().add_field(
    field.cutplane(0.5 * mm, z=-0.5 * th * mil).scalar("Ez", "complex"),
    symmetrize=True,
    clim_crop_factor=0.6,
)
model.display.show()
