import emerge as em
from emerge.plot import plot_sp
import numpy as np

""" LUMPED COMPONENT FILTER DEMO

This demo constructs a microstrip filter using discrete lumped inductors and
capacitors mounted on a PCB. We extract the characteristic impedance, place
lumped elements in the trace, mesh the structure, and compute S-parameters.
"""

# --- Unit definitions -----------------------------------------------------
mm = 0.001           # meters per millimeter
pF = 1e-12           # picofarad in farads
fF = 1e-15           # femtofarad in farads
pH = 1e-12           # picohenry in henrys
nH = 1e-9            # nanohenry in henrys

# --- Lumped element impedance functions ---------------------------------
def Lf(L):
    """Return series impedance function for inductor L."""
    return lambda f: 1j * 2 * np.pi * f * L

def Cf(C):
    """Return shunt admittance function for capacitor C."""
    return lambda f: 1 / (1j * 2 * np.pi * f * C)

# --- PCB and lumped-component parameters ---------------------------------
pack = '0603'         # package footprint for lumped components
# Create simulation and PCB layouter with substrate thickness and material
m = em.Simulation('LumpedFilter')
m.check_version("0.6.7") # Checks version compatibility.

th = 0.5         # substrate thickness (meters)
Hair = 2.0
pcb = em.geo.PCB(th, unit=mm, cs=em.GCS,
                          material=em.lib.DIEL_RO4003C, layers=2)
# Compute 50-ohm microstrip width
w0 = pcb.calc.z0(50)

# Define lengths and lumped element values
l0 = 2            # straight segment length (mm)
Lshunt = 35 * nH         # first inductor value
Cshunt = 35 * pF         # first capacitor value
Lseries = 70 * nH         # second inductor
Cseries = 18 * pF         # second capacitor

# --- Route trace with lumped elements via method chaining ----------------
# Input matching section
pcb.new(0, 0, w0, (1, 0)).store('p1').straight(3)\
    .lumped_element(Cf(Cseries), pack).straight(l0)\
    .lumped_element(Lf(Lseries), pack).straight(l0)\
    .split((0, -1)).straight(l0)\
    .lumped_element(Cf(Cshunt), pack).straight(l0 / 2, w0)\
    .via(pcb.z(1), w0 / 6, False).merge()\
    .split((0, 1)).straight(l0)\
    .lumped_element(Lf(Lshunt), pack).straight(l0 / 2, w0)\
    .via(pcb.z(1), w0 / 6, False).merge()\
    .straight(l0).lumped_element(Cf(Cseries), pack).straight(l0)\
    .lumped_element(Lf(Lseries), pack).straight(l0)\
    .straight(3).store('p2')           

# Retrieve lumped element and via objects
LEs = pcb.lumped_elements
vias = pcb.generate_vias(merge=True)
# Compile trace geometry and determine bounds
traces = pcb.compile_paths(merge=True)
pcb.determine_bounds(leftmargin=0, topmargin=1, rightmargin=0, bottommargin=1)

# --- Define modal ports and generate environment ------------------------
mp1 = pcb.modal_port(pcb.load('p1'), Hair)
mp2 = pcb.modal_port(pcb.load('p2'), Hair)
diel = pcb.generate_pcb()                   # substrate dielectric block
air = pcb.generate_air(Hair)                  # surrounding air block

# Add all geometry to simulation
m.commit_geometry()

# --- Solver and mesh settings -------------------------------------------
m.mw.set_frequency_range(0.05e9, 0.3e9, 51)       # 50–300 MHz sweep
m.mesher.set_boundary_size(traces, 0.5 * mm)
# Refine mesh around lumped component faces
for le in LEs:
    m.mesher.set_face_size(le, 0.1 * mm)
# Domain mesh refinement
m.mesher.set_domain_size(diel, 1 * mm)
m.mesher.set_domain_size(air, 3 * mm)

# Build mesh and view
m.generate_mesh()
m.view()

# --- Boundary conditions -----------------------------------------------
# Define modal (TEM) ports at input and output
p1 = m.mw.bc.ModalPort(mp1, 1, TEM=True)
p2 = m.mw.bc.ModalPort(mp2, 2, TEM=True)
# Add lumped element BCs for each element
for le in LEs:
    m.mw.bc.LumpedElement(le)
# Perfect conductor on copper traces and vias
m.mw.bc.PEC(traces)
m.mw.bc.PEC(vias.boundary())

# --- Run frequency-domain simulation ------------------------------------
data = m.mw.run_sweep(parallel=True, njobs=4, frequency_groups=8)

# --- Post-processing: plot S-parameters ---------------------------------
f = data.scalar.grid.freq
S11 = data.scalar.grid.S(1, 1)
S21 = data.scalar.grid.S(2, 1)
plot_sp(f, [S11, S21], xunit='MHz', labels=['S11', 'S21'])

# --- Visualize field distribution ---------------------------------------
m.display.add_object(diel, opacity=0.1)
m.display.add_object(traces, opacity=0.1)
# Cut-plane of Ez field through substrate center
cut = data.field.find(freq=2.25e9)\
    .cutplane(0.1 * mm, z=-th/2 * mm)
m.display.add_surf(*cut.scalar('Ez', 'real'))
m.display.show()
