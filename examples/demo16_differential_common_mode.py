""" DIFFERENTIAL AND COMMON MODE SIMULATIONS

In this simulation we look at a trivial model with two parallel microstripline traces. We show how we can
combine two ports into one to yield common and differential mode ports.

"""

import emerge as em
from emerge.plot import plot_sp

mm = 0.001  # meter units helper: 1 mm

model = em.Simulation('DiffCommon')  # simulation container
model.check_version('3.0.0') # Checks version compatibility.

# PCB stack: 1 mm substrate, RO3003 dielectric, 0.05 mm copper thickness, thick traces enabled
pcb = em.geo.PCBNew(1, mm, material=em.lib.DIEL_RO3003, trace_thickness=0.05*mm, thick_traces=True)

w0 = pcb.calc.z0(50)  # compute trace width for 50 Ω single-ended impedance

# Route two parallel single-ended microstrips. Store endpoints to attach lumped ports.
# Upper line: p1 -> straight -> p3
pcb.new(0, 2, w0, (1,0)).store('p1').straight(20).store('p3')
# Lower line: p2 -> straight -> p4
pcb.new(0, -2, w0, (1,0)).store('p2').straight(20).store('p4')

traces = pcb.compile_paths(merge=True)  # polygonize traces for meshing, merged for convenience

# Create conservative board boundary and air box margins
pcb.determine_bounds(5, 5, 5, 5)

diel = pcb.generate_pcb()    # substrate solid
air = pcb.generate_air(5)    # surrounding air region

# Create four lumped-port terminals at the stored endpoints
lp1 = pcb.lumped_port(pcb.load('p1'), 1)
lp2 = pcb.lumped_port(pcb.load('p2'), 2)
lp3 = pcb.lumped_port(pcb.load('p3'), 3)
lp4 = pcb.lumped_port(pcb.load('p4'), 4)

model.commit_geometry()  # hand geometry to the solver
model.mw.set_frequency_range(2e9, 4e9, 11)  # coarse sweep: 2–4 GHz, 5 points

# Improve mesh around conductor edges
model.mesher.set_boundary_size(traces, 1*mm)

model.generate_mesh()  # build volume mesh

model.view()  # optional mesh preview


data = model.mw.run_sweep(True, 4)  # frequency sweep, parallel workers=4

g = data.scalar.grid  # reshape scalar results to an N-D grid helper

# Plot raw single-ended S-parameters before any port combination
plot_sp(g.freq, [g.S(1,1), g.S(2,1), g.S(3,1), g.S(4,1)], labels=['S11','S21','S31','S41'])

# Combine port pairs (1,2) and (3,4).
# After this, port indices are re-interpreted as modal ports:
#   1: differential of original (1,2)
#   2: common of original (1,2)
#   3: differential of original (3,4)
#   4: common of original (3,4)
g.combine_ports(1,2)
g.combine_ports(3,4)

# Plot S-parameters of the new modal ports. Useful to see DM/CM isolation and balance.
plot_sp(g.freq, [g.S(1,1), g.S(2,1), g.S(3,1), g.S(4,1)], labels=['S11','S21','S31','S41'])

field = data.field.find(freq=3e9)  # pick a field solution at 3 GHz

# Visualize single-ended excitations first: excite each original port, show Ez cutplane
for i, text in enumerate(['Port 1','Port 2','Port 3','Port 4']):
    field.excite_port(i+1)  # single-ended excitation i+1
    model.display.add_objects(*model.all_geos())
    model.display.animate().add_field(field.cutplane(1*mm, z=-0.7*mm).scalar('Ez','real'), symmetrize=True)
    model.display.add_title(text)
    model.display.show()

# Combine field ports into modal pairs as above.
# After this, field.excite_port(1) gives differential of (1,2), 2 gives common of (1,2), etc.
field.combine_ports(1,2)
field.combine_ports(3,4)

# Visualize modal excitations: DM and CM field patterns for both ends
for i, text in enumerate(['Port 1 differential mode','Port 1 common mode','Port 2 differential mode','Port 2 common mode']):
    field.excite_port(i+1)  # modal excitation i+1 after combination
    model.display.add_objects(*model.all_geos())
    model.display.animate().add_field(field.cutplane(1*mm, z=-0.7*mm).scalar('Ez','real'), symmetrize=True)
    model.display.add_title(text)
    model.display.show()