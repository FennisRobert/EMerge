""" FLOQUET PORTS: A WIRE GRATING AND ITS DIFFRACTION ORDERS

In this demonstration we simulate an infinite grating of metal wires with Floquet ports above and below it.

The field above and below a periodic structure is a sum of Floquet modes, also called diffraction orders.
Order (m, n) is a plane wave whose in-plane wave vector is shifted by m and n times the reciprocal lattice
vectors: along the X-axis with a period P, kx = k0·sin(θ_in) + m·2π/P. An order propagates when |kx| < k0,
at the angle sin(θ_m) = kx/k0, and is evanescent otherwise. Each order comes as a TE and a TM mode.

The wires run along the Y-axis with a period of 15 mm. At normal incidence:
 - below 20 GHz (λ > period) only the specular order (0, 0) propagates. The grating is a wire-grid polarizer:
   E parallel to the wires (TE) is reflected and E perpendicular to the wires (TM) is transmitted.
 - above 20 GHz the (±1, 0) orders propagate as well. Part of the power is diffracted into beams at
   θ = ±asin(λ/period), which move towards the grating plane as the frequency drops.

With number_of_modes=None the Floquet ports include every order that propagates at the highest frequency.
"""
# [Written by Claude]
from dataclasses import replace
import numpy as np
import emerge as em
from emerge.plot import plot, plot_sp, plot_ff

mm = 0.001
period = 15 * mm     # wire period along X
depth = 4 * mm       # cell size along Y, the direction of the wires
radius = 2 * mm      # wire radius
H = 20 * mm          # distance from the wires to each port
C0 = 299792458

model = em.Simulation("WireGrating")
model.check_version("3.0.5")  # Checks version compatibility

# The unit cell is a rectangle of period x depth in the XY-plane
cell = em.RectCell(period, depth)
model.set_periodic_cell(cell)

# The air of the cell from z = -H to z = H and one wire along the Y-axis through the middle. The wire touches
# both Y-faces of the cell, so with the periodic boundaries it is an infinitely long wire.
air = cell.volume(-H, H)
wire = em.geo.Cylinder(radius, depth, em.YAX.construct_cs((0, -depth / 2, 0))).set_material(em.lib.PEC)

# The port faces at the top and the bottom of the cell
top = cell.port_face(H)
bottom = cell.port_face(-H)

model.commit_geometry()

# 5.5 to 29.5 GHz in steps of 1 GHz. We skip exactly 20 GHz: there the (±1, 0) orders graze along the grating (a Rayleigh anomaly).
model.mw.set_frequency_range(5.5e9, 29.5e9, 25)
model.mw.set_resolution(0.15)
model.generate_mesh()
model.view()

# The Floquet ports. With number_of_modes=None the ports include the TE and TM mode of every order that
# propagates at the highest frequency: (0, 0), (-1, 0) and (1, 0), so six modes per port.
port1 = model.mw.bc.floquet_port(top, 1, number_of_modes=None)
port2 = model.mw.bc.floquet_port(bottom, 2, number_of_modes=None)

# Normal incidence
cell.set_scanangle(0, 0)

data = model.mw.run_sweep()
g = data.scalar.grid
freqs = g.freq

# Each port mode has an S-matrix index, a Floquet order (m, n) and a polarization
print("Port modes of port 1:")
for index, m, n, pol in port1.floquet_modes():
    print(f"  S-matrix index {index}: order ({m:+d}, {n:+d}) {pol}")

# A small helper dictionary to find the S-matrix index of an order and polarization at each port
index1 = {(m, n, pol): index for index, m, n, pol in port1.floquet_modes()}
index2 = {(m, n, pol): index for index, m, n, pol in port2.floquet_modes()}

############################################################
#                  THE WIRE-GRID POLARIZER                 #
############################################################

# The incident wave comes from the top port. In the (0, 0) order TE means E along Y (parallel to the wires),
# TM means E along X (perpendicular to the wires).
te_in = index1[(0, 0, "TE")]
tm_in = index1[(0, 0, "TM")]
plot_sp(freqs, [g.S(te_in, te_in), g.S(index2[(0, 0, "TE")], te_in),
                g.S(tm_in, tm_in), g.S(index2[(0, 0, "TM")], tm_in)],
        labels=["Reflection, E ∥ wires", "Transmission, E ∥ wires",
                "Reflection, E ⊥ wires", "Transmission, E ⊥ wires"])

############################################################
#                  THE DIFFRACTION ORDERS                  #
############################################################

# With 1 W port modes |S|² is the fraction of the power that goes into an order. The (-1, 0) and (+1, 0)
# orders carry the same power by symmetry. Below 20 GHz they are evanescent and carry no power.

def efficiency(index_out, index_in):
    return np.abs(g.S(index_out, index_in).ravel()) ** 2

propagating = freqs > C0 / period
reflected_0 = efficiency(index1[(0, 0, "TE")], te_in)
reflected_1 = np.where(propagating, efficiency(index1[(1, 0, "TE")], te_in), 0)
transmitted_0 = efficiency(index2[(0, 0, "TE")], te_in)
transmitted_1 = np.where(propagating, efficiency(index2[(1, 0, "TE")], te_in), 0)

plot(freqs / 1e9, [reflected_0, reflected_1, transmitted_0, transmitted_1],
     labels=["Reflected (0, 0)", "Reflected (+1, 0), each", "Transmitted (0, 0)", "Transmitted (+1, 0), each"],
     xlabel="Frequency (GHz)", ylabel="Fraction of the power, E ∥ wires")

# The direction of the diffracted beams. The diffraction orders are the far field of the grating: an infinite
# grating radiates only in these directions. As the frequency drops to 20 GHz the beams tilt down to 90°.
angles = [port2.order_direction(f, 1, 0) for f in freqs]
theta = np.array([np.nan if a is None else a[0] for a in angles])
plot(freqs / 1e9, [theta], labels=["Order (+1, 0)"], xlabel="Frequency (GHz)", ylabel="Beam angle θ (deg)")

f_show = freqs[np.argmin(np.abs(freqs - 27.5e9))]
i_show = int(np.argmin(np.abs(freqs - f_show)))
for m in (-1, 0, 1):
    direction = port2.order_direction(f_show, m, 0)
    r = efficiency(index1[(m, 0, "TE")], te_in)[i_show]
    t = efficiency(index2[(m, 0, "TE")], te_in)[i_show]
    print(f"  order ({m:+d}, 0) at θ = {direction[0]:5.1f}°: reflected {100 * r:5.1f} %, transmitted {100 * t:5.1f} %")

############################################################
#              A FINITE GRATING: GRATING LOBES             #
############################################################

# A finite grating of N x M cells radiates beams in the same directions as the orders of the infinite one, with
# a width of about λ / (N · period). We predict its far field with array theory: the far field of one cell
# times the array factor of N x M copies of it. The far field of one cell follows from the transmitted field on
# the bottom port face. At normal incidence all cells are excited in phase. The edges of a finite grating are
# not part of this prediction.
N, M = 8, 30  # 8 cells along X and 30 along Y: an aperture of 120 x 120 mm
field = data.field.find(freq=f_show)
field.excite_port(te_in)
k0 = field.k0

def array_factor(ff):
    """The array factor of N x M cells in the directions of the far field ff."""
    ux = np.sin(ff.theta) * np.cos(ff.phi)
    uy = np.sin(ff.theta) * np.sin(ff.phi)
    afx = np.exp(1j * k0 * np.multiply.outer(np.arange(N) * period, ux)).sum(axis=0)
    afy = np.exp(1j * k0 * np.multiply.outer(np.arange(M) * depth, uy)).sum(axis=0)
    return afx * afy


def finite_grating(ff):
    """The far field of N x M cells for a total incident power of 1 W (1 / (N·M) W per cell)."""
    scale = array_factor(ff) / np.sqrt(N * M)
    return replace(ff, _E=ff._E * scale, _H=ff._H * scale)


# The XZ-plane below the grating, from θ = -90° to 90° around the -Z axis
cell_ff = field.farfield_2d((0, 0, -1), (0, 1, 0), bottom, (-90, 90), Npoints=721)
grating_ff = finite_grating(cell_ff)
plot_ff(cell_ff.ang * 180 / np.pi, [cell_ff.gain.norm, grating_ff.gain.norm], dB=True,
        labels=["One cell", f"{N} x {M} cells"], xlabel="Angle from -Z in the XZ-plane (deg)", ylabel="Gain (dBi)",
        ylim=(-20, 35))

# Each beam has the gain of the aperture, 4πA·cos(θ)/λ², times the fraction of the power in its order
area = N * period * M * depth
wavelength = C0 / f_show
print(f"\nGain of the {N} x {M} grating at {f_show / 1e9:.1f} GHz, E ∥ wires:")
for m in (0, 1):
    theta_m = port2.order_direction(f_show, m, 0)[0]
    expected = 4 * np.pi * area * np.cos(np.radians(theta_m)) / wavelength**2 * efficiency(index2[(m, 0, "TE")], te_in)[i_show]
    near = np.abs(np.degrees(cell_ff.ang) - theta_m) < 2
    print(f"  order ({m:+d}, 0) at θ = {theta_m:4.1f}°: {10 * np.log10(np.max(grating_ff.gain.norm[near] ** 2)):5.1f} dBi "
          f"(from the order efficiency: {10 * np.log10(expected):5.1f} dBi)")

# The 3D far field of the lower half space, shown below the cell. The (±1, 0) grating lobes lie in the XZ-plane.
grating_3d = finite_grating(field.farfield_3d(bottom, thetas=np.linspace(np.pi / 2, np.pi, 91),
                                              phis=np.linspace(-np.pi, np.pi, 361)))

############################################################
#                   THE DIFFRACTED FIELD                   #
############################################################

# The field of the incident TE wave at 27.5 GHz in the XZ-plane. Above the wires the incident and reflected
# waves interfere, below the wires the transmitted (0, 0) and (±1, 0) orders do. The far field of the finite
# grating is drawn below the cell.
model.display.add_farfield3d(grating_3d, dB=True, dBfloor=-30, rmax=15 * mm, offset=(0, 0, -H))
model.display.add_object(wire)
model.display.add_object(air, opacity=0.1)
model.display.animate().cbar("Ey").add_field(field.cutplane(0.5 * mm, y=0).scalar("Ey", "complex"), symmetrize=True)
model.display.show()
