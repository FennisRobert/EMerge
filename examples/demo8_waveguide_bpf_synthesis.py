import emerge as em
import numpy as np
from emerge.plot import plot, plot_sp

""" BANDPASS FILTER SYNTHESIS DEMO

This demo synthesizes a 7-section rectangular waveguide bandpass filter
using iris inverters. First, iris scattering parameters are simulated to
extract K-inverter and phase data. Then the full filter geometry is
assembled and S-parameters are plotted. """


# --- Physical constants and units ---------------------------------------
mm = 0.001  # meters per millimeter
dielectric_constant = 1.0  # (not used for air-filled waveguide)
c0 = 299792458  # speed of light in vacuum (m/s)

# --- Waveguide and frequency definitions --------------------------------
wga = 22.86 * mm  # a-dimension of WR-90 waveguide
wgb = 10.16 * mm  # b-dimension (height) of waveguide
# frequency band for final filter response
f1 = 9.6e9  # start frequency (Hz)
f2 = 10e9  # stop frequency (Hz)

# feed line length and iris thickness
Lfeed = 20 * mm  # length of input/output waveguide sections
t_thickness = 1 * mm  # iris plate thickness

# Prototype filter g-coefficients for 7 sections (n+2 values)
gs = [1.000, 1.1811, 1.4228, 2.0966, 1.5733, 2.0966, 1.4228, 1.1811, 1.000]


# --- Helper functions for wave propagation and inverter theory ---------
# Calculate axial propagation constant for TE10 mode at frequency f
def kz(f: float):
    beta = 2 * np.pi * f / c0
    k_c = np.pi / wga  # cutoff wavenumber for TE10
    return np.sqrt(beta**2 - k_c**2)


# Center frequency and fractional bandwidth
f0 = np.sqrt(f1 * f2)
beta0 = kz(f0)
# Fractional electrical length difference between band edges
delta = (2 * np.pi / kz(f1) - 2 * np.pi / kz(f2)) / (2 * np.pi / kz(f0))

# Number of inverter sections (exclude terminations)
N = len(gs) - 2
Kvals = []  # to store theoretical inverter values

# Compute K-inverter coefficients for prototype
for i, (g_prev, g_next) in enumerate(zip(gs[:-1], gs[1:])):
    if i == 0 or i == N:
        # end inverters use square-root formula
        Kvals.append(np.sqrt(np.pi * delta / (2 * g_prev * g_next)))
    else:
        # interior inverters use linear formula
        Kvals.append(np.pi * delta / (2 * np.sqrt(g_prev * g_next)))


# Functions to extract susceptances from scattering parameters
def XsZ0(S11, S12, S21, S22):
    # series susceptance (normalized)
    return np.real(((1 - S12 + S11) / (1 - S11 + S12)) / 1j)


def XpZ0(S11, S12, S21, S22):
    # parallel susceptance (normalized)
    return np.real((2 * S12 / ((1 - S11) ** 2 - S12**2)) / 1j)


def phif(S11, S12, S21, S22):
    # electrical phase shift of inverter
    xs = XsZ0(S11, S12, S21, S22)
    xp = XpZ0(S11, S12, S21, S22)
    return -np.arctan(2 * xp + xs) - np.arctan(xs)


def KZ0(S11, S12, S21, S22):
    # compute actual K/Z0 from phase and susceptance
    return np.abs(
        np.tan(phif(S11, S12, S21, S22) / 2 + np.arctan(XsZ0(S11, S12, S21, S22)))
    )


# --- Iris parameter sweep to characterize inverter -----------------------
# Sweep iris gap from 1 mm to 20 mm and compute K and phi at f0
wgaps = np.linspace(1 * mm, 20 * mm, 21)
Ks = []
hphis = []

with em.Simulation("IrisSim") as model_iris:
    model_iris.check_version("2.8.4")  # Checks version compatibility.

    for wgap in model_iris.parameter_sweep(True, wgap=wgaps):
        # Define two short waveguide sections separated by iris plate
        wg1 = em.geo.Box(wga, Lfeed, wgb, (-wga / 2, -Lfeed - t_thickness / 2, 0))
        iris = em.geo.Box(wgap, t_thickness, wgb, (-wgap / 2, -t_thickness / 2, 0))
        wg2 = em.geo.Box(wga, Lfeed, wgb, (-wga / 2, t_thickness / 2, 0))

        model_iris.commit_geometry()
        model_iris.mw.set_frequency(f0)
        model_iris.mw.set_resolution(0.10)
        model_iris.mesher.set_domain_size(iris, 2 * mm)
        model_iris.generate_mesh()

        # Port BCs on front/back faces for TE10 excitation
        model_iris.mw.bc.RectangularWaveguide(wg1.front, 1)
        model_iris.mw.bc.RectangularWaveguide(wg2.back, 2)

        data = model_iris.mw.run_sweep()

        # Compensate phase from feed line length
        S11 = data.scalar.select(wgap=wgap).S(1, 1) * np.exp(1j * 2 * kz(f0) * Lfeed)
        S12 = data.scalar.select(wgap=wgap).S(1, 2) * np.exp(1j * 2 * kz(f0) * Lfeed)
        S21 = data.scalar.select(wgap=wgap).S(2, 1) * np.exp(1j * 2 * kz(f0) * Lfeed)
        S22 = data.scalar.select(wgap=wgap).S(2, 2) * np.exp(1j * 2 * kz(f0) * Lfeed)

        Ks.append(KZ0(S11, S12, S21, S22))
        hphis.append(phif(S11, S12, S21, S22))

# --- Plot the K-inverter graph ------------------------------------------

Ks = np.array(Ks)
plot(wgaps * 1000, Ks, xlabel="Iris gap[mm]", ylabel="K/Z0")


# --- Interpolate iris widths and electrical lengths ---------------------
Ks = np.array(Ks)
# Map theoretical Kvals to physical iris gaps
interp_widths = [np.interp(K, Ks, wgaps) for K in Kvals]
# Map theoretical phase shifts to physical iris gaps
interp_phis = [np.interp(K, Ks, np.array(hphis)) for K in Kvals]
# Compute required cavity lengths from phase delays
cavity_lengths = (
    1
    / beta0
    * np.array(
        [
            np.pi + 0.5 * (interp_phis[i] + interp_phis[i + 1])
            for i in range(len(Kvals) - 1)
        ]
    )
).real

# --- Build and simulate full filter -------------------------------------
with em.Simulation("FullFilter") as model_final:
    # Input feed section
    feed1 = em.geo.Box(wga, Lfeed, wgb, (-wga / 2, -Lfeed, 0))
    # Create cavities and irises sequentially
    y0 = 0
    cavities = []
    irises = []
    for L, W in zip(cavity_lengths, interp_widths[:-1]):
        ir = em.geo.Box(W, t_thickness, wgb, (-W / 2, y0, 0))
        cav = em.geo.Box(wga, L, wgb, (-wga / 2, y0 + t_thickness, 0))
        y0 += L + t_thickness
        irises.append(ir)
        cavities.append(cav)
    # Last iris and output feed
    last_iris = em.geo.Box(
        interp_widths[-1], t_thickness, wgb, (-interp_widths[-1] / 2, y0, 0)
    )
    feed2 = em.geo.Box(wga, Lfeed, wgb, (-wga / 2, y0 + t_thickness, 0))

    # Define the full filter geometry
    model_final.commit_geometry()

    # Simulation settings and mesh
    model_final.mw.set_frequency_range(f1 - 0.2e9, f2 + 0.2e9, 31)
    model_final.mw.set_resolution(0.10)
    for ir in irises:
        model_final.mesher.set_domain_size(ir, 2 * mm)
    model_final.generate_mesh()

    model_final.view()
    model_final.view(plot_mesh=True)
    # Boundary conditions for feed ports
    p1 = model_final.mw.bc.RectangularWaveguide(feed1.front, 1)
    p2 = model_final.mw.bc.RectangularWaveguide(feed2.back, 2)

    # Run frequency-domain sweep and extract S-parameters
    data = model_final.mw.run_sweep(parallel=True, n_workers=3, frequency_groups=9)
    grid = data.scalar.grid
    freqs = grid.freq
    fdense = np.linspace(freqs[0], freqs[-1], 2001)

    S11 = grid.model_S(1, 1, fdense)
    S21 = grid.model_S(2, 1, fdense)

    # Plot the filter response (dB)

    plot_sp(fdense, [S11, S21], labels=["S11", "S21"])

    # Visualize geometry and mode shapes
    model_final.display.add_object(feed1, opacity=0.1)
    model_final.display.add_object(feed2, opacity=0.1)
    for obj in irises + cavities:
        model_final.display.add_object(obj, opacity=0.1)
    # Show electric field cut-plane at center frequency
    cut = data.field.find(freq=f0).cutplane(1 * mm, z=wgb / 2)
    model_final.display.animate().add_field(
        cut.scalar("Ez", "complex"), symmetrize=True
    )
    model_final.display.add_portmode(p1, k0=data.field.find(freq=f0).k0)
    model_final.display.add_portmode(p2, k0=data.field.find(freq=f0).k0)
    model_final.display.show()
