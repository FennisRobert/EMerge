import emerge as em
import numpy as np
import matplotlib.pyplot as plt

""" BANDPASS FILTER SYNTHESIS DEMO

This demo synthesizes a 7-section rectangular waveguide bandpass filter
using iris inverters. First, iris scattering parameters are simulated to
extract K-inverter and phase data. Then the full filter geometry is
assembled and S-parameters are plotted. """


# --- Physical constants and units ---------------------------------------
mm = 0.001                # meters per millimeter
dielectric_constant = 1.0 # (not used for air-filled waveguide)
c0 = 299792458            # speed of light in vacuum (m/s)

# --- Waveguide and frequency definitions --------------------------------
wga = 22.86 * mm          # a-dimension of WR-90 waveguide
wgb = 10.16 * mm              # b-dimension (height) of waveguide
# frequency band for final filter response
f1 = 9.6e9                # start frequency (Hz)
f2 = 10e9                # stop frequency (Hz)

# feed line length and iris thickness
Lfeed = 20 * mm           # length of input/output waveguide sections
t_thickness = 1 * mm      # iris plate thickness

# Prototype filter g-coefficients for 7 sections (n+2 values)
gs = [1.000, 1.1811, 1.4228, 2.0966, 1.5733, 2.0966, 1.4228, 1.1811, 1.000]

# --- Helper functions for wave propagation and inverter theory ---------
# Calculate axial propagation constant for TE10 mode at frequency f
def kz(f: float):
    beta = 2 * np.pi * f / c0
    k_c = np.pi / wga       # cutoff wavenumber for TE10
    return np.sqrt(beta**2 - k_c**2)

# Center frequency and fractional bandwidth
f0 = np.sqrt(f1 * f2)
beta0 = kz(f0)
# Fractional electrical length difference between band edges
delta = (2*np.pi/kz(f1) - 2*np.pi/kz(f2)) / (2*np.pi/kz(f0))

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
    return np.real((2 * S12 / ((1 - S11)**2 - S12**2)) / 1j)

def phif(S11, S12, S21, S22):
    # electrical phase shift of inverter
    xs = XsZ0(S11, S12, S21, S22)
    xp = XpZ0(S11, S12, S21, S22)
    return -np.arctan(2*xp + xs) - np.arctan(xs)

def KZ0(S11, S12, S21, S22):
    # compute actual K/Z0 from phase and susceptance
    return np.abs(np.tan(phif(S11,S12,S21,S22)/2 + np.arctan(XsZ0(S11,S12,S21,S22))))

# --- Iris parameter sweep to characterize inverter -----------------------
# Sweep iris gap from 1 mm to 20 mm and compute K and phi at f0
wgaps = np.linspace(1*mm, 20*mm, 21)
Ks = []
hphis = []
with em.Simulation('IrisSim') as sim:
    sim.check_version("0.6.7") # Checks version compatibility.
    for (wgap,) in sim.parameter_sweep(True, wgap=wgaps):
        # Define two short waveguide sections separated by iris plate
        wg1 = em.geo.Box(wga, Lfeed, wgb, (-wga/2, -Lfeed - t_thickness/2, 0))
        iris = em.geo.Box(wgap, t_thickness, wgb, (-wgap/2, -t_thickness/2, 0))
        wg2 = em.geo.Box(wga, Lfeed, wgb, (-wga/2, t_thickness/2, 0))

        sim.commit_geometry()
        sim.mw.set_frequency(f0)
        sim.mw.set_resolution(0.10)
        sim.mesher.set_domain_size(iris, 2*mm)
        sim.generate_mesh()

        # Port BCs on front/back faces for TE10 excitation
        sim.mw.bc.RectangularWaveguide(wg1.face('front'), 1)
        sim.mw.bc.RectangularWaveguide(wg2.face('back'), 2)

        data = sim.mw.run_sweep()
        # Compensate phase from feed line length
        S11 = data.scalar.select(wgap=wgap).S(1,1) * np.exp(1j*2*kz(f0)*Lfeed)
        S12 = data.scalar.select(wgap=wgap).S(1,2) * np.exp(1j*2*kz(f0)*Lfeed)
        S21 = data.scalar.select(wgap=wgap).S(2,1) * np.exp(1j*2*kz(f0)*Lfeed)
        S22 = data.scalar.select(wgap=wgap).S(2,2) * np.exp(1j*2*kz(f0)*Lfeed)

        Ks.append(KZ0(S11, S12, S21, S22))
        hphis.append(phif(S11, S12, S21, S22))

# --- Plot the K-inverter graph ------------------------------------------

fig, ax = plt.subplots()
ax.plot(wgaps*1000, Ks)
ax.grid(True)
ax.set_xlabel('Iris gap [mm]')
ax.set_ylabel('K/Z0')
plt.show()

# --- Interpolate iris widths and electrical lengths ---------------------
Ks = np.array(Ks)
# Map theoretical Kvals to physical iris gaps
interp_widths = [np.interp(K, Ks, wgaps) for K in Kvals]
# Map theoretical phase shifts to physical iris gaps
interp_phis = [np.interp(K, Ks, np.array(hphis)) for K in Kvals]
# Compute required cavity lengths from phase delays
cavity_lengths = (1/beta0 * np.array([
    np.pi + 0.5*(interp_phis[i] + interp_phis[i+1])
    for i in range(len(Kvals)-1)
])).real

# --- Build and simulate full filter -------------------------------------
with em.Simulation('FullFilter') as mf:
    # Input feed section
    feed1 = em.geo.Box(wga, Lfeed, wgb, (-wga/2, -Lfeed, 0))
    # Create cavities and irises sequentially
    y0 = 0
    cavities = []
    irises = []
    for L, W in zip(cavity_lengths, interp_widths[:-1]):
        ir = em.geo.Box(W, t_thickness, wgb, (-W/2, y0, 0))
        cav = em.geo.Box(wga, L, wgb, (-wga/2, y0 + t_thickness, 0))
        y0 += L + t_thickness
        irises.append(ir)
        cavities.append(cav)
    # Last iris and output feed
    last_iris = em.geo.Box(interp_widths[-1], t_thickness, wgb, (-interp_widths[-1]/2, y0, 0))
    feed2 = em.geo.Box(wga, Lfeed, wgb, (-wga/2, y0 + t_thickness, 0))

    # Define the full filter geometry
    mf.commit_geometry(feed1, feed2, last_iris, *(cavities + irises))

    # Simulation settings and mesh
    mf.mw.set_frequency_range(f1 - 0.2e9, f2 + 0.2e9, 51)
    mf.mw.set_resolution(0.10)
    for ir in irises:
        mf.mesher.set_domain_size(ir, 2*mm)
    mf.generate_mesh()

    # Boundary conditions for feed ports
    p1 = mf.mw.bc.RectangularWaveguide(feed1.face('front'), 1)
    p2 = mf.mw.bc.RectangularWaveguide(feed2.face('back'), 2)

    # Run frequency-domain sweep and extract S-parameters
    data = mf.mw.run_sweep(parallel=True, njobs=3, frequency_groups=8)
    grid = data.scalar.grid
    freqs = grid.freq
    fdense = np.linspace(freqs[0], freqs[-1], 2001)

    S11 = grid.model_S(1,1,fdense)
    S21 = grid.model_S(2,1,fdense)

    # Plot the filter response (dB)
    fig, ax = plt.subplots()
    ax.plot(fdense/1e9, 20*np.log10(np.abs(S11)), label='S11')
    ax.plot(fdense/1e9, 20*np.log10(np.abs(S21)), label='S21')
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('S-parameter [dB]')
    ax.grid(True)
    ax.legend()
    plt.show()

    # Visualize geometry and mode shapes
    mf.display.add_object(feed1, opacity=0.1)
    mf.display.add_object(feed2, opacity=0.1)
    for obj in irises + cavities:
        mf.display.add_object(obj, opacity=0.1)
    # Show electric field cut-plane at center frequency
    cut = data.field.find(freq=f0).cutplane(2*mm, z=wgb/2)
    mf.display.add_surf(*cut.scalar('Ez','real'))
    mf.display.add_portmode(p1)
    mf.display.add_portmode(p2)
    mf.display.show()
