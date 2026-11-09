# [Written by Claude]
"""Conductor loss of a ThinConductor stripline in air against Wheeler's incremental inductance rule.

The trace is a zero-thickness copper sheet centred between PEC ground planes, so only the strip is lossy.
Wheeler's formulas describe a strip of the real copper thickness. The edge current of a zero-thickness
sheet is singular, so the simulated loss depends on the edge mesh size. For exactly this geometry and
mesh (domain size 1 mm, trace boundary size 0.5 mm) the theory loss is 0.81 times the simulated loss.
This is a regression test of the ThinConductor loss for that fixed setup, not a general validation.
"""
import numpy as np

ETA0 = 376.730313
MU0 = 4e-7 * np.pi
CORRECTION_FACTOR = 0.81  # theory loss / simulated loss for this geometry and mesh


def z0_stripline(w: float, b: float, t: float) -> float:
    """Characteristic impedance of a centred stripline in air (Wheeler 1978, about 0.5 % accurate)."""
    x = t / b
    m = 2 / (1 + 2 / 3 * x / (1 - x))
    dw = (b - t) * x / (np.pi * (1 - x)) * (
        1 - 0.5 * np.log((x / (2 - x)) ** 2 + (0.0796 * x / (w / b + 1.1 * x)) ** m)
    )
    r = (b - t) / (w + dw)
    return ETA0 / (4 * np.pi) * np.log(1 + 4 / np.pi * r * (8 / np.pi * r + np.sqrt((8 / np.pi * r) ** 2 + 6.27)))


def strip_conductor_loss(f: np.ndarray, w: float, b: float, t: float, sigma: float) -> np.ndarray:
    """Conductor attenuation (Np/m) of the strip only (Wheeler's incremental inductance rule)."""
    rs = np.sqrt(np.pi * f * MU0 / sigma)
    z0 = z0_stripline(w, b, t)
    dn = 1e-9
    dz_dn = (z0_stripline(w - 2 * dn, b, t - 2 * dn) - z0) / dn
    return rs / (2 * ETA0 * z0) * dz_dn


def test_thin_conductor_stripline_loss():
    import emerge as em

    mm = 0.001
    th, w0, ds, L = 1.0 * mm, 1.3 * mm, 3 * mm, 100 * mm

    em.cleanup()
    sim = em.Simulation("ThinConductorLoss", loglevel="ERROR", reset_settings=True)
    pcb = em.geo.PCB(th, unit=1.0, layers=3, thick_traces=False, trace_material=em.lib.COPPER)
    pcb.new(0, 0, w0, (1, 0), 1)[0].straight(L)[1]
    trace = pcb.compile_paths(True)
    pcb.modal_port(0, 1, 0, width=2 * ds)
    pcb.modal_port(1, 2, 0, width=2 * ds)
    pcb.determine_bounds(0, ds, 0, ds)
    diels = pcb.generate_pcb(True)
    sim.commit_geometry()

    sim.mw.set_frequency_range(1e9, 10e9, 3)
    sim.mesher.set_boundary_size(trace, 0.5 * mm)
    sim.mesher.set_domain_size(diels, 1 * mm)
    sim.generate_mesh()
    data = sim.mw.run_sweep()

    g = data.scalar.grid
    p_loss_sim = 1 - np.abs(g.S(1, 1)) ** 2 - np.abs(g.S(2, 1)) ** 2

    alpha = strip_conductor_loss(g.freq, w0, th, pcb.trace_thickness, em.lib.COPPER.cond.scalar(1e9))
    p_loss_theory = (1 - np.exp(-2 * alpha * L)) / CORRECTION_FACTOR

    rel_error = np.abs(p_loss_sim / p_loss_theory - 1)
    assert np.all(rel_error < 0.05), dict(freq=g.freq, sim=p_loss_sim, theory=p_loss_theory, rel_error=rel_error)
    em.cleanup()


# [Written by Claude]
def test_thin_conductor_sheet_resistance():
    """A ThinConductor strip much thinner than the skin depth has the DC resistance of its thickness (issue #95).

    A 20 x 100 µm strip with σt = 1 S (Rs = 1 Ω/sq, R = 5 Ω), 10 µm over a PEC ground, fed by a lumped port and
    shorted at the far end. At 10 MHz the skin depth (~100 µm) is far larger than t = 0.2 µm. Both sides of the
    sheet carry the surface impedance, so each side must be 2/(σt) for the strip to have Re Z = 5 Ω.
    """
    import emerge as em

    um, S = 1e-6, 1000.0
    gu = um * S
    metal = em.Material(cond=5e6, er=1, name="metal", _metal=True)

    em.cleanup()
    m = em.Simulation("ThinConductorSheet", loglevel="ERROR", reset_settings=True)
    m.settings.mw_3d_surfimplim = 3.9e7
    m.mw.set_basis_space(element_order=1, complete=True)
    m.set_scale_factor(S)

    def plate(o, a, b, name):
        return em.geo.Plate(tuple(v * gu for v in o), tuple(v * gu for v in a), tuple(v * gu for v in b), name=name)

    air = em.geo.Box(220 * gu, 300 * gu, 110 * gu, position=(-100 * gu, -100 * gu, -50 * gu), name="Air")
    air.set_material(em.lib.AIR).background()
    plate((-40, -40, 0), (100, 0, 0), (0, 180, 0), "gnd").set_material(em.lib.PEC)
    plate((0, 100, 0), (20, 0, 0), (0, 0, 10), "ret").set_material(em.lib.PEC)
    port = plate((0, 0, 0), (20, 0, 0), (0, 0, 10), "port")
    strip = plate((0, 0, 10), (20, 0, 0), (0, 100, 0), "strip")
    m.mw.set_frequency([1e7])
    m.commit_geometry()
    m.mesher.max_size = 30 * um
    m.mesher.min_size = 0.05 * um
    m.mesher.set_face_size(strip, 4 * gu)
    m.generate_mesh()
    m.mw.bc.LumpedPort(port, 1, width=20 * um, height=10 * um, direction=em.ZAX, Z0=5.0)
    m.mw.bc.ThinConductor(strip, metal, thickness=0.2 * um)
    m.mw.bc.AbsorbingBoundary(air.boundary())
    s = np.asarray(m.mw.run_sweep().scalar.grid.S(1, 1)).ravel()[0]
    z = 5.0 * (1 + s) / (1 - s)
    assert abs(z.real - 5.0) < 0.05, z
    em.cleanup()
