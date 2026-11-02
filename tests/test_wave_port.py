# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np

import emerge as em

""" WavePort: a fixed set of modes over frequency, evanescent modes, and the ModalPort β extrapolation."""

mm = 0.001
C0 = 299792458.0


def _guide(a, b, L, freqs, material=None, resolution=0.15):
    em.cleanup()
    sim = em.Simulation("guide", loglevel="ERROR")
    box = em.geo.Box(a, b, L)
    if material is not None:
        box.set_material(material)
    sim.commit_geometry()
    sim.mw.set_frequency(freqs)
    sim.mw.set_resolution(resolution)
    sim.generate_mesh()
    return sim, box


def test_modal_port_dielectric_filled_waveguide():
    # The modes of a ModalPort without mixed_materials were computed at one frequency and extrapolated with
    # β² = β0² + k0² - k00², without εr: a uniform dielectric-filled guide reflected up to |S11| = 0.4.
    a, b, L, er = 22.86 * mm, 10.16 * mm, 40 * mm, 2.2
    freqs = np.array([6e9, 9e9, 12e9])
    sim, box = _guide(a, b, L, freqs, em.Material(er=er, name="diel"))
    sim.mw.bc.ModalPort(box.face("-z"), 1)
    sim.mw.bc.ModalPort(box.face("+z"), 2)
    g = sim.mw.run_sweep().scalar.grid
    em.cleanup()
    k0 = 2 * np.pi * freqs / C0
    beta = np.sqrt(er * k0**2 - (np.pi / a) ** 2)
    assert np.abs(g.S(1, 1).ravel()).max() < 0.02
    dphase = np.angle(g.S(2, 1).ravel() * np.exp(1j * beta * L), deg=True)
    assert np.abs(dphase).max() < 2, dphase


def test_wave_port_evanescent_modes():
    # 10 x 10.16 mm guide: TE01 (14.76 GHz) and TE10 (14.99 GHz) are the two modes with the largest β at 18 GHz.
    # At 12 GHz both are evanescent: they stay in the port (β = -jα), are absorbed without reflection and
    # arrive at the other port attenuated by exp(-αL).
    a, b, L = 10 * mm, 10.16 * mm, 15 * mm
    freqs = np.array([12e9, 14e9, 16e9, 18e9])
    sim, box = _guide(a, b, L, freqs)
    p1 = sim.mw.bc.WavePort(box.face("-z"), 1, number_of_modes=2)
    sim.mw.bc.WavePort(box.face("+z"), 2, number_of_modes=2)
    data = sim.mw.run_sweep()
    Sp = data.scalar.grid.Sp
    for fi, fr in enumerate(freqs):
        k0 = 2 * np.pi * fr / C0
        betas = [complex(p1.get_beta(k0, i)) for i in (1, 2)]
        exact = [np.sqrt(complex(k0**2 - kc2)) for kc2 in ((np.pi / b) ** 2, (np.pi / a) ** 2)]
        exact = [np.conj(e) if e.imag > 0 else e for e in exact]
        np.testing.assert_allclose(betas, exact, rtol=2e-2, atol=2.0)
        S = Sp[fi]  # indices 1.1, 1.2, 2.1, 2.2
        assert np.abs(S[:2, :2]).max() < 0.03, S
        for m, beta in enumerate(betas):
            assert abs(abs(S[2 + m, m]) - abs(np.exp(-1j * beta * L))) < 0.02, (fr, m, S[2 + m, m], beta)
    em.cleanup()


def test_wave_port_all_propagating_modes():
    # 40 x 15 mm guide at 9 and 9.5 GHz: TE10 (3.75 GHz) and TE20 (7.49 GHz) propagate, TE01 (10 GHz) does not.
    sim, box = _guide(40 * mm, 15 * mm, 20 * mm, np.array([9e9, 9.5e9]), resolution=0.2)
    p1 = sim.mw.bc.WavePort(box.face("-z"), 1, number_of_modes=None)
    sim.mw.bc.WavePort(box.face("+z"), 2, number_of_modes=None)
    S = sim.mw.run_sweep().scalar.grid.Sp
    em.cleanup()
    assert p1._desired_number_of_modes == 2
    assert S.shape[1:] == (4, 4)
    for Sf in S:
        assert np.abs(Sf.conj().T @ Sf - np.eye(4)).max() < 1e-2


def test_port_mode_flip_polarity():
    # E and H of a port mode interpolate the same coefficient array; flip_polarity flipped it twice.
    from emerge._emerge.elements.nedleg2 import FieldFunctionClass
    from emerge._emerge.physics.microwave.bcs.portmode import PortMode

    field = np.ones(3, dtype=complex)
    E = FieldFunctionClass(field, None, None, None, None, "E")
    H = FieldFunctionClass(field, None, np.zeros((3, 1)), np.zeros((3, 1), dtype=int), None, "H")
    mode = PortMode(field, E, H, k0=1.0, beta=1.0)
    mode.flip_polarity()
    assert np.all(E.field == -1) and np.all(H.field == -1)


def test_wave_port_homogeneous_scaled_modes():
    # Square 20 x 20 mm guide, 4 modes fixed at 12 GHz: TE10/TE01 (7.49 GHz, degenerate) and TE11/TM11
    # (10.6 GHz, degenerate). The port face is homogeneous: the modes are solved once and scaled to 9 GHz,
    # where TE11/TM11 are evanescent. The mixed TE11/TM11 pair is split into pure TE and TM, whose fields scale
    # differently. Against the analytic β and |S21| = |exp(-jβL)|, and against solving every frequency.
    a, L = 20 * mm, 10 * mm
    freqs = np.array([9e9, 12e9])
    results = []
    for scale in (True, False):
        sim, box = _guide(a, a, L, freqs, resolution=0.15)
        if not scale:
            sim.mw._wave_port_scalable = lambda port: False
        p1 = sim.mw.bc.WavePort(box.face("-z"), 1, number_of_modes=4)
        sim.mw.bc.WavePort(box.face("+z"), 2, number_of_modes=4)
        Sp = sim.mw.run_sweep().scalar.grid.Sp
        assert p1._scalable == scale
        kc2 = {"TE10": (np.pi / a) ** 2, "TE11": 2 * (np.pi / a) ** 2}
        types = [m.modetype for m in p1.get_modes(2 * np.pi * freqs[-1] / C0)]
        if scale:
            assert sorted(types) == ["TE", "TE", "TE", "TM"], types
        for fi, f in enumerate(freqs):
            k0 = 2 * np.pi * f / C0
            betas = np.array([complex(p1.get_beta(k0, i)) for i in range(1, 5)])
            exact = np.array([np.sqrt(complex(k0**2 - kc2["TE10" if i < 2 else "TE11"])) for i in range(4)])
            exact = np.where(exact.imag > 0, np.conj(exact), exact)
            np.testing.assert_allclose(betas, exact, rtol=2e-2, atol=2.0)
            S = Sp[fi]
            assert np.abs(S[:4, :4]).max() < 0.03, (scale, f)
            # Each port picks its own basis of the degenerate TE10/TE01 pair, so the transmission of a pair is a
            # 2 x 2 block with singular values |exp(-jβL)|. TE11/TM11 are split into a unique TE and TM mode.
            T = S[4:, :4]
            for pair in ([0, 1], [2, 3]):
                sv = np.linalg.svd(T[np.ix_(pair, pair)], compute_uv=False)
                np.testing.assert_allclose(sv, np.abs(np.exp(-1j * betas[pair] * L)), atol=0.02)
            assert np.abs(T[2:, :2]).max() < 0.03 and np.abs(T[:2, 2:]).max() < 0.03, (scale, f)
            assert np.abs(T[2, 3]) < 0.03 and np.abs(T[3, 2]) < 0.03, (scale, f)
        results.append(Sp)
        em.cleanup()
    for pair in ([0, 1], [2, 3]):
        for fi in range(len(freqs)):
            svs = [np.linalg.svd(Sp[fi][4:, :4][np.ix_(pair, pair)], compute_uv=False) for Sp in results]
            np.testing.assert_allclose(svs[0], svs[1], atol=0.01)


def test_wave_port_split_mixed_degenerate_modes():
    # An eigen solver may return any complex mixture of the degenerate TE11/TM11 pair. The split of a homogeneous
    # port recovers the pure TE11 and TM11 mode, which scale differently with frequency.
    from emerge._emerge.physics.microwave.port_functions import compute_modal_power_flux

    a = 20 * mm
    sim, box = _guide(a, a, 10 * mm, np.array([12e9]), resolution=0.15)
    p1 = sim.mw.bc.WavePort(box.face("-z"), 1, number_of_modes=4)
    sim.mw.bc.WavePort(box.face("+z"), 2, number_of_modes=4)
    sim.mw.run_sweep()
    k0 = 2 * np.pi * 12e9 / C0
    modes = p1.get_modes(k0)
    te11, tm11 = modes[2], modes[3]
    assert (te11.modetype, tm11.modetype) == ("TE", "TM")
    f1, f2 = (m.modefield * m.E_function.constant for m in (te11, tm11))
    mixes = [sim.mw._make_wave_port_mode(p1, c1 * f1 + c2 * f2, te11.beta, k0, 12e9, "TEM")
             for c1, c2 in ((0.8, 0.6j), (-0.6, 0.8j))]
    p1.available_modes[k0] = modes[:2] + mixes
    assert sim.mw._split_te_tm_modes(p1, k0)
    nlf, _, normal = p1._mode_basis
    split = p1.available_modes[k0][2:]
    assert [m.modetype for m in split] == ["TE", "TM"]
    for m, ref, other in ((split[0], te11, tm11), (split[1], tm11, te11)):
        assert abs(abs(compute_modal_power_flux(nlf, m.E_function, ref.H_function, normal)) - 1) < 1e-3
        assert abs(compute_modal_power_flux(nlf, m.E_function, other.H_function, normal)) < 1e-6
    em.cleanup()
