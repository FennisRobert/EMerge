"""
Test suite for thermal FEM solver: steady-state, convection, radiation, and thermal contact.

Each test compares the FEM solution against an analytical reference along the z-axis
of a simple box geometry.
"""

import numpy as np
import emerge as em


def analytical_conduction(z, q, kappa, T_bottom):
    return T_bottom + z * q / kappa


def analytical_convection(z, q, kappa, h, T_amb):
    return T_amb + q / h + z * q / kappa


def analytical_radiation(z, q, kappa, H, T_amb, emissivity=1.0):
    sigma = 5.670374419e-8
    T_top = ((q / (sigma * emissivity)) + T_amb**4) ** 0.25
    dTdz = -q / kappa
    T_bottom = T_top - dTdz * H
    return T_bottom + dTdz * np.asarray(z)


def analytical_contact(z, q, kappa, H, T_amb, h_contact, emissivity=1.0):
    T_no_contact = analytical_radiation(z, q, kappa, H, T_amb, emissivity)
    N = len(z)
    T_no_contact[: N // 2] += q / h_contact
    return T_no_contact


W = 0.25
H = 1.0
T0 = 100.0
Q = 10.0
KAPPA = 2.0


def interpolate_z(field, H, n_points=1001):
    zs = np.linspace(0, H, n_points)
    xs = np.zeros_like(zs)
    ys = np.zeros_like(zs)
    T = field.interpolate(xs, ys, zs).T
    return zs, T


############################################################
#                           TESTS                          #
############################################################


def test_linear_gradient():
    sim = em.Simulation("test_conduction")
    sim.set_physics(False, True)

    mat = em.Material(cond_thermal=KAPPA)
    box = em.geo.Box(W, W, H, (-W / 2, -W / 2, 0)).set_material(mat)

    sim.commit_geometry()
    sim.generate_mesh()

    sim.hc.bc.FixedTemperatureBoundary(box.bottom, T0)
    sim.hc.bc.HeatFluxBoundary(box.top, Q)

    data = sim.hc.run_steady_state()
    field = data.field[0]

    zs, T_fem = interpolate_z(field, H)
    T_ref = analytical_conduction(zs, Q, KAPPA, T0)

    np.testing.assert_allclose(
        T_fem,
        T_ref,
        rtol=1e-2,
        err_msg="Conduction profile does not match analytical solution",
    )

    em.cleanup()


def test_volumetric_heatflux():
    sim = em.Simulation("test_conduction")
    sim.set_physics(False, True)

    mat = em.Material(cond_thermal=KAPPA)
    box = em.geo.Box(W, W, H, (-W / 2, -W / 2, 0)).set_material(mat)

    sim.commit_geometry()
    sim.generate_mesh()

    sim.hc.bc.FixedTemperatureBoundary(box.bottom, T0)
    sim.hc.bc.HeatFluxVolume(box, Q)

    data = sim.hc.run_steady_state()
    field = data.field[0]

    zs, T_fem = interpolate_z(field, H)
    T_ref = T0 + Q / (2 * KAPPA) * (1 - (zs - 1) ** 2)

    np.testing.assert_allclose(
        T_fem,
        T_ref,
        rtol=1e-2,
        err_msg="Conduction profile does not match analytical solution",
    )

    em.cleanup()


def test_convection_gradient():
    sim = em.Simulation("test_convection")
    sim.set_physics(False, True)

    h_conv = 3.0
    mat = em.Material(cond_thermal=KAPPA)
    box = em.geo.Box(W, W, H, (-W / 2, -W / 2, 0)).set_material(mat)

    sim.commit_geometry()
    sim.generate_mesh()

    sim.hc.bc.Convection(box.bottom, h_conv, T0)
    sim.hc.bc.HeatFluxBoundary(box.top, Q)

    data = sim.hc.run_steady_state()
    field = data.field[0]

    zs, T_fem = interpolate_z(field, H)
    T_ref = analytical_convection(zs, Q, KAPPA, h_conv, T0)

    np.testing.assert_allclose(
        T_fem,
        T_ref,
        rtol=1e-2,
        err_msg="Convection profile does not match analytical solution",
    )

    em.cleanup()


def test_radiation_gradient():
    sim = em.Simulation("test_radiation")
    sim.set_physics(False, True)

    mat = em.Material(cond_thermal=KAPPA)
    box = em.geo.Box(W, W, H, (-W / 2, -W / 2, 0)).set_material(mat)

    sim.commit_geometry()
    sim.generate_mesh()

    sim.hc.bc.HeatFluxBoundary(box.bottom, Q)
    sim.hc.bc.BlackBodyRadiation(box.top, 1.0, T0)

    data = sim.hc.run_steady_state_nl()
    field = data.field[0]

    zs, T_fem = interpolate_z(field, H)
    T_ref = analytical_radiation(zs, Q, KAPPA, H, T0)

    np.testing.assert_allclose(
        T_fem,
        T_ref,
        rtol=1e-2,
        err_msg="Radiation profile does not match analytical solution",
    )

    em.cleanup()


def test_contact_jump():
    sim = em.Simulation("test_contact")
    sim.set_physics(False, True)

    h_contact = 10.0
    mat = em.Material(cond_thermal=KAPPA)
    box1 = em.geo.Box(W, W, H / 2, (-W / 2, -W / 2, 0)).set_material(mat)
    box2 = em.geo.Box(W, W, H / 2, (-W / 2, -W / 2, H / 2)).set_material(mat)

    sim.commit_geometry()
    sim.generate_mesh()

    sim.hc.bc.HeatFluxBoundary(box1.bottom, Q)
    sim.hc.bc.ThermalContact(box1.top, h_contact)
    sim.hc.bc.BlackBodyRadiation(box2.top, 1.0, T0)

    data = sim.hc.run_steady_state_nl()
    field = data.field[0]

    zs, T_fem = interpolate_z(field, H)
    T_ref = analytical_contact(zs, Q, KAPPA, H, T0, h_contact)

    # Check temperature jump at contact (z = H/2)
    z_below = H / 2 - 0.01
    z_above = H / 2 + 0.01
    T_below = field.interpolate(
        np.array([0.0]), np.array([0.0]), np.array([z_below])
    ).T[0]
    T_above = field.interpolate(
        np.array([0.0]), np.array([0.0]), np.array([z_above])
    ).T[0]

    expected_jump = Q / h_contact
    actual_jump = T_below - T_above

    assert abs(actual_jump - expected_jump) / expected_jump < 0.1, (
        f"Contact jump {actual_jump:.3f} K != expected {expected_jump:.3f} K"
    )

    # Check overall profile
    np.testing.assert_allclose(
        T_fem,
        T_ref,
        rtol=5e-2,
        err_msg="Contact profile does not match analytical solution",
    )

    em.cleanup()


if __name__ == "__main__":
    test_volumetric_heatflux()
    test_contact_jump()
    test_convection_gradient()
    test_linear_gradient()
    test_radiation_gradient()
