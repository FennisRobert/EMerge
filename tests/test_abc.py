import emerge as em
from emerge_config import config
import numpy as np
import pytest

# Configure threads for acceleration
config.set_acc_threads(10)


def predict_abc_reflection_linear(
    theta: float | np.ndarray,
    c1: float,
    c2: float = 0.0,
    order: int = 2,
    degrees: bool = True,
) -> float | np.ndarray:
    """Predicts the theoretical linear reflection coefficient |R| vs. incidence angle."""
    angles_rad = np.radians(theta) if degrees else np.asarray(theta)
    u = np.cos(angles_rad)
    s2 = np.sin(angles_rad) ** 2

    if order == 1:
        z_eff = c1
    elif order == 2:
        z_eff = c1 + c2 * s2
    else:
        raise ValueError("ABC order must be 1 or 2.")

    return np.abs((u - z_eff) / (u + z_eff))


@pytest.mark.parametrize(
    "order, abctype, opt_angle",
    [
        (1, "A", None),
        (2, "A", None),
        (2, "", 75),
    ],
)
def test_abc_reflection_coefficient(order: int, abctype: str, opt_angle: int | None):
    """Tests simulation S11 against theoretical ABC predictions and prints 

    the maximum absolute error in dB: 20*log10(max(|S11_sim - S11_theory|)).
    """
    mm = 0.001
    angles = np.linspace(0, 89, 21)

    try:
        # 1. Setup Simulation
        sim = em.Simulation("ABCTest", loglevel="INFO")

        rectcell = em.RectCell(10 * mm, 10 * mm)
        sim.set_periodic_cell(rectcell)

        airvol = rectcell.volume(0, 40 * mm)
        fptop = rectcell.port_face(40 * mm)

        sim.commit_geometry()

        sim.mw.set_frequency(10e9)
        sim.mw.set_resolution(0.1)
        sim.generate_mesh()

        # 2. Setup Boundary Conditions
        sim.mw.bc.floquet_port(fptop, 1)
        abc_bot = sim.mw.bc.AbsorbingBoundary(
            airvol.face("-z"), order=order, abctype=abctype
        )

        if opt_angle is not None:
            abc_bot.optimize_for_maximum_angle(opt_angle)

        # 3. Retrieve ABC coefficients used
        if order == 1:
            c1, c2 = 1.0, 0.0
        elif order == 2 and opt_angle is None:
            c1, c2 = abc_bot.o2coeffs[abctype]
        else:
            c1, c2 = abc_bot._coeffset

        # 4. Run Angle Sweep
        for ang in sim.parameter_sweep(False, angle=angles):
            rectcell.set_scanangle(ang, 0)
            sim.mw.run_sweep()

        # 5. Extract Simulated Linear Magnitudes |S11|
        S11_sim_linear = np.abs(sim.data.mw.scalar.grid.S(1, 1).squeeze())

        # 6. Compute Theoretical Linear Magnitudes
        R_theory_linear = predict_abc_reflection_linear(
            angles, c1=c1, c2=c2, order=order, degrees=True
        )

        # 7. Compute Maximum Absolute Error in dB: 20*log10(max(|S_sim - S_theory|))
        abs_linear_error = np.abs(S11_sim_linear - R_theory_linear)
        max_abs_error_linear = np.max(abs_linear_error)
        
        # Add epsilon to prevent log10(0) if exact match
        max_abs_error_db = 20 * np.log10(np.maximum(max_abs_error_linear, 1e-15))

        print(
            f"\n[ABC Order {order}, Type {abctype}] "
            f"Max Absolute Error: {max_abs_error_db:.2f} dB "
            f"(Linear Max Diff: {max_abs_error_linear:.6e})"
        )

        # 8. Assertion Tolerance (adjust -30 dB threshold if needed)
        assert max_abs_error_db < -30.0, (
            f"ABC Order {order} Type {abctype} max absolute error ({max_abs_error_db:.2f} dB) "
            f"exceeded threshold of -30.0 dB."
        )

    finally:
        em.cleanup()

if __name__ == "__main__":
    import sys

    # Runs pytest on this file with stdout enabled (-s) and verbose reporting (-v)
    sys.exit(pytest.main(["-s", "-v", __file__]))