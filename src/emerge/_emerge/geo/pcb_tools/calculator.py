# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025 Robert Fennis.
# Copyright (C) 2026 Yawar (Wayri).

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""PCB transmission-line, cable, and waveguide calculations.

Core functions use metres, hertz, and ohms. ``PCBCalculator`` converts its
configured stackup unit to metres. Closed-form and empirical estimates
assume the cross sections described by their individual functions.
"""

import numpy as np
from emsutil import Material
from scipy.special import ellipk, ellipkm1, jv, yv

n0 = 376.73031366857
PI = np.pi
TAU = 2 * PI
C0 = 299_792_458.0
MU0 = 4e-7 * PI


############################################################
#                     NUMERIC HELPERS                      #
############################################################


def _asf(x):
    """Convert a scalar or array-like input to a float NumPy array."""
    return np.asarray(x, dtype=float)


def _ellipk_agm(k):
    """Evaluate complete elliptic K(k) using the arithmetic-geometric mean."""
    k = np.clip(_asf(k), 0.0, 1.0 - 1e-15)
    a = np.ones_like(k)
    b = np.sqrt(1.0 - k * k)
    for _ in range(32):
        an = 0.5 * (a + b)
        bn = np.sqrt(a * b)
        if np.all(np.abs(an - bn) < 1e-15):
            a = an
            break
        a, b = an, bn
    return PI / (2.0 * a)


def _ellip_ratio(k):
    """Return K(k)/K(sqrt(1-k**2)) without small-modulus cancellation."""
    k = _asf(k)
    if np.any(~np.isfinite(k)) or np.any(k < 0.0) or np.any(k > 1.0):
        raise ValueError("Elliptic modulus must lie in [0, 1]")
    m = k * k
    # ellipkm1(m) evaluates K(1-m) without cancellation when m is tiny.
    return ellipk(m) / ellipkm1(m)


def _coth(x):
    """Return coth(x), guarding the removable numerical division near zero."""
    x = _asf(x)
    s = np.sinh(x)
    s = np.where(np.abs(s) < 1e-30, np.sign(s) * 1e-30 + (s == 0) * 1e-30, s)
    return np.cosh(x) / s


def _sech(x):
    """Return the hyperbolic secant, 1/cosh(x)."""
    return 1.0 / np.cosh(_asf(x))


def _jn(n: int, x: float) -> float:
    """Return Bessel J of integer order n at x as a Python float."""
    return float(jv(n, x))


def _yn(n: int, x: float) -> float:
    """Return Bessel Y of integer order n at x as a Python float."""
    return float(yv(n, x))


def _jnp(n: int, x: float) -> float:
    """Return the derivative of Bessel J_n using adjacent-order recurrence."""
    n = int(n)
    if n == 0:
        return -_jn(1, x)
    return 0.5 * (_jn(n - 1, x) - _jn(n + 1, x))


def _ynp(n: int, x: float) -> float:
    """Return the derivative of Bessel Y_n using adjacent-order recurrence."""
    n = int(n)
    if n == 0:
        return -_yn(1, x)
    return 0.5 * (_yn(n - 1, x) - _yn(n + 1, x))


def _material_er(mat: Material, f0: float) -> float:
    """Read relative permittivity from a material at frequency f0 in hertz."""
    er = getattr(mat, "er", None)
    if er is None:
        raise ValueError("Dielectric material has no relative permittivity")
    if hasattr(er, "scalar"):
        return float(er.scalar(f0))
    if callable(er):
        return float(er(f0))
    return float(er)


def _inverse_from_samples(target: float, xs, ys) -> float:
    """Interpolate an inverse from finite sampled points; reject out-of-range targets."""
    x = _asf(xs)
    y = _asf(ys)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        raise ValueError("No finite samples available for inverse solve")
    if x.size < 2:
        raise ValueError("At least two finite samples are required for inverse solve")

    dy = np.diff(y)
    if np.all(dy >= 0.0):
        yk, xk = y, x
    elif np.all(dy <= 0.0):
        yk, xk = y[::-1], x[::-1]
    else:
        return float(x[np.argmin(np.abs(y - target))])

    lo = float(min(yk[0], yk[-1]))
    hi = float(max(yk[0], yk[-1]))
    if not np.isfinite(target) or target < lo or target > hi:
        raise ValueError(f"Target {target} is outside the achievable range [{lo}, {hi}]")
    return float(np.interp(target, yk, xk))


def _inverse_bounds_m(x_min, x_max, unit: float, scale_m: float):
    """Convert optional search bounds from stackup units to SI metres."""
    if not np.isfinite(unit) or unit <= 0 or not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("Stackup unit and reference distance must be positive and finite")
    lo = 0.1 * scale_m if x_min is None else float(x_min) * unit
    hi = 10.0 * scale_m if x_max is None else float(x_max) * unit
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= lo:
        raise ValueError("Inverse search bounds must be positive, finite and increasing")
    return lo, hi


def _scan_inverse(target: float, fn, x_min: float, x_max: float, n: int = 501) -> float:
    """Find a bracket on logarithmic samples and refine an inverse by bisection."""
    if not np.isfinite(target) or not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Inverse target and bounds must be finite")
    x0 = float(x_min)
    x1 = float(x_max)
    if x0 <= 0.0 or x1 <= x0 or int(n) < 2:
        raise ValueError("Inverse search needs positive, increasing bounds and at least two samples")
    xs = np.geomspace(x0, x1, int(n))
    ys = _asf(fn(xs))
    x_est = _inverse_from_samples(target, xs, ys)

    m = np.isfinite(xs) & np.isfinite(ys)
    xk = _asf(xs)[m]
    yk = _asf(ys)[m]
    if xk.size < 2:
        raise ValueError("Inverse solve has fewer than two finite model samples")

    d = yk - float(target)
    crossings = np.where((d[:-1] == 0.0) | (d[1:] == 0.0) | (d[:-1] * d[1:] < 0.0))[0]
    if crossings.size == 0:
        if np.min(np.abs(d)) <= 1e-9 * max(abs(float(target)), 1.0):
            return float(xk[np.argmin(np.abs(d))])
        raise ValueError("Target has no solution within the requested search bounds")

    mids = 0.5 * (xk[crossings] + xk[crossings + 1])
    i = int(crossings[np.argmin(np.abs(mids - float(x_est)))])
    xa = float(xk[i])
    xb = float(xk[i + 1])

    def _f1(x: float) -> float:
        y = _asf(fn(np.asarray([x], dtype=float)))
        if y.size == 0 or not np.isfinite(y[0]):
            return np.nan
        return float(y[0]) - float(target)

    fa = _f1(xa)
    fb = _f1(xb)
    if not np.isfinite(fa) or not np.isfinite(fb):
        return float(x_est)
    if fa == 0.0:
        return xa
    if fb == 0.0:
        return xb
    if fa * fb > 0.0:
        return float(x_est)

    for _ in range(64):
        xm = 0.5 * (xa + xb)
        fm = _f1(xm)
        if not np.isfinite(fm):
            break
        if abs(fm) < 1e-12:
            return float(xm)
        if fa * fm <= 0.0:
            xb, fb = xm, fm
        else:
            xa, fa = xm, fm
        if abs(xb - xa) <= 1e-12 * max(1.0, abs(xm)):
            break
    return float(0.5 * (xa + xb))


def _odd_even_from_k(z0, k):
    """Convert uncoupled Z0 and coupling k to even/odd modal impedances."""
    k = np.clip(_asf(k), 0.0, 0.95)
    zo = _asf(z0) * np.sqrt((1.0 - k) / (1.0 + k))
    ze = _asf(z0) * np.sqrt((1.0 + k) / (1.0 - k))
    return ze, zo


############################################################
#                        MICROSTRIP                        #
############################################################


def microstrip_z0(W: float, th: float, er: float, t: float = 0.0):
    """Single-ended quasi-static microstrip impedance in ohms.

    Piecewise Hammerstad-style air impedance divided by sqrt(epsilon_eff); finite
    t changes electrical width.

    Args:
        W (float): Conductor width in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        t (float): Conductor thickness in metres.

    Returns:
        Single-ended quasi-static microstrip impedance in ohms.
    """
    W = _asf(W)
    h = float(th)
    u = np.maximum(W / h, 1e-12)

    if t is not None and t > 0.0:
        thn = float(t) / h
        x = np.sqrt(6.517 * u)
        du1 = (thn / PI) * np.log(1.0 + (4.0 * np.e) / (thn * _coth(x) ** 2))
        dur = 0.5 * du1 * (1.0 + _sech(np.sqrt(np.maximum(er - 1.0, 0.0))))
        u = u + dur

    eeff = microstrip_eeff(W, h, er, t=t)

    return np.where(
        u <= 1.0,
        (60.0 / np.sqrt(eeff)) * np.log(8.0 / u + 0.25 * u),
        (120.0 * PI / np.sqrt(eeff)) / (u + 1.393 + 0.667 * np.log(u + 1.444)),
    )


def microstrip_eeff(W: float, th: float, er: float, t: float = 0.0):
    """Quasi-static microstrip effective relative permittivity.

    Air/dielectric filling approximation with a narrow-line term and finite-
    thickness width correction.

    Args:
        W (float): Conductor width in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        t (float): Conductor thickness in metres.

    Returns:
        Quasi-static microstrip effective relative permittivity.
    """
    W = _asf(W)
    h = float(th)
    u = np.maximum(W / h, 1e-12)

    thickness_factor = 1.0
    if t is not None and t > 0.0:
        thn = float(t) / h
        x = np.sqrt(6.517 * u)
        du1 = (thn / PI) * np.log(1.0 + (4.0 * np.e) / (thn * _coth(x) ** 2))
        dur = 0.5 * du1 * (1.0 + _sech(np.sqrt(np.maximum(er - 1.0, 0.0))))
        # Hammerstad/Jensen uses different width corrections in air and in
        # dielectric; their air-impedance ratio also corrects epsilon_eff.
        z_air_1 = microstrip_z0((u + du1) * h, h, 1.0)
        z_air_r = microstrip_z0((u + dur) * h, h, 1.0)
        thickness_factor = (z_air_1 / z_air_r) ** 2
        u = u + dur

    eeff = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 / np.sqrt(1.0 + 12.0 / u))
    eeff = eeff + np.where(u < 1.0, 0.04 * (1.0 - u) ** 2 * (er - 1.0) / 2.0, 0.0)
    return eeff * thickness_factor


def microstrip_eeff_dispersion(
    W: float, th: float, er: float, f: float, t: float = 0.0
):
    """Frequency-dependent microstrip effective relative permittivity.

    Kirschning/Jansen empirical interpolation: er - (er - eeff(0))/(1 + P).

    Args:
        W (float): Conductor width in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        f (float): Frequency in hertz.
        t (float): Conductor thickness in metres.

    Returns:
        Frequency-dependent microstrip effective relative permittivity.
    """
    W = _asf(W)
    h = float(th)
    f = float(f)
    ee0 = _asf(microstrip_eeff(W, h, er, t=t))
    if f <= 0.0:
        return ee0

    u = np.maximum(W / h, 1e-12)
    fn = f * h / 1e6  # normalized frequency in GHz*mm
    p1 = (
        0.27488
        + u * (0.6315 + 0.525 / np.power(1.0 + 0.0157 * fn, 20.0))
        - 0.065683 * np.exp(-8.7513 * u)
    )
    p2 = 0.33622 * (1.0 - np.exp(-0.03442 * er))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1.0 - np.exp(-np.power(fn / 38.7, 4.97)))
    p4 = 1.0 + 2.751 * (1.0 - np.exp(-np.power(er / 15.916, 8.0)))
    p = p1 * p2 * np.power(np.maximum((p3 * p4 + 0.1844) * fn, 1e-30), 1.5763)
    return er - (er - ee0) / (1.0 + p)


def microstrip_z0_dispersion(W: float, th: float, er: float, f: float, t: float = 0.0):
    """Frequency-dependent microstrip impedance in ohms.

    Kirschning/Jansen correction: Z0(f) = Z0(0)*(R13/R14)**R17 within the checked
    range.

    Args:
        W (float): Conductor width in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        f (float): Frequency in hertz.
        t (float): Conductor thickness in metres.

    Returns:
        Frequency-dependent microstrip impedance in ohms.
    """
    W = _asf(W)
    h = float(th)
    f = float(f)
    if not np.isfinite(h) or h <= 0 or np.any(~np.isfinite(W)) or np.any(W <= 0):
        raise ValueError("Microstrip width and substrate height must be positive and finite")
    if not np.isfinite(er) or not 1.0 <= er <= 18.0 or not np.isfinite(f) or f < 0:
        raise ValueError("Microstrip dispersion requires 1 <= er <= 18 and finite f >= 0")
    if np.any(W / h < 0.1) or np.any(W / h > 10.0) or h * f / C0 > 0.1:
        raise ValueError("Microstrip impedance dispersion is outside its published geometry/frequency range")
    z0_0 = _asf(microstrip_z0(W, h, er, t=t))
    ee0 = _asf(microstrip_eeff(W, h, er, t=t))
    if f <= 0.0:
        return z0_0

    eef = _asf(microstrip_eeff_dispersion(W, h, er, f=f, t=t))
    u = np.maximum(W / h, 1e-12)
    fn = f * h / 1e6

    r1 = 0.03891 * er ** (1.4)
    r2 = np.clip(0.267 * u**7.0, a_min=None, a_max=20)
    r3 = 4.766 * np.exp(-3.228 * u**0.641)
    r4 = 0.016 + (0.0514 * er) ** 4.524
    r5 = (fn / 28.843) ** 12.0
    r6 = np.clip(22.2 * (u**1.92), a_min=None, a_max=20)
    r7 = 1.206 - 0.3144 * np.exp(-r1) * (1.0 - np.exp(-r2))
    r8 = 1.0 + 1.275 * (
        1.0 - np.exp(-0.004625 * r3 * er**1.674 * (fn / 18.365) ** 2.745)
    )
    tmp = (er - 1.0) ** 6.0
    r9 = (
        5.086
        * (r4 * r5 / (0.3838 + 0.386 * r4))
        * (np.exp(-r6) / (1.0 + 1.2992 * r5))
        * (tmp / (1.0 + 10.0 * tmp))
    )
    r10 = 0.00044 * er**2.136 + 0.0184
    tmp = (fn / 19.47) ** 6.0
    r11 = tmp / (1.0 + 0.0962 * tmp)
    r12 = 1.0 / (1.0 + 0.00245 * u * u)
    r13 = 0.9408 * (np.maximum(eef, 1e-30) ** r8) - 0.9603
    r14 = (0.9408 - r9) * (np.maximum(ee0, 1e-30) ** r8) - 0.9603
    r15 = 0.707 * r10 * (fn / 12.3) ** 1.097
    r16 = 1.0 + 0.0503 * er * er * r11 * (1.0 - np.exp(-((u / 15.0) ** 6.0)))
    r17 = r7 * (
        1.0 - 1.1241 * (r12 / r16) * np.exp(-0.026 * np.power(fn, 1.15656) - r15)
    )
    ratio = r13 / r14
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("Microstrip dispersion ratio is not physical for this geometry")
    d = np.power(ratio, r17)
    return z0_0 * d


############################################################
#                         STRIPLINE                        #
############################################################


def stripline_z0(W: float, b: float, er: float, t: float = 0.0):
    """Centered, homogeneous stripline impedance in ohms.

    Zero-thickness Cohn elliptic-integral form; positive thickness uses the
    finite-t logarithmic approximation.

    Args:
        W (float): Conductor width in metres.
        b (float): Reference-plane spacing in metres.
        er (float): Relative permittivity.
        t (float): Conductor thickness in metres.

    Returns:
        Centered, homogeneous stripline impedance in ohms.
    """
    W = _asf(W)
    b = float(b)
    t = float(t)
    if not np.all(np.isfinite(W)) or np.any(W <= 0.0):
        raise ValueError("Stripline width must be finite and positive.")
    if not np.isfinite(b) or b <= 0.0 or not np.isfinite(t) or t < 0.0 or t >= b:
        raise ValueError("Stripline requires finite b > t >= 0.")
    if not np.isfinite(er) or er <= 0.0:
        raise ValueError("Relative permittivity must be finite and positive.")

    if t <= 0.0:
        x = PI * W / (2.0 * b)
        k = _sech(x)
        return (n0 / (4.0 * np.sqrt(er))) * _ellip_ratio(k)

    x = t / b
    m = 2.0 / (1.0 + (2.0 * x / 3.0) * (1.0 - x))
    u = np.maximum(W / b, 1e-15)
    frac = (x / (2.0 - x)) ** 2 + np.power((0.0796 * x) / (u + 1.1 * x), m)
    bc = (x / (PI * (1.0 - x))) * (1.0 - 0.5 * np.log(np.maximum(frac, 1e-30)))
    A = 1.0 / (W / np.maximum(b - t, 1e-15) + bc)
    p = (8.0 / PI) * A
    return (30.0 / np.sqrt(er)) * np.log(
        1.0 + (4.0 / PI) * A * (p + np.sqrt(p * p + 6.27))
    )


def coupled_stripline_zodd(W: float, S: float, b: float, er: float):
    """Odd-mode impedance of zero-thickness edge-coupled stripline in ohms.

    Cohn conformal-map modulus k from W, S, b, then eta0*K(k)/(4*sqrt(er)*K(k')).

    Args:
        W (float): Conductor width in metres.
        S (float): Edge gap or coplanar slot in metres.
        b (float): Reference-plane spacing in metres.
        er (float): Relative permittivity.

    Returns:
        Odd-mode impedance of zero-thickness edge-coupled stripline in ohms.
    """
    W = _asf(W)
    b = float(b)
    s = _asf(S)
    x1 = PI * W / (2.0 * b)
    x2 = PI * (W + s) / (2.0 * b)
    k0p = np.tanh(x1) * _coth(x2)
    k0p = np.clip(k0p, 0.0, 1.0 - 1e-15)
    k0 = np.sqrt(1.0 - k0p * k0p)
    return (n0 / (4.0 * np.sqrt(er))) * _ellip_ratio(k0)


def coupled_stripline_zdiff(W: float, S: float, b: float, er: float):
    """Differential edge-coupled stripline impedance in ohms.

    Equal and opposite excitation gives Zdiff = 2*Zodd.

    Args:
        W (float): Conductor width in metres.
        S (float): Edge gap or coplanar slot in metres.
        b (float): Reference-plane spacing in metres.
        er (float): Relative permittivity.

    Returns:
        Differential edge-coupled stripline impedance in ohms.
    """
    return 2.0 * coupled_stripline_zodd(W, S, b, er)


def broadside_stripline_zdiff_zcm(W: float, G: float, b: float, er: float):
    # Full Cohn broadside-coupled stripline (zero-thickness conductors):
    #   Z0e = (188.3/sqrt(er)) * K(k')/K(k)
    #   Z0o = (296.1*s)/(sqrt(er)*atanh(k))
    # with implicit relation for width ratio (w = W/b, s = G/b):
    #   w = (2/pi) * atanh(R) - s * atanh(R/k)
    #   R = sqrt((k - s) / (1/k - s))
    """Return (differential, common-mode) broadside stripline impedances in ohms.

    Solve Cohn's implicit width/modulus equation, then use Zdiff=2*Zodd and
    Zcm=Zeven/2.

    Args:
        W (float): Conductor width in metres.
        G (float): Broadside spacing in metres.
        b (float): Reference-plane spacing in metres.
        er (float): Relative permittivity.

    Returns:
        Return (differential, common-mode) broadside stripline impedances in ohms.
    """
    ws = _asf(W)
    g = float(G)
    b = float(b)
    er = float(er)
    if g <= 0.0 or b <= 0.0 or er <= 0.0:
        raise ValueError("G, b and er must be > 0 for broadside-coupled stripline.")
    if g >= b:
        raise ValueError("Broadside spacing G must be smaller than cavity height b.")

    s = g / b
    if s <= 0.0 or s >= 1.0:
        raise ValueError("Broadside spacing ratio s=G/b must satisfy 0 < s < 1.")

    def _w_from_k(k: float) -> float:
        num = max(k - s, 1e-30)
        den = max((1.0 / k) - s, 1e-30)
        r = np.sqrt(num / den)
        r = float(np.clip(r, 1e-15, 1.0 - 1e-15))
        rk = float(np.clip(r / max(k, 1e-15), 1e-15, 1.0 - 1e-15))
        return (2.0 / PI) * (np.arctanh(r) - s * np.arctanh(rk))

    def _k_from_w(wratio: float) -> float:
        if wratio <= 0.0:
            raise ValueError("W must be > 0 for broadside-coupled stripline.")
        lo = max(s + 1e-12, 1e-9)
        hi = 1.0 - 1e-12
        flo = _w_from_k(lo) - wratio
        fhi = _w_from_k(hi) - wratio
        if not np.isfinite(flo) or not np.isfinite(fhi):
            raise ValueError(
                "Broadside k-solve failed due to non-finite endpoint value."
            )
        if flo > 0.0 or fhi < 0.0:
            raise ValueError("Broadside width has no bracketed modal solution")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = _w_from_k(mid) - wratio
            if fm >= 0.0:
                hi = mid
            else:
                lo = mid
        return float(0.5 * (lo + hi))

    out_zd = np.empty_like(ws, dtype=float)
    out_zc = np.empty_like(ws, dtype=float)
    for i, w in np.ndenumerate(ws):
        k = _k_from_w(float(w) / b)
        kp = np.sqrt(max(1.0 - k * k, 1e-30))
        z0e = (188.3 / np.sqrt(er)) * _ellip_ratio(kp)
        z0o = (296.1 * s) / (np.sqrt(er) * max(np.arctanh(k), 1e-30))
        out_zd[i] = 2.0 * z0o
        out_zc[i] = 0.5 * z0e

    return out_zd, out_zc


############################################################
#                    COPLANAR WAVEGUIDE                    #
############################################################


def cpw_z0(
    W: float,
    S: float,
    th: float,
    er: float,
    t: float = 0.0,
    has_metal_backside: bool = False,
):
    """Single-ended CPW or grounded-CPW impedance in ohms.

    Conformal-map elliptic ratios give air/dielectric filling; optional t applies
    the first-order slot correction.

    Args:
        W (float): Conductor width in metres.
        S (float): Edge gap or coplanar slot in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        t (float): Conductor thickness in metres.
        has_metal_backside (bool): Include an ideal continuous backside ground plane.

    Returns:
        Single-ended CPW or grounded-CPW impedance in ohms.
    """
    W = _asf(W)
    h = float(th)
    s = float(S)
    a = W
    b = W + 2.0 * s
    k1 = a / b
    q1 = _ellip_ratio(k1)

    if has_metal_backside:
        k3 = np.tanh(PI * a / (4.0 * h)) / np.tanh(PI * b / (4.0 * h))
        q3 = _ellip_ratio(k3)
        qz = 1.0 / (q1 + q3)
        eeff = 1.0 + q3 * qz * (er - 1.0)
        zr = n0 / 2.0 * qz
    else:
        k2 = np.sinh((PI / 4.0) * a / h) / np.sinh((PI / 4.0) * b / h)
        q2 = _ellip_ratio(k2)
        eeff = 1.0 + (er - 1.0) / 2.0 * q2 / q1
        zr = n0 / 4.0 / q1

    if t is not None and t > 0.0:
        d = (
            1.25
            * float(t)
            / PI
            * (1.0 + np.log(4.0 * PI * np.maximum(W, 1e-18) / float(t)))
        )
        ke = k1 + (1.0 - k1 * k1) * d / (2.0 * s)
        qe = _ellip_ratio(ke)
        if has_metal_backside:
            qz = 1.0 / (qe + q3)
            zr = n0 / 2.0 * qz
        else:
            zr = n0 / 4.0 / qe
        eeff = eeff - (0.7 * (eeff - 1.0) * float(t) / s) / (q1 + (0.7 * float(t) / s))

    return zr / np.sqrt(eeff)


def cpw_eeff(
    W: float,
    S: float,
    th: float,
    er: float,
    t: float = 0.0,
    has_metal_backside: bool = False,
):
    """Effective relative permittivity of CPW or grounded CPW.

    Partial capacitance filling factor from conformal-map elliptic ratios.

    Args:
        W (float): Conductor width in metres.
        S (float): Edge gap or coplanar slot in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        t (float): Conductor thickness in metres.
        has_metal_backside (bool): Include an ideal continuous backside ground plane.

    Returns:
        Effective relative permittivity of CPW or grounded CPW.
    """
    W = _asf(W)
    h = float(th)
    s = float(S)
    a = W
    b = W + 2.0 * s
    k1 = a / b
    q1 = _ellip_ratio(k1)

    if has_metal_backside:
        k3 = np.tanh(PI * a / (4.0 * h)) / np.tanh(PI * b / (4.0 * h))
        q3 = _ellip_ratio(k3)
        qz = 1.0 / (q1 + q3)
        eeff = 1.0 + q3 * qz * (er - 1.0)
    else:
        k2 = np.sinh((PI / 4.0) * a / h) / np.sinh((PI / 4.0) * b / h)
        q2 = _ellip_ratio(k2)
        eeff = 1.0 + (er - 1.0) / 2.0 * q2 / q1

    if t is not None and t > 0.0:
        eeff = eeff - (0.7 * (eeff - 1.0) * float(t) / s) / (q1 + (0.7 * float(t) / s))
    return eeff


def cpw_eeff_dispersion(
    W: float,
    S: float,
    th: float,
    er: float,
    f: float,
    t: float = 0.0,
    has_metal_backside: bool = False,
):
    """Frequency-dependent CPW or grounded-CPW effective permittivity.

    Empirical Qucs interpolation in sqrt(epsilon_eff) toward sqrt(er).

    Args:
        W (float): Conductor width in metres.
        S (float): Edge gap or coplanar slot in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        f (float): Frequency in hertz.
        t (float): Conductor thickness in metres.
        has_metal_backside (bool): Include an ideal continuous backside ground plane.

    Returns:
        Frequency-dependent CPW or grounded-CPW effective permittivity.
    """
    ee0 = _asf(cpw_eeff(W, S, th, er, t=t, has_metal_backside=has_metal_backside))
    f = float(f)
    if f <= 0.0 or er <= 1.0:
        return ee0

    w = _asf(W)
    h = float(th)
    s = float(S)
    fte = (C0 / 4.0) / (h * np.sqrt(max(er - 1.0, 1e-15)))
    p = np.log(np.maximum(w / h, 1e-15))
    u = 0.54 - (0.64 - 0.015 * p) * p
    v = 0.43 - (0.86 - 0.54 * p) * p
    g = np.exp(u * np.log(np.maximum(w / s, 1e-15)) + v)
    sr_er0 = np.sqrt(np.maximum(ee0, 1e-30))
    sr_er = np.sqrt(er)
    sr_er_f = sr_er0 + (sr_er - sr_er0) / (
        1.0 + g * np.power(np.maximum(f / fte, 1e-30), -1.8)
    )
    return sr_er_f * sr_er_f


def cpw_z0_dispersion(
    W: float,
    S: float,
    th: float,
    er: float,
    f: float,
    t: float = 0.0,
    has_metal_backside: bool = False,
):
    """Frequency-dependent CPW or grounded-CPW impedance in ohms.

    Scale quasi-static Z0 by sqrt(epsilon_eff(0)/epsilon_eff(f)).

    Args:
        W (float): Conductor width in metres.
        S (float): Edge gap or coplanar slot in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        f (float): Frequency in hertz.
        t (float): Conductor thickness in metres.
        has_metal_backside (bool): Include an ideal continuous backside ground plane.

    Returns:
        Frequency-dependent CPW or grounded-CPW impedance in ohms.
    """
    z0_qs = _asf(cpw_z0(W, S, th, er, t=t, has_metal_backside=has_metal_backside))
    ee0 = _asf(cpw_eeff(W, S, th, er, t=t, has_metal_backside=has_metal_backside))
    eef = _asf(
        cpw_eeff_dispersion(
            W, S, th, er, f=f, t=t, has_metal_backside=has_metal_backside
        )
    )
    return z0_qs * np.sqrt(np.maximum(ee0, 1e-30) / np.maximum(eef, 1e-30))


def _cpw_cap_per_len(
    W: float,
    S: float,
    th: float,
    er: float,
    t: float = 0.0,
    has_metal_backside: bool = False,
    f: float | None = None,
):
    """Convert CPW impedance and effective permittivity to capacitance per metre."""
    if f is None:
        z = _asf(cpw_z0(W, S, th, er, t=t, has_metal_backside=has_metal_backside))
        ee = _asf(cpw_eeff(W, S, th, er, t=t, has_metal_backside=has_metal_backside))
        z_air = _asf(cpw_z0(W, S, th, 1.0, t=t, has_metal_backside=has_metal_backside))
    else:
        z = _asf(
            cpw_z0_dispersion(
                W, S, th, er, f=float(f), t=t, has_metal_backside=has_metal_backside
            )
        )
        ee = _asf(
            cpw_eeff_dispersion(
                W, S, th, er, f=float(f), t=t, has_metal_backside=has_metal_backside
            )
        )
        z_air = _asf(
            cpw_z0_dispersion(
                W, S, th, 1.0, f=float(f), t=t, has_metal_backside=has_metal_backside
            )
        )
    c = np.sqrt(np.maximum(ee, 1e-15)) / (C0 * np.maximum(z, 1e-15))
    c_air = 1.0 / (C0 * np.maximum(z_air, 1e-15))
    return c, c_air, z


############################################################
#                      COAXIAL LINES                       #
############################################################


def coax_z0(d_inner: float, d_outer: float, er: float):
    """TEM coaxial-line impedance in ohms.

    eta0*ln(d_outer/d_inner)/(2*pi*sqrt(er)).

    Args:
        d_inner (float): Inner conductor diameter in metres.
        d_outer (float): Outer conductor diameter in metres.
        er (float): Relative permittivity.

    Returns:
        TEM coaxial-line impedance in ohms.
    """
    d_inner = _asf(d_inner)
    d_outer = _asf(d_outer)
    return (n0 / (2.0 * PI * np.sqrt(er))) * np.log(d_outer / d_inner)


def coax_d_for_z0(Z0: float, d_outer: float, er: float):
    """Inner coax diameter in metres for a target TEM impedance.

    Algebraic inverse of the logarithmic coax formula.

    Args:
        Z0 (float): Target impedance in ohms.
        d_outer (float): Outer conductor diameter in metres.
        er (float): Relative permittivity.

    Returns:
        Inner coax diameter in metres for a target TEM impedance.
    """
    return float(d_outer) / np.exp(2.0 * PI * np.sqrt(er) * float(Z0) / n0)


def _coax_cutoff_te_approx(
    d_inner: float, d_outer: float, er: float = 1.0, mur: float = 1.0
):
    """Return the approximate TE(1,1) cutoff when explicitly requested."""
    return C0 / (
        PI * (float(d_outer) + float(d_inner)) * np.sqrt(float(er) * float(mur))
    )


def _coax_cutoff_tm_approx(
    d_inner: float, d_outer: float, er: float = 1.0, mur: float = 1.0
):
    """Return the approximate TM(0,1) cutoff when explicitly requested."""
    return C0 / (
        2.0 * (float(d_outer) - float(d_inner)) * np.sqrt(float(er) * float(mur))
    )


def _validate_coax_cutoff(d_inner, d_outer, er, mur):
    """Reject nonphysical coax dimensions and constitutive parameters."""
    di, do, eps, mu = map(float, (d_inner, d_outer, er, mur))
    if not all(np.isfinite(value) for value in (di, do, eps, mu)):
        raise ValueError("Coax cutoff inputs must be finite.")
    if di <= 0.0 or do <= di or eps <= 0.0 or mu <= 0.0:
        raise ValueError("Coax cutoff requires d_outer > d_inner > 0 and er, mur > 0.")


def _coax_mode_char(mode: str, n: int, x: float, ratio: float) -> float:
    """Evaluate the TE/TM annular Bessel boundary-condition determinant."""
    xa = float(x)
    xb = float(ratio) * xa
    if mode == "tm":
        return _jn(n, xa) * _yn(n, xb) - _yn(n, xa) * _jn(n, xb)
    if mode == "te":
        return _jnp(n, xa) * _ynp(n, xb) - _ynp(n, xa) * _jnp(n, xb)
    raise ValueError("mode must be 'te' or 'tm'.")


def _bisect_root(fn, x0: float, x1: float, iters: int = 80) -> float:
    """Refine a sign-changing scalar root within a bracket by bisection."""
    f0 = float(fn(x0))
    f1 = float(fn(x1))
    if not np.isfinite(f0) or not np.isfinite(f1):
        raise ValueError("Non-finite bracket function values.")
    if f0 == 0.0:
        return float(x0)
    if f1 == 0.0:
        return float(x1)
    if f0 * f1 > 0.0:
        raise ValueError("Invalid bracket for bisection.")
    a, b = float(x0), float(x1)
    fa = f0
    for _ in range(int(iters)):
        m = 0.5 * (a + b)
        fm = float(fn(m))
        if not np.isfinite(fm):
            break
        if abs(fm) < 1e-13:
            return float(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
    return float(0.5 * (a + b))


def _coax_mode_root(n: int, m: int, d_inner: float, d_outer: float, mode: str):
    """Locate an annular coax TE/TM transverse-wavenumber eigen-root."""
    if int(n) != n or int(m) != m:
        raise ValueError("Coax mode indices must be integers.")
    n = int(n)
    m = int(m)
    if n < 0 or m < 1:
        raise ValueError("Mode indices must satisfy n>=0 and m>=1.")
    di = float(d_inner)
    do = float(d_outer)
    if di <= 0.0 or do <= di:
        raise ValueError("Coax diameters must satisfy d_outer > d_inner > 0.")

    a = 0.5 * di
    b = 0.5 * do
    ratio = b / a
    fn = lambda x: _coax_mode_char(mode, n, x, ratio)

    x = 1e-6
    step = 0.01
    x_max = max(120.0, (m + n + 6) * PI * 4.0)
    prev = fn(x)
    found = 0
    while x < x_max:
        xn = x + step
        cur = fn(xn)
        if np.isfinite(prev) and np.isfinite(cur):
            if prev == 0.0:
                root = x
            elif cur == 0.0:
                root = xn
            elif prev * cur < 0.0:
                root = _bisect_root(fn, x, xn)
            else:
                root = None
            if root is not None and root > 0.0:
                found += 1
                if found == m:
                    return float(root / a)
        x = xn
        prev = cur
        step = min(0.2, 0.01 + 0.001 * x)
    raise ValueError(f"Failed to find coax {mode.upper()}({n},{m}) root.")


def coax_cutoff_te(
    d_inner: float,
    d_outer: float,
    er: float = 1.0,
    mur: float = 1.0,
    n: int = 1,
    m: int = 1,
    exact: bool = True,
):
    """TE_nm coaxial higher-mode cutoff frequency in hertz.

    Solve the annular Bessel-derivative eigenvalue equation when available.

    Args:
        d_inner (float): Inner conductor diameter in metres.
        d_outer (float): Outer conductor diameter in metres.
        er (float): Relative permittivity.
        mur (float): Relative permeability.
        n (int): Mode order or index.
        m (int): Mode order or index.
        exact (bool): Solve the modal eigenvalue rather than using the cutoff estimate.

    Returns:
        TE_nm coaxial higher-mode cutoff frequency in hertz.
    """
    _validate_coax_cutoff(d_inner, d_outer, er, mur)
    if not exact:
        if (n, m) != (1, 1):
            raise ValueError("The TE cutoff estimate supports only TE(1,1).")
        return _coax_cutoff_te_approx(d_inner, d_outer, er=er, mur=mur)
    kc = _coax_mode_root(n=n, m=m, d_inner=d_inner, d_outer=d_outer, mode="te")
    return C0 * kc / (2.0 * PI * np.sqrt(float(er) * float(mur)))


def coax_cutoff_tm(
    d_inner: float,
    d_outer: float,
    er: float = 1.0,
    mur: float = 1.0,
    n: int = 0,
    m: int = 1,
    exact: bool = True,
):
    """TM_nm coaxial higher-mode cutoff frequency in hertz.

    Solve the annular Bessel-function eigenvalue equation when available.

    Args:
        d_inner (float): Inner conductor diameter in metres.
        d_outer (float): Outer conductor diameter in metres.
        er (float): Relative permittivity.
        mur (float): Relative permeability.
        n (int): Mode order or index.
        m (int): Mode order or index.
        exact (bool): Solve the modal eigenvalue rather than using the cutoff estimate.

    Returns:
        TM_nm coaxial higher-mode cutoff frequency in hertz.
    """
    _validate_coax_cutoff(d_inner, d_outer, er, mur)
    if not exact:
        if (n, m) != (0, 1):
            raise ValueError("The TM cutoff estimate supports only TM(0,1).")
        return _coax_cutoff_tm_approx(d_inner, d_outer, er=er, mur=mur)
    kc = _coax_mode_root(n=n, m=m, d_inner=d_inner, d_outer=d_outer, mode="tm")
    return C0 * kc / (2.0 * PI * np.sqrt(float(er) * float(mur)))


############################################################
#                       TWISTED PAIR                       #
############################################################


def twisted_pair_eeff(
    d_center: float,
    d_wire: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Effective relative permittivity for the empirical twisted-pair model.

    Blend inner and surrounding dielectric with a twist-dependent filling factor.

    Args:
        d_center (float): Wire centre-to-centre spacing in metres.
        d_wire (float): Wire conductor diameter in metres.
        er (float): Relative permittivity.
        er1 (float): Relative permittivity of the surrounding medium.
        twists_per_len (float): Twists per metre.
        ptfe (bool): Use the PTFE branch of the empirical dielectric model.

    Returns:
        Effective relative permittivity for the empirical twisted-pair model.
    """
    d_center = float(d_center)
    d_wire = float(d_wire)
    if not all(
        np.isfinite(value) for value in (d_center, d_wire, er, er1, twists_per_len)
    ):
        raise ValueError("Twisted-pair inputs must be finite.")
    if d_wire <= 0.0 or er <= 0.0 or er1 <= 0.0 or twists_per_len < 0.0:
        raise ValueError(
            "Twisted-pair diameters and permittivities must be positive; "
            "twist rate cannot be negative."
        )
    if d_center <= d_wire:
        raise ValueError("d_center must be greater than d_wire for twisted pair.")
    theta = np.arctan(float(twists_per_len) * PI * d_center)
    q = 0.25 + (0.001 if ptfe else 0.0004) * theta * theta
    return float(er1 + q * (er - er1))


def twisted_pair_z0(
    d_center: float,
    d_wire: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Quasi-static twisted-pair impedance in ohms.

    eta0*acosh(d_center/d_wire)/(pi*sqrt(epsilon_eff)).

    Args:
        d_center (float): Wire centre-to-centre spacing in metres.
        d_wire (float): Wire conductor diameter in metres.
        er (float): Relative permittivity.
        er1 (float): Relative permittivity of the surrounding medium.
        twists_per_len (float): Twists per metre.
        ptfe (bool): Use the PTFE branch of the empirical dielectric model.

    Returns:
        Quasi-static twisted-pair impedance in ohms.
    """
    eeff = twisted_pair_eeff(
        d_center, d_wire, er, er1=er1, twists_per_len=twists_per_len, ptfe=ptfe
    )
    arg = float(d_center) / float(d_wire)
    return n0 / (PI * np.sqrt(eeff)) * np.arccosh(arg)


def twisted_pair_d_center_for_z0(
    z0: float,
    d_wire: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Centre spacing in metres for target twisted-pair impedance.

    Numerical inverse of the twist-dependent impedance relation.

    Args:
        z0 (float): Target impedance in ohms.
        d_wire (float): Wire conductor diameter in metres.
        er (float): Relative permittivity.
        er1 (float): Relative permittivity of the surrounding medium.
        twists_per_len (float): Twists per metre.
        ptfe (bool): Use the PTFE branch of the empirical dielectric model.

    Returns:
        Centre spacing in metres for target twisted-pair impedance.
    """
    target = float(z0)
    wire = float(d_wire)
    if not np.isfinite(target) or target <= 0.0 or not np.isfinite(wire) or wire <= 0.0:
        raise ValueError("Target impedance and wire diameter must be finite and positive.")
    lower = np.nextafter(wire, np.inf)
    upper = 2.0 * wire

    def impedance(spacing):
        return twisted_pair_z0(
            spacing,
            wire,
            er,
            er1=er1,
            twists_per_len=twists_per_len,
            ptfe=ptfe,
        )

    for _ in range(80):
        if impedance(upper) >= target:
            break
        upper *= 2.0
    else:
        raise ValueError("No finite centre-spacing solution for target impedance.")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if middle <= lower or middle >= upper:
            break
        if impedance(middle) < target:
            lower = middle
        else:
            upper = middle
    return float(0.5 * (lower + upper))


def twisted_pair_d_wire_for_z0(
    z0: float,
    d_center: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Wire diameter in metres for target twisted-pair impedance.

    Numerical inverse of the twist-dependent impedance relation.

    Args:
        z0 (float): Target impedance in ohms.
        d_center (float): Wire centre-to-centre spacing in metres.
        er (float): Relative permittivity.
        er1 (float): Relative permittivity of the surrounding medium.
        twists_per_len (float): Twists per metre.
        ptfe (bool): Use the PTFE branch of the empirical dielectric model.

    Returns:
        Wire diameter in metres for target twisted-pair impedance.
    """
    eeff = twisted_pair_eeff(
        d_center=d_center,
        d_wire=max(0.5 * float(d_center), 1e-12),
        er=er,
        er1=er1,
        twists_per_len=twists_per_len,
        ptfe=ptfe,
    )
    k = np.cosh(PI * float(z0) * np.sqrt(eeff) / n0)
    if k <= 1.0:
        raise ValueError(
            "No valid d_wire solution for requested twisted-pair impedance."
        )
    return float(float(d_center) / k)


############################################################
#                   RECTANGULAR WAVEGUIDE                  #
############################################################


def rectwg_fc(
    a: float, b: float, m: int = 1, n: int = 0, er: float = 1.0, mur: float = 1.0
):
    """Rectangular-waveguide mode cutoff frequency in hertz.

    c0*sqrt((m/a)**2+(n/b)**2)/(2*sqrt(er*mur)).

    Args:
        a (float): Broad waveguide wall in metres.
        b (float): Narrow waveguide wall in metres.
        m (int): Mode order or index.
        n (int): Mode order or index.
        er (float): Relative permittivity.
        mur (float): Relative permeability.

    Returns:
        Rectangular-waveguide mode cutoff frequency in hertz.
    """
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Rectangular waveguide dimensions must be > 0.")
    if m < 0 or n < 0 or (m == 0 and n == 0):
        raise ValueError("Mode indices must satisfy m>=0, n>=0 and not both zero.")
    kxy = np.sqrt((m / a) ** 2 + (n / b) ** 2)
    return 0.5 * C0 * kxy / np.sqrt(float(er) * float(mur))


def rectwg_beta(
    f: float,
    a: float,
    b: float,
    m: int = 1,
    n: int = 0,
    er: float = 1.0,
    mur: float = 1.0,
):
    """Propagation constant beta in radians per metre above cutoff.

    beta = sqrt(k**2-kc**2); the implementation returns zero at/below cutoff.

    Args:
        f (float): Frequency in hertz.
        a (float): Broad waveguide wall in metres.
        b (float): Narrow waveguide wall in metres.
        m (int): Mode order or index.
        n (int): Mode order or index.
        er (float): Relative permittivity.
        mur (float): Relative permeability.

    Returns:
        Propagation constant beta in radians per metre above cutoff.
    """
    f = float(f)
    if f <= 0.0:
        raise ValueError("Frequency must be > 0.")
    fc = rectwg_fc(a, b, m=m, n=n, er=er, mur=mur)
    if f <= fc:
        return 0.0
    k = TAU * f * np.sqrt(float(er) * float(mur)) / C0
    kc = TAU * fc * np.sqrt(float(er) * float(mur)) / C0
    return float(np.sqrt(k * k - kc * kc))


def rectwg_z_te(
    f: float,
    a: float,
    b: float,
    m: int = 1,
    n: int = 0,
    er: float = 1.0,
    mur: float = 1.0,
):
    """TE-mode wave impedance in ohms.

    eta/sqrt(1-(fc/f)**2); infinite at/below cutoff.

    Args:
        f (float): Frequency in hertz.
        a (float): Broad waveguide wall in metres.
        b (float): Narrow waveguide wall in metres.
        m (int): Mode order or index.
        n (int): Mode order or index.
        er (float): Relative permittivity.
        mur (float): Relative permeability.

    Returns:
        TE-mode wave impedance in ohms.
    """
    f = float(f)
    fc = rectwg_fc(a, b, m=m, n=n, er=er, mur=mur)
    if f <= fc:
        return np.inf
    return float(n0 * np.sqrt(float(mur) / float(er)) / np.sqrt(1.0 - (fc / f) ** 2))


def rectwg_z_tm(
    f: float,
    a: float,
    b: float,
    m: int = 1,
    n: int = 1,
    er: float = 1.0,
    mur: float = 1.0,
):
    """TM-mode wave impedance in ohms.

    eta*sqrt(1-(fc/f)**2); zero at/below cutoff.

    Args:
        f (float): Frequency in hertz.
        a (float): Broad waveguide wall in metres.
        b (float): Narrow waveguide wall in metres.
        m (int): Mode order or index.
        n (int): Mode order or index.
        er (float): Relative permittivity.
        mur (float): Relative permeability.

    Returns:
        TM-mode wave impedance in ohms.
    """
    f = float(f)
    fc = rectwg_fc(a, b, m=m, n=n, er=er, mur=mur)
    if f <= fc:
        return 0.0
    return float(n0 * np.sqrt(float(mur) / float(er)) * np.sqrt(1.0 - (fc / f) ** 2))


def rectwg_lambda_g(
    f: float,
    a: float,
    b: float,
    m: int = 1,
    n: int = 0,
    er: float = 1.0,
    mur: float = 1.0,
):
    """Guided wavelength in metres above cutoff.

    2*pi/beta; infinite at/below cutoff.

    Args:
        f (float): Frequency in hertz.
        a (float): Broad waveguide wall in metres.
        b (float): Narrow waveguide wall in metres.
        m (int): Mode order or index.
        n (int): Mode order or index.
        er (float): Relative permittivity.
        mur (float): Relative permeability.

    Returns:
        Guided wavelength in metres above cutoff.
    """
    beta = rectwg_beta(f, a, b, m=m, n=n, er=er, mur=mur)
    if beta <= 0.0:
        return np.inf
    return float(TAU / beta)


def rectwg_a_for_fc(fc: float, er: float = 1.0, mur: float = 1.0, m: int = 1):
    """Broad-wall dimension in metres for target cutoff frequency.

    Algebraic inverse of the n=0 rectangular-waveguide cutoff formula.

    Args:
        fc (float): Cutoff frequency in hertz.
        er (float): Relative permittivity.
        mur (float): Relative permeability.
        m (int): Mode order or index.

    Returns:
        Broad-wall dimension in metres for target cutoff frequency.
    """
    fc = float(fc)
    if fc <= 0.0 or m <= 0:
        raise ValueError("fc and mode index m must be > 0.")
    return float(m * C0 / (2.0 * fc * np.sqrt(float(er) * float(mur))))


def rectwg_te10_a_for_z0(z0: float, f: float, er: float = 1.0, mur: float = 1.0):
    """Broad-wall dimension in metres for target TE10 wave impedance.

    Infer cutoff from target impedance and frequency, then invert cutoff.

    Args:
        z0 (float): Target impedance in ohms.
        f (float): Frequency in hertz.
        er (float): Relative permittivity.
        mur (float): Relative permeability.

    Returns:
        Broad-wall dimension in metres for target TE10 wave impedance.
    """
    z0 = float(z0)
    f = float(f)
    if z0 <= 0.0 or f <= 0.0:
        raise ValueError("z0 and f must be > 0.")
    q = n0 * np.sqrt(float(mur) / float(er)) / z0
    if q <= 0.0 or q >= 1.0:
        raise ValueError(
            "No propagating TE10 solution for the requested z0 at this frequency."
        )
    fc = f * np.sqrt(1.0 - q * q)
    return rectwg_a_for_fc(fc, er=er, mur=mur, m=1)


############################################################
#                    COUPLED MICROSTRIP                    #
############################################################


def coupled_microstrip_z0_even_odd(
    W: float,
    S: float,
    th: float,
    er: float,
    t: float = 0.0,
    f: float | None = None,
):
    """Return (even-mode, odd-mode) coupled-microstrip impedances in ohms.

    Kirschning/Jansen empirical modal filling and coupling; optional frequency
    dispersion.

    Args:
        W (float): Conductor width in metres.
        S (float): Edge gap or coplanar slot in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        t (float): Conductor thickness in metres.
        f (float | None): Frequency in hertz.

    Returns:
        Return (even-mode, odd-mode) coupled-microstrip impedances in ohms.
    """
    h = float(th)
    w = float(W)
    s = float(S)
    if h <= 0.0 or w <= 0.0 or s <= 0.0:
        raise ValueError("W, S and th must all be > 0.")

    u = max(w / h, 1e-12)
    g = max(s / h, 1e-12)

    def _delta_u_thickness_single(uu: float, t_h: float) -> float:
        if t_h <= 0.0:
            return 0.0
        x = (
            2.0
            + (4.0 * PI * uu - 2.0) / (1.0 + np.exp(-100.0 * (uu - 1.0 / (2.0 * PI))))
        ) / t_h
        return float((1.25 * t_h / PI) * (1.0 + np.log(max(x, 1e-30))))

    ue = u
    uo = u
    if t is not None and t > 0.0:
        t_h = float(t) / h
        du = _delta_u_thickness_single(u, t_h)
        dt = t_h / (g * er)
        due = du * (1.0 - 0.5 * np.exp(-0.69 * du / max(dt, 1e-30)))
        duo = due + dt
        ue = u + due
        uo = u + duo

    # Static modal effective permittivities.
    v = ue * (20.0 + g * g) / (10.0 + g * g) + g * np.exp(-g)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    ae = (
        1.0
        + np.log((v4 + v2 / 2704.0) / (v4 + 0.432)) / 49.0
        + np.log(1.0 + v3 / 5929.741) / 18.7
    )
    be = 0.564 * np.power((er - 0.9) / (er + 3.0), 0.053)
    q_inf_e = np.power(1.0 + 10.0 / max(v, 1e-30), -ae * be)
    ee_e0 = 0.5 * (er + 1.0) + 0.5 * (er - 1.0) * q_inf_e

    ee_single_0 = float(microstrip_eeff(w, h, er, t=0.0))
    bo = 0.747 * er / (0.15 + er)
    co = bo - (bo - 0.207) * np.exp(-0.414 * uo)
    do = 0.593 + 0.694 * np.exp(-0.562 * uo)
    q_inf_o = np.exp(-co * np.power(g, do))
    ao = 0.7287 * (ee_single_0 - 0.5 * (er + 1.0)) * (1.0 - np.exp(-0.179 * uo))
    ee_o0 = (0.5 * (er + 1.0) + ao - ee_single_0) * q_inf_o + ee_single_0

    # Static modal impedances.
    q1 = 0.8695 * np.power(ue, 0.194)
    q2 = 1.0 + 0.7519 * g + 0.189 * np.power(g, 2.31)
    q3 = (
        0.1975
        + np.power(16.6 + np.power(8.4 / g, 6.0), -0.387)
        + np.log(np.power(g, 10.0) / (1.0 + np.power(g / 3.4, 10.0))) / 241.0
    )
    q4 = (
        2.0
        * q1
        / (
            q2
            * (np.exp(-g) * np.power(ue, q3) + (2.0 - np.exp(-g)) * np.power(ue, -q3))
        )
    )
    q5 = 1.794 + 1.14 * np.log(1.0 + 0.638 / (g + 0.517 * np.power(g, 2.43)))
    q6 = (
        0.2305
        + np.log(np.power(g, 10.0) / (1.0 + np.power(g / 5.8, 10.0))) / 281.3
        + np.log(1.0 + 0.598 * np.power(g, 1.154)) / 5.1
    )
    q7 = (10.0 + 190.0 * g * g) / (1.0 + 82.3 * g * g * g)
    q8 = np.exp(-6.5 - 0.95 * np.log(g) - np.power(g / 0.15, 5.0))
    q9 = np.log(q7) * (q8 + 1.0 / 16.5)
    q10 = (q2 * q4 - q5 * np.exp(np.log(uo) * q6 * np.power(uo, -q9))) / q2

    z_single_0 = float(microstrip_z0(w, h, er, t=0.0))
    z_even_0 = (
        z_single_0
        * np.sqrt(ee_single_0 / ee_e0)
        / (1.0 - np.sqrt(ee_single_0) * q4 * z_single_0 / n0)
    )
    z_odd_0 = (
        z_single_0
        * np.sqrt(ee_single_0 / ee_o0)
        / (1.0 - np.sqrt(ee_single_0) * q10 * z_single_0 / n0)
    )

    if f is None or float(f) <= 0.0:
        return float(z_even_0), float(z_odd_0)
    if er == 1.0:
        # Homogeneous air has no dielectric frequency dispersion.
        return float(z_even_0), float(z_odd_0)

    # Frequency-dependent modal effective permittivities.
    fn = float(f) * h / 1e6
    p1 = (
        0.27488
        + (0.6315 + 0.525 / np.power(1.0 + 0.0157 * fn, 20.0)) * u
        - 0.065683 * np.exp(-8.7513 * u)
    )
    p2 = 0.33622 * (1.0 - np.exp(-0.03442 * er))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1.0 - np.exp(-np.power(fn / 38.7, 4.97)))
    p4 = 1.0 + 2.751 * (1.0 - np.exp(-np.power(er / 15.916, 8.0)))
    p8 = 0.7168 * (1.0 + 1.076 / (1.0 + 0.0576 * (er - 1.0)))
    p9 = p8 - 0.7913 * (1.0 - np.exp(-np.power(fn / 20.0, 1.424))) * np.arctan(
        2.481 * np.power(er / 8.0, 0.946)
    )
    p10 = 0.242 * np.power(er - 1.0, 0.55)
    p11 = (
        0.6366
        * (np.exp(-0.3401 * fn) - 1.0)
        * np.arctan(1.263 * np.power(u / 3.0, 1.629))
    )
    p12 = p9 + (1.0 - p9) / (1.0 + 1.183 * np.power(u, 1.376))
    p13 = 1.695 * p10 / (0.414 + 1.605 * p10)
    p14 = 0.8928 + 0.1072 * (1.0 - np.exp(-0.42 * np.power(fn / 20.0, 3.215)))
    p15 = abs(
        1.0 - 0.8928 * (1.0 + p11) * p12 * np.exp(-p13 * np.power(g, 1.092)) / p14
    )
    fo = p1 * p2 * np.power(np.maximum((p3 * p4 + 0.1844) * fn * p15, 1e-30), 1.5763)
    ee_o = er - (er - ee_o0) / (1.0 + fo)

    # Frequency-dependent modal impedances.
    ee_single_f = float(microstrip_eeff_dispersion(w, h, er, f=float(f), t=0.0))
    z_single_f = float(microstrip_z0_dispersion(w, h, er, f=float(f), t=0.0))

    q11 = 0.893 * (1.0 - 0.3 / (1.0 + 0.7 * (er - 1.0)))
    q12 = (
        2.121
        * (np.power(fn / 20.0, 4.91) / (1.0 + q11 * np.power(fn / 20.0, 4.91)))
        * np.exp(-2.87 * g)
        * np.power(g, 0.902)
    )
    q13 = 1.0 + 0.038 * np.power(er / 8.0, 5.1)
    q14 = 1.0 + 1.203 * np.power(er / 15.0, 4.0) / (1.0 + np.power(er / 15.0, 4.0))
    q15 = (
        1.887
        * np.exp(-1.5 * np.power(g, 0.84))
        * np.power(g, q14)
        / (
            1.0
            + 0.41
            * np.power(fn / 15.0, 3.0)
            * np.power(u, 2.0 / q13)
            / (0.125 + np.power(u, 1.626 / q13))
        )
    )
    q16 = (1.0 + 9.0 / (1.0 + 0.403 * np.power(er - 1.0, 2.0))) * q15
    q17 = (
        0.394
        * (1.0 - np.exp(-1.47 * np.power(u / 7.0, 0.672)))
        * (1.0 - np.exp(-4.25 * np.power(fn / 20.0, 1.87)))
    )
    q18 = (
        0.61
        * (1.0 - np.exp(-2.13 * np.power(u / 8.0, 1.593)))
        / (1.0 + 6.544 * np.power(g, 4.17))
    )
    q19 = (
        0.21
        * np.power(g, 4.0)
        / (
            (1.0 + 0.18 * np.power(g, 4.9))
            * (1.0 + 0.1 * u * u)
            * (1.0 + np.power(fn / 24.0, 3.0))
        )
    )
    q20 = (0.09 + 1.0 / (1.0 + 0.1 * np.power(er - 1.0, 2.7))) * q19
    q21 = abs(
        1.0
        - 42.54
        * np.power(g, 0.133)
        * np.exp(-0.812 * g)
        * np.power(u, 2.5)
        / (1.0 + 0.033 * np.power(u, 2.5))
    )

    re = np.power(fn / 28.843, 12.0)
    qe = 0.016 + np.power(0.0514 * er * q21, 4.524)
    pe = 4.766 * np.exp(-3.228 * np.power(u, 0.641))
    de = (
        5.086
        * qe
        * (re / (0.3838 + 0.386 * qe))
        * (np.exp(-22.2 * np.power(u, 1.92)) / (1.0 + 1.2992 * re))
        * (np.power(er - 1.0, 6.0) / (1.0 + 10.0 * np.power(er - 1.0, 6.0)))
    )
    ce = (
        1.0
        + 1.275
        * (
            1.0
            - np.exp(
                -0.004625 * pe * np.power(er, 1.674) * np.power(fn / 18.365, 2.745)
            )
        )
        - q12
        + q16
        - q17
        + q18
        + q20
    )

    r1 = 0.03891 * np.power(er, 1.4)
    r2 = 0.267 * np.power(u, 7.0)
    r7 = 1.206 - 0.3144 * np.exp(-r1) * (1.0 - np.exp(-r2))
    r10 = 0.00044 * np.power(er, 2.136) + 0.0184
    tmp = np.power(fn / 19.47, 6.0)
    r11 = tmp / (1.0 + 0.0962 * tmp)
    r12 = 1.0 / (1.0 + 0.00245 * u * u)
    r15 = 0.707 * r10 * np.power(fn / 12.3, 1.097)
    r16 = 1.0 + 0.0503 * er * er * r11 * (1.0 - np.exp(-np.power(u / 15.0, 6.0)))
    q0 = r7 * (
        1.0 - 1.1241 * (r12 / r16) * np.exp(-0.026 * np.power(fn, 1.15656) - r15)
    )

    even_ratio = (0.9408 * np.power(ee_single_f, ce) - 0.9603) / (
        (0.9408 - de) * np.power(ee_single_0, ce) - 0.9603
    )
    if not np.isfinite(even_ratio) or even_ratio <= 0.0:
        raise ValueError("Coupled microstrip even-mode dispersion ratio is not physical")
    z_even = z_even_0 * np.power(even_ratio, q0)

    q29 = 15.16 / (1.0 + 0.196 * np.power(er - 1.0, 2.0))
    tmp = np.power(er - 1.0, 3.0)
    q28 = 0.149 * tmp / (94.5 + 0.038 * tmp)
    tmp = np.power(er - 1.0, 1.5)
    q27 = 0.4 * np.power(g, 0.84) * (1.0 + 2.5 * tmp / (5.0 + tmp))
    tmp = np.power((er - 1.0) / 13.0, 12.0)
    q26 = 30.0 - 22.2 * (tmp / (1.0 + 3.0 * tmp)) - q29
    tmp = np.power(er - 1.0, 2.0)
    q25 = (0.3 * fn * fn / (10.0 + fn * fn)) * (1.0 + 2.333 * tmp / (5.0 + tmp))
    q24 = (
        2.506
        * q28
        * np.power(u, 0.894)
        * np.power((1.0 + 1.3 * u) * fn / 99.25, 4.29)
        / (3.575 + np.power(u, 0.894))
    )
    q23 = 1.0 + 0.005 * fn * q27 / (
        (1.0 + 0.812 * np.power(fn / 15.0, 1.9)) * (1.0 + 0.025 * u * u)
    )
    q22 = (
        0.925
        * np.power(fn / max(q26, 1e-30), 1.536)
        / (1.0 + 0.3 * np.power(fn / 30.0, 1.536))
    )

    z_odd = z_single_f + (z_odd_0 * np.power(ee_o / ee_o0, q22) - z_single_f * q23) / (
        1.0 + q24 + np.power(0.46 * g, 2.2) * q25
    )
    if not np.isfinite(z_even) or not np.isfinite(z_odd) or z_even <= 0.0 or z_odd <= 0.0:
        raise ValueError("Coupled microstrip modal impedance is not physical")
    return float(z_even), float(z_odd)


def differential_cpw_zdiff_zcm(
    W: float,
    S_ground: float,
    S_pair: float,
    th: float,
    er: float,
    t: float = 0.0,
    has_metal_backside: bool = False,
    f: float | None = None,
    *,
    _cells_per_feature: int = 96,
):
    """Quasi-static coupled-CPW differential and common-mode impedance.

    Solve surface potential for even and odd excitations separately. The
    homogeneous dielectric slab's Fourier-domain admittance includes an
    optional ideal backing plane. Coplanar grounds are ideal and extend away
    from the traces. Each modal impedance follows from its dielectric and air
    capacitances: ``Z = 1 / (c * sqrt(C_air * C))``. Frequency-dependent
    dispersion and multilayer dielectrics are not modeled.

    Args:
        W (float): Conductor width in metres.
        S_ground (float): Trace-to-lateral-ground slot in metres.
        S_pair (float): Gap between the two signal traces in metres.
        th (float): Substrate height in metres.
        er (float): Relative permittivity.
        t (float): Conductor thickness in metres; only zero is supported.
        has_metal_backside (bool): Include an ideal continuous backside ground plane.
        f (float | None): Frequency in hertz, used only as an applicability guard.
        _cells_per_feature (int): Internal minimum mesh density for convergence checks.

    Returns:
        Differential and common-mode impedance in ohms, respectively.

    Raises:
        ValueError: Nonphysical or under-resolved geometry.
        NotImplementedError: Finite thickness or frequency beyond the quasi-static range.
    """
    if not all(
        np.isfinite(value) and value > 0.0
        for value in (W, S_ground, S_pair, th, er)
    ):
        raise ValueError(
            "Coupled CPW widths, gaps, height and er must be finite and positive."
        )
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("Conductor thickness must be finite and nonnegative.")
    if t > 0.0:
        raise NotImplementedError(
            "Finite-thickness coupled CPW requires a 2D conductor model."
        )
    signal_span = 2.0 * W + S_pair + 2.0 * S_ground
    if not np.isfinite(signal_span):
        raise ValueError("Coupled CPW span must be finite.")
    if f is not None:
        if not np.isfinite(f) or f < 0.0:
            raise ValueError("Frequency must be finite and nonnegative.")
        if f * max(signal_span, th) * np.sqrt(er) / C0 > 0.05:
            raise NotImplementedError(
                "Coupled CPW frequency is outside the quasi-static range."
            )

    ground_pad = (
        max(4.0 * th, 2.0 * signal_span)
        if has_metal_backside
        else 2.0 * signal_span
    )
    half_domain = 0.5 * signal_span + ground_pad
    if int(_cells_per_feature) != _cells_per_feature or _cells_per_feature < 16:
        raise ValueError("Coupled CPW mesh density must be an integer of at least 16.")
    dx_target = min(W, S_ground, S_pair, th) / _cells_per_feature
    count = 1 << int(np.ceil(np.log2(2.0 * half_domain / dx_target)))
    if count > 8192:
        raise ValueError("Coupled CPW aspect ratio exceeds the 8192-cell resolution limit.")
    dx = 2.0 * half_domain / count
    x = (np.arange(count) + 0.5) * dx - half_domain
    left = (x >= -0.5 * S_pair - W) & (x <= -0.5 * S_pair)
    right = (x >= 0.5 * S_pair) & (x <= 0.5 * S_pair + W)
    ground = (x <= -0.5 * S_pair - W - S_ground) | (
        x >= 0.5 * S_pair + W + S_ground
    )
    if min(np.count_nonzero(left), np.count_nonzero(right)) < 8:
        raise ValueError("Coupled CPW conductors are under-resolved.")
    unknown = np.flatnonzero(~(left | right | ground))
    if len(unknown) > 1024:
        raise ValueError(
            "Coupled CPW has more than 1024 free-surface cells; "
            "use a narrower geometry range or lower mesh density."
        )
    signals = np.flatnonzero(left | right)
    signal_voltage = np.ones((len(signals), 2))
    signal_voltage[right[signals], 1] = -1.0
    wave_number = 2.0 * PI * np.fft.rfftfreq(count, d=dx)

    def modal_capacitance(relative_permittivity):
        """Solve free-surface potential and integrate signal charge per metre."""
        kh = wave_number * th
        if has_metal_backside:
            lower = np.empty_like(kh)
            lower[0] = relative_permittivity / th
            lower[1:] = relative_permittivity * wave_number[1:] / np.tanh(kh[1:])
        else:
            tanh_kh = np.tanh(kh)
            lower = relative_permittivity * wave_number * (
                1.0 + relative_permittivity * tanh_kh
            ) / (relative_permittivity + tanh_kh)
        admittance = (wave_number + lower) / (n0 * C0)
        kernel = np.fft.irfft(admittance, n=count)
        free_matrix = kernel[(unknown[:, None] - unknown[None, :]) % count]
        source_matrix = kernel[(unknown[:, None] - signals[None, :]) % count]
        potentials = np.zeros((count, 2))
        potentials[signals] = signal_voltage
        potentials[unknown] = np.linalg.solve(
            free_matrix, -source_matrix @ signal_voltage
        )
        charges = np.fft.irfft(
            admittance[:, None] * np.fft.rfft(potentials, axis=0),
            n=count,
            axis=0,
        )
        return np.sum(charges[left], axis=0) * dx

    capacitance_air = modal_capacitance(1.0)
    capacitance = modal_capacitance(er)
    if not np.all(np.isfinite(capacitance_air)) or not np.all(np.isfinite(capacitance)):
        raise ValueError("Coupled CPW modal capacitance is not finite.")
    if np.any(capacitance_air <= 0.0) or np.any(capacitance <= 0.0):
        raise ValueError("Coupled CPW modal capacitance is not positive.")
    z_even, z_odd = 1.0 / (C0 * np.sqrt(capacitance_air * capacitance))
    return float(2.0 * z_odd), float(0.5 * z_even)


############################################################
#                     STACKUP API VIEWS                    #
############################################################


class _MicrostripAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def z0(
        self,
        w: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """Solve microstrip characteristic impedance on a stackup pair.

        Args:
            w (float): Conductor width in stackup units.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            Single-ended impedance in ohms.
        """
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        return float(
            microstrip_z0_dispersion(
                w * self._pcb.unit, h, ee, f=f0, t=t * self._pcb.unit
            )
        )

    def eeff(
        self,
        w: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """Solve microstrip effective permittivity on a stackup pair.

        Args:
            w (float): Conductor width in stackup units.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            Dimensionless effective relative permittivity.
        """
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        return float(
            microstrip_eeff_dispersion(
                w * self._pcb.unit, h, ee, f=f0, t=t * self._pcb.unit
            )
        )

    def w_for_z0(
        self,
        Z0: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float | None = None,
        w_max: float | None = None,
        n: int = 401,
        incl_dispersion: bool = True,
    ):
        """Inverse microstrip width from target impedance.

        Args:
            Z0 (float): Target impedance in ohms.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.
            w_min (float | None): Optional inverse-search bound in stackup units.
            w_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.
            incl_dispersion (bool): Include the frequency-dispersion correction.

        Returns:
            Solved geometry in stackup units.
        """
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        w_min, w_max = _inverse_bounds_m(w_min, w_max, self._pcb.unit, h)
        if incl_dispersion:
            wm = _scan_inverse(
                Z0,
                lambda ws: microstrip_z0_dispersion(
                    ws, h, ee, f=f0, t=t * self._pcb.unit
                ),
                w_min,
                w_max,
                n,
            )
        else:
            wm = _scan_inverse(
                Z0,
                lambda ws: microstrip_z0(ws, h, ee, t=t * self._pcb.unit),
                w_min,
                w_max,
                n,
            )
        return float(wm / self._pcb.unit)

    def quarter_wave(
        self,
        f: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float | None = None,
        w: float | None = None,
        Z0: float = 50.0,
        t: float = 0.0,
    ):
        """Quarter-wave physical length helper for microstrip.

        Args:
            f (float): Frequency in hertz.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float | None): Frequency in hertz.
            w (float | None): Conductor width in stackup units.
            Z0 (float): Target impedance in ohms.
            t (float): Conductor thickness in stackup units.

        Returns:
            Physical quarter-wave length in stackup units.
        """
        if f0 is None:
            f0 = f
        if w is None:
            w = self.w_for_z0(Z0, layer=layer, ground_layer=ground_layer, f0=f0, t=t)
        eeff = self.eeff(w, layer=layer, ground_layer=ground_layer, f0=f0, t=t)
        return float((C0 / (4.0 * float(f) * np.sqrt(eeff))) / self._pcb.unit)


class _StriplineAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def z0(
        self,
        w: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """Solve centered stripline impedance between two ground layers.

        Args:
            w (float): Conductor width in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            Single-ended impedance in ohms.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        return float(stripline_z0(w * self._pcb.unit, b, ee, t=t * self._pcb.unit))

    def w_for_z0(
        self,
        Z0: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float | None = None,
        w_max: float | None = None,
        n: int = 401,
    ):
        """Inverse stripline width from target impedance.

        Args:
            Z0 (float): Target impedance in ohms.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.
            w_min (float | None): Optional inverse-search bound in stackup units.
            w_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        w_min, w_max = _inverse_bounds_m(w_min, w_max, self._pcb.unit, b)
        wm = _scan_inverse(
            Z0,
            lambda ws: stripline_z0(ws, b, ee, t=t * self._pcb.unit),
            w_min,
            w_max,
            n,
        )
        return float(wm / self._pcb.unit)


class _EdgeCoupledStriplineAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def zodd(
        self,
        w: float,
        s: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
    ):
        """Edge-coupled stripline odd-mode impedance.

        Args:
            w (float): Conductor width in stackup units.
            s (float): Edge gap or coplanar slot in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.

        Returns:
            Odd-mode impedance in ohms.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        return float(
            coupled_stripline_zodd(w * self._pcb.unit, s * self._pcb.unit, b, ee)
        )

    def zdiff(
        self,
        w: float,
        s: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
    ):
        """Edge-coupled stripline differential impedance.

        Args:
            w (float): Conductor width in stackup units.
            s (float): Edge gap or coplanar slot in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.

        Returns:
            Differential impedance in ohms.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        return float(
            coupled_stripline_zdiff(w * self._pcb.unit, s * self._pcb.unit, b, ee)
        )

    def w_for_zdiff(
        self,
        Zdiff: float,
        s: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        w_min: float | None = None,
        w_max: float | None = None,
        n: int = 501,
    ):
        """Inverse edge-coupled stripline width from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            s (float): Edge gap or coplanar slot in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            w_min (float | None): Optional inverse-search bound in stackup units.
            w_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        w_min, w_max = _inverse_bounds_m(w_min, w_max, self._pcb.unit, b)
        wm = _scan_inverse(
            Zdiff,
            lambda ws: coupled_stripline_zdiff(ws, s * self._pcb.unit, b, ee),
            w_min,
            w_max,
            n,
        )
        return float(wm / self._pcb.unit)

    def s_for_zdiff(
        self,
        Zdiff: float,
        w: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        s_min: float | None = None,
        s_max: float | None = None,
        n: int = 501,
    ):
        """Inverse edge-coupled stripline spacing from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            w (float): Conductor width in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            s_min (float | None): Optional inverse-search bound in stackup units.
            s_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        s_min, s_max = _inverse_bounds_m(s_min, s_max, self._pcb.unit, b)
        sm = _scan_inverse(
            Zdiff,
            lambda ss: coupled_stripline_zdiff(w * self._pcb.unit, ss, b, ee),
            s_min,
            s_max,
            n,
        )
        return float(sm / self._pcb.unit)


class _BroadsideCoupledStriplineAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def zdiff_zcm(
        self,
        w: float,
        g: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
    ):
        """Broadside-coupled stripline differential/common-mode impedances.

        Args:
            w (float): Conductor width in stackup units.
            g (float): Broadside spacing in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.

        Returns:
            (differential impedance, common-mode impedance) in ohms.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        zd, zc = broadside_stripline_zdiff_zcm(
            w * self._pcb.unit, g * self._pcb.unit, b, ee
        )
        return float(zd), float(zc)

    def w_for_zdiff(
        self,
        Zdiff: float,
        g: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        w_min: float | None = None,
        w_max: float | None = None,
        n: int = 501,
    ):
        """Inverse broadside stripline width from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            g (float): Broadside spacing in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            w_min (float | None): Optional inverse-search bound in stackup units.
            w_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        w_min, w_max = _inverse_bounds_m(w_min, w_max, self._pcb.unit, b)
        wm = _scan_inverse(
            Zdiff,
            lambda ws: broadside_stripline_zdiff_zcm(ws, g * self._pcb.unit, b, ee)[0],
            w_min,
            w_max,
            n,
        )
        return float(wm / self._pcb.unit)

    def g_for_zdiff(
        self,
        Zdiff: float,
        w: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        g_min: float | None = None,
        g_max: float | None = None,
        n: int = 501,
    ):
        """Inverse broadside stripline spacing from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            w (float): Conductor width in stackup units.
            gnd_top (int): Layer index in the calculator stackup.
            gnd_bot (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            g_min (float | None): Optional inverse-search bound in stackup units.
            g_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        g0, g1 = _inverse_bounds_m(g_min, g_max, self._pcb.unit, b)
        g1 = min(g1, 0.499 * b)
        if g1 <= g0:
            raise ValueError("Broadside gap bounds do not fit between reference planes")

        # Broadside Zdiff(G) is generally non-monotonic over wide ranges.
        # Restrict solve interval to the initial monotonic (increasing) branch.
        gs_probe = np.geomspace(g0, g1, 129)
        zd_probe = np.asarray(
            [
                broadside_stripline_zdiff_zcm(w * self._pcb.unit, float(gm), b, ee)[0]
                for gm in gs_probe
            ],
            dtype=float,
        )
        m = np.isfinite(zd_probe)
        if np.count_nonzero(m) >= 3:
            gp = gs_probe[m]
            zp = zd_probe[m]
            dz = np.diff(zp)
            turn = np.where(dz <= 0.0)[0]
            if turn.size > 0:
                g1 = float(gp[turn[0] + 1])
                if g1 <= g0:
                    g1 = min(float(gp[-1]), max(g0 * 1.0001, g0 + 1e-12))

        wm = w * self._pcb.unit

        def _zd(gs):
            out = np.empty_like(gs, dtype=float)
            for i, gm in enumerate(gs):
                out[i] = broadside_stripline_zdiff_zcm(wm, float(gm), b, ee)[0]
            return out

        gm = _scan_inverse(Zdiff, _zd, g0, g1, n)
        return float(gm / self._pcb.unit)


class _CPWAPI:
    def __init__(self, pcb, has_metal_backside: bool):
        self._pcb = pcb
        self._metal = bool(has_metal_backside)

    def z0(
        self,
        w: float,
        s: float,
        layer: int = -1,
        ref_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """CPW/GCPW characteristic impedance from stackup geometry.

        Args:
            w (float): Conductor width in stackup units.
            s (float): Edge gap or coplanar slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ref_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            Single-ended impedance in ohms.
        """
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
        return float(
            cpw_z0_dispersion(
                w * self._pcb.unit,
                s * self._pcb.unit,
                h,
                ee,
                f=f0,
                t=t * self._pcb.unit,
                has_metal_backside=self._metal,
            )
        )

    def eeff(
        self,
        w: float,
        s: float,
        layer: int = -1,
        ref_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """CPW/GCPW effective permittivity from stackup geometry.

        Args:
            w (float): Conductor width in stackup units.
            s (float): Edge gap or coplanar slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ref_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            Dimensionless effective relative permittivity.
        """
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
        return float(
            cpw_eeff_dispersion(
                w * self._pcb.unit,
                s * self._pcb.unit,
                h,
                ee,
                f=f0,
                t=t * self._pcb.unit,
                has_metal_backside=self._metal,
            )
        )

    def w_for_z0(
        self,
        Z0: float,
        s: float,
        layer: int = -1,
        ref_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float | None = None,
        w_max: float | None = None,
        n: int = 501,
    ):
        """Inverse CPW/GCPW center width from target impedance.

        Args:
            Z0 (float): Target impedance in ohms.
            s (float): Edge gap or coplanar slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ref_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.
            w_min (float | None): Optional inverse-search bound in stackup units.
            w_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
        w_min, w_max = _inverse_bounds_m(w_min, w_max, self._pcb.unit, h)
        wm = _scan_inverse(
            Z0,
            lambda ws: cpw_z0_dispersion(
                ws,
                s * self._pcb.unit,
                h,
                ee,
                f=f0,
                t=t * self._pcb.unit,
                has_metal_backside=self._metal,
            ),
            w_min,
            w_max,
            n,
        )
        return float(wm / self._pcb.unit)


class _EdgeCoupledMicrostripAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def even_odd(
        self,
        w: float,
        s: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """Edge-coupled microstrip even/odd modal impedances.

        Args:
            w (float): Conductor width in stackup units.
            s (float): Edge gap or coplanar slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            (even-mode impedance, odd-mode impedance) in ohms.
        """
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        return coupled_microstrip_z0_even_odd(
            w * self._pcb.unit,
            s * self._pcb.unit,
            h,
            ee,
            t=t * self._pcb.unit,
            f=f0,
        )

    def zdiff_zcm(
        self,
        w: float,
        s: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """Edge-coupled microstrip differential/common-mode impedances.

        Args:
            w (float): Conductor width in stackup units.
            s (float): Edge gap or coplanar slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            (differential impedance, common-mode impedance) in ohms.
        """
        ze, zo = self.even_odd(
            w, s, layer=layer, ground_layer=ground_layer, f0=f0, er=er, t=t
        )
        return float(2.0 * zo), float(0.5 * ze)

    def w_for_zdiff(
        self,
        Zdiff: float,
        s: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float | None = None,
        w_max: float | None = None,
        n: int = 501,
    ):
        """Inverse edge-coupled microstrip width from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            s (float): Edge gap or coplanar slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.
            w_min (float | None): Optional inverse-search bound in stackup units.
            w_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        w_min, w_max = _inverse_bounds_m(w_min, w_max, self._pcb.unit, h)

        def _zd(ws):
            out = np.empty_like(ws, dtype=float)
            sm = s * self._pcb.unit
            tm = t * self._pcb.unit
            for i, wm in enumerate(ws):
                _, zo = coupled_microstrip_z0_even_odd(wm, sm, h, ee, t=tm, f=f0)
                out[i] = 2.0 * zo
            return out

        wm = _scan_inverse(Zdiff, _zd, w_min, w_max, n)
        return float(wm / self._pcb.unit)

    def s_for_zdiff(
        self,
        Zdiff: float,
        w: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        s_min: float | None = None,
        s_max: float | None = None,
        n: int = 501,
    ):
        """Inverse edge-coupled microstrip spacing from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            w (float): Conductor width in stackup units.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.
            s_min (float | None): Optional inverse-search bound in stackup units.
            s_max (float | None): Optional inverse-search bound in stackup units.
            n (int): Inverse-search sample count.

        Returns:
            Solved geometry in stackup units.
        """
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        s_min, s_max = _inverse_bounds_m(s_min, s_max, self._pcb.unit, h)
        wm = w * self._pcb.unit
        tm = t * self._pcb.unit

        def _zd(ss):
            out = np.empty_like(ss, dtype=float)
            for i, sm in enumerate(ss):
                _, zo = coupled_microstrip_z0_even_odd(wm, sm, h, ee, t=tm, f=f0)
                out[i] = 2.0 * zo
            return out

        sm = _scan_inverse(Zdiff, _zd, s_min, s_max, n)
        return float(sm / self._pcb.unit)


class _DifferentialCPWAPI:
    def __init__(self, pcb, has_metal_backside: bool):
        self._pcb = pcb
        self._metal = bool(has_metal_backside)

    def zdiff_zcm(
        self,
        w: float,
        s_pair: float,
        s_ground: float,
        layer: int = -1,
        ref_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        """Differential CPW/DCPWG differential/common-mode impedances.

        Args:
            w (float): Conductor width in stackup units.
            s_pair (float): Gap between the two signal traces in stackup units.
            s_ground (float): Trace-to-lateral-ground slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ref_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.

        Returns:
            Differential and common-mode impedance in ohms.
        """
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
        zd, zc = differential_cpw_zdiff_zcm(
            w * self._pcb.unit,
            s_ground * self._pcb.unit,
            s_pair * self._pcb.unit,
            h,
            ee,
            t=t * self._pcb.unit,
            has_metal_backside=self._metal,
            f=f0,
        )
        return float(zd), float(zc)

    def w_for_zdiff(
        self,
        Zdiff: float,
        s_pair: float,
        s_ground: float,
        layer: int = -1,
        ref_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float | None = None,
        w_max: float | None = None,
        n: int = 31,
    ):
        """Inverse differential CPW/DCPWG width from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            s_pair (float): Gap between the two signal traces in stackup units.
            s_ground (float): Trace-to-lateral-ground slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ref_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.
            w_min (float | None): Optional lower bound; defaults to the smallest
                                  of pair gap, ground gap and slab height.
            w_max (float | None): Optional upper bound; defaults to 2.5 times
                                  slab height. Specify bounds for wider traces.
            n (int): Inverse-search sample count.

        Returns:
            Solved conductor width in stackup units.
        """
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
        default_bounds = w_min is None and w_max is None
        if w_min is None:
            w_min = min(s_pair, s_ground, h / self._pcb.unit)
        if w_max is None:
            w_max = 2.5 * h / self._pcb.unit
        w_min, w_max = _inverse_bounds_m(w_min, w_max, self._pcb.unit, h)
        def _zd(ws):
            out = np.empty_like(ws, dtype=float)
            for i, width in enumerate(ws):
                out[i] = differential_cpw_zdiff_zcm(
                    float(width),
                    s_ground * self._pcb.unit,
                    s_pair * self._pcb.unit,
                    h,
                    ee,
                    t=t * self._pcb.unit,
                    has_metal_backside=self._metal,
                    f=f0,
                )[0]
            return out

        try:
            wm = _scan_inverse(Zdiff, _zd, w_min, w_max, n)
        except ValueError as exc:
            if default_bounds and "outside the achievable range" in str(exc):
                raise ValueError(
                    f"{exc}; the default width range is "
                    f"[{w_min / self._pcb.unit:g}, {w_max / self._pcb.unit:g}] "
                    "stackup units. Set w_min and w_max to search wider traces."
                ) from exc
            raise
        return float(wm / self._pcb.unit)

    def s_for_zdiff(
        self,
        Zdiff: float,
        w: float,
        s_ground: float,
        layer: int = -1,
        ref_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        s_min: float | None = None,
        s_max: float | None = None,
        n: int = 31,
    ):
        """Inverse differential CPW/DCPWG pair spacing from target differential impedance.

        Args:
            Zdiff (float): Target impedance in ohms.
            w (float): Conductor width in stackup units.
            s_ground (float): Trace-to-lateral-ground slot in stackup units.
            layer (int): Layer index in the calculator stackup.
            ref_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            t (float): Conductor thickness in stackup units.
            s_min (float | None): Optional lower bound; defaults to half the
                                  smallest of width, ground gap and slab height.
            s_max (float | None): Optional upper bound; defaults to twice that
                                  smallest dimension. Specify wider pair gaps.
            n (int): Inverse-search sample count.

        Returns:
            Solved pair gap in stackup units.
        """
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
        default_bounds = s_min is None and s_max is None
        if s_min is None:
            s_min = 0.5 * min(w, s_ground, h / self._pcb.unit)
        if s_max is None:
            s_max = 2.0 * min(w, s_ground, h / self._pcb.unit)
        s_min, s_max = _inverse_bounds_m(s_min, s_max, self._pcb.unit, h)
        wm = w * self._pcb.unit
        sgm = s_ground * self._pcb.unit
        tm = t * self._pcb.unit

        def _zd(ss):
            out = np.empty_like(ss, dtype=float)
            for i, sp in enumerate(ss):
                out[i] = differential_cpw_zdiff_zcm(
                    wm,
                    sgm,
                    float(sp),
                    h,
                    ee,
                    t=tm,
                    has_metal_backside=self._metal,
                    f=f0,
                )[0]
            return out

        try:
            sm = _scan_inverse(Zdiff, _zd, s_min, s_max, n)
        except ValueError as exc:
            if default_bounds and "outside the achievable range" in str(exc):
                raise ValueError(
                    f"{exc}; the default pair-gap range is "
                    f"[{s_min / self._pcb.unit:g}, {s_max / self._pcb.unit:g}] "
                    "stackup units. Set s_min and s_max for wider pair gaps."
                ) from exc
            raise
        return float(sm / self._pcb.unit)


class _CoaxAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def z0(self, d_inner: float, d_outer: float, er: float = 1.0):
        """Coax characteristic impedance wrapper in user geometry units.

        Args:
            d_inner (float): Inner conductor diameter in stackup units.
            d_outer (float): Outer conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.

        Returns:
            Single-ended impedance in ohms.
        """
        return float(coax_z0(d_inner * self._pcb.unit, d_outer * self._pcb.unit, er))

    def d_inner_for_z0(self, Z0: float, d_outer: float, er: float = 1.0):
        """Coax inverse inner diameter from target impedance.

        Args:
            Z0 (float): Target impedance in ohms.
            d_outer (float): Outer conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.

        Returns:
            Solved geometry in stackup units.
        """
        di = coax_d_for_z0(Z0, d_outer * self._pcb.unit, er)
        return float(di / self._pcb.unit)

    def cutoff_te(
        self,
        d_inner: float,
        d_outer: float,
        er: float = 1.0,
        mur: float = 1.0,
        n: int = 1,
        m: int = 1,
        exact: bool = True,
    ):
        """Coax TE cutoff frequency wrapper.

        Args:
            d_inner (float): Inner conductor diameter in stackup units.
            d_outer (float): Outer conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.
            n (int): Mode order or index.
            m (int): Mode order or index.
            exact (bool): Solve the modal eigenvalue rather than using the cutoff estimate.

        Returns:
            TE-mode cutoff frequency in hertz.
        """
        di = d_inner * self._pcb.unit
        do = d_outer * self._pcb.unit
        return float(coax_cutoff_te(di, do, er=er, mur=mur, n=n, m=m, exact=exact))

    def cutoff_tm(
        self,
        d_inner: float,
        d_outer: float,
        er: float = 1.0,
        mur: float = 1.0,
        n: int = 0,
        m: int = 1,
        exact: bool = True,
    ):
        """Coax TM cutoff frequency wrapper.

        Args:
            d_inner (float): Inner conductor diameter in stackup units.
            d_outer (float): Outer conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.
            n (int): Mode order or index.
            m (int): Mode order or index.
            exact (bool): Solve the modal eigenvalue rather than using the cutoff estimate.

        Returns:
            TM-mode cutoff frequency in hertz.
        """
        di = d_inner * self._pcb.unit
        do = d_outer * self._pcb.unit
        return float(coax_cutoff_tm(di, do, er=er, mur=mur, n=n, m=m, exact=exact))

    def cutoffs(
        self,
        d_inner: float,
        d_outer: float,
        er: float = 1.0,
        mur: float = 1.0,
        exact: bool = True,
    ):
        """Coax default cutoff pair (TE11, TM01).

        Args:
            d_inner (float): Inner conductor diameter in stackup units.
            d_outer (float): Outer conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.
            exact (bool): Solve the modal eigenvalue rather than using the cutoff estimate.

        Returns:
            (TE11 cutoff, TM01 cutoff) in hertz.
        """
        di = d_inner * self._pcb.unit
        do = d_outer * self._pcb.unit
        return (
            float(coax_cutoff_te(di, do, er=er, mur=mur, n=1, m=1, exact=exact)),
            float(coax_cutoff_tm(di, do, er=er, mur=mur, n=0, m=1, exact=exact)),
        )


class _TwistedPairAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def eeff(
        self,
        d_center: float,
        d_wire: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
        """Twisted-pair effective permittivity wrapper.

        Args:
            d_center (float): Wire centre-to-centre spacing in stackup units.
            d_wire (float): Wire conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.
            er1 (float): Relative permittivity of the surrounding medium.
            twists_per_len (float): Twists per stackup length unit.
            ptfe (bool): Use the PTFE branch of the empirical dielectric model.

        Returns:
            Dimensionless effective relative permittivity.
        """
        return float(
            twisted_pair_eeff(
                d_center * self._pcb.unit,
                d_wire * self._pcb.unit,
                er,
                er1=er1,
                twists_per_len=twists_per_len / self._pcb.unit,
                ptfe=ptfe,
            )
        )

    def z0(
        self,
        d_center: float,
        d_wire: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
        """Twisted-pair impedance wrapper.

        Args:
            d_center (float): Wire centre-to-centre spacing in stackup units.
            d_wire (float): Wire conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.
            er1 (float): Relative permittivity of the surrounding medium.
            twists_per_len (float): Twists per stackup length unit.
            ptfe (bool): Use the PTFE branch of the empirical dielectric model.

        Returns:
            Single-ended impedance in ohms.
        """
        return float(
            twisted_pair_z0(
                d_center * self._pcb.unit,
                d_wire * self._pcb.unit,
                er,
                er1=er1,
                twists_per_len=twists_per_len / self._pcb.unit,
                ptfe=ptfe,
            )
        )

    def d_center_for_z0(
        self,
        z0: float,
        d_wire: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
        """Twisted-pair inverse center spacing from target impedance.

        Args:
            z0 (float): Target impedance in ohms.
            d_wire (float): Wire conductor diameter in stackup units.
            er (float): Relative permittivity of the medium.
            er1 (float): Relative permittivity of the surrounding medium.
            twists_per_len (float): Twists per stackup length unit.
            ptfe (bool): Use the PTFE branch of the empirical dielectric model.

        Returns:
            Solved geometry in stackup units.
        """
        dc = twisted_pair_d_center_for_z0(
            z0=float(z0),
            d_wire=d_wire * self._pcb.unit,
            er=er,
            er1=er1,
            twists_per_len=twists_per_len / self._pcb.unit,
            ptfe=ptfe,
        )
        return float(dc / self._pcb.unit)

    def d_wire_for_z0(
        self,
        z0: float,
        d_center: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
        """Twisted-pair inverse wire diameter from target impedance.

        Args:
            z0 (float): Target impedance in ohms.
            d_center (float): Wire centre-to-centre spacing in stackup units.
            er (float): Relative permittivity of the medium.
            er1 (float): Relative permittivity of the surrounding medium.
            twists_per_len (float): Twists per stackup length unit.
            ptfe (bool): Use the PTFE branch of the empirical dielectric model.

        Returns:
            Solved geometry in stackup units.
        """
        dw = twisted_pair_d_wire_for_z0(
            z0=float(z0),
            d_center=d_center * self._pcb.unit,
            er=er,
            er1=er1,
            twists_per_len=twists_per_len / self._pcb.unit,
            ptfe=ptfe,
        )
        return float(dw / self._pcb.unit)


class _RectangularWaveguideAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    def fc(
        self,
        a: float,
        b: float,
        m: int = 1,
        n: int = 0,
        er: float = 1.0,
        mur: float = 1.0,
    ):
        """Waveguide cutoff frequency wrapper.

        Args:
            a (float): Broad waveguide wall in stackup units.
            b (float): Narrow waveguide wall in stackup units.
            m (int): Mode order or index.
            n (int): Mode order or index.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            Mode cutoff frequency in hertz.
        """
        return float(
            rectwg_fc(a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur)
        )

    def beta(
        self,
        f: float,
        a: float,
        b: float,
        m: int = 1,
        n: int = 0,
        er: float = 1.0,
        mur: float = 1.0,
    ):
        """Waveguide propagation constant wrapper.

        Args:
            f (float): Frequency in hertz.
            a (float): Broad waveguide wall in stackup units.
            b (float): Narrow waveguide wall in stackup units.
            m (int): Mode order or index.
            n (int): Mode order or index.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            Propagation constant in radians per metre.
        """
        return float(
            rectwg_beta(
                f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
            )
        )

    def lambda_g(
        self,
        f: float,
        a: float,
        b: float,
        m: int = 1,
        n: int = 0,
        er: float = 1.0,
        mur: float = 1.0,
    ):
        """Waveguide guided wavelength wrapper.

        Args:
            f (float): Frequency in hertz.
            a (float): Broad waveguide wall in stackup units.
            b (float): Narrow waveguide wall in stackup units.
            m (int): Mode order or index.
            n (int): Mode order or index.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            Guided wavelength in stackup units.
        """
        lg = rectwg_lambda_g(
            f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
        )
        if np.isinf(lg):
            return np.inf
        return float(lg / self._pcb.unit)

    def z_te(
        self,
        f: float,
        a: float,
        b: float,
        m: int = 1,
        n: int = 0,
        er: float = 1.0,
        mur: float = 1.0,
    ):
        """Waveguide TE impedance wrapper.

        Args:
            f (float): Frequency in hertz.
            a (float): Broad waveguide wall in stackup units.
            b (float): Narrow waveguide wall in stackup units.
            m (int): Mode order or index.
            n (int): Mode order or index.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            TE-mode wave impedance in ohms.
        """
        return float(
            rectwg_z_te(
                f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
            )
        )

    def z_tm(
        self,
        f: float,
        a: float,
        b: float,
        m: int = 1,
        n: int = 1,
        er: float = 1.0,
        mur: float = 1.0,
    ):
        """Waveguide TM impedance wrapper.

        Args:
            f (float): Frequency in hertz.
            a (float): Broad waveguide wall in stackup units.
            b (float): Narrow waveguide wall in stackup units.
            m (int): Mode order or index.
            n (int): Mode order or index.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            TM-mode wave impedance in ohms.
        """
        return float(
            rectwg_z_tm(
                f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
            )
        )

    def a_for_fc(self, fc: float, er: float = 1.0, mur: float = 1.0, m: int = 1):
        """Inverse broad wall size from cutoff.

        Args:
            fc (float): Cutoff frequency in hertz.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.
            m (int): Mode order or index.

        Returns:
            Broad-wall dimension in stackup units.
        """
        return float(rectwg_a_for_fc(fc, er=er, mur=mur, m=m) / self._pcb.unit)

    def a_for_z_te10(self, z0: float, f: float, er: float = 1.0, mur: float = 1.0):
        """Inverse TE10 broad wall size from TE impedance.

        Args:
            z0 (float): Target impedance in ohms.
            f (float): Frequency in hertz.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            Broad-wall dimension in stackup units.
        """
        return float(rectwg_te10_a_for_z0(z0, f, er=er, mur=mur) / self._pcb.unit)

    def length_for_angle(
        self,
        angle_rad: float,
        f: float,
        a: float,
        b: float,
        m: int = 1,
        n: int = 0,
        er: float = 1.0,
        mur: float = 1.0,
    ):
        """Physical length for desired phase angle in selected mode.

        Args:
            angle_rad (float): Target phase angle in radians.
            f (float): Frequency in hertz.
            a (float): Broad waveguide wall in stackup units.
            b (float): Narrow waveguide wall in stackup units.
            m (int): Mode order or index.
            n (int): Inverse-search sample count.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            Physical length in stackup units.
        """
        beta = rectwg_beta(
            f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
        )
        if beta <= 0.0:
            return np.inf
        return float((float(angle_rad) / beta) / self._pcb.unit)

    def te10(self, f: float, a: float, b: float, er: float = 1.0, mur: float = 1.0):
        """Convenience TE10 report dict.

        Args:
            f (float): Frequency in hertz.
            a (float): Broad waveguide wall in stackup units.
            b (float): Narrow waveguide wall in stackup units.
            er (float): Relative permittivity of the medium.
            mur (float): Relative permeability.

        Returns:
            Dictionary with TE10 cutoff, impedance, beta, wavelength and propagation flag.
        """
        fc10 = self.fc(a, b, m=1, n=0, er=er, mur=mur)
        z = self.z_te(f, a, b, m=1, n=0, er=er, mur=mur)
        beta = self.beta(f, a, b, m=1, n=0, er=er, mur=mur)
        lg = self.lambda_g(f, a, b, m=1, n=0, er=er, mur=mur)
        return {
            "fc": fc10,
            "z_te": z,
            "beta": beta,
            "lambda_g": lg,
            "propagating": bool(np.isfinite(z) and beta > 0.0),
        }


class PCBCalculator:
    def __init__(self, layers: np.ndarray, materials: list[Material], unit: float):
        """Initialize the stackup-bound calculator namespace.

        Args:
            layers (np.ndarray): Layer z-coordinates in the configured geometry unit.
            materials (list[Material]): One dielectric material for each adjacent layer
                                        interval.
            unit (float): Metres per configured geometry unit.

        Returns:
            None.
        """
        self.layers: np.ndarray = np.asarray(layers, dtype=float)
        self.mat: list[Material] = materials
        self.unit: float = float(unit)

        self.microstrip = _MicrostripAPI(self)
        self.stripline = _StriplineAPI(self)
        self.cpw = _CPWAPI(self, has_metal_backside=False)
        self.gcpw = _CPWAPI(self, has_metal_backside=True)

        self.edge_coupled_microstrip = _EdgeCoupledMicrostripAPI(self)
        self.edge_coupled_stripline = _EdgeCoupledStriplineAPI(self)
        self.broadside_coupled_stripline = _BroadsideCoupledStriplineAPI(self)
        self.coplanar_microstrip_diff = _DifferentialCPWAPI(
            self, has_metal_backside=False
        )
        self.dcpwg = _DifferentialCPWAPI(self, has_metal_backside=True)
        self.coax = _CoaxAPI(self)
        self.twisted_pair = _TwistedPairAPI(self)
        self.rectangular_waveguide = _RectangularWaveguideAPI(self)

        # Backward compatible aliases
        self.coupled_microstrip = self.edge_coupled_microstrip
        self.coupled_stripline = self.edge_coupled_stripline

    def z0(
        self,
        Z0: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        include_dispersion: bool = True,
    ) -> float:
        """Backward-compatible alias for microstrip inverse width solve.

        Args:
            Z0 (float): Target impedance in ohms.
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.
            include_dispersion (bool): Include the frequency-dispersion correction.

        Returns:
            Solved microstrip width in stackup units (legacy alias).
        """
        return self.microstrip.w_for_z0(
            Z0,
            layer=layer,
            ground_layer=ground_layer,
            f0=f0,
            er=er,
            incl_dispersion=include_dispersion,
        )

    def layer_index(self, layer: int) -> int:
        """Normalize positive/negative layer index to absolute index.

        Args:
            layer (int): Layer index in the calculator stackup.

        Returns:
            Absolute zero-based layer index.
        """
        idx = int(layer)
        if idx < 0:
            idx += len(self.layers)
        if idx < 0 or idx >= len(self.layers):
            raise IndexError(f"Layer index out of range: {layer}")
        return idx

    def z(self, layer: int) -> float:
        """Return layer z-coordinate in stackup units.

        Args:
            layer (int): Layer index in the calculator stackup.

        Returns:
            Layer z-coordinate in stackup units.
        """
        return float(self.layers[self.layer_index(layer)])

    def layer_distance(self, a: int, b: int) -> float:
        """Physical distance between two layers.

        Args:
            a (int): Layer index.
            b (int): Layer index.

        Returns:
            Physical separation in metres.
        """
        return abs(self.z(a) - self.z(b)) * self.unit

    def effective_er(
        self, layer: int, ground_layer: int, f0: float, er: float | None = None
    ) -> float:
        """Effective dielectric constant between two layers.

        Args:
            layer (int): Layer index in the calculator stackup.
            ground_layer (int): Layer index in the calculator stackup.
            f0 (float): Frequency in hertz.
            er (float | None): Relative permittivity; overrides stackup material when
                               provided.

        Returns:
            Dimensionless relative permittivity, or ValueError for mixed/unresolved
            dielectrics.
        """
        if er is not None:
            value = float(er)
            if not np.isfinite(value) or value < 1.0:
                raise ValueError("Relative permittivity must be finite and at least 1")
            return value
        i1 = self.layer_index(layer)
        i2 = self.layer_index(ground_layer)
        if i1 == i2:
            raise ValueError("Signal layer and reference layer cannot be the same")
        lo = min(i1, i2)
        hi = max(i1, i2)

        mats = self.mat[lo:hi]
        if len(mats) != hi - lo:
            raise ValueError("Missing dielectric material between selected layers")

        ers = np.asarray([_material_er(mat, f0) for mat in mats], dtype=float)
        ths = np.abs(np.diff(self.layers))[lo:hi]
        if ths.size != ers.size or np.any(~np.isfinite(ths)) or np.any(ths <= 0):
            raise ValueError("Stackup layer spacing must be positive and finite")
        if np.any(~np.isfinite(ers)) or np.any(ers < 1.0):
            raise ValueError("Relative permittivity must be finite and at least 1")
        if not np.allclose(ers, ers[0], rtol=1e-6, atol=0.0):
            raise ValueError("Mixed dielectric intervals need a multilayer field solver or explicit effective-er override")
        return float(ers[0])
