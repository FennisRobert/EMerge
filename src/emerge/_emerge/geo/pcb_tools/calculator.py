# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

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

# Implemented functions in this module (index):
#
# Internal numeric helpers:
# - _asf, _ellipk_agm, _ellip_ratio, _coth, _sech
# - _load_bessel_runtime, _jn, _yn, _jnp, _ynp
# - _material_er
# - _inverse_from_samples, _scan_inverse
# - _odd_even_from_k
# - _cpw_cap_per_len
# - _coax_cutoff_te_approx, _coax_cutoff_tm_approx
# - _coax_mode_char, _bisect_root, _coax_mode_root
#
# Core transmission-line / waveguide formulas:
# - microstrip_z0
# - microstrip_eeff
# - microstrip_eeff_dispersion
# - microstrip_z0_dispersion
# - stripline_z0
# - coupled_stripline_zodd
# - coupled_stripline_zdiff
# - broadside_stripline_zdiff_zcm
# - cpw_z0
# - cpw_eeff
# - cpw_eeff_dispersion
# - cpw_z0_dispersion
# - coax_z0
# - coax_d_for_z0
# - coax_cutoff_te
# - coax_cutoff_tm
# - twisted_pair_eeff
# - twisted_pair_z0
# - twisted_pair_d_center_for_z0
# - twisted_pair_d_wire_for_z0
# - rectwg_fc
# - rectwg_beta
# - rectwg_z_te
# - rectwg_z_tm
# - rectwg_lambda_g
# - rectwg_a_for_fc
# - rectwg_te10_a_for_z0
# - coupled_microstrip_z0_even_odd
# - differential_cpw_zdiff_zcm
#
# Public API namespaces/methods:
# - _MicrostripAPI: z0, eeff, w_for_z0, quarter_wave
# - _StriplineAPI: z0, w_for_z0
# - _EdgeCoupledStriplineAPI: zodd, zdiff, w_for_zdiff, s_for_zdiff
# - _BroadsideCoupledStriplineAPI: zdiff_zcm, w_for_zdiff, g_for_zdiff
# - _CPWAPI: z0, eeff, w_for_z0
# - _EdgeCoupledMicrostripAPI: even_odd, zdiff_zcm, w_for_zdiff, s_for_zdiff
# - _DifferentialCPWAPI: zdiff_zcm, w_for_zdiff, s_for_zdiff
# - _CoaxAPI: z0, d_inner_for_z0, cutoff_te, cutoff_tm, cutoffs
# - _TwistedPairAPI: eeff, z0, d_center_for_z0, d_wire_for_z0
# - _RectangularWaveguideAPI: fc, beta, lambda_g, z_te, z_tm, a_for_fc,
#   a_for_z_te10, length_for_angle, te10
# - PCBCalculator: __init__, z0, layer_index, z, layer_distance, effective_er
#
# Last Cleanup: 2025-01-01
import numpy as np
from scipy.special import jv, yv
from emsutil import Material

n0 = 376.73031366857
PI = np.pi
TAU = 2 * PI
C0 = 299_792_458.0
MU0 = 4e-7 * PI


def _asf(x):
    return np.asarray(x, dtype=float)


def _ellipk_agm(k):
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
    k = np.clip(_asf(k), 0.0, 1.0 - 1e-15)
    kp = np.sqrt(1.0 - k * k)
    return _ellipk_agm(k) / _ellipk_agm(kp)


def _coth(x):
    x = _asf(x)
    s = np.sinh(x)
    s = np.where(np.abs(s) < 1e-30, np.sign(s) * 1e-30 + (s == 0) * 1e-30, s)
    return np.cosh(x) / s


def _sech(x):
    return 1.0 / np.cosh(_asf(x))


def _jn(n: int, x: float) -> float:
    return float(jv(n, x))


def _yn(n: int, x: float) -> float:
    return float(yv(n, x))


def _jnp(n: int, x: float) -> float:
    n = int(n)
    if n == 0:
        return -_jn(1, x)
    return 0.5 * (_jn(n - 1, x) - _jn(n + 1, x))


def _ynp(n: int, x: float) -> float:
    n = int(n)
    if n == 0:
        return -_yn(1, x)
    return 0.5 * (_yn(n - 1, x) - _yn(n + 1, x))


def _material_er(mat: Material, f0: float) -> float:
    er = getattr(mat, "er", None)
    if er is None:
        return 1.0
    if hasattr(er, "scalar"):
        return float(er.scalar(f0))
    if callable(er):
        return float(er(f0))
    return float(er)


def _inverse_from_samples(target: float, xs, ys) -> float:
    x = _asf(xs)
    y = _asf(ys)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        raise ValueError("No finite samples available for inverse solve")
    if x.size == 1:
        return float(x[0])

    dy = np.diff(y)
    if np.all(dy >= 0.0):
        yk, xk = y, x
    elif np.all(dy <= 0.0):
        yk, xk = y[::-1], x[::-1]
    else:
        return float(x[np.argmin(np.abs(y - target))])

    lo = float(min(yk[0], yk[-1]))
    hi = float(max(yk[0], yk[-1]))
    tgt = float(np.clip(target, lo, hi))
    return float(np.interp(tgt, yk, xk))


def _scan_inverse(target: float, fn, x_min: float, x_max: float, n: int = 501) -> float:
    x0 = max(float(x_min), 1e-15)
    x1 = max(float(x_max), x0 * 1.00001)
    xs = np.geomspace(x0, x1, int(n))
    ys = _asf(fn(xs))
    x_est = _inverse_from_samples(target, xs, ys)

    m = np.isfinite(xs) & np.isfinite(ys)
    xk = _asf(xs)[m]
    yk = _asf(ys)[m]
    if xk.size < 2:
        return float(x_est)

    d = yk - float(target)
    crossings = np.where((d[:-1] == 0.0) | (d[1:] == 0.0) | (d[:-1] * d[1:] < 0.0))[0]
    if crossings.size == 0:
        return float(x_est)

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
    k = np.clip(_asf(k), 0.0, 0.95)
    zo = _asf(z0) * np.sqrt((1.0 - k) / (1.0 + k))
    ze = _asf(z0) * np.sqrt((1.0 + k) / (1.0 - k))
    return ze, zo


def microstrip_z0(W: float, th: float, er: float, t: float = 0.0):
    """Microstrip characteristic impedance (quasi-static).

    Args:
        W (float): The trace width [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.

    Returns:
        np.ndarray: The characteristic impedance [Ohm]
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

    eeff = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 / np.sqrt(1.0 + 12.0 / u))
    eeff = eeff + np.where(u < 1.0, 0.04 * (1.0 - u) ** 2 * (er - 1.0) / 2.0, 0.0)

    return np.where(
        u <= 1.0,
        (60.0 / np.sqrt(eeff)) * np.log(8.0 / u + 0.25 * u),
        (120.0 * PI / np.sqrt(eeff)) / (u + 1.393 + 0.667 * np.log(u + 1.444)),
    )


def microstrip_eeff(W: float, th: float, er: float, t: float = 0.0):
    """Microstrip effective permittivity (quasi-static).

    Args:
        W (float): The trace width [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.

    Returns:
        np.ndarray: The effective relative permittivity
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

    eeff = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 / np.sqrt(1.0 + 12.0 / u))
    eeff = eeff + np.where(u < 1.0, 0.04 * (1.0 - u) ** 2 * (er - 1.0) / 2.0, 0.0)
    return eeff


def microstrip_eeff_dispersion(
    W: float, th: float, er: float, f: float, t: float = 0.0
):
    """Microstrip effective permittivity with frequency dispersion (Kirschning/Jansen).

    Args:
        W (float): The trace width [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        f (float): The frequency [Hz]
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.

    Returns:
        np.ndarray: The effective relative permittivity at f
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
    """Microstrip characteristic impedance with frequency dispersion (Kirschning/Jansen).

    Args:
        W (float): The trace width [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        f (float): The frequency [Hz]
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.

    Returns:
        np.ndarray: The characteristic impedance [Ohm] at f
    """
    W = _asf(W)
    h = float(th)
    f = float(f)
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
        1.0 - np.exp(-0.004625 * r3 * er * 1.674) * (fn / 18.365) ** 2.745
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
    d = np.power(np.maximum(r13 / np.maximum(r14, 1e-30), 1e-30), r17)
    return z0_0 * d


def stripline_z0(W: float, b: float, er: float, t: float = 0.0):
    """Centered stripline characteristic impedance.

    Args:
        W (float): The trace width [m]
        b (float): The ground-to-ground spacing (cavity height) [m]
        er (float): The relative permittivity
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.

    Returns:
        np.ndarray: The characteristic impedance [Ohm]
    """
    W = _asf(W)
    b = float(b)
    t = float(t)

    if t <= 0.0:
        x = PI * W / (2.0 * b)
        k = _sech(x)
        return (n0 / (4.0 * np.sqrt(er))) * _ellip_ratio(k)

    t = min(t, 0.99 * b)
    x = np.clip(t / b, 1e-15, 0.99)
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
    """Edge-coupled stripline odd-mode impedance.

    Args:
        W (float): The trace width [m]
        S (float): The edge spacing [m]
        b (float): The ground-to-ground spacing (cavity height) [m]
        er (float): The relative permittivity

    Returns:
        np.ndarray: The odd-mode impedance [Ohm]
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
    """Edge-coupled stripline differential impedance.

    Args:
        W (float): The trace width [m]
        S (float): The edge spacing [m]
        b (float): The ground-to-ground spacing (cavity height) [m]
        er (float): The relative permittivity

    Returns:
        np.ndarray: The differential impedance [Ohm]
    """
    return 2.0 * coupled_stripline_zodd(W, S, b, er)


def broadside_stripline_zdiff_zcm(W: float, G: float, b: float, er: float):
    # Full Cohn broadside-coupled stripline (zero-thickness conductors):
    #   Z0e = (188.3/sqrt(er)) * K(k')/K(k)
    #   Z0o = (296.1*s)/(sqrt(er)*atanh(k))
    # with implicit relation for width ratio (w = W/b, s = G/b):
    #   w = (2/pi) * atanh(R) - s * atanh(R/k)
    #   R = sqrt((k - s) / (1/k - s))
    """Broadside-coupled stripline differential and common-mode impedances (Cohn, zero-thickness strips).

    Args:
        W (float): The trace width [m]
        G (float): The broadside gap between the strips [m]
        b (float): The ground-to-ground spacing (cavity height) [m]
        er (float): The relative permittivity

    Returns:
        tuple: The differential and common-mode impedances (Zdiff, Zcm) [Ohm]
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
        return (2.0 / PI) * np.arctanh(r) - s * np.arctanh(rk)

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
            ks = np.linspace(lo, hi, 2001)
            fs = np.asarray([_w_from_k(float(kk)) - wratio for kk in ks], dtype=float)
            i = int(np.argmin(np.abs(fs)))
            return float(ks[i])
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


def cpw_z0(
    W: float,
    S: float,
    th: float,
    er: float,
    t: float = 0.0,
    has_metal_backside: bool = False,
):
    """CPW/GCPW characteristic impedance (quasi-static).

    Args:
        W (float): The trace width [m]
        S (float): The slot width between the center conductor and the ground [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.
        has_metal_backside (bool, optional): Model a grounded CPW (GCPW) with a metal backside. Defaults to False.

    Returns:
        np.ndarray: The characteristic impedance [Ohm]
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
    """CPW/GCPW effective permittivity (quasi-static).

    Args:
        W (float): The trace width [m]
        S (float): The slot width between the center conductor and the ground [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.
        has_metal_backside (bool, optional): Model a grounded CPW (GCPW) with a metal backside. Defaults to False.

    Returns:
        np.ndarray: The effective relative permittivity
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
    """CPW/GCPW effective permittivity with Qucs-style frequency dispersion.

    Args:
        W (float): The trace width [m]
        S (float): The slot width between the center conductor and the ground [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        f (float): The frequency [Hz]
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.
        has_metal_backside (bool, optional): Model a grounded CPW (GCPW) with a metal backside. Defaults to False.

    Returns:
        np.ndarray: The effective relative permittivity at f
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
    """CPW/GCPW characteristic impedance with Qucs-style frequency dispersion.

    Args:
        W (float): The trace width [m]
        S (float): The slot width between the center conductor and the ground [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        f (float): The frequency [Hz]
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.
        has_metal_backside (bool, optional): Model a grounded CPW (GCPW) with a metal backside. Defaults to False.

    Returns:
        np.ndarray: The characteristic impedance [Ohm] at f
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


def coax_z0(d_inner: float, d_outer: float, er: float):
    """Coax characteristic impedance.

    Args:
        d_inner (float): The inner conductor diameter [m]
        d_outer (float): The outer conductor diameter [m]
        er (float): The relative permittivity

    Returns:
        np.ndarray: The characteristic impedance [Ohm]
    """
    d_inner = _asf(d_inner)
    d_outer = _asf(d_outer)
    return (n0 / (2.0 * PI * np.sqrt(er))) * np.log(d_outer / d_inner)


def coax_d_for_z0(Z0: float, d_outer: float, er: float):
    """Coax inner conductor diameter for a target impedance.

    Args:
        Z0 (float): The target impedance [Ohm]
        d_outer (float): The outer conductor diameter [m]
        er (float): The relative permittivity

    Returns:
        float: The inner conductor diameter [m]
    """
    return float(d_outer) / np.exp(2.0 * PI * np.sqrt(er) * float(Z0) / n0)


def _coax_cutoff_te_approx(
    d_inner: float, d_outer: float, er: float = 1.0, mur: float = 1.0
):
    return C0 / (
        PI * (float(d_outer) + float(d_inner)) * np.sqrt(float(er) * float(mur))
    )


def _coax_cutoff_tm_approx(
    d_inner: float, d_outer: float, er: float = 1.0, mur: float = 1.0
):
    return C0 / (
        2.0 * (float(d_outer) - float(d_inner)) * np.sqrt(float(er) * float(mur))
    )


def _coax_mode_char(mode: str, n: int, x: float, ratio: float) -> float:
    xa = float(x)
    xb = float(ratio) * xa
    if mode == "tm":
        return _jn(n, xa) * _yn(n, xb) - _yn(n, xa) * _jn(n, xb)
    if mode == "te":
        return _jnp(n, xa) * _ynp(n, xb) - _ynp(n, xa) * _jnp(n, xb)
    raise ValueError("mode must be 'te' or 'tm'.")


def _bisect_root(fn, x0: float, x1: float, iters: int = 80) -> float:
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
    fa, fb = f0, f1
    for _ in range(int(iters)):
        m = 0.5 * (a + b)
        fm = float(fn(m))
        if not np.isfinite(fm):
            break
        if abs(fm) < 1e-13:
            return float(m)
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(0.5 * (a + b))


def _coax_mode_root(n: int, m: int, d_inner: float, d_outer: float, mode: str):
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
    """Coax TE(n,m) mode cutoff frequency.

    Args:
        d_inner (float): The inner conductor diameter [m]
        d_outer (float): The outer conductor diameter [m]
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.
        n (int, optional): The azimuthal mode index. Defaults to 1.
        m (int, optional): The radial mode index. Defaults to 1.
        exact (bool, optional): Solve for the exact Bessel root instead of using the approximation. Defaults to True.

    Returns:
        float: The cutoff frequency [Hz]
    """
    if not exact:
        return _coax_cutoff_te_approx(d_inner, d_outer, er=er, mur=mur)
    try:
        kc = _coax_mode_root(n=n, m=m, d_inner=d_inner, d_outer=d_outer, mode="te")
        return C0 * kc / (2.0 * PI * np.sqrt(float(er) * float(mur)))
    except Exception:
        return _coax_cutoff_te_approx(d_inner, d_outer, er=er, mur=mur)


def coax_cutoff_tm(
    d_inner: float,
    d_outer: float,
    er: float = 1.0,
    mur: float = 1.0,
    n: int = 0,
    m: int = 1,
    exact: bool = True,
):
    """Coax TM(n,m) mode cutoff frequency.

    Args:
        d_inner (float): The inner conductor diameter [m]
        d_outer (float): The outer conductor diameter [m]
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.
        n (int, optional): The azimuthal mode index. Defaults to 0.
        m (int, optional): The radial mode index. Defaults to 1.
        exact (bool, optional): Solve for the exact Bessel root instead of using the approximation. Defaults to True.

    Returns:
        float: The cutoff frequency [Hz]
    """
    if not exact:
        return _coax_cutoff_tm_approx(d_inner, d_outer, er=er, mur=mur)
    try:
        kc = _coax_mode_root(n=n, m=m, d_inner=d_inner, d_outer=d_outer, mode="tm")
        return C0 * kc / (2.0 * PI * np.sqrt(float(er) * float(mur)))
    except Exception:
        return _coax_cutoff_tm_approx(d_inner, d_outer, er=er, mur=mur)


def twisted_pair_eeff(
    d_center: float,
    d_wire: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Twisted-pair effective permittivity.

    Args:
        d_center (float): The center-to-center wire spacing [m]
        d_wire (float): The wire diameter [m]
        er (float): The relative permittivity of the wire insulation
        er1 (float, optional): The relative permittivity of the surrounding medium. Defaults to 1.0.
        twists_per_len (float, optional): The number of twists per meter. Defaults to 0.0.
        ptfe (bool, optional): Use the PTFE insulation correction. Defaults to False.

    Returns:
        float: The effective relative permittivity
    """
    d_center = float(d_center)
    d_wire = float(d_wire)
    if d_center <= d_wire:
        raise ValueError("d_center must be greater than d_wire for twisted pair.")
    theta = np.arctan(float(twists_per_len) * PI * d_center)
    q = 0.001 if ptfe else 0.25 + 0.0004 * theta * theta
    return float(er1 + q * (er - er1))


def twisted_pair_z0(
    d_center: float,
    d_wire: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Twisted-pair characteristic impedance.

    Args:
        d_center (float): The center-to-center wire spacing [m]
        d_wire (float): The wire diameter [m]
        er (float): The relative permittivity of the wire insulation
        er1 (float, optional): The relative permittivity of the surrounding medium. Defaults to 1.0.
        twists_per_len (float, optional): The number of twists per meter. Defaults to 0.0.
        ptfe (bool, optional): Use the PTFE insulation correction. Defaults to False.

    Returns:
        float: The characteristic impedance [Ohm]
    """
    eeff = twisted_pair_eeff(
        d_center, d_wire, er, er1=er1, twists_per_len=twists_per_len, ptfe=ptfe
    )
    arg = max(float(d_center) / float(d_wire), 1.0 + 1e-12)
    return n0 / (PI * np.sqrt(eeff)) * np.arccosh(arg)


def twisted_pair_d_center_for_z0(
    z0: float,
    d_wire: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Twisted-pair center spacing for a target impedance.

    Args:
        z0 (float): The target impedance [Ohm]
        d_wire (float): The wire diameter [m]
        er (float): The relative permittivity of the wire insulation
        er1 (float, optional): The relative permittivity of the surrounding medium. Defaults to 1.0.
        twists_per_len (float, optional): The number of twists per meter. Defaults to 0.0.
        ptfe (bool, optional): Use the PTFE insulation correction. Defaults to False.

    Returns:
        float: The center-to-center spacing [m]
    """
    eeff = twisted_pair_eeff(
        d_center=max(2.0 * float(d_wire), float(d_wire) + 1e-12),
        d_wire=d_wire,
        er=er,
        er1=er1,
        twists_per_len=twists_per_len,
        ptfe=ptfe,
    )
    k = np.cosh(PI * float(z0) * np.sqrt(eeff) / n0)
    return float(float(d_wire) * k)


def twisted_pair_d_wire_for_z0(
    z0: float,
    d_center: float,
    er: float,
    er1: float = 1.0,
    twists_per_len: float = 0.0,
    ptfe: bool = False,
):
    """Twisted-pair wire diameter for a target impedance.

    Args:
        z0 (float): The target impedance [Ohm]
        d_center (float): The center-to-center wire spacing [m]
        er (float): The relative permittivity of the wire insulation
        er1 (float, optional): The relative permittivity of the surrounding medium. Defaults to 1.0.
        twists_per_len (float, optional): The number of twists per meter. Defaults to 0.0.
        ptfe (bool, optional): Use the PTFE insulation correction. Defaults to False.

    Returns:
        float: The wire diameter [m]
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


def rectwg_fc(
    a: float, b: float, m: int = 1, n: int = 0, er: float = 1.0, mur: float = 1.0
):
    """Rectangular waveguide cutoff frequency of mode (m, n).

    Args:
        a (float): The broad wall width [m]
        b (float): The narrow wall height [m]
        m (int, optional): The first mode index. Defaults to 1.
        n (int, optional): The second mode index. Defaults to 0.
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.

    Returns:
        float: The cutoff frequency [Hz]
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
    """Rectangular waveguide propagation constant of mode (m, n).

    Args:
        f (float): The frequency [Hz]
        a (float): The broad wall width [m]
        b (float): The narrow wall height [m]
        m (int, optional): The first mode index. Defaults to 1.
        n (int, optional): The second mode index. Defaults to 0.
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.

    Returns:
        float: The propagation constant [rad/m], 0 below cutoff
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
    """Rectangular waveguide TE mode wave impedance.

    Args:
        f (float): The frequency [Hz]
        a (float): The broad wall width [m]
        b (float): The narrow wall height [m]
        m (int, optional): The first mode index. Defaults to 1.
        n (int, optional): The second mode index. Defaults to 0.
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.

    Returns:
        float: The wave impedance [Ohm], inf below cutoff
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
    """Rectangular waveguide TM mode wave impedance.

    Args:
        f (float): The frequency [Hz]
        a (float): The broad wall width [m]
        b (float): The narrow wall height [m]
        m (int, optional): The first mode index. Defaults to 1.
        n (int, optional): The second mode index. Defaults to 1.
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.

    Returns:
        float: The wave impedance [Ohm], 0 below cutoff
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
    """Rectangular waveguide guided wavelength.

    Args:
        f (float): The frequency [Hz]
        a (float): The broad wall width [m]
        b (float): The narrow wall height [m]
        m (int, optional): The first mode index. Defaults to 1.
        n (int, optional): The second mode index. Defaults to 0.
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.

    Returns:
        float: The guided wavelength [m], inf below cutoff
    """
    beta = rectwg_beta(f, a, b, m=m, n=n, er=er, mur=mur)
    if beta <= 0.0:
        return np.inf
    return float(TAU / beta)


def rectwg_a_for_fc(fc: float, er: float = 1.0, mur: float = 1.0, m: int = 1):
    """Rectangular waveguide broad wall width for a target cutoff frequency.

    Args:
        fc (float): The cutoff frequency [Hz]
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.
        m (int, optional): The mode index along the broad wall. Defaults to 1.

    Returns:
        float: The broad wall width [m]
    """
    fc = float(fc)
    if fc <= 0.0 or m <= 0:
        raise ValueError("fc and mode index m must be > 0.")
    return float(m * C0 / (2.0 * fc * np.sqrt(float(er) * float(mur))))


def rectwg_te10_a_for_z0(z0: float, f: float, er: float = 1.0, mur: float = 1.0):
    """Rectangular waveguide broad wall width for a target TE10 wave impedance.

    Args:
        z0 (float): The target TE10 wave impedance [Ohm]
        f (float): The frequency [Hz]
        er (float, optional): The relative permittivity. Defaults to 1.0.
        mur (float, optional): The relative permeability. Defaults to 1.0.

    Returns:
        float: The broad wall width [m]
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


def coupled_microstrip_z0_even_odd(
    W: float,
    S: float,
    th: float,
    er: float,
    t: float = 0.0,
    f: float | None = None,
):
    """Coupled microstrip even/odd mode impedances (Kirschning/Jansen) with optional frequency dispersion.

    Args:
        W (float): The trace width [m]
        S (float): The edge spacing [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.
        f (float | None, optional): The frequency [Hz] for dispersion, None for the quasi-static result. Defaults to None.

    Returns:
        tuple[float, float]: The even and odd mode impedances (Zeven, Zodd) [Ohm]
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
    p5 = 0.334 * np.exp(-3.3 * np.power(er / 15.0, 3.0)) + 0.746
    p6 = p5 * np.exp(-np.power(fn / 18.0, 0.368))
    p7 = 1.0 + 4.069 * p6 * np.power(g, 0.479) * np.exp(
        -1.347 * np.power(g, 0.595) - 0.17 * np.power(g, 2.5)
    )
    fe = p1 * p2 * np.power(np.maximum((p3 * p4 + 0.1844 * p7) * fn, 1e-30), 1.5763)
    ee_e = er - (er - ee_e0) / (1.0 + fe)

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

    z_even = (
        z_even_0
        * np.power(0.9408 * np.power(ee_single_f, ce) - 0.9603, q0)
        / np.power((0.9408 - de) * np.power(ee_single_0, ce) - 0.9603, q0)
    )

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
):
    """Differential CPW/DCPWG differential and common-mode impedances.

    Args:
        W (float): The trace width [m]
        S_ground (float): The trace-to-ground slot width [m]
        S_pair (float): The gap between the two traces [m]
        th (float): The substrate height [m]
        er (float): The relative permittivity
        t (float, optional): The conductor thickness [m]. Defaults to 0.0.
        has_metal_backside (bool, optional): Model a grounded CPW (GCPW) with a metal backside. Defaults to False.
        f (float | None, optional): The frequency [Hz] for dispersion, None for the quasi-static result. Defaults to None.

    Returns:
        tuple: The differential and common-mode impedances (Zdiff, Zcm) [Ohm]
    """
    w = _asf(W)
    sg = float(S_ground)
    sp = float(S_pair)
    h = float(th)
    if np.any(w <= 0.0) or sg <= 0.0 or sp <= 0.0 or h <= 0.0:
        raise ValueError(
            "W, S_ground, S_pair and th must be > 0 for differential CPW/DCPWG."
        )

    # Quasi-static decomposition:
    # - even mode: pair gap carries negligible E-field -> dominated by outer CPW slots
    # - odd mode: adds inner-slot capacitance between the two traces
    c_out, c_air_out, z_out = _cpw_cap_per_len(
        w, sg, h, er, t=t, has_metal_backside=has_metal_backside, f=f
    )
    c_in, c_air_in, _ = _cpw_cap_per_len(
        w, 0.5 * sp, h, er, t=t, has_metal_backside=has_metal_backside, f=f
    )

    c_odd = c_out + c_in
    c_air_odd = c_air_out + c_air_in
    z_odd = 1.0 / (C0 * np.sqrt(np.maximum(c_odd * c_air_odd, 1e-30)))

    z_even = z_out
    return 2.0 * z_odd, 0.5 * z_even


class _MicrostripAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    # Solve microstrip characteristic impedance on a stackup pair.
    # Args: w width [unit], layer/ground_layer indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def z0(
        self,
        w: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        return float(
            microstrip_z0_dispersion(
                w * self._pcb.unit, h, ee, f=f0, t=t * self._pcb.unit
            )
        )

    # Solve microstrip effective permittivity on a stackup pair.
    # Args: w width [unit], layer/ground_layer indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def eeff(
        self,
        w: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        return float(
            microstrip_eeff_dispersion(
                w * self._pcb.unit, h, ee, f=f0, t=t * self._pcb.unit
            )
        )

    # Inverse microstrip width from target impedance.
    # Args: Z0 target impedance, layer/ground_layer indices, f0 frequency [Hz], er override, t thickness [unit], w_min/w_max search bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def w_for_z0(
        self,
        Z0: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float = 1e-6,
        w_max: float = 1e-1,
        n: int = 401,
        incl_dispersion: bool = True,
    ):
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
        w_min = 0.1 * h
        w_max = 10 * h
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

    # Quarter-wave physical length helper for microstrip.
    # Args: f design frequency [Hz], layer/ground_layer indices, f0 material/model frequency [Hz], w optional fixed width [unit], Z0 target impedance, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        if f0 is None:
            f0 = f
        if w is None:
            w = self.w_for_z0(Z0, layer=layer, ground_layer=ground_layer, f0=f0, t=t)
        eeff = self.eeff(w, layer=layer, ground_layer=ground_layer, f0=f0, t=t)
        return float((C0 / (4.0 * float(f) * np.sqrt(eeff))) / self._pcb.unit)


class _StriplineAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    # Solve centered stripline impedance between two ground layers.
    # Args: w width [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def z0(
        self,
        w: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        return float(stripline_z0(w * self._pcb.unit, b, ee, t=t * self._pcb.unit))

    # Inverse stripline width from target impedance.
    # Args: Z0 target impedance, gnd_top/gnd_bot indices, f0 frequency [Hz], er override, t thickness [unit], w_min/w_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def w_for_z0(
        self,
        Z0: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float = 1e-6,
        w_max: float = 1e-1,
        n: int = 401,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
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

    # Edge-coupled stripline odd-mode impedance.
    # Args: w width [unit], s edge spacing [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override dielectric.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def zodd(
        self,
        w: float,
        s: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        return float(
            coupled_stripline_zodd(w * self._pcb.unit, s * self._pcb.unit, b, ee)
        )

    # Edge-coupled stripline differential impedance.
    # Args: w width [unit], s edge spacing [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override dielectric.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def zdiff(
        self,
        w: float,
        s: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        return float(
            coupled_stripline_zdiff(w * self._pcb.unit, s * self._pcb.unit, b, ee)
        )

    # Inverse edge-coupled stripline width from target differential impedance.
    # Args: Zdiff target differential impedance, s fixed spacing [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override, w_min/w_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def w_for_zdiff(
        self,
        Zdiff: float,
        s: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        w_min: float = 1e-6,
        w_max: float = 1e-1,
        n: int = 501,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        wm = _scan_inverse(
            Zdiff,
            lambda ws: coupled_stripline_zdiff(ws, s * self._pcb.unit, b, ee),
            w_min,
            w_max,
            n,
        )
        return float(wm / self._pcb.unit)

    # Inverse edge-coupled stripline spacing from target differential impedance.
    # Args: Zdiff target differential impedance, w fixed width [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override, s_min/s_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def s_for_zdiff(
        self,
        Zdiff: float,
        w: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        s_min: float = 1e-6,
        s_max: float = 1e-1,
        n: int = 501,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
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

    # Broadside-coupled stripline differential/common-mode impedances.
    # Args: w strip width [unit], g broadside spacing [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override dielectric.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def zdiff_zcm(
        self,
        w: float,
        g: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        zd, zc = broadside_stripline_zdiff_zcm(
            w * self._pcb.unit, g * self._pcb.unit, b, ee
        )
        return float(zd), float(zc)

    # Inverse broadside stripline width from target differential impedance.
    # Args: Zdiff target differential impedance, g fixed broadside spacing [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override, w_min/w_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def w_for_zdiff(
        self,
        Zdiff: float,
        g: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        w_min: float = 1e-6,
        w_max: float = 1e-1,
        n: int = 501,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        wm = _scan_inverse(
            Zdiff,
            lambda ws: broadside_stripline_zdiff_zcm(ws, g * self._pcb.unit, b, ee)[0],
            w_min,
            w_max,
            n,
        )
        return float(wm / self._pcb.unit)

    # Inverse broadside stripline spacing from target differential impedance.
    # Args: Zdiff target differential impedance, w fixed width [unit], gnd_top/gnd_bot indices, f0 frequency [Hz], er override, g_min/g_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def g_for_zdiff(
        self,
        Zdiff: float,
        w: float,
        gnd_top: int,
        gnd_bot: int,
        f0: float = 1e9,
        er: float | None = None,
        g_min: float = 1e-6,
        g_max: float = 1e-1,
        n: int = 501,
    ):
        b = self._pcb.layer_distance(gnd_top, gnd_bot)
        ee = self._pcb.effective_er(gnd_top, gnd_bot, f0, er=er)
        g0 = max(float(g_min), 1e-15)
        g1 = min(float(g_max), 0.499 * b)
        if g1 <= g0:
            g1 = max(g0 * 1.0001, min(0.499 * b, g0 * 10.0))

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

    # CPW/GCPW characteristic impedance from stackup geometry.
    # Args: w center width [unit], s slot [unit], layer/ref_layer indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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

    # CPW/GCPW effective permittivity from stackup geometry.
    # Args: w center width [unit], s slot [unit], layer/ref_layer indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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

    # Inverse CPW/GCPW center width from target impedance.
    # Args: Z0 target impedance, s fixed slot [unit], layer/ref_layer indices, f0 frequency [Hz], er override, t thickness [unit], w_min/w_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def w_for_z0(
        self,
        Z0: float,
        s: float,
        layer: int = -1,
        ref_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float = 1e-6,
        w_max: float = 1e-1,
        n: int = 501,
    ):
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
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

    # Edge-coupled microstrip even/odd modal impedances.
    # Args: w width [unit], s edge spacing [unit], layer/ground_layer indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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

    # Edge-coupled microstrip differential/common-mode impedances.
    # Args: w width [unit], s edge spacing [unit], layer/ground_layer indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        ze, zo = self.even_odd(
            w, s, layer=layer, ground_layer=ground_layer, f0=f0, er=er, t=t
        )
        return float(2.0 * zo), float(0.5 * ze)

    # Inverse edge-coupled microstrip width from target differential impedance.
    # Args: Zdiff target differential impedance, s fixed spacing [unit], layer/ground_layer indices, f0 frequency [Hz], er override, t thickness [unit], w_min/w_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def w_for_zdiff(
        self,
        Zdiff: float,
        s: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        w_min: float = 1e-6,
        w_max: float = 1e-1,
        n: int = 501,
    ):
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)

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

    # Inverse edge-coupled microstrip spacing from target differential impedance.
    # Args: Zdiff target differential impedance, w fixed width [unit], layer/ground_layer indices, f0 frequency [Hz], er override, t thickness [unit], s_min/s_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def s_for_zdiff(
        self,
        Zdiff: float,
        w: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        t: float = 0.0,
        s_min: float = 1e-6,
        s_max: float = 1e-1,
        n: int = 501,
    ):
        h = self._pcb.layer_distance(layer, ground_layer)
        ee = self._pcb.effective_er(layer, ground_layer, f0, er=er)
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

    # Differential CPW/DCPWG differential/common-mode impedances.
    # Args: w width [unit], s_pair pair gap [unit], s_ground trace-to-ground slot [unit], layer/ref_layer indices, f0 frequency [Hz], er override dielectric, t thickness [unit].
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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

    # Inverse differential CPW/DCPWG width from target differential impedance.
    # Args: Zdiff target differential impedance, s_pair fixed pair gap [unit], s_ground fixed trace-to-ground slot [unit], layer/ref_layer indices, f0 frequency [Hz], er override, t thickness [unit], w_min/w_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        w_min: float = 1e-6,
        w_max: float = 1e-1,
        n: int = 501,
    ):
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
        wm = _scan_inverse(
            Zdiff,
            lambda ws: differential_cpw_zdiff_zcm(
                ws,
                s_ground * self._pcb.unit,
                s_pair * self._pcb.unit,
                h,
                ee,
                t=t * self._pcb.unit,
                has_metal_backside=self._metal,
                f=f0,
            )[0],
            w_min,
            w_max,
            n,
        )
        return float(wm / self._pcb.unit)

    # Inverse differential CPW/DCPWG pair spacing from target differential impedance.
    # Args: Zdiff target differential impedance, w fixed width [unit], s_ground fixed trace-to-ground slot [unit], layer/ref_layer indices, f0 frequency [Hz], er override, t thickness [unit], s_min/s_max bounds [unit], n sample count.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        s_min: float = 1e-6,
        s_max: float = 1e-1,
        n: int = 501,
    ):
        h = self._pcb.layer_distance(layer, ref_layer)
        ee = self._pcb.effective_er(layer, ref_layer, f0, er=er)
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

        sm = _scan_inverse(Zdiff, _zd, s_min, s_max, n)
        return float(sm / self._pcb.unit)


class _CoaxAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    # Coax characteristic impedance wrapper in user geometry units.
    # Args: d_inner/d_outer diameters [unit], er relative permittivity.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def z0(self, d_inner: float, d_outer: float, er: float = 1.0):
        return float(coax_z0(d_inner * self._pcb.unit, d_outer * self._pcb.unit, er))

    # Coax inverse inner diameter from target impedance.
    # Args: Z0 target impedance, d_outer outer diameter [unit], er relative permittivity.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def d_inner_for_z0(self, Z0: float, d_outer: float, er: float = 1.0):
        di = coax_d_for_z0(Z0, d_outer * self._pcb.unit, er)
        return float(di / self._pcb.unit)

    # Coax TE cutoff frequency wrapper.
    # Args: d_inner/d_outer diameters [unit], er/mur medium constants, n/m mode indices, exact exact-root flag.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        di = d_inner * self._pcb.unit
        do = d_outer * self._pcb.unit
        return float(coax_cutoff_te(di, do, er=er, mur=mur, n=n, m=m, exact=exact))

    # Coax TM cutoff frequency wrapper.
    # Args: d_inner/d_outer diameters [unit], er/mur medium constants, n/m mode indices, exact exact-root flag.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        di = d_inner * self._pcb.unit
        do = d_outer * self._pcb.unit
        return float(coax_cutoff_tm(di, do, er=er, mur=mur, n=n, m=m, exact=exact))

    # Coax default cutoff pair (TE11, TM01).
    # Args: d_inner/d_outer diameters [unit], er/mur medium constants, exact exact-root flag.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def cutoffs(
        self,
        d_inner: float,
        d_outer: float,
        er: float = 1.0,
        mur: float = 1.0,
        exact: bool = True,
    ):
        di = d_inner * self._pcb.unit
        do = d_outer * self._pcb.unit
        return (
            float(coax_cutoff_te(di, do, er=er, mur=mur, n=1, m=1, exact=exact)),
            float(coax_cutoff_tm(di, do, er=er, mur=mur, n=0, m=1, exact=exact)),
        )


class _TwistedPairAPI:
    def __init__(self, pcb):
        self._pcb = pcb

    # Twisted-pair effective permittivity wrapper.
    # Args: d_center center spacing [unit], d_wire wire diameter [unit], er/er1 dielectric terms, twists_per_len turns per unit length, ptfe branch flag.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def eeff(
        self,
        d_center: float,
        d_wire: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
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

    # Twisted-pair impedance wrapper.
    # Args: d_center center spacing [unit], d_wire wire diameter [unit], er/er1 dielectric terms, twists_per_len turns per unit length, ptfe branch flag.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def z0(
        self,
        d_center: float,
        d_wire: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
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

    # Twisted-pair inverse center spacing from target impedance.
    # Args: z0 target impedance, d_wire wire diameter [unit], er/er1 dielectric terms, twists_per_len turns per unit length, ptfe branch flag.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def d_center_for_z0(
        self,
        z0: float,
        d_wire: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
        dc = twisted_pair_d_center_for_z0(
            z0=float(z0),
            d_wire=d_wire * self._pcb.unit,
            er=er,
            er1=er1,
            twists_per_len=twists_per_len / self._pcb.unit,
            ptfe=ptfe,
        )
        return float(dc / self._pcb.unit)

    # Twisted-pair inverse wire diameter from target impedance.
    # Args: z0 target impedance, d_center center spacing [unit], er/er1 dielectric terms, twists_per_len turns per unit length, ptfe branch flag.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def d_wire_for_z0(
        self,
        z0: float,
        d_center: float,
        er: float,
        er1: float = 1.0,
        twists_per_len: float = 0.0,
        ptfe: bool = False,
    ):
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

    # Waveguide cutoff frequency wrapper.
    # Args: a/b waveguide dimensions [unit], m/n mode indices, er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def fc(
        self,
        a: float,
        b: float,
        m: int = 1,
        n: int = 0,
        er: float = 1.0,
        mur: float = 1.0,
    ):
        return float(
            rectwg_fc(a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur)
        )

    # Waveguide propagation constant wrapper.
    # Args: f frequency [Hz], a/b dimensions [unit], m/n mode indices, er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        return float(
            rectwg_beta(
                f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
            )
        )

    # Waveguide guided wavelength wrapper.
    # Args: f frequency [Hz], a/b dimensions [unit], m/n mode indices, er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        lg = rectwg_lambda_g(
            f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
        )
        if np.isinf(lg):
            return np.inf
        return float(lg / self._pcb.unit)

    # Waveguide TE impedance wrapper.
    # Args: f frequency [Hz], a/b dimensions [unit], m/n mode indices, er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        return float(
            rectwg_z_te(
                f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
            )
        )

    # Waveguide TM impedance wrapper.
    # Args: f frequency [Hz], a/b dimensions [unit], m/n mode indices, er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        return float(
            rectwg_z_tm(
                f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
            )
        )

    # Inverse broad wall size from cutoff.
    # Args: fc cutoff frequency [Hz], er/mur medium constants, m mode index.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def a_for_fc(self, fc: float, er: float = 1.0, mur: float = 1.0, m: int = 1):
        return float(rectwg_a_for_fc(fc, er=er, mur=mur, m=m) / self._pcb.unit)

    # Inverse TE10 broad wall size from TE impedance.
    # Args: z0 target TE impedance, f frequency [Hz], er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def a_for_z_te10(self, z0: float, f: float, er: float = 1.0, mur: float = 1.0):
        return float(rectwg_te10_a_for_z0(z0, f, er=er, mur=mur) / self._pcb.unit)

    # Physical length for desired phase angle in selected mode.
    # Args: angle_rad target phase angle [rad], f frequency [Hz], a/b dimensions [unit], m/n mode indices, er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
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
        beta = rectwg_beta(
            f, a * self._pcb.unit, b * self._pcb.unit, m=m, n=n, er=er, mur=mur
        )
        if beta <= 0.0:
            return np.inf
        return float((float(angle_rad) / beta) / self._pcb.unit)

    # Convenience TE10 report dict.
    # Args: f frequency [Hz], a/b dimensions [unit], er/mur medium constants.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def te10(self, f: float, a: float, b: float, er: float = 1.0, mur: float = 1.0):
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
    # Build calculator from stackup geometry and dielectric list.
    # Args: layers z-coordinates [unit], materials dielectric objects per layer interval, unit conversion to meters.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def __init__(self, layers: np.ndarray, materials: list[Material], unit: float):
        """Transmission line calculator for a PCB stack.

        Args:
            layers (np.ndarray): The z-coordinates of the copper layers (in PCB units)
            materials (list[Material]): The dielectric material between each pair of layers
            unit (float): The PCB unit in meters
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

    # Backward-compatible alias for microstrip inverse width solve.
    # Args: Z0 target impedance, layer/ground_layer indices, f0 frequency [Hz], er override dielectric.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def z0(
        self,
        Z0: float,
        layer: int = -1,
        ground_layer: int = 0,
        f0: float = 1e9,
        er: float | None = None,
        include_dispersion: bool = True,
    ) -> float:
        """The microstrip trace width for a target characteristic impedance.

        Args:
            Z0 (float): The target impedance [Ohm]
            layer (int, optional): The trace layer. Defaults to -1.
            ground_layer (int, optional): The reference ground layer. Defaults to 0.
            f0 (float, optional): The design frequency [Hz]. Defaults to 1e9.
            er (float | None, optional): Override the relative permittivity of the stack. Defaults to None.
            include_dispersion (bool, optional): Include frequency dispersion. Defaults to True.

        Returns:
            float: The trace width (in PCB units)
        """
        return self.microstrip.w_for_z0(
            Z0,
            layer=layer,
            ground_layer=ground_layer,
            f0=f0,
            er=er,
            incl_dispersion=include_dispersion,
        )

    # Normalize positive/negative layer index to absolute index.
    # Args: layer signed layer index.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def layer_index(self, layer: int) -> int:
        """Resolve a (possibly negative) layer number to a layer index.

        Args:
            layer (int): The layer number, negative numbers count from the top

        Returns:
            int: The layer index
        """
        idx = int(layer)
        if idx < 0:
            idx += len(self.layers)
        if idx < 0 or idx >= len(self.layers):
            raise IndexError(f"Layer index out of range: {layer}")
        return idx

    # Return layer z-coordinate in stackup units.
    # Args: layer signed layer index.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def z(self, layer: int) -> float:
        """The z-coordinate of a layer.

        Args:
            layer (int): The layer number

        Returns:
            float: The z-coordinate (in PCB units)
        """
        return float(self.layers[self.layer_index(layer)])

    # Physical distance between two layers.
    # Args: a/b signed layer indices.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def layer_distance(self, a: int, b: int) -> float:
        """The distance between two layers.

        Args:
            a (int): The first layer number
            b (int): The second layer number

        Returns:
            float: The distance [m]
        """
        return abs(self.z(a) - self.z(b)) * self.unit

    # Effective dielectric constant between two layers.
    # Args: layer/ground_layer signed indices, f0 frequency [Hz], er optional direct override.
    # Returns: Numeric result for the requested quantity; may be a tuple or dict for grouped outputs.
    # Notes: Geometry args are in stackup units and are converted internally using self._pcb.unit.
    def effective_er(
        self, layer: int, ground_layer: int, f0: float, er: float | None = None
    ) -> float:
        """The thickness weighted relative permittivity of the dielectrics between two layers.

        Args:
            layer (int): The trace layer
            ground_layer (int): The reference ground layer
            f0 (float): The frequency at which the material permittivity is evaluated [Hz]
            er (float | None, optional): If given, this value is returned instead. Defaults to None.

        Returns:
            float: The relative permittivity
        """
        if er is not None:
            return float(er)
        i1 = self.layer_index(layer)
        i2 = self.layer_index(ground_layer)
        if i1 == i2:
            raise ValueError("Signal layer and reference layer cannot be the same")
        lo = min(i1, i2)
        hi = max(i1, i2)

        mats = self.mat[lo:hi]
        if not mats:
            return 1.0

        ers = np.asarray([_material_er(mat, f0) for mat in mats], dtype=float)
        ths = np.abs(np.diff(self.layers))[lo:hi]
        if ths.size != ers.size:
            return float(np.mean(ers))
        sw = float(np.sum(ths))
        if sw <= 0.0:
            return float(np.mean(ers))
        return float(np.sum(ers * ths) / sw)
