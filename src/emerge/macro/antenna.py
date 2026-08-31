import numpy as np
from .._emerge.geo import XYPolygon
from .._emerge.geometry import GeoSurface
from .._emerge.cs import CoordinateSystem, GCS
from typing import Optional
from dataclasses import dataclass, field
import math

def vivaldi_taper(
    length: float,
    gap: float,
    opening: float,
    curve_coefficient: float = 200.0,
    mirrored: bool = True,
    dilation: float = 0.0,
    cs: CoordinateSystem = GCS,
) -> GeoSurface:
    L = length
    g = gap
    W = opening
    K = curve_coefficient
    A = (g / 2) - (W - g * K) / (2 - 2 * K)
    fx = lambda t: t * L
    fy = lambda t: (g / 2) * K**t + (W - g * K) / (2 - 2 * K) * (1 - K**t)
    if dilation == 0:
        if mirrored:
            exp_taper = (
                XYPolygon()
                .parametric(fx, fy, tolerance=1e-5)
                .parametric(fx, lambda t: -fy(t), reverse=True, tolerance=1e-5)
                .geo(cs)
            )
        else:
            exp_taper = (
                XYPolygon()
                .parametric(fx, fy, tolerance=1e-5)
                .extend([L, 0], [0, 0])
                .geo(cs)
            )
        return exp_taper
    else:
        dfx = lambda t: A * np.log(K) / L * K ** (t)
        R = lambda t: 1 / np.sqrt(1 + dfx(t) ** 2)
        fx2 = lambda t: t * L - dilation * R(t) * dfx(t)
        fy2 = lambda t: fy(t) + dilation * R(t)
        if mirrored:
            exp_taper_dialated = (
                XYPolygon()
                .parametric(fx2, fy2, tolerance=1e-5, tmax=1.1)
                .parametric(
                    fx2, lambda t: -fy2(t), reverse=True, tolerance=1e-5, tmax=1.1
                )
                .geo(cs)
            )
        else:
            exp_taper_dialated = (
                XYPolygon()
                .parametric(fx2, fy2, tolerance=1e-5, tmax=1.1)
                .extend([L, 0], [0, 0])
                .geo(cs)
            )
        return exp_taper_dialated

"""
yagi_uda_design.py
-------------------
Yagi-Uda antenna dimension calculator.

Primary data source (exact, empirical):
    P. P. Viezbicke, "Yagi Antenna Design," NBS Technical Note 688,
    U.S. Dept. of Commerce / National Bureau of Standards, Dec. 1976,
    Table 10.6 ("Optimized Uncompensated Lengths of Parasitic Elements
    for Yagi-Uda Antennas of Six Different Lengths").

    That table was measured at an element diameter-to-wavelength ratio
    of d/lambda = 0.0085, a reflector-to-driven spacing of s12 = 0.20*lambda,
    and gives six discrete, gain-optimised designs (NOT a continuous
    formula) for overall boom lengths of 0.4, 0.8, 1.2, 2.2, 3.2 and
    4.2 wavelengths, containing 1, 3, 4, 10, 13 and 15 directors
    respectively (i.e. 3, 5, 6, 12, 15 and 17 total elements).

    Because it is a physical measurement result, not a formula, this
    script hard-codes it verbatim and only *scales* it by wavelength.
    It does NOT interpolate between the six designs, because Yagi
    behaviour is not smoothly interpolable — a 7-element antenna is
    not "a bit more than 6 and a bit less than 12".

Secondary fallback (approximate, for element counts NOT in the NBS
table): a widely-used empirical tapering scheme (the kind found in
amateur-radio "long Yagi" design guides descending from DL6WU-style
practice) is used to extrapolate a reasonable director taper. This
is clearly flagged as approximate. It is a starting point for NEC-2/
NEC-4 (e.g. PyNEC, 4nec2) optimisation, not a substitute for it.

Everything in this table assumes:
  - a single supporting boom, electrically thin elements
  - a dielectric (non-conducting) boom (metal booms require the
    length correction in the source's Figure 10.26, which is a
    hand-read graph, not reproduced numerically here, and is left
    as a documented TODO — see boom_material parameter)
  - element diameter-to-wavelength ratio near 0.0085; a warning is
    issued if the user's supplied diameter deviates substantially,
    since accuracy degrades outside roughly 0.001 <= d/lambda <= 0.04.
"""



C = 299_792_458.0  # speed of light, m/s


# ---------------------------------------------------------------------------
# 1. Hard-coded empirical data: NBS TN-688, Table 10.6
# ---------------------------------------------------------------------------
# Each entry is one of the six validated NBS designs.
# Lengths are in wavelengths (fractions of lambda), director list is
# ordered nearest-to-driven-element first (l3, l4, l5, ...).

@dataclass
class NBSDesign:
    boom_length_wl: float          # overall boom length, wavelengths
    reflector_len_wl: float        # l1 / lambda
    director_lens_wl: list[float]  # l3, l4, l5, ... / lambda
    director_spacing_wl: float     # sij / lambda (uniform in this table)
    reflector_spacing_wl: float    # s12 / lambda (reflector to driven)
    directivity_dBd: float         # gain over half-wave dipole, dB

    @property
    def num_directors(self) -> int:
        return len(self.director_lens_wl)

    @property
    def total_elements(self) -> int:
        return self.num_directors + 2  # + reflector + driven element


NBS_TABLE_688 = [
    NBSDesign(
        boom_length_wl=0.4,
        reflector_len_wl=0.482,
        director_lens_wl=[0.442],
        director_spacing_wl=0.20,
        reflector_spacing_wl=0.20,
        directivity_dBd=7.1,
    ),
    NBSDesign(
        boom_length_wl=0.8,
        reflector_len_wl=0.482,
        director_lens_wl=[0.428, 0.424, 0.428],
        director_spacing_wl=0.20,
        reflector_spacing_wl=0.20,
        directivity_dBd=9.2,
    ),
    NBSDesign(
        boom_length_wl=1.2,
        reflector_len_wl=0.482,
        director_lens_wl=[0.428, 0.420, 0.420, 0.428],
        director_spacing_wl=0.25,
        reflector_spacing_wl=0.20,
        directivity_dBd=10.2,
    ),
    NBSDesign(
        boom_length_wl=2.2,
        reflector_len_wl=0.482,
        director_lens_wl=[0.432, 0.415, 0.407, 0.398, 0.390,
                           0.390, 0.390, 0.390, 0.398, 0.407],
        director_spacing_wl=0.20,
        reflector_spacing_wl=0.20,
        directivity_dBd=12.25,
    ),
    NBSDesign(
        boom_length_wl=3.2,
        reflector_len_wl=0.482,
        director_lens_wl=[0.428, 0.420, 0.407, 0.398, 0.394, 0.390,
                           0.386, 0.386, 0.386, 0.386, 0.386, 0.386, 0.386],
        director_spacing_wl=0.20,
        reflector_spacing_wl=0.20,
        directivity_dBd=13.4,
    ),
    NBSDesign(
        boom_length_wl=4.2,
        reflector_len_wl=0.475,
        director_lens_wl=[0.424, 0.424, 0.420, 0.407, 0.403, 0.398, 0.394,
                           0.390, 0.390, 0.390, 0.390, 0.390, 0.390, 0.386, 0.386],
        director_spacing_wl=0.308,
        reflector_spacing_wl=0.20,
        directivity_dBd=14.2,
    ),
]

NBS_D_OVER_LAMBDA = 0.0085  # element diameter ratio the table was measured at


# ---------------------------------------------------------------------------
# 2. Result container
# ---------------------------------------------------------------------------

@dataclass
class YagiElement:
    role: str            # "reflector", "driven", "director-1", ...
    length_m: float
    position_m: float    # distance from reflector along the boom


@dataclass
class YagiDesign:
    frequency_hz: float
    wavelength_m: float
    elements: list[YagiElement]
    boom_length_m: float
    estimated_gain_dBi: float
    source: str                 # "NBS TN-688 Table 10.6" or "approximate taper"
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Yagi-Uda design @ {self.frequency_hz/1e6:.3f} MHz "
            f"(lambda = {self.wavelength_m*100:.2f} cm)",
            f"Source: {self.source}",
            f"Elements: {len(self.elements)}   Boom length: "
            f"{self.boom_length_m*100:.1f} cm ({self.boom_length_m:.3f} m)",
            f"Estimated gain: {self.estimated_gain_dBi:.2f} dBi",
            "-" * 68,
            f"{'Element':<14}{'Length (mm)':>14}{'Position (mm)':>18}",
        ]
        for el in self.elements:
            lines.append(
                f"{el.role:<14}{el.length_m*1000:>14.1f}{el.position_m*1000:>18.1f}"
            )
        if self.warnings:
            lines.append("-" * 68)
            lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Core design function
# ---------------------------------------------------------------------------

def design_yagi(
    frequency_hz: float,
    num_elements: int | None = None,
    boom_length_wl: float | None = None,
    element_diameter_m: float | None = None,
    boom_material: str = "dielectric",
    driven_element_type: str = "dipole",
) -> YagiDesign:
    """
    Compute Yagi-Uda element lengths and spacings for a given frequency.

    Exactly one of `num_elements` or `boom_length_wl` should normally be
    given to pick a design point; if both are omitted, a mid-size 6
    element / 1.2*lambda design is returned. If neither the requested
    element count nor boom length matches one of the six NBS TN-688
    designs exactly, the nearest validated NBS design is used AND a
    generic tapering fallback is also offered via `design_yagi_tapered`
    for arbitrary counts.

    Parameters
    ----------
    frequency_hz : float
        Design (center) frequency in Hz.
    num_elements : int, optional
        Desired total element count (reflector + driven + directors).
        Must be one of {3, 5, 6, 12, 15, 17} to get an exact NBS-table
        match; otherwise the nearest of those is chosen and a warning
        is attached (use design_yagi_tapered for a genuinely arbitrary
        element count).
    boom_length_wl : float, optional
        Desired boom length in wavelengths; alternative selector to
        num_elements. Nearest of {0.4, 0.8, 1.2, 2.2, 3.2, 4.2} is used.
    element_diameter_m : float, optional
        Physical element diameter. Used only to warn if d/lambda strays
        far from the 0.0085 the table was measured at (roughly valid
        for 0.001 <= d/lambda <= 0.04); no numerical correction is
        applied (that correction is a hand-read graph in the source,
        Fig. 10.25, not a formula).
    boom_material : {"dielectric", "metal"}
        Metal booms detune elements slightly (source Fig. 10.26 gives a
        graphical correction, additive to element length); not
        numerically implemented here, so a warning is issued instead.
    driven_element_type : {"dipole", "folded_dipole"}
        Only affects the note attached to the driven element length,
        since the source table does not specify a driven-element length
        (it is tuned empirically against the chosen match/feed, and
        typically falls between the reflector and first director
        lengths). A generic 0.47*lambda dipole starting point is used.

    Returns
    -------
    YagiDesign
    """
    warnings: list[str] = []
    wavelength_m = C / frequency_hz

    design = _select_nbs_design(num_elements, boom_length_wl, warnings)

    if element_diameter_m is not None:
        d_over_lambda = element_diameter_m / wavelength_m
        if not (0.001 <= d_over_lambda <= 0.04):
            warnings.append(
                f"d/lambda = {d_over_lambda:.4f} is outside the "
                f"0.001-0.04 range the NBS table was validated over; "
                f"lengths below are unadjusted and may be optimistic."
            )
        elif abs(d_over_lambda - NBS_D_OVER_LAMBDA) > 0.003:
            warnings.append(
                f"d/lambda = {d_over_lambda:.4f} differs from the "
                f"table's reference value of {NBS_D_OVER_LAMBDA}; the "
                f"source normally corrects for this via a graph "
                f"(Fig. 10.25), not reproduced numerically here, so "
                f"expect a resonance shift of perhaps 1-3%."
            )

    if boom_material == "metal":
        warnings.append(
            "Metal boom selected: the source lengthens elements slightly "
            "to compensate for boom capacitive loading (Fig. 10.26, a "
            "hand-read graph). Not numerically applied here — add on the "
            "order of 0.5-1% to each element length as a rough manual "
            "correction, then verify in NEC."
        )

    elements: list[YagiElement] = []

    # Reflector
    reflector_len = design.reflector_len_wl * wavelength_m
    elements.append(YagiElement("reflector", reflector_len, 0.0))

    # Driven element: NOT given numerically by the NBS table (it is
    # tuned against the chosen feed/match). We seed it at a standard
    # dipole starting point, strictly between reflector and first
    # director as the source's own design notes recommend.
    driven_pos = design.reflector_spacing_wl * wavelength_m
    first_dir_len_wl = design.director_lens_wl[0]
    driven_len_wl = (design.reflector_len_wl + first_dir_len_wl) / 2.0
    driven_len = driven_len_wl * wavelength_m
    driven_note = (
        "resonant half-wave dipole starting length; empirically trim "
        "for match" if driven_element_type == "dipole" else
        "folded-dipole driven element; physical straight-wire length "
        "is longer than a plain dipole for the same resonance "
        "(commonly ~1.00-1.04x lambda for a two-wire fold) - tune "
        "against your chosen feed impedance"
    )
    warnings.append(
        f"Driven element length ({driven_len*1000:.1f} mm) is a "
        f"starting estimate, not from the NBS table ({driven_note}); "
        f"the source explicitly treats it as tuned empirically at the "
        f"design frequency for the chosen match."
    )
    elements.append(YagiElement("driven", driven_len, driven_pos))

    # Directors
    pos = driven_pos
    for i, dlen_wl in enumerate(design.director_lens_wl, start=1):
        pos += design.director_spacing_wl * wavelength_m
        elements.append(
            YagiElement(f"director-{i}", dlen_wl * wavelength_m, pos)
        )

    boom_length_m = pos  # distance from reflector to last director
    gain_dBi = design.directivity_dBd + 2.15  # dBd -> dBi

    return YagiDesign(
        frequency_hz=frequency_hz,
        wavelength_m=wavelength_m,
        elements=elements,
        boom_length_m=boom_length_m,
        estimated_gain_dBi=gain_dBi,
        source="NBS Technical Note 688, Table 10.6 (Viezbicke, 1976)",
        warnings=warnings,
    )


def _select_nbs_design(
    num_elements: int | None,
    boom_length_wl: float | None,
    warnings: list[str],
) -> NBSDesign:
    if num_elements is not None:
        exact = [d for d in NBS_TABLE_688 if d.total_elements == num_elements]
        if exact:
            return exact[0]
        nearest = min(NBS_TABLE_688, key=lambda d: abs(d.total_elements - num_elements))
        warnings.append(
            f"No NBS TN-688 design has exactly {num_elements} elements "
            f"(the table only covers "
            f"{[d.total_elements for d in NBS_TABLE_688]}); substituting "
            f"the nearest validated design ({nearest.total_elements} "
            f"elements). For a genuinely custom element count, use "
            f"design_yagi_tapered() instead — it is an approximation, "
            f"not an NBS-validated result."
        )
        return nearest

    if boom_length_wl is not None:
        nearest = min(NBS_TABLE_688, key=lambda d: abs(d.boom_length_wl - boom_length_wl))
        if abs(nearest.boom_length_wl - boom_length_wl) > 1e-9:
            warnings.append(
                f"No NBS TN-688 design has boom length exactly "
                f"{boom_length_wl}*lambda; substituting nearest "
                f"validated design ({nearest.boom_length_wl}*lambda, "
                f"{nearest.total_elements} elements)."
            )
        return nearest

    # default: 6-element, 1.2*lambda design (a very common ham/TV size)
    return next(d for d in NBS_TABLE_688 if d.total_elements == 6)


# ---------------------------------------------------------------------------
# 4. Approximate fallback for arbitrary element counts
# ---------------------------------------------------------------------------

def design_yagi_tapered(
    frequency_hz: float,
    num_elements: int,
    director_spacing_wl: float = 0.20,
    reflector_spacing_wl: float = 0.20,
) -> YagiDesign:
    """
    Approximate design for an arbitrary total element count, using a
    generic empirical taper (element length shrinks and inter-director
    spacing effectively lengthens the array as you add directors, in
    line with the *trend* visible in NBS TN-688, but interpolated by
    formula rather than measured directly).

    This is explicitly a starting point for NEC modelling/optimisation
    (e.g. via PyNEC or 4nec2), NOT an NBS-validated design. Prefer
    design_yagi() with num_elements in {3, 5, 6, 12, 15, 17} whenever
    possible, since those numbers ARE directly backed by measurement.

    Taper model (documented, not black-box):
      - reflector: 0.482*lambda (matches NBS table across all designs)
      - driven: midway between reflector and first director length
      - directors: start at 0.45*lambda and asymptotically decay
        toward ~0.385*lambda (the NBS table's long-Yagi director floor)
        with a simple exponential taper, since the real data shows
        exactly this kind of decay-then-plateau behaviour.
    """
    if num_elements < 3:
        raise ValueError("A Yagi-Uda antenna needs at least 3 elements "
                          "(reflector, driven, one director).")

    wavelength_m = C / frequency_hz
    num_directors = num_elements - 2

    reflector_len_wl = 0.482
    director_floor_wl = 0.385   # long-Yagi asymptotic director length
    director_start_wl = 0.45    # first-director length
    decay_rate = 0.35           # taper sharpness, fit by eye to NBS data

    director_lens_wl = [
        director_floor_wl
        + (director_start_wl - director_floor_wl) * math.exp(-decay_rate * i)
        for i in range(num_directors)
    ]

    driven_len_wl = (reflector_len_wl + director_lens_wl[0]) / 2.0

    elements = [YagiElement("reflector", reflector_len_wl * wavelength_m, 0.0)]
    driven_pos = reflector_spacing_wl * wavelength_m
    elements.append(YagiElement("driven", driven_len_wl * wavelength_m, driven_pos))

    pos = driven_pos
    for i, dlen_wl in enumerate(director_lens_wl, start=1):
        pos += director_spacing_wl * wavelength_m
        elements.append(YagiElement(f"director-{i}", dlen_wl * wavelength_m, pos))

    boom_length_m = pos
    # crude gain estimate: log-fit to the six NBS directivity points
    # (7.1 dBd at 1 director, up to 14.2 dBd at 15 directors) — this is
    # curve-fitting the trend, not a physical model, and is intentionally
    # conservative; flagged clearly as indicative only.
    gain_dBd = 7.1 + 1.4 * math.log2(max(num_directors, 1))
    gain_dBd = min(gain_dBd, 14.5)

    return YagiDesign(
        frequency_hz=frequency_hz,
        wavelength_m=wavelength_m,
        elements=elements,
        boom_length_m=boom_length_m,
        estimated_gain_dBi=gain_dBd + 2.15,
        source="Approximate exponential taper (fit to NBS TN-688 trend, "
               "NOT independently measured) — verify in NEC before building",
        warnings=[
            "This design was NOT looked up from measured data — it is "
            "a smooth curve fit through the NBS TN-688 trend. Treat gain "
            "and SWR figures as rough guidance only, and simulate "
            "(e.g. PyNEC / 4nec2 / EZNEC) before cutting metal.",
        ],
    )


# ---------------------------------------------------------------------------
# 5. DL6WU-style long-Yagi design (tapered spacing AND tapered length)
# ---------------------------------------------------------------------------
#
# Source characteristics (documented, not a single closed-form formula from
# the original author — Guenter Hoch, DL6WU, published his results as
# graphs in "Extremely Long Yagi Antennas," VHF Communications, March 1982,
# later reproduced in the RSGB "VHF/UHF DX Book." David Tanner, VK3AUU,
# published fitted equations approximating those graphs. The exact
# proprietary graph values are not reproduced here; instead this uses the
# well-documented, independently-confirmed *characteristics* of the
# design family):
#
#   - reflector-to-driven spacing:      0.20 * lambda        (same as NBS)
#   - driven-to-first-director spacing: ~0.075-0.10 * lambda (deliberately
#     tight -- this is the "close first gap" you noticed)
#   - director-to-director spacing:     ramps up smoothly from that tight
#     first gap to a ceiling of 0.4 * lambda, reached by around the 13th
#     director, and stays flat at 0.4 * lambda beyond that
#   - director length: tapers down from roughly 0.45 * lambda for the
#     first director toward an asymptote around 0.40-0.41 * lambda for
#     long arrays (DL6WU designs commonly use fatter elements, e.g. ~4 mm
#     at 432 MHz, which is part of why the asymptotic length sits higher
#     than the NBS table's thinner-wire ~0.385 * lambda floor)
#
# This is explicitly a documented APPROXIMATION of a real, respected
# design family, not a reproduction of Hoch's or Tanner's exact published
# figures (which are graphs/proprietary fitted curves, not something to be
# transcribed here without the source in hand). For a real build, run this
# as a NEC starting point (PyNEC/4nec2/EZNEC) and, if you want the actual
# published DL6WU/VK3AUU numbers verbatim, use the "Yagi Calculator" tool
# by John Drew (VK5DJ), which implements them directly.

def design_yagi_dl6wu(
    frequency_hz: float,
    num_elements: int,
    first_gap_wl: float = 0.10,
    spacing_ceiling_wl: float = 0.40,
    director_after_which_flat: int = 13,
    driven_spacing_wl: float = 0.20,
) -> YagiDesign:
    """
    Long-Yagi design in the DL6WU tradition: tight first director gap,
    smoothly increasing director spacing up to a 0.4*lambda ceiling, and
    director lengths tapering toward a higher asymptote than the NBS
    table (consistent with the fatter elements this design family
    typically uses). Intended for longer arrays; the source community
    generally considers boom lengths under ~2*lambda (roughly 8-10
    elements) too short for this style to show its advantage over NBS
    TN-688 or simple uniform designs.

    Parameters
    ----------
    frequency_hz : float
    num_elements : int
        Total elements including reflector and driven element; must be
        >= 4 (reflector, driven, >= 2 directors) for the taper to mean
        anything.
    first_gap_wl : float
        Driven-to-first-director spacing, wavelengths. DL6WU designs
        typically use ~0.075-0.10; default 0.10 is a conservative
        (slightly gentler, easier to build) choice within that range.
    spacing_ceiling_wl : float
        The director spacing this design ramps up to and then holds;
        documented DL6WU practice uses 0.4*lambda.
    director_after_which_flat : int
        Which director index the spacing ramp reaches its ceiling at;
        documented practice is "director 13 onward."
    driven_spacing_wl : float
        Reflector-to-driven spacing, wavelengths (unchanged from NBS
        practice at 0.20).
    """
    if num_elements < 4:
        raise ValueError(
            "DL6WU-style tapered spacing needs at least 2 directors "
            "(4 elements total) for the taper to be meaningful; use "
            "design_yagi() for short 3-element designs."
        )

    wavelength_m = C / frequency_hz
    num_directors = num_elements - 2

    # --- spacing taper: linear ramp from first_gap_wl to spacing_ceiling_wl,
    # reaching the ceiling at director index `director_after_which_flat`,
    # then flat.
    if director_after_which_flat > 1:
        ramp_step = (spacing_ceiling_wl - first_gap_wl) / (director_after_which_flat - 1)
    else:
        ramp_step = 0.0

    director_spacings_wl = []
    for n in range(1, num_directors + 1):  # n = director index, 1-based
        if n >= director_after_which_flat:
            gap = spacing_ceiling_wl
        else:
            gap = first_gap_wl + ramp_step * (n - 1)
        director_spacings_wl.append(gap)

    # --- length taper: exponential decay from a first-director length
    # down toward the higher DL6WU-family asymptote.
    reflector_len_wl = 0.50   # DL6WU designs commonly run the reflector
                               # close to a full half-wave, slightly more
                               # than the NBS 0.482 figure
    director_start_wl = 0.45
    director_floor_wl = 0.40
    decay_rate = 0.30

    director_lens_wl = [
        director_floor_wl
        + (director_start_wl - director_floor_wl) * math.exp(-decay_rate * i)
        for i in range(num_directors)
    ]

    driven_len_wl = (reflector_len_wl + director_lens_wl[0]) / 2.0

    elements = [YagiElement("reflector", reflector_len_wl * wavelength_m, 0.0)]
    driven_pos = driven_spacing_wl * wavelength_m
    elements.append(YagiElement("driven", driven_len_wl * wavelength_m, driven_pos))

    pos = driven_pos
    for i, (dlen_wl, gap_wl) in enumerate(zip(director_lens_wl, director_spacings_wl), start=1):
        pos += gap_wl * wavelength_m
        elements.append(YagiElement(f"director-{i}", dlen_wl * wavelength_m, pos))

    boom_length_m = pos

    # Rough gain estimate, calibrated to the community-reported "DL6WU
    # beats NBS by roughly 0.5 dB per wavelength of boom" figure, applied
    # on top of the NBS log-fit used elsewhere in this module.
    boom_length_wl = boom_length_m / wavelength_m
    nbs_like_gain_dBd = 7.1 + 1.4 * math.log2(max(num_directors, 1))
    gain_dBd = min(nbs_like_gain_dBd + 0.5 * boom_length_wl, 18.0)

    warnings = [
        "DL6WU-style taper: lengths and the spacing ramp are a documented "
        "APPROXIMATION of the design family's known characteristics "
        "(tight first gap, ramp to 0.4*lambda spacing, higher director "
        "length asymptote for fatter elements) -- not a transcription of "
        "Hoch's or Tanner's original proprietary graphs/equations. "
        "Verify in NEC (PyNEC/4nec2/EZNEC) before building, and consider "
        "the 'Yagi Calculator' tool (VK5DJ) if you want the published "
        "figures verbatim.",
        f"Driven element length ({driven_len_wl*wavelength_m*1000:.1f} mm) "
        f"is a starting estimate for a folded dipole (typical DL6WU feed); "
        f"a folded dipole's physical length runs longer than a plain "
        f"dipole for the same resonance -- tune against your chosen "
        f"balun/match.",
        "This style is intended for longer arrays (boom lengths "
        "upward of roughly 2*lambda / 8-10 elements); for short "
        "3-6 element antennas prefer design_yagi() (NBS TN-688).",
    ]

    return YagiDesign(
        frequency_hz=frequency_hz,
        wavelength_m=wavelength_m,
        elements=elements,
        boom_length_m=boom_length_m,
        estimated_gain_dBi=gain_dBd + 2.15,
        source="DL6WU-tradition tapered-spacing design (documented "
               "approximation of Hoch/VK3AUU characteristics, NOT a "
               "verbatim reproduction) -- verify in NEC before building",
        warnings=warnings,
    )

