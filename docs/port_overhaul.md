# Port overhaul: multi-mode WavePort and FloquetPort

All changes are uncommitted on `main` (on top of `07c2eb9`). The theory and the validation are in
`docs/wave_port.typ` (→ `docs/wave_port.pdf`). This file lists what changed where.

Not part of this work (your own uncommitted edits): `examples/demo3_coupled_line_filter.py`,
`src/emerge/_emerge/geo/pcb.py`, `src/emerge/_emerge/simmodel.py` (OCC target unit), `demo3_new.step`.

## Behaviour changes for users

| Change | Effect |
|---|---|
| `WavePort` is now the multi-mode wave port (`number_of_modes`, `None` = all propagating at f_max) | New API. The mode set is fixed at the highest frequency and tracked down; evanescent modes stay in the S-matrix (β = −jα). |
| `WavePortIH` is deprecated | Still works: `WavePort(..., modetype="TEM")`, with a warning. `MultiModeWavePort` (hertzon PR) is removed, merged into `WavePort`. |
| **Bug fix:** `ModalPort` β extrapolation missed εr·μr | Dielectric-filled guides gave \|S11\| up to 0.41. Modes are now only reused when exact (homogeneous face, TEM modes); other `ModalPort`s are re-solved per frequency, which is slower. Microstrip `ModalPort` results change (dispersion is now included). |
| **Bug fix:** `PortMode.flip_polarity` was a no-op | E and H share one array and were flipped twice. TEM ports whose voltage came out negative now flip: their S-parameters rotate by 180°. |
| `FloquetPort` absorbs TE00 and TM00 exactly | Previously one blended γ for both polarizations. Default `number_of_modes=1` keeps the old single-mode S-matrix (label = port number). |
| `floquet_port(..., number_of_modes, polarization_deg, er)` | `2` = TE00/TM00 as `p.1`/`p.2`; more or `None` adds grating modes (TE and TM per harmonic). |
| **Bug fix:** `set_scanangle` before creating ports / periodic BCs was lost | The cell now stores the scan angle and applies it to BCs and ports created later. |
| Homogeneous `WavePort` faces are solved once, at f_max | Auto-detected: one isotropic, lossless (σ = 0), non-dispersive material. The other frequencies scale the modes analytically (TM E_t ∝ β). Same S-parameters (1.5e-5 on the step test). Port-mode time on the step dropped from 28 s to 2 s. Otherwise the port is solved and tracked at every frequency. |
| New public port methods | `port.smatrix_index(mode_nr)` (all ports), `FloquetPort.floquet_modes()` (S-matrix index, order (m, n) and polarization per mode), `FloquetPort.order_direction(freq, m, n)` (beam angles of an order, `None` if evanescent). |
| **Bug fix:** iterative `modal_analysis` dropped TM modes and degenerate partners | `SolverARPACK_BMA` rejected candidates with xᴴBx < 0.1 (unit vector); TM11 of a round guide has 0.04. Now xᴴBx > 0, candidates taken closest to the shift first. |
| `ModalPort` with fewer propagating modes than `number_of_modes` | Clear `BoundaryConditionError` before the sweep instead of a `ZeroDivisionError` after it. |
| **Bug fix:** PARDISO non-symmetric CSC solved Aᵀx = b again (PR #92) | MKL's default IPARM is loaded only in the symbolic phase, so `IPARM[11] = 2` survives. |
| S-parameter post-processing interpolates the solution once per port | Same results; the 30-mode Floquet case went from 9.0 s to 3.0 s, multi-mode WavePort sweeps also speed up. |

## Changes per file

### `physics/microwave/bcs/port_bcs.py`
- `PortBC`: public `smatrix_index(mode_nr)`. New per-mode hooks used by the post-processing instead of `isinstance` checks:
  `mode_beta`, `mode_Z0`, `is_propagating` (default `True`), `sparam_projection` (default `None`),
  `_self_normalized` (default `False`).
- `ModalPort`:
  - `get_beta` extrapolates with β² = β0² + εr·μr(k0² − k00²) (was missing εr·μr).
  - New attributes `_homogeneous`, `_ermur`, `_ermur_max`, `_nmodes_solved`, and the `_frequency_scalable` property.
  - `reset()` also clears the PEC-island cache.
- `WavePort` (new, replaces `WavePortIH` and the hertzon `MultiModeWavePort`):
  - Accepts evanescent modes (`try_add_mode`).
  - Per-mode `get_beta`/`portZ0`/`is_propagating`.
  - Excitation `get_Uinc`, `get_modepf_kappa_all` for the dense WPBC, and the S-parameter projection on γ_m e_tm + ∇e_zm (`sparam_projection`).
  - Mode set and tracking logic: `_has_mode_set`, `_fix_mode_set` (selection at f_max, degeneracy warning, alignment vectors), `_predict_beta2` (secant in k0²), `_continue_tracks` (CMAC + Hungarian assignment, phase alignment).
  - `solved_frequencies` property.
  - The `_scalable`, `_k0_ref` and `_mode_basis` attributes of the homogeneous path.
- `WavePortIH`: deprecated subclass of `WavePort` with `modetype="TEM"`.
- `FloquetPort` (rewritten):
  - Analytic Floquet harmonics (m, n), TE and TM, from the reciprocal lattice (`_reciprocal`, `_kt`, `_basis_mode`, `basis_field`, `basis_gamma`, `basis_norm`).
  - 1 W normalization, mode selection at f_max (`_prepare`, which also reads the normal and the area from the mesh).
  - Polarization mix for one mode (`_set_port_modes`), conjugate S-parameter projection.
  - `get_gamma` is now the residual first-order γ_r = j k0 √εr.
  - Public `floquet_modes()` and `order_direction(freq, m, n)`.

### `physics/microwave/bcs/portmode.py`
- `flip_polarity`: flips H only if it does not share its array with E.

### `physics/microwave/bcs/boundary_condition_set.py`, `bcs/__init__.py`
- Registers `WavePort` and `WavePortIH` and removes `MultiModeWavePort`. Exports `FloquetPort`.
- `floquet_port()`: new arguments `number_of_modes`, `polarization_deg`, `er`. It sets the lattice vectors (`RectCell`, `HexCell`) and applies the cell's scan angle.

### `periodic.py`
- `PeriodicCell` stores `scan_theta`/`scan_phi`.
- `_apply_scanangle()` is called by `set_scanangle`, `generate_bcs` and `floquet_port`.

### `physics/microwave/microwave_3d.py`
- `_compute_modes`:
  - `FloquetPort._prepare` per sweep.
  - `WavePort` goes through `_wave_port_modes`.
  - A `ModalPort` reuses modes only when `_frequency_scalable`, or when modes already exist at that k0.
- New:
  - `_wave_port_modes` (scales the modes of a homogeneous port, otherwise marches from the nearest solved frequency).
  - `_wave_port_scalable`, the homogeneity check: σ = 0 and no dispersion between f_min and f_max.
  - `_split_te_tm_modes`, which splits mixed degenerate TE/TM groups.
  - `_scale_wave_port_modes` and `_make_wave_port_mode`.
  - `_tri_materials(freq)`, the per-triangle material lookup, moved out of `modal_analysis`.
  - `_check_modal_port_count`: a `ModalPort` with fewer propagating modes than requested raises before the sweep.
  - `_select_wave_port_modes` and `_track_wave_port_step`. These only run the eigen solves; the decisions are on `WavePort`.
  - `_port_islands`, a cache for the TEM conductor search.
- `modal_analysis`:
  - Stores the surface basis, μr and normal of the solve on a `WavePort` (`_mode_basis`).
  - `neff2_shifts` argument (shift-invert around predicted β²) and the β branch Im β ≤ 0.
  - Homogeneity detection.
  - WavePort normalization |½∬(e×h)·n| = 1, with a sign check on the conjugate power for propagating modes only.
  - No gauge warning for WavePort.
- Post-processing:
  - Per-mode β/Z0 via `mode_beta`/`mode_Z0`; port power correction skipped for `_self_normalized` ports.
  - |S| caps only on propagating modes (`is_propagating`); port-specific projection via `sparam_projection`.
  - `_reuse_last_evaluation`: the solution is interpolated once per port.

### `physics/microwave/assembly/`
- `wpbc_multimode.py` (new, from hertzon):
  - `assemble_wpbc_multimode`: one rank-1 WPBC term per WavePort mode.
  - `assemble_floquet_port`: Hermitian rank-1 term (γ_i − γ_r)/N_i · G_i G_iᴴ per Floquet basis mode, plus the excitations.
- `assembler.py`: `WavePort` → dense multi-mode block; `FloquetPort` → Robin term (γ_r) plus the dense projection block. The single-mode `assemble_wpbc` path for `WavePortIH` is gone.
- `system_pattern.py`: the dense block pattern is registered for `WavePort` and `FloquetPort`. A `FloquetPort` also keeps its Robin term.

### Solver refactor: `solver.py` → `solveroutine.py` + `solvers/`
`solver.py` (3000 lines) and `solve_interfaces/` are split into one module per solver. Each solver module holds the
interface to its library and the `Solver` class. The code moved verbatim, except for the port-mode solver.

```
_emerge/
├── solveroutine.py            SolveRoutine, AutomaticRoutine
└── solvers/
    ├── __init__.py            ALL_SOLVERS, PUBLIC_SOLVERS, available(), the removed EMSolver enum
    ├── solver_base.py         Solver, EigSolver, Sorter, Preconditioner, SolveReport, MatrixType, helpers
    ├── preconditioners.py     ReverseCuthillMckee, ILU/Jacobi/ICC
    ├── iterative.py           CG, BiCGSTAB, GCROT(m,k), GMRES
    ├── superlu.py  umfpack.py  cholmod.py  rslab.py  sparta.py  corvid.py
    ├── pardiso.py  mumps.py  aasds.py  cudss.py     interface + Solver (cudss.py is imported in a try block)
    ├── eigen.py               SolverLAPACK, SolverARPACK (eigenmode studies)
    ├── boundary_modes.py      SolverARPACK_BMA: the port-mode solver of ModalPort and WavePort
    └── domain_decomposition.py  (was solver_dd.py)
```

- Import paths: `emerge._emerge.solver` → `emerge._emerge.solveroutine` (routines) and `emerge._emerge.solvers` (solvers,
  `SolveReport`, ...).
- **`EMSolver` removed; solvers are selected by class.** The public `em.solvers` module lists them under short names
  (`em.solvers.Pardiso`, `SuperLU`, `UMFPACK`, `MUMPS`, `AASDS`, `CuDSS`, `CHOLMOD`, `RSLab`, `Sparta`, `BiCGSTAB`, `CG`,
  `Corvid`, `LAPACK`, `ARPACK`):
  - `model.set_solver(em.solvers.Pardiso)` returns the `Pardiso` solver object (typed: `set_solver(type[S] | S) -> S`,
    `get_solver(type[S]) -> S`).
  - Every solver class has `available` and `requires` (also in its docstring). `em.solvers.available()` lists the
    installed ones; selecting a missing one raises "The PARDISO solver is not available on this system. It requires …".
  - `em.EMSolver` keeps its members for autocomplete, but accessing one (`em.EMSolver.PARDISO`) raises
    "em.EMSolver was removed in EMerge 3.0. Replace em.EMSolver.PARDISO with em.solvers.Pardiso …". Remove it before
    the final 3.0 release.
  - Fixed: `get_solver` with a solver object or a missing solver raised `KeyError`; `AutomaticRoutine` asked for
    `CORVID` with a `KeyError` when it is not installed (now falls back).
  - `SolveRoutine.solvers` is keyed by class; `disable()` takes classes. Saved results store
  `SolveReport` by class name, so old `.EMResults` files still load.
- Removed: the unused `filter_unique_eigenpairs` and `filter_and_rank_modes`, and `SolveRoutine._get_eig_solver_bma`.
- **One port-mode solver** (`SolverARPACK_BMA.eig(A, B, nmodes, shifts, kmax2, direct, keep_evanescent, descending)`),
  called through `SolveRoutine.eig_boundary`. It replaces the old BMA solver and `SolveRoutine.eig_boundary_shifts`:
  - Dense LAPACK below 500 DOFs or with `direct=True`, otherwise shift-invert ARPACK per cluster of shifts.
  - Modes are accepted on their eigenvalue (|λ| > 1e-4·kmax², β² < 2·kmax²) and residual (< 1e-6), and copies are
    removed by CMAC. The old curl/mass ratio and the xᴴBx > 0.1 threshold (which dropped TM11, issue 90 case 2) are gone.
  - If fewer than `nmodes` modes are found: a larger Krylov subspace, then the search widens to β·(1.41, 0.71, 2, 0.5)
    around a propagating shift (the old BMA fallback).
  - `ModalPort`: one shift at target β, `keep_evanescent=False`. `WavePort`: the tracking shifts, evanescent modes kept.
  - The dense path for a `ModalPort` now also returns the `nmodes` modes closest to the target (it used to return every
    propagating mode, with a hard-coded εr μr = 4 limit).
- `solvers/pardiso.py`: `_configure(A, symbolic=True)` loads MKL's defaults only in the symbolic phase (PR #92).

### `physics/microwave/port_functions.py`
- `compute_modal_power_flux`: the non-conjugate ½∬(E×H)·n used to normalize evanescent modes.

## Tests (new)
- `tests/test_wave_port.py`: dielectric-filled `ModalPort` (the εr bug), evanescent WavePort modes (|S21| = e^(−αL)), `number_of_modes=None`, `flip_polarity`, scaled modes of a homogeneous square guide (against analytic results and against solving every frequency), and the split of mixed TE11/TM11 modes.
- `tests/test_multimode_wave_port.py`: from hertzon, renamed to `WavePort`.
- `tests/test_degenerate_port_modes.py`: full TE01/TM11 cluster of a round guide from the iterative solver; clear error for too few propagating `ModalPort` modes.
- `tests/test_floquet_port.py`: oblique slab TE/TM vs analytic, scan angle set before the ports (hex cell), grating modes vs RCWA (through the public `floquet_modes`/`order_direction`).

Last run (after the solver refactor and the EMSolver removal): full `tests/`: 79 passed, 1 skipped (PARDISO, no MKL
on this Mac).
- `tests/test_demo10.py` (fixed): it failed since `07c2eb9` (far field on interior surfaces). That commit is right; it
  corrected 300 normals on the air/PML interface that pointed into the air. The test compared the full near-field H
  at R = 10 km with r̂×E/Z0; the transverse part agrees to 2e-6, the difference is a radial H (≤ 5.5 % of the peak)
  from the (n·H) term, which only cancels for exact surface fields. The test now checks transverse H (1e-3) and
  bounds the radial residual (< 10 %).
`tests/test_wpbc.py` still uses `WavePortIH` (passes, with the deprecation warning).

## Examples
- `examples/demo23_multimode_ports.py` (new): a 40 × 10 mm guide with two 2-mode WavePorts. Straight, it
  shows absorption without mode conversion. With an asymmetric post, it shows the full 2-mode S-matrix and
  power conservation. Not run yet.
- `examples/demo24_floquet_wire_grating.py` (new): a grating of PEC wires (15 mm period, 2 mm radius) between
  two Floquet ports with `number_of_modes=None`, 5.5–29.5 GHz. Below 20 GHz it is a wire-grid polarizer
  (\|S21\| 0.21 for E ∥ wires against 0.99 for E ⊥ at 5.5 GHz). Above 20 GHz the (±1, 0) orders diffract,
  for example into 46.6° beams at 27.5 GHz. It plots the power per order, the beam angle against frequency
  and the field. Run headless (no plots): power conserved to 1e-4, 7 s.

## Dev scripts (`dev/`, gitignored)
| Script | What |
|---|---|
| `modal_port_beta_bug.py` | Reproduces the `ModalPort` εr bug. |
| `waveport_step.py` | WR90 → 10 mm step, 8–30 GHz, two multi-mode WavePorts vs mode matching. |
| `waveport_dispersion.py` | Mode tracking only: slab-loaded WR90 and shielded microstrip; slopes, predictor error, CMAC. |
| `waveport_figures.py` | Figures of the WavePort sections. |
| `floquet_ports.py` | Slab (TE/TM, 0–70°), lamellar grating vs RCWA, hex lattice with evanescent modes. |
| `floquet_figures.py` | Figures of the Floquet section. |

## Docs
- `docs/wave_port.typ` / `.pdf`: theory, the predictor bound, validation, and §8 on Floquet ports.
- `docs/figures/*.svg`: generated by the two figure scripts.

## Open items
- CHANGELOG entries (not written yet).
- `tests/test_wpbc.py` could switch to `WavePort`.
- Commit `07c2eb9` is still unpushed; nothing of this overhaul is committed.
