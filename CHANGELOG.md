# Changelog

## 3.0.0b5 — changes since 2.8.9

1. [New features](#1-new-features): what you can do now that you couldn't before
2. [Breaking API changes](#2-breaking-api-changes): changes that can break existing scripts, plus
   a migration checklist
3. [Bug fixes, accuracy and performance](#3-bug-fixes-accuracy-and-performance)
4. [Internal changes](#4-internal-changes): changes in how things are handled that most users
   won't notice

---

## 1. New features

### Meshing and adaptivity

- **New adaptive mesh refinement (AURA).** `Simulation.adaptive_mesh_refinement()` refines the
  existing mesh locally, without remeshing in GMSH:
  - New nodes are projected onto curved CAD faces and curves.
  - Works on periodic models: every split on a periodic face is mirrored on its paired face, so the periodic meshes keep matching.
  - You control how fast the mesh grows: `growth` (a float, a per-pass list, `"auto"` or a
    callable), `growth_limits`, `order` and `theta`.
  - Further options: how far refinement spreads (`max_depth`, `lepp_ratio`), quality
    improvement (`improve_sweeps`, `improve_layers`, `improve_initial`, `improve_global`) and
    thin-strip regularization (`strip_ratio`, `strip_max_growth`, `strip_gated`).
  - `show_mesh` shows the mesh between passes.
- **New adaptive frequency sweep.** `run_adaptive_sweep(tol, max_samples)` fits an AAA rational
  model and adds samples until the estimated error is below `tol`. Use `model_S()`/`model_Smat()`
  with `method='aaa'|'vf'|'auto'` and `tol` for a dense response.
- `Mesher.set_global_max_size(size)` sets a global maximum element size in meters.
- `Mesher.set_pec_domain(obj, size)` uses a coarse mesh inside PEC volumes.
- `Mesher.set_algorithm(alg, nthreads)` sets the number of GMSH threads.
- `Simulation.set_scale_factor(f)` (experimental) lets you model tiny structures, such as bond
  wires, `f` times larger and scale the mesh back.

### Physical attributes and automatic boundary conditions

- Geometries carry a `properties` set. You can attach `FiniteThickness`, `SurfaceRoughness`,
  `MetalCoating`, `VoidAttribute`, `WavePortAttribute`, `LumpedPortAttribute` and
  `LumpedElementAttribute`, combined with `+`, for example
  `obj.properties += em.FiniteThickness(35e-6) + em.SurfaceRoughness(1e-6)`.
- Ports, lumped elements and voids are created from these attributes. Surface impedance and
  thin conductor boundary conditions use the thickness and roughness.
- **Voids:** a volume with `VoidAttribute` is cut out of all other volumes and its boundary
  becomes PEC. A `Void` boundary condition is also available as `model.mw.bc.Void`.

### Post-processing

- `MWField.nearfield(x, y, z, faces, origin, syms, structure)` computes E and H at any point
  outside a closed surface, including points outside the mesh. It accepts inputs of any shape
  and returns an `EHField` of the same shape.
- `MWField.trace_poynting_lines(...)` traces Poynting vector streamlines. Seeds come from points,
  surfaces or domains, optionally weighted by field strength. Plot them with
  `display.add_particle_lines()`.
- `MWField.in_domain(selection, nodes, edges, tris, tets)` samples the field inside a domain.
- `MWField.cache` reuses interpolation data for repeated evaluations at the same points.
- `current_boundary(selection, eps, n_samples)` takes extra arguments and returns per-face
  currents, centres and areas in `aux`.
- `combine_ports(p1, p2, Z0renorm=...)` renormalizes all ports to a common reference before
  combining them into differential and common mode.
- `save_smatrix(..., Z0_override=...)`.
- `embed_external_component()` can be called repeatedly on different ports.
- `display.add_solution_error()` plots the error estimate.
- `display.add_backgroundfields()` shows incident plane waves.
- `display.show(zoom=...)`.
- `populate(skip_air=True)` hides air objects.
- `minimize_opacity` option on `add_object()`, `add_objects()` and `populate()`.

### Boundary conditions and ports

- `AbsorbingBoundary.optimize_for_maximum_angle(angle_deg)` and
  `set_zero_angles(ang1_deg, ang2_deg)` tune the second-order ABC for a range of incidence
  angles.
- `FloquetPort(polarization_deg=...)` and `FloquetPort.set_polarization()`.
- `ModalPort(modetype='any')` disables forcing a mode type.
- `ScatteredField(definition='EA'|'SPH', material_correction=...)`. With
  `material_correction=True`, the absorbing constant is scaled with the adjacent permittivity.
- Spherical-coordinate background fields.
- `model.mw.bc.get_port(n)` and `model.mw.bc.get_ports()`.
- `model.mw.conducting_surfaces()`, `model.conducting_surfaces(level)` and
  `model.conducting_volumes(level)` select all conductors at once.

### Geometry and selections

- `geo.connect_faces(face1, face2, ...)` lofts between two faces.
- `geo.thick_wall(thickness, *volumes)` creates a wall of the given thickness around volumes.
- `geo.duplicate(entity)` copies a geometry or selection.
- `geo.SectionedDisc(radius, Nsections, ...)` is a disc with N perimeter segments, for clean
  lofts.
- `Cylinder(..., compensate_radius=True)` compensates the radius of a polygonal cylinder.
- `Curve(..., multiplicities=...)` and `Curve.pipe(..., trihedron=...)`.
- `CustomCell1D(u, ou)` and `CustomCell2D(u, ou, v, ov)` are periodic cells with arbitrary
  lattice vectors.
- `em.uniaxial_tensor(axis, ax_value, plane_value)` and `Axis.tensor(...)` build a uniaxial
  material tensor ε⊥·I + (ε∥ − ε⊥)·n̂n̂ᵀ.
- New `Selection` filters:
  - `constrain(func, complete)`;
  - `below()`, `above()`, `left_of()`, `right_of()`, `front_of()`, `behind()`;
  - `around(distance, coordinate)`;
  - `getCAs` returns `(centre, area)` for each tag.

### PCB

- `set_board_outline(xs, ys)` sets an arbitrary board outline polygon.
- `generate_all(merge, height_air_top, height_air_bottom)` generates the board, air and vias in
  one call.
- `generate_silkscreen(material, thickenss_top, thickness_bottom)`.
- Ports from `modal_port()`, `lumped_port()` and `lumped_port_pts()` are created automatically.
- `via(..., annular_ring_radius=...)`.
- `taper(distance, width, dx, dy, offset_r)`.
- `pturn()` supports `'miter'`.
- `turn()`, `pturn()` and `to()` remember the last used corner settings.
- `move()`.
- `pcb_to_dxf()` exports the PCB to DXF.

### File import and export

- `emerge.read.GDSFile` with `GDSLayer` stack-up definitions imports GDSII.
- `emerge.read.GDSFromXML` imports GDSII with a gds2palace stack-up XML:
  - derived (boolean) layers, via merging, dielectric stacks and `size_scale`;
  - ports from markers and text labels.
- `emerge.read.ODBImport` imports ODB++ (needs `emcad`).
- `FileBasedPCB.layer_from_file(..., curved=True)` builds Gerber layers from exact arcs instead
  of polygons.
- `emerge.write.export_dxf` and `pcb_to_dxf`.
- `emerge.optycal.surface_model()` and `farfield_model()` export data for Optycal.

### Solvers

- `EMSolver.RSLAB` (`rslab` package) is a direct solver.
- `EMSolver.SPARTA` (`sparta>=0.2.0`) is a direct solver with GPU, CPU and hybrid modes
  (`set_options()`).
- `SolverCuDSS.set_precision('auto'|'single'|'double')`. `'auto'` uses single precision on GPUs
  with slow double precision.
- `Simulation.get_solver(solver)`; `set_solver()` returns the solver.
- `Optimizer.set_method(...)` accepts any `scipy.optimize.minimize` method or `'direct'` (DIRECT
  global search).
- `Optimizer.print_python()`.

### Macros

- Yagi-Uda antenna design: `design_yagi`, `design_yagi_dl6wu` and `design_yagi_tapered`.
- `SMAConnector` model.

### Installation and tools

- **The Apple Accelerate solver is installed automatically on macOS.** `emerge-aasds` is now a
  precompiled PyPI package and a dependency of EMerge on macOS, so the fast Accelerate direct
  solver works out of the box.
- `pip install emerge[spyder]` installs the packages needed to use EMerge in Spyder.

### Experimental

- `Microwave3D.solve_cnosdd(...)`: domain decomposition solver.
- `ArrayTiling`/`ArrayLayout` define finite-array tilings.
- `MLPreconData`: exports training data for ML preconditioners.

---

## 2. Breaking API changes

### Installation and imports

- **numba is removed.** EMerge ships compiled wheels; building from source needs a Rust
  toolchain. The `emerge compile` command is removed.
- The **`gerber` extra is renamed to `cad`**.
- **The `emerge` command line tool is removed** (`emerge install-solver`, `python -m emerge`). Install
  optional solvers with pip, e.g. `pip install emerge[cad]`.
- The minimum `emsutil` version is 1.1.3.
- **`emerge.beta` is removed:**
  - `emerge.beta.gerber.FileBasedPCB` → `emerge.read.FileBasedPCB`
  - `emerge.beta.dxf.import_dxf` → `emerge.read.import_dxf`
  - `emerge.beta.dxf.export_dxf`, `pcb_to_dxf` → `emerge.write.export_dxf`, `emerge.write.pcb_to_dxf`
- `SimulationBeta` is removed.
- The module-level singletons (`_GEOMANAGER`, `LOG_CONTROLLER`, `DEBUG_COLLECTOR`, `SELECTOR_OBJ`,
  `_PCB_MANAGER`, `DEFAULT_ROUTINE`, …) are removed. `em.cleanup()` still works.

### Simulation

- **`commit_geometry()` takes no arguments anymore.**
- **The default save format is `msgpack`** (was `joblib`). msgpack cannot store Python functions,
  for example lumped element impedance functions; use `store_system="joblib"` for those.
- **`generate_mesh()` no longer removes orphan surfaces by default.** Pass `remove_orphans=True`.
- **`adaptive_mesh_refinement()` has new arguments.** `refinement_ratio`, `growth_rate`,
  `minimum_refinement_percentage`, `error_field_inclusion_percentage` and `ensure_mesh_growth`
  are removed. `adaptive_mesh_refinement_new()` is removed.
- `data.sim.iterate()` yields `DataEntry` objects (`.vars`, `.data`, `.reports`) instead of
  `(vars, dict)` tuples.

### Microwave

- **`run_adaptive_sweep()`**: `n_max_new` is replaced by `tol` and `max_samples`.
- **`MWField.farfield()` returns an `EHFieldFF`** (use `.E`, `.H`) instead of an `(E, H, Ptot)`
  tuple.
- **`farfield_3d()` arrays are indexed `(theta, phi)`** (were `(phi, theta)`).
- **`ScatteredField` uses standard spherical angles by default** (`definition='SPH'`). Pass
  `definition='EA'` for the old convention.
- **`embed_external_component()`:**
  - `MWField.embed_external_component()` no longer takes `Smat`.
  - Embedded ports are no longer deleted. They stay in `Sp` as `NaN` and keep their numbers.
  - The `MWScalarNdim` version modifies the dataset in place.
  - Exciting an embedded port raises `KeyError`.
- `MWScalarNdim.Smat` returns `Sp`, including `NaN` entries for embedded ports.
- `eigenmode(deep_search=...)` is removed.
- `Microwave3D.frequency_domain()` (the alias of `run_sweep()`) is removed.
- `BoundaryCondition.details()` is removed.
- `get_Uinc()` moved from the individual port classes to the `PortBC` base class.

### Solvers

- **`EMSolver` values are renumbered.** `SMART_ARPACK` is removed and `TEST`, `RSLAB` and
  `SPARTA` are added. Refer to solvers by name.
- `SmartARPACK` is removed and `SmartARPACK_BMA` is renamed to `SolverARPACK_BMA`.
- `CuDSSInterface.set_algorithm('COLAM_BT')` is now `'COLAMD_BT'`.

### Geometry, selections and meshing

- **`face()`, `faces()`, `boundary()` and `all_faces()` exist only on `GeoVolume`.** They are no
  longer available on surfaces, edges or points.
- **`GeoObject.material` is read-only.** Use `set_material()` or `obj.properties += mat`.
  `_mdi`, `_store()` and `_load()` are removed.
- **`Selection.exclude()`: the `plane` argument is removed.**
- `Selection.code()`, `encode_data()` and `decode_data()` are removed.
- `geo.stick()`: the first argument is renamed from `object` to `obj`.
- `Mesher.set_face_size()` on a non-surface raises `MeshError`. It used to warn and forward.
- `Mesher.set_algorithm()` uses 1 GMSH thread by default (was 16).
- The old AMR helpers are removed: `Mesher.add_refinement_points()`,
  `set_error_size_field()`, `clear_error_size_field()`, `AMRPoints` and the `adaptive_mesh`
  helper functions.

### PCB

- **`PCBNew` is renamed to `PCB`.** `PCBNew` is kept as an alias. The old deprecated `PCB`
  wrapper is removed.
- **z-coordinates are replaced by layer indices** (counted from 0; negative indices count from
  the top):

  | 2.8.9 | 3.0 |
  |---|---|
  | `pcb.new(x, y, w, dir, z=pcb.z(1))` | `pcb.new(x, y, w, dir, trace_layer=1)` (default `-1`, top) |
  | `pcb.plane(pcb.z(0))` | `pcb.plane(0)` |
  | `.via(pcb.z(0), radius)` | `.via(0, radius)` |
  | `pcb.add_poly(xs, ys, z=...)` / `add_hole(...)` | `(xs, ys, layer=0)` |
  | `pcb.add_vias(..., z1=, z2=)` | `pcb.add_vias(..., layer1=0, layer2=-1)` |
  | `pcb.radial_stub(..., z=)` | `pcb.radial_stub(..., layer=-1)` |
  | `lumped_element_from_sp(..., z_gnd=)` | `lumped_element_from_sp(..., ground_layer=0)` |
  | `StripLine/StripTurn/StripCurve/PCBPoly(..., z, ...)` | `(..., layer, ...)` |

- **Port methods take a port number and create the boundary condition:**

  | 2.8.9 | 3.0 |
  |---|---|
  | `mp = pcb.modal_port(pcb.load("p1"), height)` + `model.mw.bc.ModalPort(mp, 1)` | `pcb.modal_port("p1", 1, height)` |
  | `lp = pcb.lumped_port(pcb.load("p1"))` + `model.mw.bc.LumpedPort(lp, 1)` | `pcb.lumped_port("p1", 1)` |
  | `pcb.lumped_port_pts(p1, p2, z, z_ground)` | `pcb.lumped_port_pts(p1, p2, layer, port_number, ground_layer=0)` |

  Remove the separate `model.mw.bc.ModalPort(...)`/`LumpedPort(...)` calls and get the ports with
  `model.mw.bc.get_port(n)`.
- `trace_thickness` defaults to 35 µm (was `None`).

### Changed defaults that change results

- `Settings.qtem_limit` 0.05 → 0.1: more quasi-TEM modes are treated as TEM.
- `run_sweep()`/`run_scattered()`: `harddisc_path` is relative to the model directory.

### Migrating a 2.8.9 script

1. Update the version check: `model.check_version("3.0.0")`.
2. Replace `em.geo.PCBNew` with `em.geo.PCB`, and `pcb.z(i)` arguments with layer indices.
3. Give `pcb.modal_port()`/`pcb.lumped_port()` a port number, delete the matching
   `model.mw.bc.ModalPort(...)`/`LumpedPort(...)` calls, and use `model.mw.bc.get_port(n)`.
4. Remove arguments from `model.commit_geometry()`.
5. Use `store_system="joblib"` if you save lumped element functions.
6. Read far fields from the returned `EHFieldFF` (`ff.E`, `ff.H`), and check the axis order of
   `farfield_3d()` arrays.
7. Pass `definition='EA'` to `ScatteredField` if your angles use the old convention.
8. Replace `emerge.beta.*` imports with `emerge.read`/`emerge.write`.
9. Update `adaptive_mesh_refinement()` and `run_adaptive_sweep()` calls to the new arguments.

---

## 3. Bug fixes, accuracy and performance

### Bug fixes

- **`ThinConductor`:** the two sides of a thin conductor sheet were not separated in the solve.
  The volume was assembled with the unsplit numbering, only one tet per sheet triangle got the
  copied DoFs (not the tets that touch the sheet through an edge), and the copied side was picked
  at random per triangle. Now every tet on one side of the sheet uses the copies, and the solve,
  the fields, the far fields and the error estimate all use the same numbering.
- **Multithreaded sweeps:** every worker thread got the same solve routine, so threads replaced
  each other's solvers and could factorize in the same PARDISO handle at once (occasional
  segmentation faults in MKL). Each thread now gets its own routine and solvers, also for a solver
  forced with `set_solver()`.
- **Selections:** `Selection.exclude()` and `isolate()` returned wrong selections (`~` was used
  on Python booleans).
- **`ModalPort`/`WavePortIH`:** passing `cs=...` raised
  `ValueError: No Coordinate System could be derived`. A copy of the given `cs` is now used.
- **`SurfaceImpedance(surface_conductivity=...)`** was ignored.
- **`combine_ports()`:**
  - Non-sequential port numbers now work.
  - `Z0renorm` passed the impedances to the renormalization in the wrong order.
  - The new reference impedance is now stored.
- **`embed_external_component()`** used wrong port indices when port numbers were not 1…N.
- **`ScatteredField.set_excitations()`:** each theta × phi × polarization combination is now
  solved exactly once. Equal-length arrays were broadcast and then combined, which repeated
  combinations.
- **Thermal `FixedTemperature`** crashed (`node_to_ield` typo).
- **`HexCell`** used `max(z2, z2)` for the cell height.
- **Periodic face pairing** also compares face areas, so faces with the same centre are no longer
  mispaired.
- **Pardiso:** the zero right-hand-side vector had the wrong shape.
- **`XYPolygon` fillets** were inserted at shifted positions when more than one fillet was used.
- **STEP import:** integer values were rejected where floats were expected.
- **`Curve`** no longer opens the FLTK GUI.
- **cuDSS:**
  - Works with nvmath-python 0.5 and 1.0 and cuDSS 0.8.
  - `set_algorithm()` now takes effect.
  - A failing cuDSS import no longer breaks `import emerge`.
- Boolean operations no longer try to assign a missing material.

### Accuracy improvements

- **Triangle quadrature:** the 7-point rule (`STRANG6`) had a typo in two barycentric coordinates.
  Order-4 surface integrals and boundary terms that used it were slightly inaccurate.
- **Second-order ABC:** separate TE (surface curl) and TM (surface divergence) correction terms
  improve absorption at oblique incidence.
- **Modal analysis:**
  - The εr/µr estimate is taken over the port only (it used to include values from the whole
    mesh).
  - H-field interpolation uses the port permeability.
  - Non-propagating modes are ignored.
  - Mode signs are made consistent between meshes (a warning is shown when this is ambiguous).
- **Surface currents (`current_boundary()`):** samples several offsets, weights by triangle area
  and removes the normal component.
- **Robin boundary conditions** use a higher-order quadrature.
- **Inconsistent GMSH meshes** (tetrahedra that reference missing nodes) are detected and
  reported with a fix, instead of failing later with an unexplained `KeyError`.
- Volumes with negative orientation are rebuilt automatically.
- Lumped port and lumped element mesh-size constraints follow the model scale factor.

### Performance

- No JIT compilation: the first-run delay is gone.
- Faster assembly:
  - Rust kernels.
  - A sparsity pattern that is cached across frequencies.
  - Tetrahedra inside PEC or highly conducting volumes are skipped.
- Lower memory use in Robin boundary condition assembly.
- SuperLU solves several right-hand sides in one call.
- ARPACK uses an explicit shift-invert LU with an oversampled Krylov space. `SolverARPACK_BMA`
  filters modes with a cross-modal assurance criterion.
- With multiprocessing, the worker pool and each worker's solver state are reused across
  frequency groups, so cached symbolic factorizations survive.

---

## 4. Internal changes

- All numba code is ported to the Rust crate `emr`, built with maturin. This covers curl-curl,
  Robin, ABC, periodic, eigenmode, Stratton-Chu, error estimation, interpolation, integrals and
  heat kernels.
- Global state lives in one `GLOBALHANDLER`: geometry manager, key generator, PCB manager,
  simulation states, selector, log controller and debug collector.
- `MaterialAssignment` replaces `_get_material_assignment()`. Material properties are evaluated
  with tetrahedron centres and IDs.
- `run_sweep()` and `run_scattered()` share one frequency-domain driver (`_run_frequency_domain`,
  `_dispatch_jobs`).
- The default boundary conditions are generated once (`_autogenerate_bcs`), and the boundary
  condition data is initialized once per mesh.
- Assembly uses a `SystemPattern` with a cached COO→CSC map. Explicit zero elimination is
  removed.
- `diagnose_matrix()` raises `MatrixDiagnosisError` with the coordinates of the problematic
  degrees of freedom.
- The microwave and heat physics of a `Simulation` share one solve routine.
- Face tags of named faces are cached on `GeoVolume`.
- `GeoObject._become()` replaces an object in place, for example when voids are subtracted.
- Launch messages and the version banner are only shown in the main process. Spyder is detected and
  the PyVista backend is set automatically.
- More informative logging:
  - every boundary condition and its attributes;
  - the port impedance with its definition;
  - the mesh size after meshing;
  - frequency sanity warnings for all solve entry points.
- The face-label view shows `object<face>` names.
- `emerge.plot` is its own module, re-exporting the emsutil plotting functions.
- New test suite: thermal, WPBC, solvers, save/load, ABC, AMR, combine ports, background fields,
  impedances, power, near field and demo references.
- The release workflow builds abi3 wheels for Linux, macOS and Windows and publishes to PyPI
  when the version in `pyproject.toml` is new.
