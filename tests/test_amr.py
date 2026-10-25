def _post_in_waveguide():
    import emerge as em

    em.cleanup()
    model = em.Simulation("AMR", loglevel="INFO")
    a, b, L = 0.02286, 0.01016, 0.04
    box = em.geo.Box(a, L, b)
    post = em.geo.Cylinder(0.002, b, em.ZAX.construct_cs((a / 3, L / 2, 0)))
    wg = em.geo.subtract(box, post)
    model.commit_geometry()
    model.mw.set_frequency_range(9e9, 11e9, 3)
    model.mw.set_resolution(0.2)
    model.generate_mesh()
    p1 = model.mw.bc.ModalPort(wg.face("front", tool=box), 1)
    p2 = model.mw.bc.ModalPort(wg.face("back", tool=box), 2)
    p1.align_modes(em.ZAX)
    p2.align_modes(em.ZAX)
    return model


def test_amr_local_refinement():
    """Three local refinement passes: the mesh grows, stays valid and the S-parameters settle."""
    import emerge as em
    import numpy as np

    model = _post_in_waveguide()
    n0 = model.mesh.n_tets
    model.adaptive_mesh_refinement(max_steps=3, frequency=10e9, convergence=1e-9)
    n1 = model.mesh.n_tets
    assert n1 > 1.3 * n0, (n0, n1)
    model.mesh.diagnose()

    data = model.mw.run_sweep(False)
    S21 = np.abs(data.scalar.grid.S(2, 1))
    assert np.all(S21 > 0.05) and np.all(S21 < 1.0 + 1e-6)
    em.cleanup()


def test_amr_thin_conductor():
    """AMR on a finite conductivity (ThinConductor) trace: the refined basis needs its DoF split too."""
    import emerge as em
    import numpy as np

    em.cleanup()
    mm = 0.001
    model = em.Simulation("AMRThin", loglevel="ERROR")
    pcb = em.geo.PCB(1.6, unit=mm, material=em.Material(er=4.2), trace_material=em.Material(cond=5.8e7))
    pcb.new(0, 0, 3.0, (1, 0)).store("p1").straight(20).store("p2")
    pcb.compile_paths(merge=True)
    pcb.determine_bounds(topmargin=10, bottommargin=10)
    pcb.generate_pcb()
    pcb.generate_air(5)
    pcb.modal_port(pcb["p1"], 1, width_multiplier=5, height=5)
    pcb.modal_port(pcb["p2"], 2, width_multiplier=5, height=5)
    model.commit_geometry()
    model.mw.set_frequency_range(2e9, 3e9, 2)
    model.mw.set_resolution(0.3)
    model.generate_mesh()

    n0 = model.mesh.n_tets
    model.adaptive_mesh_refinement(max_steps=2, frequency=3e9, convergence=1e-9)
    from emerge._emerge.physics.microwave.bcs import ThinConductor
    assert len(model.mw.bc.oftype(ThinConductor)) > 0
    assert model.mesh.n_tets > n0

    S21 = np.abs(model.mw.run_sweep(False).scalar.grid.S(2, 1))
    assert np.all(S21 > 0.9) and np.all(S21 < 1.0 + 1e-6)
    em.cleanup()


def test_amr_periodic():
    """AMR on a periodic unit cell: paired faces are refined identically and power is conserved."""
    import emerge as em
    import numpy as np
    from emerge._emerge.aura.emerge_tools import periodic_report

    mm = 1e-3
    em.cleanup()
    model = em.Simulation("AMRPeriodic", loglevel="ERROR")
    cell = em.RectCell(10 * mm, 10 * mm)
    model.set_periodic_cell(cell)
    cell.volume(0, 30 * mm)
    em.geo.Sphere(3 * mm, position=(0, 0, 15 * mm)).set_material(em.Material(6))
    top, bot = cell.port_face(30 * mm), cell.port_face(0)
    cell.set_scanangle(30, 20)
    model.commit_geometry()
    model.mw.set_frequency(10e9)
    model.mw.set_resolution(0.2)
    model.generate_mesh()
    model.mw.bc.floquet_port(top, 1)
    model.mw.bc.floquet_port(bot, 2)

    periodic = [(bc.face1.tags, bc.face2.tags, np.asarray(bc.dv)) for bc in model.mesher._get_periodic_bcs()]
    tris_before = [r["n_tris"][0] for r in periodic_report(model.mesh, periodic)]
    n0 = model.mesh.n_tets
    model.adaptive_mesh_refinement(max_steps=3, frequency=10e9, convergence=1e-9)
    assert model.mesh.n_tets > 1.3 * n0

    reports = periodic_report(model.mesh, periodic)
    assert sum(r["n_tris"][0] for r in reports) > sum(tris_before), "periodic faces were not refined"
    for r in reports:
        assert r["n_tris"][0] == r["n_tris"][1]
        assert r["unmatched"] == 0
        assert r["order_preserving"]

    model.mw.run_sweep()
    powers = model.mw._check_port_powers()
    assert abs(abs(powers[1]) - 1.0) < 1e-4
    assert abs(abs(powers[2]) - 1.0) < 1e-4
    em.cleanup()


# [Written by Claude]
def test_aura_reflex_edge_on_sphere():
    """A surface triangulation over the reflex diagonal of four sphere points leaves an air wedge tet on the
    concave side. Its edge midpoint cannot be projected onto the sphere without inverting the wedge, which left
    the new node on the chord (a dent). AURA moves the wedge to the other region first, so no node of the
    refined sphere surface lies off the sphere."""
    import numpy as np
    from emerge._emerge.aura import refine
    from emerge._emerge.aura.brep import SURF_SPHERE, BrepSnapshot

    s, t = 0.3, 0.5
    c, d = np.sqrt(1 - s * s), np.sqrt(1 - t * t)
    nodes = np.array([[s, 0, c], [-s, 0, c], [0, t, d], [0, -t, d], [0, 0, 0.3], [0, 0, 3.0]])
    # wedge (0, 1, 2, 3) in the air (region 1), the sphere surface over the reflex diagonal (2, 3)
    tets = np.array([[0, 1, 2, 3], [0, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 5], [0, 1, 3, 5]])
    region = np.array([1, 2, 2, 1, 1], dtype=np.int32)
    face_tris = np.array([[0, 2, 3], [1, 2, 3]])
    face_tags = np.array([7, 7], dtype=np.int32)
    brep = BrepSnapshot.empty()
    brep.surf_tags = np.array([7], dtype=np.int32)
    brep.surf_kind = np.array([SURF_SPHERE], dtype=np.int32)
    brep.surf_params = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float64)
    error = np.array([1.0, 0.1, 0.1, 0.0, 0.0])
    res = refine(nodes, tets, error, tet_region=region, face_tris=face_tris, face_tri_tags=face_tags, brep=brep,
                 growth=0.5, improve_sweeps=0)
    assert res.stats["n_wedge_flips"] == 1
    assert res.stats["n_projection_reduced"] == 0 and res.stats["n_projection_rejected"] == 0
    on = np.unique(np.asarray(res.face_tris)[np.asarray(res.face_tri_tags) == 7])
    r = np.linalg.norm(np.asarray(res.nodes)[on], axis=1)
    assert np.allclose(r, 1.0, atol=1e-12), r


# [Written by Claude]
def _aura_passes(obj_factory, passes, cavity, two_sided=False):
    """A few AURA passes on an object in an air box with an error that peaks at the object surface.
    ``min_projected_volume=0.9`` makes almost every projection of a new surface node fail for plain
    bisection, so the cavity re-meshing (hzAURA) has to do all of them."""
    import emerge as em
    import gmsh
    import numpy as np
    from scipy.spatial import cKDTree
    from emerge._emerge.aura import refine
    from emerge._emerge.aura.brep import snapshot_gmsh
    from emerge._emerge.aura.emerge_tools import mesh_to_aura_inputs

    em.cleanup()
    model = em.Simulation("hzAURA", loglevel="ERROR")
    air = em.geo.Box(1.2, 1.2, 1.2, position=(-0.6, -0.6, -0.6))
    obj = obj_factory(em)
    if two_sided:
        obj.set_material(em.Material(er=4.0))
        obj.prio_up()
    else:
        em.geo.subtract(air, obj)
    model.commit_geometry()
    model.mw.set_frequency(1e9)
    model.mw.set_resolution(0.25)
    model.generate_mesh()
    brep = snapshot_gmsh()
    curved = [t for _, t in gmsh.model.getEntities(2) if gmsh.model.getType(2, t) != "Plane"]
    x = mesh_to_aura_inputs(model.mesh)
    nodes, tets, region = x["nodes"], x["tets"], x["tet_region"]
    ft, fg, ce, cg = x["face_tris"], x["face_tri_tags"], x["curve_edges"], x["curve_edge_tags"]
    rng = np.random.default_rng(1)
    stats = {}
    for _ in range(passes):
        surf = np.unique(ft[np.isin(fg, curved)])
        d, _ = cKDTree(nodes[surf]).query(nodes[tets].mean(axis=1))
        err = rng.uniform(0.5, 1.5, len(tets)) / (d + 1e-3)
        r = refine(nodes, tets, err, region, ft, fg, ce, cg, None, brep, growth=0.3, min_projected_volume=0.9,
                   cavity=cavity)
        nodes, tets, region = r.nodes, r.tets, r.tet_region
        ft, fg, ce, cg = r.face_tris, r.face_tri_tags, r.curve_edges, r.curve_edge_tags
        for k, v in r.stats.items():
            stats[k] = stats.get(k, 0) + v
    # valid and conforming: same orientation everywhere, every boundary face is a tagged face
    p = nodes[tets]
    v = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    assert np.all(v * np.sign(v[0]) > 0)
    faces = np.sort(np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]]), axis=1)
    uf, cnt = np.unique(faces, axis=0, return_counts=True)
    assert cnt.max() <= 2
    assert {tuple(f) for f in uf[cnt == 1]} <= {tuple(f) for f in np.sort(ft, axis=1)}
    surf = np.unique(ft[np.isin(fg, curved)])
    em.cleanup()
    return nodes[surf], stats


# [Written by Claude]
def test_hzaura_sphere_and_cylinder():
    """hzAURA: where bisection cannot move a new node onto the curved face, the cavity around it is re-meshed
    and the node lies on the surface; plain AURA leaves it on the chord (a dent)."""
    import numpy as np

    sphere = lambda em: em.geo.Sphere(0.25)  # noqa: E731
    cylinder = lambda em: em.geo.Cylinder(0.2, 0.5, em.ZAX.construct_cs((0, 0, -0.25)))  # noqa: E731
    for factory, dist, two_sided in [
        (sphere, lambda x: np.abs(np.linalg.norm(x, axis=1) - 0.25), False),
        (cylinder, lambda x: np.abs(np.linalg.norm(x[:, :2], axis=1) - 0.2), True),
    ]:
        x, st = _aura_passes(factory, 3, cavity=False, two_sided=two_sided)
        assert st["n_projection_reduced"] > 0 and dist(x).max() > 1e-4
        x, st = _aura_passes(factory, 3, cavity=True, two_sided=two_sided)
        assert st["n_cavity"] > 0, st
        assert st["n_cavity_failed"] == 0 and dist(x).max() < 1e-9, (dist(x).max(), st)


# [Written by Claude]
def test_hzaura_freeform_face():
    """hzAURA on a lofted B-spline body (sampled face, bicubic interpolation of the CAD parameter grid):
    new surface nodes stay within 0.5 mm of the true face."""
    import gmsh
    import numpy as np
    from emerge._emerge.geometry import GeoVolume

    def blob(em):
        occ = gmsh.model.occ
        wires = []
        for z, r0, a in [(-0.25, 0.18, 0.04), (0.0, 0.24, -0.05), (0.25, 0.15, 0.03)]:
            pts = [occ.addPoint((r0 + a * np.cos(3 * t + 4 * z)) * np.cos(t), (r0 + a * np.cos(3 * t + 4 * z)) * np.sin(t), z)
                   for t in np.linspace(0, 2 * np.pi, 13)[:-1]]
            wires.append(occ.addWire([occ.addBSpline(pts + [pts[0]])]))
        vols = occ.addThruSections(wires, makeSolid=True, makeRuled=False)
        occ.synchronize()
        blob.tags = [t for d, t in gmsh.model.getEntities(2) if gmsh.model.getType(2, t) != "Plane"]
        return GeoVolume([t for d, t in vols if d == 3])

    import emerge as em  # noqa: F401
    x, st = _aura_passes(blob, 3, cavity=True)
    assert st["n_cavity"] > 0, st
    # _aura_passes cleans up gmsh, so measure against a fresh copy of the same body
    gmsh.initialize() if not gmsh.isInitialized() else None
    gmsh.model.add("blob")
    blob(None)
    best = np.full(len(x), np.inf)
    for tag in blob.tags:
        cp = np.asarray(gmsh.model.getClosestPoint(2, tag, x.ravel())[0]).reshape(-1, 3)
        best = np.minimum(best, np.linalg.norm(cp - x, axis=1))
    gmsh.finalize()
    # nodes of the flat caps are far from the side face; only nodes near it count
    near = best < 0.02
    assert near.sum() > 100
    assert best[near].max() < 5e-4, best[near].max()
