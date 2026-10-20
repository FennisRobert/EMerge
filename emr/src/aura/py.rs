//! Python bindings of the AURA local mesh refinement: numpy arrays in, refined mesh dictionary out.
use crate::aura::geom::{Brep, Curve, Surface, P3};
use crate::aura::improve::{ImproveParams, Improver};
use crate::aura::refine::{mean_ratio, Params, Refiner};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

fn rows3(a: &PyReadonlyArray2<f64>) -> PyResult<Vec<P3>> {
    let v = a.as_array();
    if v.ncols() != 3 {
        return Err(PyValueError::new_err("expected an (n, 3) float array"));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn rows_u32<const K: usize>(a: &PyReadonlyArray2<i64>) -> PyResult<Vec<[u32; K]>> {
    let v = a.as_array();
    if v.nrows() > 0 && v.ncols() != K {
        return Err(PyValueError::new_err(format!("expected an (n, {K}) integer array")));
    }
    Ok(v.rows().into_iter().map(|r| std::array::from_fn(|k| r[k] as u32)).collect())
}

fn to_2d<T: numpy::Element>(py: Python<'_>, data: Vec<T>, ncol: usize) -> PyObject {
    let n = if ncol == 0 { 0 } else { data.len() / ncol };
    Array2::from_shape_vec((n, ncol), data)
        .expect("shape")
        .into_pyarray_bound(py)
        .into_any()
        .unbind()
}

/// Build the BREP snapshot from flat descriptor arrays.
///
/// Surface kinds: 0 flat, 1 sphere [c, r], 2 cylinder [p, axis, r], 3 sampled.
/// Curve kinds:   0 straight, 1 circle [c, n, r], 2 polyline.
#[allow(clippy::too_many_arguments)]
fn build_brep(
    surf_tags: &[i32],
    surf_kind: &[i32],
    surf_params: &[[f64; 7]],
    samp_owner: &[i32],
    samp_pts: &[P3],
    samp_nrm: &[P3],
    curve_tags: &[i32],
    curve_kind: &[i32],
    curve_params: &[[f64; 7]],
    poly_owner: &[i32],
    poly_pts: &[P3],
) -> PyResult<Brep> {
    let mut brep = Brep::default();
    for (i, &tag) in surf_tags.iter().enumerate() {
        let p = surf_params[i];
        let s = match surf_kind[i] {
            0 => Surface::Flat,
            1 => Surface::Sphere {
                c: [p[0], p[1], p[2]],
                r: p[3],
            },
            2 => Surface::Cylinder {
                p: [p[0], p[1], p[2]],
                axis: [p[3], p[4], p[5]],
                r: p[6],
            },
            3 => {
                let idx: Vec<usize> = (0..samp_owner.len()).filter(|&k| samp_owner[k] == tag).collect();
                Surface::Sampled {
                    pts: idx.iter().map(|&k| samp_pts[k]).collect(),
                    nrm: idx.iter().map(|&k| samp_nrm[k]).collect(),
                }
            }
            k => return Err(PyValueError::new_err(format!("unknown surface kind {k}"))),
        };
        brep.surfaces.insert(tag, s);
    }
    for (i, &tag) in curve_tags.iter().enumerate() {
        let p = curve_params[i];
        let c = match curve_kind[i] {
            0 => Curve::Straight,
            1 => Curve::Circle {
                c: [p[0], p[1], p[2]],
                n: [p[3], p[4], p[5]],
                r: p[6],
            },
            2 => Curve::Polyline {
                pts: (0..poly_owner.len())
                    .filter(|&k| poly_owner[k] == tag)
                    .map(|k| poly_pts[k])
                    .collect(),
            },
            k => return Err(PyValueError::new_err(format!("unknown curve kind {k}"))),
        };
        brep.curves.insert(tag, c);
    }
    Ok(brep)
}

fn params7(a: &PyReadonlyArray2<f64>) -> Vec<[f64; 7]> {
    a.as_array()
        .rows()
        .into_iter()
        .map(|r| std::array::from_fn(|k| if k < r.len() { r[k] } else { 0.0 }))
        .collect()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    nodes, tets, error, tet_region, face_tris, face_tags, curve_edges, curve_edge_tags, periodic_pairs,
    surf_tags, surf_kind, surf_params, samp_owner, samp_pts, samp_nrm,
    curve_tags, curve_kind, curve_params, poly_owner, poly_pts,
    growth, child_factor, max_depth, min_projected_volume, lepp_ratio,
    improve_sweeps, flip_quality, smooth_iters, improve_layers, improve_global,
    strip_ratio, strip_max_growth, strip_gated
))]
fn refine<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    tets: PyReadonlyArray2<'py, i64>,
    error: PyReadonlyArray1<'py, f64>,
    tet_region: PyReadonlyArray1<'py, i32>,
    face_tris: PyReadonlyArray2<'py, i64>,
    face_tags: PyReadonlyArray1<'py, i32>,
    curve_edges: PyReadonlyArray2<'py, i64>,
    curve_edge_tags: PyReadonlyArray1<'py, i32>,
    periodic_pairs: PyReadonlyArray2<'py, i64>,
    surf_tags: PyReadonlyArray1<'py, i32>,
    surf_kind: PyReadonlyArray1<'py, i32>,
    surf_params: PyReadonlyArray2<'py, f64>,
    samp_owner: PyReadonlyArray1<'py, i32>,
    samp_pts: PyReadonlyArray2<'py, f64>,
    samp_nrm: PyReadonlyArray2<'py, f64>,
    curve_tags: PyReadonlyArray1<'py, i32>,
    curve_kind: PyReadonlyArray1<'py, i32>,
    curve_params: PyReadonlyArray2<'py, f64>,
    poly_owner: PyReadonlyArray1<'py, i32>,
    poly_pts: PyReadonlyArray2<'py, f64>,
    growth: f64,
    child_factor: f64,
    max_depth: u8,
    min_projected_volume: f64,
    lepp_ratio: f64,
    improve_sweeps: usize,
    flip_quality: f64,
    smooth_iters: usize,
    improve_layers: usize,
    improve_global: bool,
    strip_ratio: f64,
    strip_max_growth: f64,
    strip_gated: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let nodes = rows3(&nodes)?;
    let tets = rows_u32::<4>(&tets)?;
    let tets_len = tets.len();
    let error = error.as_slice()?.to_vec();
    if error.len() != tets.len() {
        return Err(PyValueError::new_err("error must have one value per tetrahedron"));
    }
    let region_in = tet_region.as_slice()?.to_vec();
    if region_in.len() != tets.len() {
        return Err(PyValueError::new_err("tet_region must have one value per tetrahedron"));
    }
    let ft = rows_u32::<3>(&face_tris)?;
    let ftag = face_tags.as_slice()?;
    let faces: Vec<([u32; 3], i32)> = ft.into_iter().zip(ftag.iter().copied()).collect();
    let ce = rows_u32::<2>(&curve_edges)?;
    let ctag = curve_edge_tags.as_slice()?;
    let curves: Vec<((u32, u32), i32)> = ce.into_iter().map(|e| (e[0], e[1])).zip(ctag.iter().copied()).collect();

    let brep = build_brep(
        surf_tags.as_slice()?,
        surf_kind.as_slice()?,
        &params7(&surf_params),
        samp_owner.as_slice()?,
        &rows3(&samp_pts)?,
        &rows3(&samp_nrm)?,
        curve_tags.as_slice()?,
        curve_kind.as_slice()?,
        &params7(&curve_params),
        poly_owner.as_slice()?,
        &rows3(&poly_pts)?,
    )?;
    let params = Params {
        growth,
        child_factor,
        max_depth,
        min_projected_volume,
        lepp_ratio,
    };

    let pairs: Vec<(usize, u32, u32)> = rows_u32::<3>(&periodic_pairs)?
        .into_iter()
        .map(|r| (r[0] as usize, r[1], r[2]))
        .collect();
    let iparams = ImproveParams {
        sweeps: improve_sweeps,
        flip_quality,
        smooth_iters,
        layers: improve_layers,
        global: improve_global,
    };

    let (mut out, region, istats) = py.allow_threads(|| {
        let mut r = Refiner::new(nodes, tets, error, &faces, &curves, &brep, params);
        r.set_periodic(&pairs);
        let limit = |n: usize| (n as f64 * (1.0 + strip_max_growth)).round() as usize;
        if strip_ratio > 0.0 && !strip_gated {
            r.regularize_strips(strip_ratio, limit(tets_len));
        }
        r.run();
        if strip_ratio > 0.0 && strip_gated {
            let n = r.n_tets();
            r.regularize_strips_gated(strip_ratio, limit(n));
        }
        let periodic_nodes = r.periodic_nodes();
        let mut out = r.finish();
        let region: Vec<i32> = out.tet_root.iter().map(|&i| region_in[i as usize]).collect();
        if iparams.sweeps == 0 {
            return (out, region, Default::default());
        }
        let curve_set: HashSet<(u32, u32)> = out.curve_edges.iter().copied().collect();
        let tets = std::mem::take(&mut out.tets);
        let root = std::mem::take(&mut out.tet_root);
        let active = std::mem::take(&mut out.tet_new);
        let face_list: Vec<([u32; 3], i32)> = out.face_tris.iter().copied().zip(out.face_tri_tags.iter().copied()).collect();
        let planar: HashSet<i32> = brep
            .surfaces
            .iter()
            .filter(|(_, srf)| matches!(srf, Surface::Flat))
            .map(|(t, _)| *t)
            .collect();
        let mut imp = Improver::new(&mut out.nodes, tets, region, root, active, &face_list, &curve_set, iparams);
        imp.lock_nodes(&periodic_nodes);
        imp.set_planar_tags(&planar);
        imp.run();
        let istats = imp.stats.clone();
        let (t, r, o, a, faces) = imp.finish();
        out.tets = t;
        out.tet_root = o;
        out.tet_new = a;
        out.face_tris = faces.iter().map(|x| x.0).collect();
        out.face_tri_tags = faces.iter().map(|x| x.1).collect();
        (out, r, istats)
    });
    let tet_new: Vec<bool> = std::mem::take(&mut out.tet_new);

    let d = PyDict::new_bound(py);
    d.set_item("nodes", to_2d(py, out.nodes.concat(), 3))?;
    d.set_item("tets", to_2d(py, out.tets.iter().flatten().map(|&x| x as i64).collect(), 4))?;
    d.set_item(
        "tet_root",
        out.tet_root.iter().map(|&x| x as i64).collect::<Vec<_>>().into_pyarray_bound(py),
    )?;
    d.set_item("tet_region", region.into_pyarray_bound(py))?;
    d.set_item("tet_new", tet_new.into_pyarray_bound(py))?;
    d.set_item(
        "face_tris",
        to_2d(py, out.face_tris.iter().flatten().map(|&x| x as i64).collect(), 3),
    )?;
    d.set_item("face_tri_tags", out.face_tri_tags.into_pyarray_bound(py))?;
    d.set_item(
        "curve_edges",
        to_2d(py, out.curve_edges.iter().flat_map(|e| [e.0 as i64, e.1 as i64]).collect(), 2),
    )?;
    d.set_item("curve_edge_tags", out.curve_edge_tags.into_pyarray_bound(py))?;
    let s = PyDict::new_bound(py);
    s.set_item("n_splits", out.stats.n_splits)?;
    s.set_item("n_projected", out.stats.n_projected)?;
    s.set_item("n_projection_reduced", out.stats.n_projection_reduced)?;
    s.set_item("n_projection_rejected", out.stats.n_projection_rejected)?;
    s.set_item("n_strip_splits", out.stats.n_strip_splits)?;
    s.set_item("n_flip23", istats.n_flip23)?;
    s.set_item("n_flip32", istats.n_flip32)?;
    s.set_item("n_flip44", istats.n_flip44)?;
    s.set_item("n_flip_large", istats.n_flip_large)?;
    s.set_item("n_improve_zone", istats.n_zone)?;
    s.set_item("n_smoothed", istats.n_smoothed)?;
    s.set_item("n_slid", istats.n_slid)?;
    s.set_item("n_surface_flips", istats.n_surface_flips)?;
    d.set_item("stats", s)?;
    Ok(d)
}

/// Mean-ratio quality per tetrahedron (1 = regular, 0 = degenerate).
#[pyfunction]
fn quality<'py>(py: Python<'py>, nodes: PyReadonlyArray2<'py, f64>, tets: PyReadonlyArray2<'py, i64>) -> PyResult<PyObject> {
    let nodes = rows3(&nodes)?;
    let tets = rows_u32::<4>(&tets)?;
    let q: Vec<f64> = tets
        .iter()
        .map(|t| {
            let x = t.map(|i| nodes[i as usize]);
            mean_ratio(x[0], x[1], x[2], x[3])
        })
        .collect();
    Ok(q.into_pyarray_bound(py).into_any().unbind())
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(refine, m)?)?;
    m.add_function(wrap_pyfunction!(quality, m)?)?;
    Ok(())
}
