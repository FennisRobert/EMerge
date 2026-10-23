// [Written by Claude] All code in this module was written by Claude (AI).
//! Conforming local refinement of a tetrahedral mesh by longest-edge bisection.
//!
//! Refinement is done exclusively by *edge splits*: an edge `(a, b)` receives a
//! midpoint `m` and every tetrahedron in the shell around that edge is replaced
//! by its two halves. The mesh is therefore conforming after every single
//! operation, so no hanging-node closure step is required.
//!
//! To keep element quality bounded, a tetrahedron is only ever bisected through
//! its own longest edge (Rivara's LEPP strategy): before splitting edge `e`,
//! every tet in its shell whose longest edge is not `e` is first refined
//! itself. Edges are compared with a strict total order (length, then node ids)
//! so the propagation path strictly increases and always terminates.
//!
//! Geometric classification is carried along with every split:
//! * a sub-segment of a tagged curve edge keeps the curve tag(s),
//! * a sub-triangle of a tagged face triangle keeps the face tag(s),
//! * a child tetrahedron remembers the index of its original root tetrahedron.
//!
//! New nodes on a tagged curve or face are projected back onto the BREP snapshot.
//!
//! Periodic boundaries are supported through node maps (one per periodic pair,
//! source face node -> target face node). An edge on a periodic face is always
//! split together with all of its periodic images, so paired faces stay exact
//! translated copies. The new midpoints of such an orbit get consecutive ids in
//! the lexicographic order of their parent edges, which keeps every map order
//! preserving (x < y <=> M(x) < M(y)), as EMerge requires for consistent edge
//! orientations across periodic faces.

use crate::aura::geom::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

pub type Edge = (u32, u32);
pub type Face = [u32; 3];

#[inline]
fn ekey(a: u32, b: u32) -> Edge {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[inline]
fn fkey(a: u32, b: u32, c: u32) -> Face {
    let mut f = [a, b, c];
    f.sort_unstable();
    f
}

const TET_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

#[inline]
pub fn vol6(a: P3, b: P3, c: P3, d: P3) -> f64 {
    dot(sub(b, a), cross(sub(c, a), sub(d, a)))
}

/// Mean-ratio quality: 1 for a regular tetrahedron, 0 for a degenerate one.
pub fn mean_ratio(a: P3, b: P3, c: P3, d: P3) -> f64 {
    let v = vol6(a, b, c, d).abs() / 6.0;
    let pts = [a, b, c, d];
    let mut l2 = 0.0;
    for (i, j) in TET_EDGES {
        let e = sub(pts[i], pts[j]);
        l2 += dot(e, e);
    }
    if l2 == 0.0 {
        return 0.0;
    }
    12.0 * (3.0 * v).powf(2.0 / 3.0) / l2
}

#[derive(Clone, Copy, PartialEq)]
struct Prio(f64, u32);
impl Eq for Prio {}
impl PartialOrd for Prio {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Prio {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then_with(|| o.1.cmp(&self.1))
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    /// Target relative increase of the number of tetrahedra (0.3 = +30 %).
    pub growth: f64,
    /// Error multiplier assigned to both children of a bisected tetrahedron.
    pub child_factor: f64,
    /// Maximum number of bisection generations per pass for a selected tet.
    pub max_depth: u8,
    /// Minimum child volume (relative to an exact half) accepted after projection.
    pub min_projected_volume: f64,
    /// A neighbour may be bisected through a shared edge that is not its own
    /// longest edge if that edge is at least this fraction of its longest edge.
    /// 1.0 is strict longest-edge propagation (LEPP); lower values limit how far
    /// refinement spreads into large neighbouring elements, at some cost in quality.
    pub lepp_ratio: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            growth: 0.3,
            child_factor: 2f64.powf(-(2.0 / 3.0 + 0.5)),
            max_depth: 6,
            min_projected_volume: 0.2,
            lepp_ratio: 0.7,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub n_splits: usize,
    pub n_projected: usize,
    pub n_projection_reduced: usize,
    pub n_projection_rejected: usize,
    /// Curve edges split by the narrow-strip regularisation (step 0).
    pub n_strip_splits: usize,
}

pub struct Refiner<'a> {
    pub nodes: Vec<P3>,
    pub tets: Vec<[u32; 4]>,
    alive: Vec<bool>,
    pub root: Vec<u32>,
    err: Vec<f64>,
    depth: Vec<u8>,
    shell: HashMap<Edge, Vec<u32>>,
    pub face_tags: HashMap<Face, Vec<i32>>,
    pub curve_tags: HashMap<Edge, Vec<i32>>,
    brep: &'a Brep,
    params: Params,
    n_alive: usize,
    heap: BinaryHeap<Prio>,
    n0: usize,
    /// Lowest original error among the tets chosen by the error-driven refinement
    /// of this pass (the "high error" cut-off used to gate strip regularisation).
    refined_cutoff: f64,
    /// When set, only strip edges next to a tet whose original error is at least
    /// this value are regularised.
    strip_gate: Option<f64>,
    /// Periodic node maps, one (forward, inverse) pair per periodic boundary.
    pmap: Vec<(HashMap<u32, u32>, HashMap<u32, u32>)>,
    pub stats: Stats,
}

pub struct Output {
    pub nodes: Vec<P3>,
    pub tets: Vec<[u32; 4]>,
    pub tet_root: Vec<u32>,
    /// True for tets created in this pass (as opposed to untouched input tets).
    pub tet_new: Vec<bool>,
    pub face_tris: Vec<Face>,
    pub face_tri_tags: Vec<i32>,
    pub curve_edges: Vec<Edge>,
    pub curve_edge_tags: Vec<i32>,
    pub stats: Stats,
}

impl<'a> Refiner<'a> {
    pub fn new(
        nodes: Vec<P3>,
        tets: Vec<[u32; 4]>,
        err: Vec<f64>,
        face_tris: &[(Face, i32)],
        curve_edges: &[(Edge, i32)],
        brep: &'a Brep,
        params: Params,
    ) -> Self {
        let n = tets.len();
        let mut shell: HashMap<Edge, Vec<u32>> = HashMap::with_capacity(n * 2);
        for (it, t) in tets.iter().enumerate() {
            for (i, j) in TET_EDGES {
                shell.entry(ekey(t[i], t[j])).or_default().push(it as u32);
            }
        }
        let mut face_tags: HashMap<Face, Vec<i32>> = HashMap::new();
        for (f, tag) in face_tris {
            let v = face_tags.entry(fkey(f[0], f[1], f[2])).or_default();
            if !v.contains(tag) {
                v.push(*tag);
            }
        }
        let mut curve_tags: HashMap<Edge, Vec<i32>> = HashMap::new();
        for (e, tag) in curve_edges {
            let v = curve_tags.entry(ekey(e.0, e.1)).or_default();
            if !v.contains(tag) {
                v.push(*tag);
            }
        }
        let heap = err
            .iter()
            .enumerate()
            .map(|(i, &e)| Prio(if e.is_finite() { e } else { 0.0 }, i as u32))
            .collect();
        Refiner {
            nodes,
            alive: vec![true; n],
            root: (0..n as u32).collect(),
            depth: vec![0; n],
            err,
            tets,
            shell,
            face_tags,
            curve_tags,
            brep,
            params,
            n_alive: n,
            heap,
            n0: n,
            refined_cutoff: f64::INFINITY,
            strip_gate: None,
            pmap: vec![],

            stats: Stats::default(),
        }
    }

    /// Register periodic node pairs as (map index, source node, target node).
    pub fn set_periodic(&mut self, pairs: &[(usize, u32, u32)]) {
        let n_maps = pairs.iter().map(|p| p.0 + 1).max().unwrap_or(0);
        self.pmap = vec![(HashMap::new(), HashMap::new()); n_maps];
        for &(k, a, b) in pairs {
            self.pmap[k].0.insert(a, b);
            self.pmap[k].1.insert(b, a);
        }
    }

    /// All nodes that take part in a periodic map.
    pub fn periodic_nodes(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.pmap.iter().flat_map(|(f, _)| f.keys().chain(f.values()).copied()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The edge together with all its periodic images, sorted lexicographically.
    fn orbit(&self, e: Edge) -> Vec<Edge> {
        let mut out = vec![e];
        if self.pmap.is_empty() {
            return out;
        }
        let mut i = 0;
        while i < out.len() {
            let (a, b) = out[i];
            for (fwd, inv) in &self.pmap {
                for m in [fwd, inv] {
                    if let (Some(&x), Some(&y)) = (m.get(&a), m.get(&b)) {
                        let img = ekey(x, y);
                        if !out.contains(&img) && self.shell.contains_key(&img) {
                            out.push(img);
                        }
                    }
                }
            }
            i += 1;
        }
        out.sort_unstable();
        out
    }

    #[inline]
    fn len2(&self, e: Edge) -> f64 {
        let d = sub(self.nodes[e.0 as usize], self.nodes[e.1 as usize]);
        dot(d, d)
    }

    /// Strict total order on edges: length first, node ids as tie-breaker.
    #[inline]
    fn edge_cmp(&self, a: Edge, b: Edge) -> Ordering {
        self.len2(a).total_cmp(&self.len2(b)).then_with(|| a.cmp(&b))
    }

    fn longest_edge(&self, it: u32) -> Edge {
        let t = self.tets[it as usize];
        let mut best = ekey(t[0], t[1]);
        for (i, j) in TET_EDGES.iter().skip(1) {
            let e = ekey(t[*i], t[*j]);
            if self.edge_cmp(e, best) == Ordering::Greater {
                best = e;
            }
        }
        best
    }

    /// Refine tet `it` through its longest edge, refining neighbours first where
    /// needed so that every split edge is the longest edge of all tets around it.
    pub fn refine_tet(&mut self, it: u32) {
        let mut stack = vec![it];
        while let Some(&s) = stack.last() {
            if !self.alive[s as usize] {
                stack.pop();
                continue;
            }
            let e = self.longest_edge(s);
            let lim2 = self.len2(e) / (self.params.lepp_ratio * self.params.lepp_ratio);
            // Periodic images of e are split together with it, so their shells
            // must satisfy the propagation rule as well.
            let orbit = self.orbit(e);
            let blocker = orbit.iter().find_map(|&oe| {
                self.shell[&oe].iter().copied().find(|&u| {
                    if u == s {
                        return false;
                    }
                    let lu = self.longest_edge(u);
                    lu != oe && self.len2(lu) > lim2
                })
            });
            match blocker {
                Some(u) => stack.push(u),
                None => {
                    self.split_edge(e);
                    stack.pop();
                }
            }
        }
    }

    /// Projection target for the midpoint of `e`, if `e` lies on a curved curve/face.
    fn projected_point(&self, e: Edge, shell: &[u32], mid: P3) -> Option<P3> {
        if let Some(tags) = self.curve_tags.get(&e) {
            for t in tags {
                if let Some(c) = self.brep.curves.get(t) {
                    return match c {
                        Curve::Straight => None,
                        _ => Some(c.project(mid)),
                    };
                }
            }
            // A curve edge without a curve descriptor stays straight; do not
            // fall through to a face projection, which would pull it off the curve.
            return None;
        }
        for &it in shell {
            let t = self.tets[it as usize];
            for &o in t.iter() {
                if o == e.0 || o == e.1 {
                    continue;
                }
                if let Some(tags) = self.face_tags.get(&fkey(e.0, e.1, o)) {
                    for tag in tags {
                        if let Some(s) = self.brep.surfaces.get(tag) {
                            return match s {
                                Surface::Flat => None,
                                _ => Some(s.project(mid)),
                            };
                        }
                    }
                }
            }
        }
        None
    }

    /// Check that inserting `p` on edge `e` keeps every child well shaped.
    fn placement_ok(&self, e: Edge, shell: &[u32], p: P3) -> bool {
        let lim = self.params.min_projected_volume;
        for &it in shell {
            let t = self.tets[it as usize];
            let x = t.map(|i| self.nodes[i as usize]);
            let v = vol6(x[0], x[1], x[2], x[3]);
            for drop in [e.0, e.1] {
                let mut y = x;
                for k in 0..4 {
                    if t[k] == drop {
                        y[k] = p;
                    }
                }
                let vc = vol6(y[0], y[1], y[2], y[3]);
                // Same orientation as the parent and at least `lim` of an exact half.
                if vc * v.signum() < lim * 0.5 * v.abs() {
                    return false;
                }
            }
        }
        true
    }

    fn push_tet(&mut self, t: [u32; 4], root: u32, err: f64, depth: u8) {
        let it = self.tets.len() as u32;
        self.tets.push(t);
        self.alive.push(true);
        self.root.push(root);
        self.err.push(err);
        self.depth.push(depth);
        for (i, j) in TET_EDGES {
            self.shell.entry(ekey(t[i], t[j])).or_default().push(it);
        }
        self.heap.push(Prio(err, it));
        self.n_alive += 1;
    }

    fn kill_tet(&mut self, it: u32, skip: Edge) {
        let t = self.tets[it as usize];
        for (i, j) in TET_EDGES {
            let k = ekey(t[i], t[j]);
            if k == skip {
                continue;
            }
            if let Some(v) = self.shell.get_mut(&k) {
                if let Some(pos) = v.iter().position(|&x| x == it) {
                    v.swap_remove(pos);
                }
                if v.is_empty() {
                    self.shell.remove(&k);
                }
            }
        }
        self.alive[it as usize] = false;
        self.n_alive -= 1;
    }

    /// Split `e` and, if it lies on a periodic face, all of its periodic images.
    pub fn split_edge(&mut self, e: Edge) {
        let orbit = self.orbit(e);
        if orbit.len() == 1 {
            self.split_single(e, true);
            return;
        }
        // Midpoints get consecutive ids in the (lexicographic) order of the
        // parent edges, which keeps every periodic map order preserving.
        let mids: Vec<u32> = orbit.iter().map(|&oe| self.split_single(oe, false)).collect();
        for (fwd, inv) in self.pmap.iter_mut() {
            for (i, &(a, b)) in orbit.iter().enumerate() {
                let (Some(&x), Some(&y)) = (fwd.get(&a), fwd.get(&b)) else { continue };
                if let Some(j) = orbit.iter().position(|&oe| oe == ekey(x, y)) {
                    fwd.insert(mids[i], mids[j]);
                    inv.insert(mids[j], mids[i]);
                }
            }
        }
    }

    /// Split a single edge; returns the new node id.
    fn split_single(&mut self, e: Edge, project: bool) -> u32 {
        let shell = self.shell.remove(&e).expect("split_edge: edge not in mesh");
        let (a, b) = e;
        let mid = scale(add(self.nodes[a as usize], self.nodes[b as usize]), 0.5);

        let mut p = mid;
        let target = if project { self.projected_point(e, &shell, mid) } else { None };
        if let Some(target) = target {
            self.stats.n_projected += 1;
            let mut q = target;
            let mut ok = false;
            for attempt in 0..5 {
                if self.placement_ok(e, &shell, q) {
                    ok = true;
                    if attempt > 0 {
                        self.stats.n_projection_reduced += 1;
                    }
                    break;
                }
                q = scale(add(q, mid), 0.5);
            }
            if ok {
                p = q;
            } else {
                self.stats.n_projection_rejected += 1;
            }
        }

        let m = self.nodes.len() as u32;
        self.nodes.push(p);

        for &it in &shell {
            let t = self.tets[it as usize];
            self.kill_tet(it, e);
            let root = self.root[it as usize];
            let err = self.err[it as usize] * self.params.child_factor;
            let depth = self.depth[it as usize].saturating_add(1);
            // Replacing one vertex by m keeps the orientation of the parent.
            let c1 = t.map(|x| if x == b { m } else { x });
            let c2 = t.map(|x| if x == a { m } else { x });
            self.push_tet(c1, root, err, depth);
            self.push_tet(c2, root, err, depth);

            for &o in t.iter() {
                if o == a || o == b {
                    continue;
                }
                if let Some(tags) = self.face_tags.remove(&fkey(a, b, o)) {
                    self.face_tags.insert(fkey(a, m, o), tags.clone());
                    self.face_tags.insert(fkey(m, b, o), tags);
                }
            }
        }
        if let Some(tags) = self.curve_tags.remove(&e) {
            self.curve_tags.insert(ekey(a, m), tags.clone());
            self.curve_tags.insert(ekey(m, b), tags);
        }
        self.stats.n_splits += 1;
        m
    }

    /// Step 0: geometric regularisation of narrow strips.
    ///
    /// A narrow strip (e.g. a thin PCB trace) shows up as a tagged face triangle
    /// with one edge on a geometric curve and its apex on a *different* curve that
    /// does not touch that edge. Its height is the local strip width `w`. When the
    /// node spacing along the strip is much larger than `w`, every triangle across
    /// the strip is a sliver whatever the triangulation. Such curve edges are split
    /// (conformingly, with their periodic images) until they are at most `ratio * w`
    /// long. The long diagonals this leaves inside the strip are removed afterwards
    /// by the surface flips of the improvement step. Needs no field information,
    /// only the face/curve tags.
    ///
    /// Stops early once `max_tets` tetrahedra exist. Returns the number of splits.
    pub fn regularize_strips(&mut self, ratio: f64, max_tets: usize) -> usize {
        let mut total = 0;
        for _iter in 0..30 {
            let order = self.strip_edges(ratio);
            if order.is_empty() {
                break;
            }
            let mut split_any = false;
            for (_, e) in order {
                if self.n_alive >= max_tets {
                    return total;
                }
                if self.shell.contains_key(&e) {
                    self.split_edge(e);
                    self.stats.n_strip_splits += 1;
                    total += 1;
                    split_any = true;
                }
            }
            if !split_any {
                break;
            }
        }
        total
    }

    /// Curve edges along narrow strips that are longer than `ratio` times the
    /// local strip width, most over-long first.
    ///
    /// A triangle on a tagged face with an edge on curve A and its apex on another
    /// curve B is a strip crossing only if the edge lies *opposite* curve B: the
    /// point of B closest to the edge midpoint must be inside B, not one of B's end
    /// points. The strip width is that distance. This rejects in-line
    /// configurations, e.g. two lines of slightly different width meeting at a tiny
    /// step: there B continues away from A, so its closest point is its own end and
    /// the apparent "width" would only be the tiny step size.
    fn strip_edges(&self, ratio: f64) -> Vec<(f64, Edge)> {
        let mut node_curves: HashMap<u32, Vec<i32>> = HashMap::new();
        let mut curve_segs: HashMap<i32, Vec<Edge>> = HashMap::new();
        for (e, tags) in &self.curve_tags {
            for t in tags {
                curve_segs.entry(*t).or_default().push(*e);
            }
            for v in [e.0, e.1] {
                let nc = node_curves.entry(v).or_default();
                for t in tags {
                    if !nc.contains(t) {
                        nc.push(*t);
                    }
                }
            }
        }
        // Curve end points: nodes used by exactly one segment of that curve.
        let mut curve_ends: HashMap<i32, Vec<u32>> = HashMap::new();
        for (t, segs) in &curve_segs {
            let mut count: HashMap<u32, usize> = HashMap::new();
            for e in segs {
                *count.entry(e.0).or_default() += 1;
                *count.entry(e.1).or_default() += 1;
            }
            curve_ends.insert(*t, count.into_iter().filter(|&(_, c)| c == 1).map(|(v, _)| v).collect());
        }
        // Distance from p to curve `tag`, and whether the closest point is a curve end.
        let closest_on_curve = |tag: i32, p: P3| -> (f64, bool) {
            let mut best = (f64::INFINITY, false);
            for e in &curve_segs[&tag] {
                let (a, b) = (self.nodes[e.0 as usize], self.nodes[e.1 as usize]);
                let ab = sub(b, a);
                let l2 = dot(ab, ab);
                let t = if l2 > 0.0 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
                let d = norm(sub(p, add(a, scale(ab, t))));
                if d < best.0 {
                    let ends = &curve_ends[&tag];
                    let at_end = (t <= 1e-9 && ends.contains(&e.0)) || (t >= 1.0 - 1e-9 && ends.contains(&e.1));
                    best = (d, at_end);
                }
            }
            best
        };
        let none: Vec<i32> = vec![];
        let mut cand: HashMap<Edge, f64> = HashMap::new();
        for f in self.face_tags.keys() {
            for k in 0..3 {
                let e = ekey(f[k], f[(k + 1) % 3]);
                let apex = f[(k + 2) % 3];
                let Some(ct) = self.curve_tags.get(&e) else { continue };
                let Some(oc) = node_curves.get(&apex) else { continue };
                // The apex must lie on some curve that touches neither the edge
                // nor its endpoints; otherwise this is a corner, not a strip.
                let (c0, c1) = (node_curves.get(&e.0).unwrap_or(&none), node_curves.get(&e.1).unwrap_or(&none));
                if !oc.iter().any(|c| !ct.contains(c) && !c0.contains(c) && !c1.contains(c)) {
                    continue;
                }
                let mid = scale(add(self.nodes[e.0 as usize], self.nodes[e.1 as usize]), 0.5);
                // Strip width = distance from the edge midpoint to an opposite curve B.
                let mut w = f64::INFINITY;
                for &tag in oc.iter().filter(|c| !ct.contains(c) && !c0.contains(c) && !c1.contains(c)) {
                    let (d, at_end) = closest_on_curve(tag, mid);
                    if !at_end {
                        w = w.min(d);
                    }
                }
                let len = self.len2(e).sqrt();
                if let Some(gate) = self.strip_gate {
                    let hot = self.shell.get(&e).map_or(false, |sh| sh.iter().any(|&t| self.original_error(t) >= gate));
                    if !hot {
                        continue;
                    }
                }
                if w.is_finite() && w > 0.0 && len > ratio * w {
                    let s = cand.entry(e).or_insert(0.0);
                    *s = s.max(len / (ratio * w));
                }
            }
        }
        let mut order: Vec<(f64, Edge)> = cand.into_iter().map(|(e, s)| (s, e)).collect();
        order.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        order
    }

    /// Greedily refine the highest-error tets until the target count is reached.
    pub fn run(&mut self) {
        let target = ((1.0 + self.params.growth) * self.n_alive as f64).round() as usize;
        while self.n_alive < target {
            let Some(Prio(_, it)) = self.heap.pop() else { break };
            if !self.alive[it as usize] || self.depth[it as usize] >= self.params.max_depth {
                continue;
            }
            let e0 = self.original_error(it);
            self.refined_cutoff = self.refined_cutoff.min(e0);
            self.refine_tet(it);
        }
    }

    /// Current number of tetrahedra.
    pub fn n_tets(&self) -> usize {
        self.n_alive
    }

    /// Error of the input tet that `it` descends from (children inherit it).
    #[inline]
    fn original_error(&self, it: u32) -> f64 {
        let e = self.err[self.root[it as usize] as usize];
        if e.is_finite() {
            e
        } else {
            0.0
        }
    }

    /// Error-gated strip regularisation, to be called after [`Refiner::run`].
    ///
    /// Like [`Refiner::regularize_strips`], but only narrow strips next to tets
    /// whose original error reached the level of the tets the error-driven
    /// refinement selected in this pass. Strips in quiet regions keep their mesh.
    /// Returns the number of splits (0 if `run` refined nothing).
    pub fn regularize_strips_gated(&mut self, ratio: f64, max_tets: usize) -> usize {
        if !self.refined_cutoff.is_finite() {
            return 0;
        }
        self.strip_gate = Some(self.refined_cutoff);
        let n = self.regularize_strips(ratio, max_tets);
        self.strip_gate = None;
        n
    }

    pub fn finish(self) -> Output {
        let mut tets = Vec::with_capacity(self.n_alive);
        let mut tet_root = Vec::with_capacity(self.n_alive);
        let mut tet_new = Vec::with_capacity(self.n_alive);
        for (i, t) in self.tets.iter().enumerate() {
            if self.alive[i] {
                tets.push(*t);
                tet_root.push(self.root[i]);
                tet_new.push(i >= self.n0);
            }
        }
        let mut faces: Vec<(Face, i32)> = self
            .face_tags
            .iter()
            .flat_map(|(f, tags)| tags.iter().map(move |t| (*f, *t)))
            .collect();
        faces.sort_unstable();
        let mut curves: Vec<(Edge, i32)> = self
            .curve_tags
            .iter()
            .flat_map(|(e, tags)| tags.iter().map(move |t| (*e, *t)))
            .collect();
        curves.sort_unstable();
        Output {
            nodes: self.nodes,
            tets,
            tet_root,
            tet_new,
            face_tris: faces.iter().map(|x| x.0).collect(),
            face_tri_tags: faces.iter().map(|x| x.1).collect(),
            curve_edges: curves.iter().map(|x| x.0).collect(),
            curve_edge_tags: curves.iter().map(|x| x.1).collect(),
            stats: self.stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Unit cube split into 6 tets (Kuhn triangulation), all positively oriented.
    fn cube() -> (Vec<P3>, Vec<[u32; 4]>) {
        let mut nodes = vec![];
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    nodes.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        let paths = [[1, 2, 4], [1, 4, 2], [2, 1, 4], [2, 4, 1], [4, 1, 2], [4, 2, 1]];
        let mut tets = vec![];
        for p in paths {
            let v1 = p[0];
            let v2 = v1 + p[1];
            let mut t = [0u32, v1, v2, 7];
            let x = t.map(|i| nodes[i as usize]);
            if vol6(x[0], x[1], x[2], x[3]) < 0.0 {
                t.swap(2, 3);
            }
            tets.push(t);
        }
        (nodes, tets)
    }

    fn boundary_faces(tets: &[[u32; 4]]) -> HashMap<Face, usize> {
        let mut count: HashMap<Face, usize> = HashMap::new();
        for t in tets {
            for skip in 0..4 {
                let v: Vec<u32> = (0..4).filter(|&k| k != skip).map(|k| t[k]).collect();
                *count.entry(fkey(v[0], v[1], v[2])).or_default() += 1;
            }
        }
        count
    }

    #[test]
    fn refinement_is_conforming_and_tagged() {
        let (nodes, tets) = cube();
        let n0 = tets.len();
        let err: Vec<f64> = (0..n0).map(|i| if i == 0 { 10.0 } else { 1.0 }).collect();
        // Tag every boundary face of the cube with tag 1, every boundary edge on x=0,y=0 with curve 7.
        let bf: Vec<(Face, i32)> = boundary_faces(&tets)
            .into_iter()
            .filter(|(_, c)| *c == 1)
            .map(|(f, _)| (f, 1))
            .collect();
        let curve = vec![((0u32, 4u32), 7)];
        let brep = Brep::default();
        let params = Params { growth: 5.0, ..Default::default() };
        let mut r = Refiner::new(nodes, tets, err, &bf, &curve, &brep, params);
        r.run();
        let out = r.finish();
        assert!(out.tets.len() >= 6 * 6);

        // Volume preserved and all tets positively oriented.
        let mut vol = 0.0;
        for t in &out.tets {
            let x = t.map(|i| out.nodes[i as usize]);
            let v = vol6(x[0], x[1], x[2], x[3]);
            assert!(v > 0.0);
            vol += v / 6.0;
        }
        assert!((vol - 1.0).abs() < 1e-12);

        // Conformity: each face shared by at most two tets, and boundary faces
        // (shared by one) are exactly the tagged faces.
        let count = boundary_faces(&out.tets);
        assert!(count.values().all(|&c| c <= 2));
        let bnd: HashSet<Face> = count.iter().filter(|(_, &c)| c == 1).map(|(f, _)| *f).collect();
        let tagged: HashSet<Face> = out.face_tris.iter().copied().collect();
        assert_eq!(bnd, tagged);

        // Curve edges still cover the original segment from node 0 to node 4.
        let total: f64 = out
            .curve_edges
            .iter()
            .map(|e| norm(sub(out.nodes[e.0 as usize], out.nodes[e.1 as usize])))
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn periodic_faces_stay_matched_and_order_preserving() {
        // 3x1x1 row of cubes; x=0 face is periodic with x=3 face (dv = +3 x).
        let n = 3u32;
        let id = |i: u32, j: u32, k: u32| i + (n + 1) * (j + 2 * k);
        let mut nodes = vec![];
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..=n {
                    nodes.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        let mut tets = vec![];
        for i in 0..n {
            let c = |b: u32| id(i + (b & 1), (b >> 1) & 1, (b >> 2) & 1);
            for p in [[1, 2, 4], [1, 4, 2], [2, 1, 4], [2, 4, 1], [4, 1, 2], [4, 2, 1]] {
                let t = [c(0), c(p[0]), c(p[0] + p[1]), c(7)];
                let x = t.map(|v| nodes[v as usize]);
                tets.push(if vol6(x[0], x[1], x[2], x[3]) < 0.0 { [t[0], t[1], t[3], t[2]] } else { t });
            }
        }
        let pairs: Vec<(usize, u32, u32)> =
            (0..2).flat_map(|k| (0..2).map(move |j| (0usize, id(0, j, k), id(n, j, k)))).collect();
        // Error concentrated on the source face only.
        let err: Vec<f64> = tets
            .iter()
            .map(|t| if t.iter().any(|&v| nodes[v as usize][0] == 0.0) { 10.0 } else { 1.0 })
            .collect();
        let brep = Brep::default();
        let mut r = Refiner::new(nodes, tets, err, &[], &[], &brep, Params { growth: 4.0, ..Default::default() });
        r.set_periodic(&pairs);
        r.run();
        let fwd = r.pmap[0].0.clone();
        let out = r.finish();

        let face = |x: f64| -> HashSet<Face> {
            boundary_faces(&out.tets)
                .into_iter()
                .filter(|(f, c)| *c == 1 && f.iter().all(|&v| out.nodes[v as usize][0] == x))
                .map(|(f, _)| f)
                .collect()
        };
        let (f1, f2) = (face(0.0), face(n as f64));
        assert!(f1.len() > 2, "source face was not refined");
        assert_eq!(f1.len(), f2.len());
        for f in &f1 {
            let img = f.map(|v| fwd[&v]);
            // Maps are order preserving, so the image of a sorted face is sorted.
            assert!(img[0] < img[1] && img[1] < img[2]);
            assert!(f2.contains(&img), "image of {f:?} missing on target face");
            for (&v, &w) in f.iter().zip(img.iter()) {
                let (p, q) = (out.nodes[v as usize], out.nodes[w as usize]);
                assert!((q[0] - p[0] - n as f64).abs() < 1e-12 && (q[1] - p[1]).abs() < 1e-12 && (q[2] - p[2]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn narrow_strip_is_regularised() {
        // 10 x 0.2 x 1 slab, one cell per unit length: the top face is a narrow
        // strip with 1.0-long curve edges along a 0.2-wide strip (ratio 5).
        let (nx, w) = (10u32, 0.2);
        let id = |i: u32, j: u32, k: u32| i + (nx + 1) * (j + 2 * k);
        let mut nodes = vec![];
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..=nx {
                    nodes.push([i as f64, j as f64 * w, k as f64]);
                }
            }
        }
        let mut tets = vec![];
        for i in 0..nx {
            let c = |b: u32| id(i + (b & 1), (b >> 1) & 1, (b >> 2) & 1);
            for p in [[1, 2, 4], [1, 4, 2], [2, 1, 4], [2, 4, 1], [4, 1, 2], [4, 2, 1]] {
                let t = [c(0), c(p[0]), c(p[0] + p[1]), c(7)];
                let x = t.map(|v| nodes[v as usize]);
                tets.push(if vol6(x[0], x[1], x[2], x[3]) < 0.0 { [t[0], t[1], t[3], t[2]] } else { t });
            }
        }
        // Top face (the strip) is tag 1; all other boundary faces get tag 2 so
        // that, as in EMerge, every boundary node is on a tagged face.
        let top: Vec<(Face, i32)> = boundary_faces(&tets)
            .into_iter()
            .filter(|(_, c)| *c == 1)
            .map(|(f, _)| (f, if f.iter().all(|&v| nodes[v as usize][2] == 1.0) { 1 } else { 2 }))
            .collect();
        let mut curves = vec![];
        for i in 0..nx {
            curves.push(((id(i, 0, 1), id(i + 1, 0, 1)), 10)); // long side y=0
            curves.push(((id(i, 1, 1), id(i + 1, 1, 1)), 11)); // long side y=w
        }
        curves.push(((id(0, 0, 1), id(0, 1, 1)), 12));
        curves.push(((id(nx, 0, 1), id(nx, 1, 1)), 13));
        let tri_q = |n: &[P3], f: &Face| {
            let x = f.map(|v| n[v as usize]);
            let a = norm(cross(sub(x[1], x[0]), sub(x[2], x[0]))) / 2.0;
            let l2: f64 = [(0, 1), (1, 2), (0, 2)].iter().map(|&(i, j)| dot(sub(x[i], x[j]), sub(x[i], x[j]))).sum();
            4.0 * 3f64.sqrt() * a / l2
        };
        let q0 = top.iter().filter(|x| x.1 == 1).map(|(f, _)| tri_q(&nodes, f)).fold(f64::INFINITY, f64::min);
        let n0 = tets.len();
        let brep = Brep::default();
        let err = vec![0.0; n0];
        let mut r = Refiner::new(nodes, tets, err, &top, &curves, &brep, Params { growth: 0.0, ..Default::default() });
        let ratio = 2.0;
        let splits = r.regularize_strips(ratio, 100 * n0);
        let out = r.finish();
        assert!(splits > 0);
        // Long-side curve edges are now at most ratio * w.
        for (e, tag) in out.curve_edges.iter().zip(&out.curve_edge_tags) {
            if *tag == 10 || *tag == 11 {
                let l = norm(sub(out.nodes[e.0 as usize], out.nodes[e.1 as usize]));
                assert!(l <= ratio * w + 1e-12, "curve edge {l} longer than {}", ratio * w);
            }
        }
        // Surface flips remove the long diagonals left inside the strip.
        let mut out = out;
        let faces: Vec<(Face, i32)> = out.face_tris.iter().copied().zip(out.face_tri_tags.iter().copied()).collect();
        let curve_set: std::collections::HashSet<Edge> = out.curve_edges.iter().copied().collect();
        let m = out.tets.len();
        let tets = std::mem::take(&mut out.tets);
        let mut imp = crate::aura::improve::Improver::new(
            &mut out.nodes, tets, vec![0; m], vec![0; m], vec![true; m], &faces, &curve_set,
            crate::aura::improve::ImproveParams { sweeps: 6, ..Default::default() },
        );
        imp.set_planar_tags(&[1].into_iter().collect());
        imp.run();
        assert!(imp.stats.n_surface_flips > 0, "{:?}", imp.stats);
        let (tets, _, _, _, faces) = imp.finish();
        out.tets = tets;
        out.face_tris = faces.iter().filter(|x| x.1 == 1).map(|x| x.0).collect();
        let q1 = out.face_tris.iter().map(|f| tri_q(&out.nodes, f)).fold(f64::INFINITY, f64::min);
        assert!(q1 > 2.0 * q0, "strip triangle quality {q0} -> {q1}");
        let vol: f64 = out.tets.iter().map(|t| {
            let x = t.map(|v| out.nodes[v as usize]);
            let v = vol6(x[0], x[1], x[2], x[3]);
            assert!(v > 0.0);
            v / 6.0
        }).sum();
        assert!((vol - nx as f64 * w).abs() < 1e-12);
    }

    #[test]
    fn gated_strip_regularisation_stays_near_high_error() {
        // Same 10 x 0.2 x 1 slab as above; the error is high only for x < 2.
        let (nx, w) = (10u32, 0.2);
        let id = |i: u32, j: u32, k: u32| i + (nx + 1) * (j + 2 * k);
        let mut nodes = vec![];
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..=nx {
                    nodes.push([i as f64, j as f64 * w, k as f64]);
                }
            }
        }
        let mut tets = vec![];
        for i in 0..nx {
            let c = |b: u32| id(i + (b & 1), (b >> 1) & 1, (b >> 2) & 1);
            for p in [[1, 2, 4], [1, 4, 2], [2, 1, 4], [2, 4, 1], [4, 1, 2], [4, 2, 1]] {
                let t = [c(0), c(p[0]), c(p[0] + p[1]), c(7)];
                let x = t.map(|v| nodes[v as usize]);
                tets.push(if vol6(x[0], x[1], x[2], x[3]) < 0.0 { [t[0], t[1], t[3], t[2]] } else { t });
            }
        }
        let faces: Vec<(Face, i32)> = boundary_faces(&tets)
            .into_iter()
            .filter(|(_, c)| *c == 1)
            .map(|(f, _)| (f, if f.iter().all(|&v| nodes[v as usize][2] == 1.0) { 1 } else { 2 }))
            .collect();
        let mut curves = vec![];
        for i in 0..nx {
            curves.push(((id(i, 0, 1), id(i + 1, 0, 1)), 10));
            curves.push(((id(i, 1, 1), id(i + 1, 1, 1)), 11));
        }
        curves.push(((id(0, 0, 1), id(0, 1, 1)), 12));
        curves.push(((id(nx, 0, 1), id(nx, 1, 1)), 13));
        let err: Vec<f64> = tets
            .iter()
            .map(|t| {
                let cx = t.iter().map(|&v| nodes[v as usize][0]).sum::<f64>() / 4.0;
                if cx < 2.0 { 10.0 } else { 0.01 }
            })
            .collect();
        let n0 = tets.len();
        let brep = Brep::default();
        let mut r = Refiner::new(nodes, tets, err, &faces, &curves, &brep, Params { growth: 0.2, ..Default::default() });
        r.run();
        let splits = r.regularize_strips_gated(2.0, 100 * n0);
        let out = r.finish();
        assert!(splits > 0);
        let (mut near_max, mut far_min) = (0.0f64, f64::INFINITY);
        for (e, tag) in out.curve_edges.iter().zip(&out.curve_edge_tags) {
            if *tag != 10 && *tag != 11 {
                continue;
            }
            let (a, b) = (out.nodes[e.0 as usize], out.nodes[e.1 as usize]);
            let l = norm(sub(a, b));
            let xm = 0.5 * (a[0] + b[0]);
            if xm < 1.5 {
                near_max = near_max.max(l);
            } else if xm > 5.0 {
                far_min = far_min.min(l);
            }
        }
        assert!(near_max <= 0.4 + 1e-12, "strip near the high error was not regularised ({near_max})");
        assert!(far_min >= 1.0 - 1e-12, "strip far from the high error was refined ({far_min})");
    }

    #[test]
    fn quality_stays_bounded() {
        let (nodes, tets) = cube();
        let q0 = tets
            .iter()
            .map(|t| {
                let x = t.map(|i| nodes[i as usize]);
                mean_ratio(x[0], x[1], x[2], x[3])
            })
            .fold(f64::INFINITY, f64::min);
        let err = vec![1.0; tets.len()];
        let brep = Brep::default();
        let params = Params { growth: 60.0, max_depth: 30, ..Default::default() };
        let mut r = Refiner::new(nodes, tets, err, &[], &[], &brep, params);
        r.run();
        let out = r.finish();
        let q = out
            .tets
            .iter()
            .map(|t| {
                let x = t.map(|i| out.nodes[i as usize]);
                mean_ratio(x[0], x[1], x[2], x[3])
            })
            .fold(f64::INFINITY, f64::min);
        assert!(q > 0.3 * q0, "min quality dropped from {q0} to {q}");
    }
}
