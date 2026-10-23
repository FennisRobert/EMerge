// [Written by Claude] All code in this module was written by Claude (AI).
//! Local mesh quality improvement after refinement: topological flips and
//! smart Laplacian smoothing of interior nodes.
//!
//! * 2-3 flip: two tets sharing a face become three tets around a new edge.
//! * Edge removal (3-2, 4-4, 5-6, 6-8, 7-10): the tets around an edge with 3..=7
//!   tets are re-triangulated without that edge, trying every triangulation of
//!   the ring of nodes around it.
//! * Surface flip: the diagonal between two triangles of the same planar
//!   geometric face is swapped, re-triangulating the tets on either side of the
//!   surface (one side for an outer boundary, two for a sheet or interface).
//!   This repairs slivers on the surface itself, e.g. along narrow strips.
//! * Smoothing: a free node moves (partially) towards the centroid of its
//!   neighbours if that improves the worst incident tet. Nodes on a single
//!   planar geometric face (not on a curve) may slide within that plane.
//!
//! The work zone is the set of tets created in this pass plus `layers` rings of
//! node-adjacent neighbours, so tets of the old mesh next to the refined region
//! (where new and old elements meet) are improved too.
//!
//! Every operation is only accepted if it strictly improves the minimum
//! mean-ratio quality of the tets it replaces. Tagged faces, tagged curve
//! edges, their nodes and region (volume) boundaries are never modified, so
//! all geometry/tag information stays valid.

use crate::aura::geom::*;
use crate::aura::refine::{mean_ratio, vol6, Edge, Face};
use std::collections::{HashMap, HashSet};

const TET_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

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

#[derive(Clone, Debug)]
pub struct ImproveParams {
    /// Number of flip + smoothing sweeps.
    pub sweeps: usize,
    /// Only tets below this quality are considered for flips.
    pub flip_quality: f64,
    /// Number of smoothing iterations per sweep.
    pub smooth_iters: usize,
    /// Rings of node-adjacent neighbours around the new tets added to the work zone.
    pub layers: usize,
    /// Work on the whole mesh instead of the refined region only.
    pub global: bool,
}

impl Default for ImproveParams {
    fn default() -> Self {
        ImproveParams { sweeps: 3, flip_quality: 0.6, smooth_iters: 2, layers: 2, global: false }
    }
}

#[derive(Default, Clone, Debug)]
pub struct ImproveStats {
    pub n_flip23: usize,
    pub n_flip32: usize,
    pub n_flip44: usize,
    /// Edge removals with 5..=7 tets around the edge.
    pub n_flip_large: usize,
    pub n_smoothed: usize,
    /// Smoothing moves of nodes sliding within a planar face.
    pub n_slid: usize,
    /// Diagonal swaps on planar geometric faces.
    pub n_surface_flips: usize,
    /// Number of tets in the work zone (new tets plus halo) at the start.
    pub n_zone: usize,
}

pub struct Improver<'a> {
    pub nodes: &'a mut Vec<P3>,
    pub tets: Vec<[u32; 4]>,
    pub region: Vec<i32>,
    pub root: Vec<u32>,
    pub active: Vec<bool>,
    /// Work zone: active tets plus their neighbourhood.
    zone: Vec<bool>,
    alive: Vec<bool>,
    shell: HashMap<Edge, Vec<u32>>,
    /// Tagged face triangles and their tags (changed only by surface flips).
    faces: HashMap<Face, Vec<i32>>,
    curve_edges: HashSet<Edge>,
    /// Tags of planar faces (surface flips and sliding are allowed there).
    planar: HashSet<i32>,
    /// Nodes that must never move or be part of a surface flip (periodic faces).
    locked: HashSet<u32>,
    fixed_edges: HashSet<Edge>,
    fixed_nodes: HashSet<u32>,
    /// Nodes on tagged curves (never move).
    curve_nodes: HashSet<u32>,
    /// Nodes that may slide within a planar face: (unit normal, neighbours on that face).
    slide: HashMap<u32, (P3, Vec<u32>)>,
    /// Sign of the orientation convention of the input mesh (+1 or -1).
    sign: f64,
    params: ImproveParams,
    pub stats: ImproveStats,
}

impl<'a> Improver<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nodes: &'a mut Vec<P3>,
        tets: Vec<[u32; 4]>,
        region: Vec<i32>,
        root: Vec<u32>,
        active: Vec<bool>,
        face_list: &[(Face, i32)],
        curve_edges: &HashSet<Edge>,
        params: ImproveParams,
    ) -> Self {
        let mut faces: HashMap<Face, Vec<i32>> = HashMap::new();
        for (f, tag) in face_list {
            let v = faces.entry(fkey(f[0], f[1], f[2])).or_default();
            if !v.contains(tag) {
                v.push(*tag);
            }
        }
        let mut shell: HashMap<Edge, Vec<u32>> = HashMap::with_capacity(tets.len() * 2);
        let mut npos = 0usize;
        for (it, t) in tets.iter().enumerate() {
            for (i, j) in TET_EDGES {
                shell.entry(ekey(t[i], t[j])).or_default().push(it as u32);
            }
            let x = t.map(|i| nodes[i as usize]);
            if vol6(x[0], x[1], x[2], x[3]) > 0.0 {
                npos += 1;
            }
        }
        let sign = if 2 * npos >= tets.len() { 1.0 } else { -1.0 };
        let mut fixed_edges: HashSet<Edge> = curve_edges.clone();
        let mut fixed_nodes: HashSet<u32> = HashSet::new();
        for f in faces.keys() {
            fixed_edges.insert(ekey(f[0], f[1]));
            fixed_edges.insert(ekey(f[1], f[2]));
            fixed_edges.insert(ekey(f[0], f[2]));
            fixed_nodes.extend(f.iter().copied());
        }
        let curve_nodes: HashSet<u32> = curve_edges.iter().flat_map(|e| [e.0, e.1]).collect();
        fixed_nodes.extend(curve_nodes.iter().copied());
        let n = tets.len();
        Improver {
            nodes,
            tets,
            region,
            root,
            zone: active.clone(),
            active,
            alive: vec![true; n],
            shell,
            faces,
            curve_edges: curve_edges.clone(),
            planar: HashSet::new(),
            locked: HashSet::new(),
            fixed_edges,
            fixed_nodes,
            curve_nodes,
            slide: HashMap::new(),
            sign,
            params,
            stats: ImproveStats::default(),
        }
    }

    /// Declare which face tags are planar. On those faces nodes may slide within
    /// the plane (if they lie on exactly one tagged face and on no curve) and
    /// diagonals may be swapped by surface flips.
    pub fn set_planar_tags(&mut self, planar_tags: &HashSet<i32>) {
        self.planar = planar_tags.clone();
        self.rebuild_slide();
    }

    fn rebuild_slide(&mut self) {
        let mut node_tags: HashMap<u32, Vec<i32>> = HashMap::new();
        for (f, tags) in &self.faces {
            for &v in f {
                let e = node_tags.entry(v).or_default();
                for t in tags {
                    if !e.contains(t) {
                        e.push(*t);
                    }
                }
            }
        }
        self.slide.clear();
        for (f, tags) in &self.faces {
            if tags.len() != 1 || !self.planar.contains(&tags[0]) {
                continue;
            }
            for &v in f {
                if self.curve_nodes.contains(&v) || self.locked.contains(&v) || node_tags[&v].len() != 1 {
                    continue;
                }
                let x = f.map(|i| self.nodes[i as usize]);
                let n = cross(sub(x[1], x[0]), sub(x[2], x[0]));
                let l = norm(n);
                if l == 0.0 {
                    continue;
                }
                let entry = self.slide.entry(v).or_insert_with(|| (scale(n, 1.0 / l), vec![]));
                for &w in f {
                    if w != v && !entry.1.contains(&w) {
                        entry.1.push(w);
                    }
                }
            }
        }
    }

    /// Never move these nodes (e.g. nodes on periodic faces, which must stay
    /// exact translated copies of their partner nodes).
    pub fn lock_nodes(&mut self, nodes: &[u32]) {
        for v in nodes {
            self.slide.remove(v);
            self.fixed_nodes.insert(*v);
            self.locked.insert(*v);
        }
    }

    // ------------------------------------------------------------------
    // Geometry helpers
    // ------------------------------------------------------------------

    #[inline]
    fn signed_vol(&self, t: [u32; 4]) -> f64 {
        let x = t.map(|i| self.nodes[i as usize]);
        vol6(x[0], x[1], x[2], x[3])
    }

    /// Quality with orientation check: inverted or flat tets score 0.
    #[inline]
    fn q(&self, t: [u32; 4]) -> f64 {
        let x = t.map(|i| self.nodes[i as usize]);
        if vol6(x[0], x[1], x[2], x[3]) * self.sign <= 0.0 {
            return 0.0;
        }
        mean_ratio(x[0], x[1], x[2], x[3])
    }

    /// Orient a tet to the mesh convention.
    #[inline]
    fn oriented(&self, mut t: [u32; 4]) -> [u32; 4] {
        if self.signed_vol(t) * self.sign < 0.0 {
            t.swap(2, 3);
        }
        t
    }

    /// Validate a replacement: the new tets must exactly tile the volume of
    /// the old ones (no overlap/inversion) and improve the minimum quality.
    /// Returns the oriented new tets if accepted.
    fn evaluate(&self, old: &[u32], new: &[[u32; 4]]) -> Option<(Vec<[u32; 4]>, f64)> {
        let (out, q_new, q_old) = self.valid(old, new)?;
        if q_new > q_old + 1e-6 {
            Some((out, q_new))
        } else {
            None
        }
    }

    /// Validity only: the new tets exactly tile the old ones. Returns the oriented
    /// new tets with the new and old minimum quality.
    fn valid(&self, old: &[u32], new: &[[u32; 4]]) -> Option<(Vec<[u32; 4]>, f64, f64)> {
        let v_old: f64 = old.iter().map(|&i| self.signed_vol(self.tets[i as usize]).abs()).sum();
        let q_old = old.iter().map(|&i| self.q(self.tets[i as usize])).fold(f64::INFINITY, f64::min);
        let mut v_new = 0.0;
        let mut out = Vec::with_capacity(new.len());
        for t in new {
            let v = self.signed_vol(*t).abs();
            if v <= 1e-12 * v_old {
                return None;
            }
            v_new += v;
            out.push(self.oriented(*t));
        }
        if (v_new - v_old).abs() > 1e-9 * v_old {
            return None;
        }
        let q_new = out.iter().map(|t| self.q(*t)).fold(f64::INFINITY, f64::min);
        Some((out, q_new, q_old))
    }

    /// Shape quality of a triangle: 1 for equilateral, 0 for degenerate.
    fn tri_quality(&self, f: Face) -> f64 {
        let x = f.map(|i| self.nodes[i as usize]);
        let a = norm(cross(sub(x[1], x[0]), sub(x[2], x[0]))) / 2.0;
        let l2: f64 = [(0, 1), (1, 2), (0, 2)].iter().map(|&(i, j)| dot(sub(x[i], x[j]), sub(x[i], x[j]))).sum();
        if l2 == 0.0 {
            0.0
        } else {
            4.0 * 3f64.sqrt() * a / l2
        }
    }

    // ------------------------------------------------------------------
    // Connectivity helpers
    // ------------------------------------------------------------------

    fn add_tet(&mut self, t: [u32; 4], region: i32, root: u32) {
        let it = self.tets.len() as u32;
        self.tets.push(t);
        self.region.push(region);
        self.root.push(root);
        self.active.push(true);
        self.zone.push(true);
        self.alive.push(true);
        for (i, j) in TET_EDGES {
            self.shell.entry(ekey(t[i], t[j])).or_default().push(it);
        }
    }

    fn kill_tet(&mut self, it: u32) {
        let t = self.tets[it as usize];
        for (i, j) in TET_EDGES {
            let k = ekey(t[i], t[j]);
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
    }

    fn replace(&mut self, old: &[u32], new: Vec<[u32; 4]>) {
        let region = self.region[old[0] as usize];
        let root = self.root[old[0] as usize];
        for &i in old {
            self.kill_tet(i);
        }
        for t in new {
            self.add_tet(t, region, root);
        }
    }

    /// Tets containing all three nodes of a face.
    fn face_tets(&self, f: Face) -> Vec<u32> {
        match self.shell.get(&ekey(f[0], f[1])) {
            Some(v) => v.iter().copied().filter(|&i| self.tets[i as usize].contains(&f[2])).collect(),
            None => vec![],
        }
    }

    fn same_region(&self, tets: &[u32]) -> bool {
        let r = self.region[tets[0] as usize];
        tets.iter().all(|&i| self.region[i as usize] == r)
    }

    // ------------------------------------------------------------------
    // Flips
    // ------------------------------------------------------------------

    fn try_flip23(&mut self, it: u32, local_face: usize) -> bool {
        let t = self.tets[it as usize];
        let [i, j, k] = TET_FACES[local_face];
        let (a, b, c) = (t[i], t[j], t[k]);
        let f = fkey(a, b, c);
        if self.faces.contains_key(&f) {
            return false;
        }
        let nb: Vec<u32> = self.face_tets(f).into_iter().filter(|&u| u != it).collect();
        if nb.len() != 1 {
            return false;
        }
        let u = nb[0];
        if !self.same_region(&[it, u]) {
            return false;
        }
        let d = t[local_face];
        let e = *self.tets[u as usize].iter().find(|&&x| x != a && x != b && x != c).unwrap();
        if self.shell.contains_key(&ekey(d, e)) {
            return false;
        }
        let new = [[a, b, d, e], [b, c, d, e], [c, a, d, e]];
        match self.evaluate(&[it, u], &new) {
            Some((out, _)) => {
                self.replace(&[it, u], out);
                self.stats.n_flip23 += 1;
                true
            }
            None => false,
        }
    }

    /// Ring of nodes around an interior edge, in cyclic order.
    fn ring(&self, e: Edge, shell: &[u32]) -> Option<Vec<u32>> {
        let pairs: Vec<(u32, u32)> = shell
            .iter()
            .map(|&i| {
                let o: Vec<u32> =
                    self.tets[i as usize].iter().copied().filter(|&x| x != e.0 && x != e.1).collect();
                (o[0], o[1])
            })
            .collect();
        let mut ring = vec![pairs[0].0, pairs[0].1];
        let mut used = vec![false; pairs.len()];
        used[0] = true;
        while ring.len() < pairs.len() {
            let last = *ring.last().unwrap();
            let next = (0..pairs.len()).find(|&k| !used[k] && (pairs[k].0 == last || pairs[k].1 == last))?;
            used[next] = true;
            let (p, q) = pairs[next];
            ring.push(if p == last { q } else { p });
        }
        // Closed ring: the remaining pair connects the last node back to the first.
        let k = (0..pairs.len()).find(|&k| !used[k])?;
        let (p, q) = pairs[k];
        let last = *ring.last().unwrap();
        if (p == last && q == ring[0]) || (q == last && p == ring[0]) {
            Some(ring)
        } else {
            None
        }
    }

    fn try_edge_removal(&mut self, e: Edge) -> bool {
        if self.fixed_edges.contains(&e) {
            return false;
        }
        let shell = match self.shell.get(&e) {
            Some(v) if (3..=7).contains(&v.len()) => v.clone(),
            _ => return false,
        };
        if !self.same_region(&shell) {
            return false;
        }
        let Some(r) = self.ring(e, &shell) else { return false };
        let (a, b) = e;
        let n = r.len();

        let mut best: Option<(Vec<[u32; 4]>, f64)> = None;
        'tri: for tris in polygon_triangulations(n) {
            let mut new = Vec::with_capacity(2 * tris.len());
            for &[i, j, k] in &tris {
                let (p, q, w) = (r[i], r[j], r[k]);
                // Ring diagonals become new edges and must not exist yet; the
                // triangles become new faces and must not exist either.
                for (x, y) in [(i, j), (j, k), (i, k)] {
                    let is_ring_edge = (x + 1) % n == y || (y + 1) % n == x;
                    if !is_ring_edge && self.shell.contains_key(&ekey(r[x], r[y])) {
                        continue 'tri;
                    }
                }
                if !self.face_tets(fkey(p, q, w)).is_empty() {
                    continue 'tri;
                }
                new.push([p, q, w, a]);
                new.push([p, q, w, b]);
            }
            if let Some((out, qn)) = self.evaluate(&shell, &new) {
                if best.as_ref().map_or(true, |(_, qb)| qn > *qb) {
                    best = Some((out, qn));
                }
            }
        }
        let Some((out, _)) = best else { return false };
        self.replace(&shell, out);
        match n {
            3 => self.stats.n_flip32 += 1,
            4 => self.stats.n_flip44 += 1,
            _ => self.stats.n_flip_large += 1,
        }
        true
    }

    fn flip_sweep(&mut self) -> usize {
        let mut cand: Vec<(f64, u32)> = (0..self.tets.len())
            .filter(|&i| self.alive[i] && self.zone[i])
            .map(|i| (self.q(self.tets[i]), i as u32))
            .filter(|(q, _)| *q < self.params.flip_quality)
            .collect();
        cand.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut n = 0;
        for (_, it) in cand {
            if !self.alive[it as usize] {
                continue;
            }
            let t = self.tets[it as usize];
            let mut done = false;
            for (i, j) in TET_EDGES {
                if self.try_edge_removal(ekey(t[i], t[j])) {
                    done = true;
                    break;
                }
            }
            if !done {
                for f in 0..4 {
                    if self.try_flip23(it, f) {
                        done = true;
                        break;
                    }
                }
            }
            n += done as usize;
        }
        n
    }

    /// Swap the diagonal `e` between two triangles of the same planar face.
    fn try_surface_flip(&mut self, e: Edge) -> bool {
        let (a, b) = e;
        if self.curve_edges.contains(&e) || self.locked.contains(&a) || self.locked.contains(&b) {
            return false;
        }
        let Some(shell) = self.shell.get(&e).cloned() else { return false };
        let mut tris: Vec<(u32, Face)> = vec![];
        for &it in &shell {
            for &o in self.tets[it as usize].iter() {
                if o != a && o != b {
                    let f = fkey(a, b, o);
                    if self.faces.contains_key(&f) && !tris.iter().any(|(_, g)| *g == f) {
                        tris.push((o, f));
                    }
                }
            }
        }
        if tris.len() != 2 {
            return false;
        }
        let ((c, fc), (d, fd)) = (tris[0], tris[1]);
        let tags = self.faces[&fc].clone();
        if tags.len() != 1 || self.faces[&fd] != tags || !self.planar.contains(&tags[0]) {
            return false;
        }
        if self.locked.contains(&c) || self.locked.contains(&d) || self.shell.contains_key(&ekey(c, d)) {
            return false;
        }
        // The quad a-c-b-d must be convex: c, d on opposite sides of ab and a, b
        // on opposite sides of cd.
        let p = |i: u32| self.nodes[i as usize];
        let n = cross(sub(p(b), p(a)), sub(p(c), p(a)));
        let side = |u: u32, v: u32, w: u32| dot(cross(sub(p(v), p(u)), sub(p(w), p(u))), n);
        if side(a, b, c) * side(a, b, d) >= 0.0 || side(c, d, a) * side(c, d, b) >= 0.0 {
            return false;
        }
        let q2_old = self.tri_quality(fc).min(self.tri_quality(fd));
        let q2_new = self.tri_quality(fkey(a, c, d)).min(self.tri_quality(fkey(b, c, d)));
        if q2_new <= q2_old + 1e-6 {
            return false;
        }

        // The shell splits into one chain of tets from c to d per side of the surface.
        let pairs: Vec<(u32, u32, u32)> = shell
            .iter()
            .map(|&it| {
                let o: Vec<u32> = self.tets[it as usize].iter().copied().filter(|&x| x != a && x != b).collect();
                (o[0], o[1], it)
            })
            .collect();
        let mut used = vec![false; pairs.len()];
        let mut chains: Vec<(Vec<u32>, Vec<u32>)> = vec![];
        while used.iter().any(|u| !u) {
            let (mut ring, mut ctets, mut cur) = (vec![c], vec![], c);
            loop {
                let Some(k) = (0..pairs.len()).find(|&k| !used[k] && (pairs[k].0 == cur || pairs[k].1 == cur)) else {
                    return false;
                };
                used[k] = true;
                cur = if pairs[k].0 == cur { pairs[k].1 } else { pairs[k].0 };
                ctets.push(pairs[k].2);
                ring.push(cur);
                if cur == d {
                    break;
                }
                if ring.len() > 7 {
                    return false;
                }
            }
            chains.push((ring, ctets));
            if chains.len() > 2 {
                return false;
            }
        }

        // Re-triangulate each side without edge ab; the closing chord d-c becomes
        // the new surface edge.
        let mut plan: Vec<(Vec<u32>, Vec<[u32; 4]>)> = vec![];
        let (mut q3_old, mut q3_new) = (f64::INFINITY, f64::INFINITY);
        for (r, ctets) in &chains {
            let nr = r.len();
            if nr < 3 || !self.same_region(ctets) {
                return false;
            }
            let mut best: Option<(Vec<[u32; 4]>, f64, f64)> = None;
            'tri: for tris in polygon_triangulations(nr) {
                let mut new = Vec::with_capacity(2 * tris.len());
                for &[i, j, k] in &tris {
                    for (x, y) in [(i, j), (j, k), (i, k)] {
                        let polygon_edge = x.abs_diff(y) == 1 || x.abs_diff(y) == nr - 1;
                        if !polygon_edge && self.shell.contains_key(&ekey(r[x], r[y])) {
                            continue 'tri;
                        }
                    }
                    if !self.face_tets(fkey(r[i], r[j], r[k])).is_empty() {
                        continue 'tri;
                    }
                    new.push([r[i], r[j], r[k], a]);
                    new.push([r[i], r[j], r[k], b]);
                }
                if let Some((out, qn, qo)) = self.valid(ctets, &new) {
                    if best.as_ref().map_or(true, |(_, qb, _)| qn > *qb) {
                        best = Some((out, qn, qo));
                    }
                }
            }
            let Some((out, qn, qo)) = best else { return false };
            q3_new = q3_new.min(qn);
            q3_old = q3_old.min(qo);
            plan.push((ctets.clone(), out));
        }
        // Accept if the surface improves without making the volume noticeably worse.
        if q3_new < 0.8 * q3_old {
            return false;
        }
        for (old, new) in plan {
            self.replace(&old, new);
        }
        self.faces.remove(&fc);
        self.faces.remove(&fd);
        self.faces.insert(fkey(a, c, d), tags.clone());
        self.faces.insert(fkey(b, c, d), tags);
        self.fixed_edges.remove(&e);
        self.fixed_edges.insert(ekey(c, d));
        self.stats.n_surface_flips += 1;
        true
    }

    fn surface_sweep(&mut self) -> usize {
        if self.planar.is_empty() {
            return 0;
        }
        let mut zone_nodes: HashSet<u32> = HashSet::new();
        for i in 0..self.tets.len() {
            if self.alive[i] && self.zone[i] {
                zone_nodes.extend(self.tets[i]);
            }
        }
        let mut cand: Vec<(f64, Face)> = self
            .faces
            .iter()
            .filter(|(f, tags)| tags.len() == 1 && self.planar.contains(&tags[0]) && f.iter().all(|v| zone_nodes.contains(v)))
            .map(|(f, _)| (self.tri_quality(*f), *f))
            .filter(|(q, _)| *q < self.params.flip_quality)
            .collect();
        cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let mut n = 0;
        for (_, f) in cand {
            if !self.faces.contains_key(&f) {
                continue;
            }
            // Try the longest edge first: that is the diagonal a sliver wants gone.
            let mut edges: Vec<Edge> = (0..3).map(|k| ekey(f[k], f[(k + 1) % 3])).collect();
            edges.sort_by(|x, y| {
                let l = |e: &Edge| dot(sub(self.nodes[e.0 as usize], self.nodes[e.1 as usize]), sub(self.nodes[e.0 as usize], self.nodes[e.1 as usize]));
                l(y).total_cmp(&l(x))
            });
            for e in edges {
                if self.try_surface_flip(e) {
                    n += 1;
                    break;
                }
            }
        }
        n
    }

    // ------------------------------------------------------------------
    // Smoothing
    // ------------------------------------------------------------------

    fn smooth_sweep(&mut self) -> usize {
        self.rebuild_slide();
        let mut node_tets: HashMap<u32, Vec<u32>> = HashMap::new();
        for (i, t) in self.tets.iter().enumerate() {
            if !self.alive[i] {
                continue;
            }
            for &v in t {
                node_tets.entry(v).or_default().push(i as u32);
            }
        }
        let mut cand: Vec<u32> = (0..self.tets.len())
            .filter(|&i| self.alive[i] && self.zone[i])
            .flat_map(|i| self.tets[i])
            .filter(|v| !self.fixed_nodes.contains(v) || self.slide.contains_key(v))
            .collect();
        cand.sort_unstable();
        cand.dedup();

        let mut moved = 0;
        for v in cand {
            let inc = &node_tets[&v];
            let mut nb: Vec<u32> = inc.iter().flat_map(|&i| self.tets[i as usize]).filter(|&x| x != v).collect();
            nb.sort_unstable();
            nb.dedup();
            let centroid = |ids: &[u32]| {
                let mut c = [0.0; 3];
                for &x in ids {
                    c = add(c, self.nodes[x as usize]);
                }
                scale(c, 1.0 / ids.len() as f64)
            };
            let p0 = self.nodes[v as usize];

            // Candidate displacements: interior nodes move freely; face nodes only
            // tangentially, towards their face neighbours or all neighbours.
            let sliding = self.fixed_nodes.contains(&v);
            let dirs: Vec<P3> = if sliding {
                let (n, fnb) = &self.slide[&v];
                let tangential = |d: P3| sub(d, scale(*n, dot(d, *n)));
                vec![tangential(sub(centroid(fnb), p0)), tangential(sub(centroid(&nb), p0))]
            } else {
                // A node whose incident tets span several regions lies on an
                // interface; those are tagged faces, but be defensive anyway.
                let r0 = self.region[inc[0] as usize];
                if inc.iter().any(|&i| self.region[i as usize] != r0) {
                    continue;
                }
                vec![sub(centroid(&nb), p0)]
            };

            let q0 = inc.iter().map(|&i| self.q(self.tets[i as usize])).fold(f64::INFINITY, f64::min);
            let mut accepted = false;
            'search: for d in dirs {
                for alpha in [1.0, 0.5, 0.25] {
                    self.nodes[v as usize] = add(p0, scale(d, alpha));
                    let q1 = inc.iter().map(|&i| self.q(self.tets[i as usize])).fold(f64::INFINITY, f64::min);
                    if q1 > q0 + 1e-6 {
                        accepted = true;
                        break 'search;
                    }
                }
            }
            if accepted {
                moved += 1;
                if sliding {
                    self.stats.n_slid += 1;
                }
            } else {
                self.nodes[v as usize] = p0;
            }
        }
        moved
    }

    /// Grow the work zone by `layers` rings of node-adjacent tets.
    fn expand_zone(&mut self) {
        if self.params.global {
            self.zone.iter_mut().for_each(|z| *z = true);
            self.stats.n_zone = self.alive.iter().filter(|&&a| a).count();
            return;
        }
        let n_nodes = self.nodes.len();
        for _ in 0..self.params.layers {
            let mut hot = vec![false; n_nodes];
            for (i, t) in self.tets.iter().enumerate() {
                if self.alive[i] && self.zone[i] {
                    for &v in t {
                        hot[v as usize] = true;
                    }
                }
            }
            for (i, t) in self.tets.iter().enumerate() {
                if self.alive[i] && t.iter().any(|&v| hot[v as usize]) {
                    self.zone[i] = true;
                }
            }
        }
        self.stats.n_zone = (0..self.tets.len()).filter(|&i| self.alive[i] && self.zone[i]).count();
    }

    pub fn run(&mut self) {
        self.expand_zone();
        for _ in 0..self.params.sweeps {
            let flips = self.flip_sweep() + self.surface_sweep();
            let mut moved = 0;
            for _ in 0..self.params.smooth_iters {
                moved += self.smooth_sweep();
            }
            self.stats.n_smoothed += moved;
            if flips == 0 && moved == 0 {
                break;
            }
        }
    }

    /// Compact the alive tets: returns (tets, region, root, active, tagged faces).
    #[allow(clippy::type_complexity)]
    pub fn finish(self) -> (Vec<[u32; 4]>, Vec<i32>, Vec<u32>, Vec<bool>, Vec<(Face, i32)>) {
        let mut t = vec![];
        let mut r = vec![];
        let mut o = vec![];
        let mut a = vec![];
        for i in 0..self.tets.len() {
            if self.alive[i] {
                t.push(self.tets[i]);
                r.push(self.region[i]);
                o.push(self.root[i]);
                a.push(self.active[i]);
            }
        }
        let mut faces: Vec<(Face, i32)> =
            self.faces.iter().flat_map(|(f, tags)| tags.iter().map(move |t| (*f, *t))).collect();
        faces.sort_unstable();
        (t, r, o, a, faces)
    }
}

/// All triangulations of a convex polygon with vertices 0..n (Catalan(n-2) of them).
fn polygon_triangulations(n: usize) -> Vec<Vec<[usize; 3]>> {
    fn rec(i: usize, j: usize) -> Vec<Vec<[usize; 3]>> {
        if j < i + 2 {
            return vec![vec![]];
        }
        let mut out = vec![];
        for k in i + 1..j {
            for l in rec(i, k) {
                for r in rec(k, j) {
                    let mut t = Vec::with_capacity(l.len() + r.len() + 1);
                    t.push([i, k, j]);
                    t.extend_from_slice(&l);
                    t.extend_from_slice(&r);
                    out.push(t);
                }
            }
        }
        out
    }
    rec(0, n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aura::refine::{Params, Refiner};

    /// n^3 grid of cubes, each split into 6 Kuhn tets, with jittered interior nodes.
    fn grid(n: usize) -> (Vec<P3>, Vec<[u32; 4]>) {
        let id = |i: usize, j: usize, k: usize| (i + (n + 1) * (j + (n + 1) * k)) as u32;
        let mut nodes = vec![];
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for k in 0..=n {
            for j in 0..=n {
                for i in 0..=n {
                    let interior = i > 0 && i < n && j > 0 && j < n && k > 0 && k < n;
                    let s = if interior { 0.3 } else { 0.0 };
                    nodes.push([i as f64 + s * rnd(), j as f64 + s * rnd(), k as f64 + s * rnd()]);
                }
            }
        }
        let mut tets = vec![];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let c = |b: usize| id(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                    for p in [[1, 2, 4], [1, 4, 2], [2, 1, 4], [2, 4, 1], [4, 1, 2], [4, 2, 1]] {
                        let t = [c(0), c(p[0]), c(p[0] + p[1]), c(7)];
                        let x = t.map(|v| nodes[v as usize]);
                        tets.push(if vol6(x[0], x[1], x[2], x[3]) < 0.0 { [t[0], t[1], t[3], t[2]] } else { t });
                    }
                }
            }
        }
        (nodes, tets)
    }

    fn faces_count(tets: &[[u32; 4]]) -> HashMap<Face, usize> {
        let mut m = HashMap::new();
        for t in tets {
            for f in TET_FACES {
                *m.entry(fkey(t[f[0]], t[f[1]], t[f[2]])).or_default() += 1;
            }
        }
        m
    }

    #[test]
    fn planar_sliding_keeps_nodes_on_their_faces() {
        let n = 4;
        let (nodes, tets) = grid(n);
        let nf = n as f64;
        // Tag each boundary triangle with the cube face it lies on (1..=6) and the
        // cube edges as curves, like a real CAD box.
        let on = |p: P3, k: usize, v: f64| (p[k] - v).abs() < 1e-12;
        let face_of = |x: [P3; 3]| -> i32 {
            for (tag, k, v) in [(1, 0, 0.0), (2, 0, nf), (3, 1, 0.0), (4, 1, nf), (5, 2, 0.0), (6, 2, nf)] {
                if x.iter().all(|p| on(*p, k, v)) {
                    return tag;
                }
            }
            0
        };
        let bnd: Vec<(Face, i32)> = faces_count(&tets)
            .into_iter()
            .filter(|(_, c)| *c == 1)
            .map(|(f, _)| (f, face_of(f.map(|i| nodes[i as usize]))))
            .collect();
        let mut curves: HashSet<Edge> = HashSet::new();
        for t in &tets {
            for (i, j) in TET_EDGES {
                let (a, b) = (nodes[t[i] as usize], nodes[t[j] as usize]);
                let planes = |p: P3| (0..3).filter(|&k| on(p, k, 0.0) || on(p, k, nf)).count();
                let shared = (0..3).filter(|&k| (on(a, k, 0.0) && on(b, k, 0.0)) || (on(a, k, nf) && on(b, k, nf))).count();
                if planes(a) >= 2 && planes(b) >= 2 && shared >= 2 {
                    curves.insert(ekey(t[i], t[j]));
                }
            }
        }
        let fixed: HashSet<Face> = bnd.iter().map(|x| x.0).collect();
        let planar: HashSet<i32> = (1..=6).collect();
        let mut nodes = nodes;
        let before = nodes.clone();
        let m = tets.len();
        let _ = &fixed;
        let mut imp = Improver::new(&mut nodes, tets, vec![0; m], vec![0; m], vec![true; m], &bnd, &curves,
                                    ImproveParams::default());
        imp.set_planar_tags(&planar);
        imp.run();
        assert!(imp.stats.n_slid > 0, "{:?}", imp.stats);
        let (tets, ..) = imp.finish();
        for (f, tag) in &bnd {
            let (k, v) = [(0, 0.0), (0, nf), (1, 0.0), (1, nf), (2, 0.0), (2, nf)][(*tag - 1) as usize];
            for &i in f {
                assert!((nodes[i as usize][k] - v).abs() < 1e-12, "node left its face plane");
            }
        }
        for e in &curves {
            assert_eq!(nodes[e.0 as usize], before[e.0 as usize]);
        }
        let vol: f64 = tets.iter().map(|t| {
            let x = t.map(|v| nodes[v as usize]);
            let v = vol6(x[0], x[1], x[2], x[3]);
            assert!(v > 0.0);
            v / 6.0
        }).sum();
        assert!((vol - nf * nf * nf).abs() < 1e-9);
    }

    #[test]
    fn triangulation_counts_are_catalan() {
        let counts: Vec<usize> = (3..=7).map(|n| polygon_triangulations(n).len()).collect();
        assert_eq!(counts, vec![1, 2, 5, 14, 42]);
        for t in polygon_triangulations(6) {
            assert_eq!(t.len(), 4);
        }
    }

    #[test]
    fn improve_keeps_mesh_valid_and_improves_quality() {
        let n = 4;
        let (nodes, tets) = grid(n);
        let bnd: Vec<(Face, i32)> =
            faces_count(&tets).into_iter().filter(|(_, c)| *c == 1).map(|(f, _)| (f, 1)).collect();
        let err: Vec<f64> = (0..tets.len()).map(|i| 1.0 + (i % 7) as f64).collect();
        let brep = Brep::default();
        let mut r = Refiner::new(nodes, tets, err, &bnd, &[], &brep, Params { growth: 1.0, ..Default::default() });
        r.run();
        let mut out = r.finish();
        let qmin = |nodes: &[P3], tets: &[[u32; 4]]| {
            tets.iter()
                .map(|t| {
                    let x = t.map(|v| nodes[v as usize]);
                    mean_ratio(x[0], x[1], x[2], x[3])
                })
                .fold(f64::INFINITY, f64::min)
        };
        let qmean = |nodes: &[P3], tets: &[[u32; 4]]| {
            tets.iter()
                .map(|t| {
                    let x = t.map(|v| nodes[v as usize]);
                    mean_ratio(x[0], x[1], x[2], x[3])
                })
                .sum::<f64>()
                / tets.len() as f64
        };
        let (q0, m0) = (qmin(&out.nodes, &out.tets), qmean(&out.nodes, &out.tets));

        let fixed: HashSet<Face> = out.face_tris.iter().copied().collect();
        let bnd_tagged: Vec<(Face, i32)> = out.face_tris.iter().copied().zip(out.face_tri_tags.iter().copied()).collect();
        let region = vec![0; out.tets.len()];
        let tets = std::mem::take(&mut out.tets);
        let active = vec![true; tets.len()];
        let root = out.tet_root.clone();
        let mut imp =
            Improver::new(&mut out.nodes, tets, region, root, active, &bnd_tagged, &HashSet::new(), ImproveParams::default());
        imp.run();
        let stats = imp.stats.clone();
        let (tets, ..) = imp.finish();
        let (q1, m1) = (qmin(&out.nodes, &tets), qmean(&out.nodes, &tets));
        assert!(stats.n_flip23 + stats.n_flip32 + stats.n_flip44 + stats.n_flip_large > 0, "{stats:?}");
        assert!(stats.n_smoothed > 0);
        assert!(q1 >= q0, "min quality {q0} -> {q1}");
        assert!(m1 > m0, "mean quality {m0} -> {m1}");

        // Valid, conforming, volume preserving, boundary unchanged.
        let mut vol = 0.0;
        for t in &tets {
            let x = t.map(|v| out.nodes[v as usize]);
            let v = vol6(x[0], x[1], x[2], x[3]);
            assert!(v > 0.0);
            vol += v / 6.0;
        }
        assert!((vol - (n * n * n) as f64).abs() < 1e-9);
        let fc = faces_count(&tets);
        assert!(fc.values().all(|&c| c <= 2));
        let bnd_after: HashSet<Face> = fc.iter().filter(|(_, &c)| c == 1).map(|(f, _)| *f).collect();
        assert_eq!(bnd_after, fixed);
    }
}
