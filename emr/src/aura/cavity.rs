// [Written by Claude] All code in this module was written by Claude (AI).
//! hzAURA: cavity re-meshing for boundary nodes that bisection cannot project.
//!
//! AURA places the midpoint of a split edge on a curved face by moving it from
//! the chord onto the CAD surface. When the tets around the edge are too thin
//! for that (the move would invert or flatten a child), plain AURA halves the
//! move back towards the chord and the node stays off the surface, a dent.
//!
//! Here the node is put on the surface anyway and its neighbourhood is
//! re-meshed instead of bisected, as in a constrained Delaunay insertion:
//!
//! 1. The cavity starts with the shell of the edge and grows into every
//!    neighbour whose circumsphere contains the new point (Bowyer-Watson).
//!    It never grows across a tagged face or a region boundary, except to the
//!    tets behind the surface triangles that are replaced (two-sided faces).
//! 2. The tagged triangles of that face inside the cavity (the patch) are
//!    replaced by a fan from the new point to the patch boundary. For a node
//!    on a curve edge the patch covers the faces on both sides of the curve,
//!    and the curve edge is split into two halves through the new node. The
//!    patch must be a disk around the split edge; triangles or tets that keep
//!    it from being one are taken out of the cavity again.
//! 3. Every other boundary face of the cavity is joined to the new point.
//!    Faces the point does not see well (inverted or flat tets) make the
//!    cavity grow over them, or shrink back where growth is not allowed.
//!    This is tried with a strict quality floor first, then a relaxed one.
//!
//! Nodes, tagged curve edges and tagged faces outside the patch stay as they
//! are. If no valid cavity is found, nothing is changed and the caller falls
//! back to the bisection with a reduced projection.

use super::{ekey, fkey, mean_ratio, vol6, Edge, Face, Refiner};
use crate::aura::geom::*;
use std::collections::{HashMap, HashSet};

/// Limits of the cavity search.
const MAX_CAVITY: usize = 160;
const MAX_ROUNDS: usize = 60;
/// New tets must reach this mean ratio, or half the worst removed tet if that is lower;
/// the second value is tried when no cavity reaches the first.
const Q_FLOORS: [f64; 2] = [0.15, 0.05];

/// Face `k` of tet `t` (the face opposite vertex `t[k]`), sorted.
#[inline]
fn face_of(t: &[u32; 4], k: usize) -> Face {
    fkey(t[(k + 1) % 4], t[(k + 2) % 4], t[(k + 3) % 4])
}

/// Circumcentre and squared radius of a tetrahedron (None if it is flat).
fn circumsphere(a: P3, b: P3, c: P3, d: P3) -> Option<(P3, f64)> {
    let (u, v, w) = (sub(b, a), sub(c, a), sub(d, a));
    let det = 2.0 * dot(u, cross(v, w));
    if det.abs() < 1e-300 {
        return None;
    }
    let num = add(
        add(scale(cross(v, w), dot(u, u)), scale(cross(w, u), dot(v, v))),
        scale(cross(u, v), dot(w, w)),
    );
    let r = scale(num, 1.0 / det);
    Some((add(a, r), dot(r, r)))
}

/// A boundary face of the cavity: the owning tet and the local index of the
/// vertex opposite the face.
#[derive(Clone, Copy)]
struct Base {
    owner: u32,
    k: usize,
}

/// Outcome of one check of a candidate cavity.
enum Check {
    Ok(Vec<Base>, Vec<(Face, Vec<i32>)>),
    /// Tets to add to the cavity, tets to take out of it.
    Repair(Vec<u32>, Vec<u32>),
    Fail,
}

impl<'a> Refiner<'a> {
    #[inline]
    fn region_of(&self, it: u32) -> i32 {
        if self.region.is_empty() {
            0
        } else {
            self.region[self.root[it as usize] as usize]
        }
    }

    /// Live tets that contain face `f` (one or two).
    fn tets_of_face(&self, f: Face) -> Vec<u32> {
        // Any edge of the face that still has a shell; the split edge's shell is already taken out.
        for (i, j, o) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)] {
            if let Some(sh) = self.shell.get(&ekey(f[i], f[j])) {
                return sh.iter().copied().filter(|&u| self.tets[u as usize].contains(&f[o])).collect();
            }
        }
        vec![]
    }

    /// Tags of the faces the midpoint of `e` is moved on: the curved face of a face edge, or
    /// every face along a curve edge. Empty if the node stays where bisection puts it.
    pub(super) fn cavity_tags(&self, e: Edge, shell: &[u32]) -> Vec<i32> {
        let on_curve = self.curve_tags.contains_key(&e);
        let mut tags: Vec<i32> = vec![];
        for &it in shell {
            for &o in self.tets[it as usize].iter() {
                if o == e.0 || o == e.1 {
                    continue;
                }
                for tag in self.face_tags.get(&fkey(e.0, e.1, o)).into_iter().flatten() {
                    match self.brep.surfaces.get(tag) {
                        Some(Surface::Flat) if !on_curve => return vec![],
                        Some(_) if !on_curve => return vec![*tag],
                        _ => {
                            if on_curve && !tags.contains(tag) {
                                tags.push(*tag);
                            }
                        }
                    }
                }
            }
        }
        tags
    }

    /// Insert `q` (on the faces `tags`) instead of bisecting edge `e` with shell `shell`.
    /// Returns the new node id, or None (mesh unchanged) if no valid cavity exists.
    pub(super) fn cavity_insert(&mut self, e: Edge, shell: &[u32], q: P3, tags: &[i32]) -> Option<u32> {
        if shell.is_empty() {
            return None;
        }
        let sgn = {
            let t = self.tets[shell[0] as usize].map(|i| self.nodes[i as usize]);
            vol6(t[0], t[1], t[2], t[3]).signum()
        };
        let periodic: HashSet<u32> = self.periodic_nodes().into_iter().collect();
        let mandatory: HashSet<u32> = shell.iter().copied().collect();
        let mut cav: HashSet<u32> = mandatory.clone();

        // 1. Bowyer-Watson growth, constrained by tagged faces and regions.
        let mut queue: Vec<u32> = shell.to_vec();
        while let Some(it) = queue.pop() {
            let t = self.tets[it as usize];
            for k in 0..4 {
                let f = face_of(&t, k);
                if self.face_tags.contains_key(&f) {
                    continue;
                }
                for n in self.tets_of_face(f) {
                    if n == it || cav.contains(&n) || self.region_of(n) != self.region_of(it) {
                        continue;
                    }
                    let x = self.tets[n as usize].map(|i| self.nodes[i as usize]);
                    let Some((c, r2)) = circumsphere(x[0], x[1], x[2], x[3]) else { continue };
                    let d = sub(q, c);
                    if dot(d, d) < r2 * (1.0 - 1e-9) && cav.len() < MAX_CAVITY / 2 {
                        cav.insert(n);
                        queue.push(n);
                    }
                }
            }
        }

        // 2./3. Check and repair until the fan from q is valid; first with the strict
        // quality floor, then with the relaxed one (the improvement step follows).
        let start = cav.clone();
        let (cav, bases, fan) = 'search: {
            for floor in Q_FLOORS {
                let mut cav = start.clone();
                let mut excluded: HashSet<u32> = HashSet::new();
                for _ in 0..MAX_ROUNDS {
                    match self.check_cavity(e, &cav, &mandatory, &excluded, &periodic, q, tags, sgn, floor) {
                        Check::Ok(b, f) => break 'search (cav, b, f),
                        Check::Fail => break,
                        Check::Repair(add, del) => {
                            for n in del {
                                cav.remove(&n);
                                excluded.insert(n);
                            }
                            for n in add {
                                cav.insert(n);
                            }
                            if cav.len() > MAX_CAVITY {
                                break;
                            }
                        }
                    }
                }
            }
            return None;
        };
        // Commit: new node, fan tets, surface fan, remove the cavity.
        let m = self.nodes.len() as u32;
        self.nodes.push(q);
        let mut new_tets = Vec::with_capacity(bases.len());
        for b in &bases {
            let t = self.tets[b.owner as usize];
            let mut nt = t;
            nt[b.k] = m;
            let bump = u8::from(mandatory.contains(&b.owner));
            new_tets.push((
                nt,
                self.root[b.owner as usize],
                self.err[b.owner as usize] * self.params.child_factor,
                self.depth[b.owner as usize].saturating_add(bump),
            ));
        }
        for &it in &cav {
            let t = self.tets[it as usize];
            for k in 0..4 {
                let f = face_of(&t, k);
                if self.face_tags.get(&f).map_or(false, |ft| ft.iter().any(|t| tags.contains(t))) {
                    self.face_tags.remove(&f);
                }
            }
            self.kill_tet(it, e);
        }
        for (f, ft) in fan {
            self.face_tags.insert(f, ft);
        }
        // A split curve edge keeps its tags on both halves.
        if let Some(ct) = self.curve_tags.remove(&e) {
            self.curve_tags.insert(ekey(e.0, m), ct.clone());
            self.curve_tags.insert(ekey(m, e.1), ct);
        }
        for (t, root, err, depth) in new_tets {
            self.push_tet(t, root, err, depth);
        }
        self.stats.n_cavity += 1;
        self.stats.n_cavity_tets += cav.len();
        Some(m)
    }

    /// Take the optional ones of `tets` out of the cavity; fail if all of them are mandatory.
    fn drop_optional(&self, tets: impl Iterator<Item = u32>, mandatory: &HashSet<u32>) -> Check {
        let mut del: Vec<u32> = tets.filter(|u| !mandatory.contains(u)).collect();
        if del.is_empty() {
            return Check::Fail;
        }
        del.sort_unstable();
        del.dedup();
        Check::Repair(vec![], del)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_cavity(
        &self,
        e: Edge,
        cav: &HashSet<u32>,
        mandatory: &HashSet<u32>,
        excluded: &HashSet<u32>,
        periodic: &HashSet<u32>,
        q: P3,
        tags: &[i32],
        sgn: f64,
        floor: f64,
    ) -> Check {
        // Faces of the cavity tets with their owners.
        let mut owners: HashMap<Face, Vec<Base>> = HashMap::new();
        for &it in cav {
            let t = self.tets[it as usize];
            if t.iter().any(|v| periodic.contains(v)) {
                return Check::Fail;
            }
            for k in 0..4 {
                owners.entry(face_of(&t, k)).or_default().push(Base { owner: it, k });
            }
        }
        let on_patch = |f: &Face| self.face_tags.get(f).map_or(false, |ft| ft.iter().any(|t| tags.contains(t)));
        let mut add: Vec<u32> = vec![];
        let mut patch: Vec<Face> = vec![];
        let mut bases: Vec<Base> = vec![];
        for (f, bs) in &owners {
            if on_patch(f) {
                // Both sides of a replaced surface triangle must be in the cavity.
                for n in self.tets_of_face(*f) {
                    if !cav.contains(&n) {
                        if excluded.contains(&n) {
                            return Check::Fail;
                        }
                        add.push(n);
                    }
                }
                patch.push(*f);
                continue;
            }
            if bs.len() == 2 {
                // Interior face of the cavity: it must not be a tagged face.
                if self.face_tags.contains_key(f) {
                    return self.drop_optional(bs.iter().map(|b| b.owner), mandatory);
                }
                continue;
            }
            bases.push(bs[0]);
        }
        if !add.is_empty() {
            add.sort_unstable();
            add.dedup();
            return Check::Repair(add, vec![]);
        }
        if patch.is_empty() {
            return Check::Fail;
        }

        // The patch must be a disk: one boundary loop, no interior vertex.
        let mut edge_count: HashMap<Edge, (usize, Face)> = HashMap::new();
        for f in &patch {
            for (i, j) in [(0, 1), (1, 2), (0, 2)] {
                let x = edge_count.entry(ekey(f[i], f[j])).or_insert((0, *f));
                x.0 += 1;
            }
        }
        if edge_count.values().any(|x| x.0 > 2) {
            return Check::Fail;
        }
        let bnd: Vec<(Edge, Face)> = edge_count.iter().filter(|(_, x)| x.0 == 1).map(|(e, x)| (*e, x.1)).collect();
        let n_v: HashSet<u32> = patch.iter().flatten().copied().collect();
        let n_e = edge_count.len();
        if n_v.len() + patch.len() != n_e + 1 || bnd.len() != n_v.len() {
            // Not a disk. Patch triangles not connected to the split edge go out of the cavity...
            let mut reached: HashSet<Face> = patch.iter().copied().filter(|f| f.contains(&e.0) && f.contains(&e.1)).collect();
            let mut stack: Vec<Face> = reached.iter().copied().collect();
            while let Some(f) = stack.pop() {
                for g in &patch {
                    let shared = g.iter().filter(|v| f.contains(v)).count();
                    if shared == 2 && reached.insert(*g) {
                        stack.push(*g);
                    }
                }
            }
            let mut off: Vec<u32> = vec![];
            for f in patch.iter().filter(|f| !reached.contains(*f)) {
                off.extend(owners[f].iter().map(|b| b.owner));
            }
            // ...and so do the tets around patch nodes that no boundary edge touches (enclosed).
            if off.is_empty() {
                let on_bnd: HashSet<u32> = bnd.iter().flat_map(|(ed, _)| [ed.0, ed.1]).collect();
                for v in n_v.iter().filter(|v| !on_bnd.contains(v)) {
                    off.extend(cav.iter().copied().filter(|&u| self.tets[u as usize].contains(v)));
                }
            }
            return self.drop_optional(off.into_iter(), mandatory);
        }
        // The split edge itself must disappear from the surface.
        if edge_count.get(&e).map_or(true, |x| x.0 != 2) {
            return Check::Fail;
        }

        // Every node, tagged curve edge and other tagged face must survive on the cavity boundary.
        let kept_nodes: HashSet<u32> = bases.iter().flat_map(|b| face_of(&self.tets[b.owner as usize], b.k)).collect();
        let mut kept_edges: HashSet<Edge> = HashSet::new();
        for b in &bases {
            let f = face_of(&self.tets[b.owner as usize], b.k);
            kept_edges.extend([ekey(f[0], f[1]), ekey(f[1], f[2]), ekey(f[0], f[2])]);
        }
        let mut del: Vec<u32> = vec![];
        for &it in cav {
            let t = self.tets[it as usize];
            for v in t {
                if !kept_nodes.contains(&v) {
                    // Enclosed node: take the optional tets around it out of the cavity.
                    let opt: Vec<u32> = cav.iter().copied().filter(|&u| !mandatory.contains(&u) && self.tets[u as usize].contains(&v)).collect();
                    if opt.is_empty() {
                        return Check::Fail;
                    }
                    del.extend(opt);
                }
            }
            for (i, j) in super::TET_EDGES {
                let ed = ekey(t[i], t[j]);
                if ed != e && self.curve_tags.contains_key(&ed) && !kept_edges.contains(&ed) {
                    let around = cav.iter().copied().filter(|&u| {
                        let x = self.tets[u as usize];
                        x.contains(&ed.0) && x.contains(&ed.1)
                    });
                    return self.drop_optional(around, mandatory);
                }
            }
        }
        if !del.is_empty() {
            del.sort_unstable();
            del.dedup();
            return Check::Repair(vec![], del);
        }

        // Fan tets: positive and not much worse than the tets they replace.
        let q_old = cav
            .iter()
            .map(|&it| {
                let x = self.tets[it as usize].map(|i| self.nodes[i as usize]);
                mean_ratio(x[0], x[1], x[2], x[3])
            })
            .fold(f64::INFINITY, f64::min);
        let q_min = floor.min(0.5 * q_old);
        let (mut grow, mut shrink): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
        for b in &bases {
            let t = self.tets[b.owner as usize];
            let mut x = t.map(|i| self.nodes[i as usize]);
            let v_old = vol6(x[0], x[1], x[2], x[3]).abs();
            x[b.k] = q;
            let v = vol6(x[0], x[1], x[2], x[3]) * sgn;
            let scale_l = {
                let f = face_of(&t, b.k).map(|i| self.nodes[i as usize]);
                norm(sub(f[1], f[0])).max(norm(sub(f[2], f[0])))
            };
            let bad = v <= 1e-9 * scale_l.powi(3).max(v_old * 1e-6) || mean_ratio(x[0], x[1], x[2], x[3]) < q_min;
            if !bad {
                continue;
            }
            // Grow over an untagged face into a neighbour of the same region...
            let f = face_of(&t, b.k);
            if !self.face_tags.contains_key(&f) {
                let n: Vec<u32> = self.tets_of_face(f).into_iter().filter(|&u| u != b.owner).collect();
                if let Some(&n) = n.first() {
                    if !excluded.contains(&n) && self.region_of(n) == self.region_of(b.owner) {
                        grow.push(n);
                        continue;
                    }
                }
            }
            // ...or shrink, if the owner is optional.
            if !mandatory.contains(&b.owner) {
                shrink.push(b.owner);
                continue;
            }
            return Check::Fail;
        }
        if !grow.is_empty() || !shrink.is_empty() {
            for v in [&mut grow, &mut shrink] {
                v.sort_unstable();
                v.dedup();
            }
            return Check::Repair(grow, shrink);
        }

        // Surface fan: the new point joined to every boundary edge of the patch.
        let fan: Vec<(Face, Vec<i32>)> = bnd
            .iter()
            .map(|((u, v), pf)| (fkey(*u, *v, self.nodes.len() as u32), self.face_tags[pf].clone()))
            .collect();
        Check::Ok(bases, fan)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Params, Refiner};
    use super::*;
    use std::collections::HashMap;

    fn faces_count(tets: &[[u32; 4]]) -> HashMap<Face, usize> {
        let mut c: HashMap<Face, usize> = HashMap::new();
        for t in tets {
            for k in 0..4 {
                *c.entry(face_of(t, k)).or_default() += 1;
            }
        }
        c
    }

    /// The reflex-diagonal wedge of the sphere (see `wedge.rs`), without the wedge
    /// flip: bisection cannot put the midpoint of (2, 3) on the sphere, the cavity can.
    #[test]
    fn cavity_puts_node_on_sphere() {
        let (s, t) = (0.3f64, 0.5f64);
        let (c, d) = ((1.0 - s * s).sqrt(), (1.0 - t * t).sqrt());
        let nodes: Vec<P3> = vec![[s, 0.0, c], [-s, 0.0, c], [0.0, t, d], [0.0, -t, d], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]];
        let mut tets = vec![[0u32, 1, 2, 3], [0, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 5], [0, 1, 3, 5]];
        for t in tets.iter_mut() {
            let x = t.map(|i| nodes[i as usize]);
            if vol6(x[0], x[1], x[2], x[3]) < 0.0 {
                t.swap(2, 3);
            }
        }
        let faces = vec![(fkey(0, 2, 3), 7), (fkey(1, 2, 3), 7)];
        let mut surfaces = HashMap::new();
        surfaces.insert(7, Surface::Sphere { c: [0.0; 3], r: 1.0 });
        let brep = Brep { surfaces, curves: HashMap::new() };
        let vol0: f64 = tets.iter().map(|t| { let x = t.map(|i| nodes[i as usize]); vol6(x[0], x[1], x[2], x[3]) }).sum();
        for cavity in [false, true] {
            let mut r = Refiner::new(nodes.clone(), tets.clone(), vec![1.0; 5], &faces, &[], &brep, Params { cavity, ..Default::default() });
            r.set_regions(vec![1, 2, 2, 1, 1]);
            r.split_edge((2, 3));
            let st = r.stats.clone();
            let out = r.finish();
            let m = out.nodes.len() - 1;
            let off = (norm(out.nodes[m]) - 1.0).abs();
            if !cavity {
                assert!(off > 1e-3, "plain bisection reached the sphere ({off})");
                continue;
            }
            assert_eq!(st.n_cavity, 1, "{st:?}");
            assert!(off < 1e-12, "node {off} off the sphere");
            // valid, conforming, every face triangle on the sphere has the new node or is untouched
            let mut vol = 0.0;
            for t in &out.tets {
                let x = t.map(|i| out.nodes[i as usize]);
                let v = vol6(x[0], x[1], x[2], x[3]);
                assert!(v > 0.0);
                vol += v;
            }
            assert!(vol > vol0 - 1e-12);
            assert!(faces_count(&out.tets).values().all(|&c| c <= 2));
            assert!(out.face_tris.iter().any(|f| f.contains(&(m as u32))));
            // the face is still a closed disk bounded by the same four nodes
            let mut ec: HashMap<Edge, usize> = HashMap::new();
            for f in &out.face_tris {
                for (i, j) in [(0, 1), (1, 2), (0, 2)] {
                    *ec.entry(ekey(f[i], f[j])).or_default() += 1;
                }
            }
            let bnd: HashSet<Edge> = ec.iter().filter(|x| *x.1 == 1).map(|x| *x.0).collect();
            let want: HashSet<Edge> = [(0, 2), (0, 3), (1, 2), (1, 3)].into_iter().collect();
            assert_eq!(bnd, want);
        }
    }
}
