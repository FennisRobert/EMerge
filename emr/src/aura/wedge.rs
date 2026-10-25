// [Written by Claude] All code in this module was written by Claude (AI).
//! Concave wedges at curved interfaces.
//!
//! The surface mesh of a curved face can contain a reflex edge: two face
//! triangles (a, b, o1) and (a, b, o2) whose common edge (a, b) lies on the
//! concave side, while the other diagonal (o1, o2) of the quad would be convex.
//! The pocket between the two diagonals is a single tetrahedron (a, b, o1, o2)
//! that has both triangles as faces. When (a, b) is split, its midpoint has to
//! move into that tetrahedron to reach the surface, which inverts the children:
//! the projection is reduced or rejected and the new node stays on the chord,
//! a dent in the refined surface.
//!
//! Moving the wedge tetrahedron to the region on the other side of the interface
//! flips the interface to the convex diagonal (o1, o2) and removes the pocket.

use crate::aura::geom::*;
use std::collections::{HashMap, HashSet};

type Face = [u32; 3];

#[inline]
fn fkey(a: u32, b: u32, c: u32) -> Face {
    let mut f = [a, b, c];
    f.sort_unstable();
    f
}

#[inline]
fn ekey(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Reassign every concave wedge tetrahedron at a curved face to the region on
/// the other side of that face and move the face triangles to its two other
/// faces. Wedges on a curve edge or with a node in `locked` (periodic nodes)
/// are left alone. Returns the number of flipped wedges.
pub fn flip_concave_wedges(
    nodes: &[P3],
    tets: &[[u32; 4]],
    region: &mut [i32],
    faces: &mut Vec<(Face, i32)>,
    curve_edges: &HashSet<(u32, u32)>,
    locked: &HashSet<u32>,
    brep: &Brep,
) -> usize {
    let mut face_tets: HashMap<Face, Vec<u32>> = HashMap::with_capacity(tets.len() * 2);
    for (it, t) in tets.iter().enumerate() {
        for k in 0..4 {
            let f = fkey(t[(k + 1) % 4], t[(k + 2) % 4], t[(k + 3) % 4]);
            face_tets.entry(f).or_default().push(it as u32);
        }
    }
    // Indices into `faces` per triangle; removed entries are dropped at the end.
    let mut face_idx: HashMap<Face, Vec<usize>> = HashMap::new();
    for (i, (f, _)) in faces.iter().enumerate() {
        face_idx.entry(fkey(f[0], f[1], f[2])).or_default().push(i);
    }
    let mut removed = vec![false; faces.len()];
    let curved = |tag: i32| matches!(brep.surfaces.get(&tag), Some(s) if !matches!(s, Surface::Flat));
    let other_side = |f: &Face, it: u32| -> Option<u32> {
        let v = face_tets.get(f)?;
        if v.len() != 2 {
            return None;
        }
        Some(if v[0] == it { v[1] } else { v[0] })
    };

    let mut n_flips = 0;
    for (it, t) in tets.iter().enumerate() {
        let it = it as u32;
        if t.iter().any(|n| locked.contains(n)) {
            continue;
        }
        // Face k is opposite vertex t[k]; collect the live tags of each face.
        let fs: Vec<Face> = (0..4).map(|k| fkey(t[(k + 1) % 4], t[(k + 2) % 4], t[(k + 3) % 4])).collect();
        let tags: Vec<Vec<i32>> = fs
            .iter()
            .map(|f| {
                face_idx
                    .get(f)
                    .map(|v| v.iter().filter(|&&i| !removed[i]).map(|&i| faces[i].1).collect())
                    .unwrap_or_default()
            })
            .collect();
        let on: Vec<usize> = (0..4).filter(|&k| !tags[k].is_empty()).collect();
        if on.len() != 2 || tags[on[0]].len() != 1 || tags[on[0]] != tags[on[1]] || !curved(tags[on[0]][0]) {
            continue;
        }
        let tag = tags[on[0]][0];
        let off: Vec<usize> = (0..4).filter(|&k| tags[k].is_empty()).collect();
        // The two tagged faces are opposite o1 and o2; the edge (a, b) is shared by both.
        let (o1, o2) = (t[on[0]], t[on[1]]);
        let (a, b) = (t[off[0]], t[off[1]]);
        if curve_edges.contains(&ekey(a, b)) {
            continue;
        }
        // Interface on one side (one region behind both tagged faces), own region behind the others.
        let r = region[it as usize];
        let behind: Vec<Option<u32>> = on.iter().map(|&k| other_side(&fs[k], it)).collect();
        let (Some(u0), Some(u1)) = (behind[0], behind[1]) else { continue };
        let r_other = region[u0 as usize];
        if r_other == r || region[u1 as usize] != r_other {
            continue;
        }
        if off.iter().any(|&k| other_side(&fs[k], it).map_or(true, |u| region[u as usize] != r)) {
            continue;
        }
        // Concave: the surface point above the middle of (a, b) lies towards (o1, o2).
        let (pa, pb) = (nodes[a as usize], nodes[b as usize]);
        let mid = scale(add(pa, pb), 0.5);
        let q = brep.surfaces[&tag].project(mid);
        let opp = scale(add(nodes[o1 as usize], nodes[o2 as usize]), 0.5);
        if dot(sub(q, mid), sub(opp, mid)) <= 0.0 {
            continue;
        }
        region[it as usize] = r_other;
        for &k in &on {
            for &i in &face_idx[&fs[k]] {
                removed[i] = true;
            }
        }
        for &k in &off {
            let i = faces.len();
            faces.push((fs[k], tag));
            removed.push(false);
            face_idx.entry(fs[k]).or_default().push(i);
        }
        n_flips += 1;
    }
    if n_flips > 0 {
        let mut keep = removed.iter().map(|&x| !x);
        faces.retain(|_| keep.next().unwrap());
    }
    n_flips
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four points on the unit sphere around the north pole. The face triangles
    /// use the lower diagonal (2, 3), the reflex one: the wedge (0, 1, 2, 3) above
    /// them lies inside the ball but belongs to the outer region 1.
    #[test]
    fn flips_reflex_diagonal() {
        let (s, t) = (0.3f64, 0.5f64);
        let (c, d) = ((1.0 - s * s).sqrt(), (1.0 - t * t).sqrt());
        let nodes: Vec<P3> = vec![
            [s, 0.0, c],
            [-s, 0.0, c],
            [0.0, t, d],
            [0.0, -t, d],
            [0.0, 0.0, 0.0], // inside
            [0.0, 0.0, 3.0], // outside
        ];
        let tets = vec![[0, 1, 2, 3], [0, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 5], [0, 1, 3, 5]];
        let mut region = vec![1, 2, 2, 1, 1];
        let mut faces = vec![(fkey(0, 2, 3), 7), (fkey(1, 2, 3), 7)];
        let mut surfaces = HashMap::new();
        surfaces.insert(7, Surface::Sphere { c: [0.0; 3], r: 1.0 });
        let brep = Brep { surfaces, curves: HashMap::new() };
        let n = flip_concave_wedges(&nodes, &tets, &mut region, &mut faces, &HashSet::new(), &HashSet::new(), &brep);
        assert_eq!(n, 1);
        assert_eq!(region[0], 2);
        let mut f: Vec<Face> = faces.iter().map(|x| x.0).collect();
        f.sort();
        assert_eq!(f, vec![fkey(0, 1, 2), fkey(0, 1, 3)]);
        // the convex side is left as it is
        let n = flip_concave_wedges(&nodes, &tets, &mut region, &mut faces, &HashSet::new(), &HashSet::new(), &brep);
        assert_eq!(n, 0);
    }
}
