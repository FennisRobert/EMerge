//! Snapshot of the original BREP geometry and point projection onto it.
//!
//! The Python side samples the CAD model once (sphere/cylinder fits, circles,
//! or dense samples for anything else). New nodes created on a geometric
//! curve or face are projected back onto that entity so that refinement
//! converges to the true geometry instead of the initial facets.

use std::collections::HashMap;

pub type P3 = [f64; 3];

#[inline]
pub fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub fn norm(a: P3) -> f64 {
    dot(a, a).sqrt()
}

#[derive(Clone, Debug)]
pub enum Surface {
    /// Planar or unknown: the edge midpoint already lies on it.
    Flat,
    Sphere { c: P3, r: f64 },
    Cylinder { p: P3, axis: P3, r: f64 },
    /// Dense samples with normals; projects onto the tangent plane of the nearest sample.
    Sampled { pts: Vec<P3>, nrm: Vec<P3> },
}

#[derive(Clone, Debug)]
pub enum Curve {
    /// Straight line or unknown: the edge midpoint already lies on it.
    Straight,
    Circle { c: P3, n: P3, r: f64 },
    /// Ordered polyline samples along the curve.
    Polyline { pts: Vec<P3> },
}

impl Surface {
    pub fn project(&self, p: P3) -> P3 {
        match self {
            Surface::Flat => p,
            Surface::Sphere { c, r } => {
                let d = sub(p, *c);
                let l = norm(d);
                if l == 0.0 {
                    return p;
                }
                add(*c, scale(d, r / l))
            }
            Surface::Cylinder { p: p0, axis, r } => {
                let d = sub(p, *p0);
                let t = dot(d, *axis);
                let radial = sub(d, scale(*axis, t));
                let l = norm(radial);
                if l == 0.0 {
                    return p;
                }
                add(add(*p0, scale(*axis, t)), scale(radial, r / l))
            }
            Surface::Sampled { pts, nrm } => {
                if pts.is_empty() {
                    return p;
                }
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for (i, s) in pts.iter().enumerate() {
                    let d = sub(p, *s);
                    let dd = dot(d, d);
                    if dd < bd {
                        bd = dd;
                        best = i;
                    }
                }
                let n = nrm[best];
                let off = dot(sub(p, pts[best]), n);
                sub(p, scale(n, off))
            }
        }
    }
}

impl Curve {
    pub fn project(&self, p: P3) -> P3 {
        match self {
            Curve::Straight => p,
            Curve::Circle { c, n, r } => {
                let v = sub(p, *c);
                let vp = sub(v, scale(*n, dot(v, *n)));
                let l = norm(vp);
                if l == 0.0 {
                    return p;
                }
                add(*c, scale(vp, r / l))
            }
            Curve::Polyline { pts } => {
                if pts.len() < 2 {
                    return p;
                }
                let mut best = p;
                let mut bd = f64::INFINITY;
                for w in pts.windows(2) {
                    let ab = sub(w[1], w[0]);
                    let l2 = dot(ab, ab);
                    let t = if l2 > 0.0 {
                        (dot(sub(p, w[0]), ab) / l2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let q = add(w[0], scale(ab, t));
                    let d = sub(p, q);
                    let dd = dot(d, d);
                    if dd < bd {
                        bd = dd;
                        best = q;
                    }
                }
                best
            }
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct Brep {
    pub surfaces: HashMap<i32, Surface>,
    pub curves: HashMap<i32, Curve>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_projection() {
        let s = Surface::Sphere { c: [1.0, 0.0, 0.0], r: 2.0 };
        let q = s.project([1.0, 0.0, 1.0]);
        assert!((norm(sub(q, [1.0, 0.0, 0.0])) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn circle_projection() {
        let c = Curve::Circle { c: [0.0; 3], n: [0.0, 0.0, 1.0], r: 1.0 };
        let q = c.project([0.5, 0.5, 0.3]);
        assert!((q[2]).abs() < 1e-12);
        assert!((norm(q) - 1.0).abs() < 1e-12);
    }
}
