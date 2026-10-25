// [Written by Claude] All code in this module was written by Claude (AI).
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
    /// [Written by Claude] hzAURA: with `grid = (nu, nv)` the samples are a regular grid of the
    /// face parametrisation (u major); the projection then follows a bicubic interpolation of it.
    Sampled { pts: Vec<P3>, nrm: Vec<P3>, grid: (usize, usize) },
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
            Surface::Sampled { pts, nrm, grid } => {
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
                if grid.0 >= 2 && grid.1 >= 2 && grid.0 * grid.1 == pts.len() {
                    return SampleGrid::new(pts, *grid).project(p, best);
                }
                // [Written by Claude] hzAURA: a local quadric through the nearest samples follows
                // the face to third order; the tangent plane of one sample only to second.
                if let Some(q) = project_quadric(pts, nrm, best, p) {
                    return q;
                }
                let n = nrm[best];
                let off = dot(sub(p, pts[best]), n);
                sub(p, scale(n, off))
            }
        }
    }
}

// [Written by Claude]
/// A regular (u, v) sample grid of a face, interpolated by Catmull-Rom splines in both
/// directions (exact for quadratics, C1). Rows or columns that repeat at the far end
/// (closed faces) wrap around.
struct SampleGrid<'a> {
    pts: &'a [P3],
    nu: usize,
    nv: usize,
    wrap_u: bool,
    wrap_v: bool,
}

impl<'a> SampleGrid<'a> {
    fn new(pts: &'a [P3], (nu, nv): (usize, usize)) -> Self {
        let span = pts.iter().fold(0.0f64, |m, p| m.max(norm(*p)));
        let tol = 1e-9 * span.max(1e-30);
        let at = |i: usize, j: usize| pts[i * nv + j];
        let wrap_u = nu > 3 && (0..nv).all(|j| norm(sub(at(0, j), at(nu - 1, j))) < tol);
        let wrap_v = nv > 3 && (0..nu).all(|i| norm(sub(at(i, 0), at(i, nv - 1))) < tol);
        SampleGrid { pts, nu, nv, wrap_u, wrap_v }
    }

    /// Sample (i, j) for any integer index: wrapped on closed directions, clamped otherwise.
    fn at(&self, i: i64, j: i64) -> P3 {
        let idx = |k: i64, n: usize, wrap: bool| -> usize {
            if wrap {
                k.rem_euclid(n as i64 - 1) as usize
            } else {
                k.clamp(0, n as i64 - 1) as usize
            }
        };
        self.pts[idx(i, self.nu, self.wrap_u) * self.nv + idx(j, self.nv, self.wrap_v)]
    }

    /// Point and first derivatives at grid coordinates (s, t).
    fn eval(&self, s: f64, t: f64) -> (P3, P3, P3) {
        // Catmull-Rom weights and their derivatives for the local coordinate x in [0, 1].
        let w = |x: f64| {
            let (x2, x3) = (x * x, x * x * x);
            (
                [-0.5 * x3 + x2 - 0.5 * x, 1.5 * x3 - 2.5 * x2 + 1.0, -1.5 * x3 + 2.0 * x2 + 0.5 * x, 0.5 * x3 - 0.5 * x2],
                [-1.5 * x2 + 2.0 * x - 0.5, 4.5 * x2 - 5.0 * x, -4.5 * x2 + 4.0 * x + 0.5, 1.5 * x2 - x],
            )
        };
        let (i0, j0) = (s.floor(), t.floor());
        let (wu, du) = w(s - i0);
        let (wv, dv) = w(t - j0);
        let (mut p, mut ps, mut pt) = ([0.0; 3], [0.0; 3], [0.0; 3]);
        for a in 0..4 {
            for b in 0..4 {
                let x = self.at(i0 as i64 + a as i64 - 1, j0 as i64 + b as i64 - 1);
                p = add(p, scale(x, wu[a] * wv[b]));
                ps = add(ps, scale(x, du[a] * wv[b]));
                pt = add(pt, scale(x, wu[a] * dv[b]));
            }
        }
        (p, ps, pt)
    }

    /// Closest point to `p`, by Gauss-Newton in (s, t) from the nearest sample `best`.
    fn project(&self, p: P3, best: usize) -> P3 {
        let (umax, vmax) = ((self.nu - 1) as f64, (self.nv - 1) as f64);
        let (mut s, mut t) = ((best / self.nv) as f64, (best % self.nv) as f64);
        let fold = |x: f64, max: f64, wrap: bool| if wrap { x.rem_euclid(max) } else { x.clamp(0.0, max) };
        for _ in 0..12 {
            let (x, xs, xt) = self.eval(s, t);
            let r = sub(p, x);
            let (a, b, c) = (dot(xs, xs), dot(xs, xt), dot(xt, xt));
            let (g1, g2) = (dot(xs, r), dot(xt, r));
            let det = a * c - b * b;
            if det.abs() < 1e-300 {
                break;
            }
            let (mut ds, mut dt) = ((c * g1 - b * g2) / det, (a * g2 - b * g1) / det);
            // at most one grid cell per step
            let l = ds.abs().max(dt.abs());
            if l > 1.0 {
                ds /= l;
                dt /= l;
            }
            s = fold(s + ds, umax, self.wrap_u);
            t = fold(t + dt, vmax, self.wrap_v);
            if l < 1e-10 {
                break;
            }
        }
        self.eval(s, t).0
    }
}

/// Number of samples a local quadric is fitted to.
const QUADRIC_SAMPLES: usize = 15;

// [Written by Claude]
/// Project `p` onto the height field z = f(x, y), a least-squares quadric through the samples
/// nearest to sample `i0`, in the tangent frame of that sample. None if the fit is singular.
fn project_quadric(pts: &[P3], nrm: &[P3], i0: usize, p: P3) -> Option<P3> {
    if pts.len() < QUADRIC_SAMPLES {
        return None;
    }
    let o = pts[i0];
    let n = nrm[i0];
    let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let t1 = {
        let v = cross(n, a);
        scale(v, 1.0 / norm(v))
    };
    let t2 = cross(n, t1);
    let local = |x: P3| {
        let d = sub(x, o);
        [dot(d, t1), dot(d, t2), dot(d, n)]
    };
    let mut near: Vec<(f64, usize)> = pts
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let d = sub(*s, o);
            (dot(d, d), k)
        })
        .collect();
    near.select_nth_unstable_by(QUADRIC_SAMPLES - 1, |x, y| x.0.total_cmp(&y.0));
    // Normal equations of z = c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2, scaled by the sample radius.
    let h = near[..QUADRIC_SAMPLES].iter().map(|x| x.0).fold(0.0, f64::max).sqrt();
    if h == 0.0 {
        return None;
    }
    let basis = |x: f64, y: f64| [1.0, x, y, x * x, x * y, y * y];
    let mut m = [[0.0f64; 7]; 6];
    for &(_, k) in &near[..QUADRIC_SAMPLES] {
        let l = local(pts[k]);
        // samples on the far side of a fold (normal turned away) do not belong to this sheet
        if dot(nrm[k], n) < 0.5 {
            continue;
        }
        let b = basis(l[0] / h, l[1] / h);
        for r in 0..6 {
            for c in 0..6 {
                m[r][c] += b[r] * b[c];
            }
            m[r][6] += b[r] * l[2];
        }
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..6 {
        let piv = (c..6).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if m[piv][c].abs() < 1e-10 {
            return None;
        }
        m.swap(c, piv);
        for r in 0..6 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..7 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    let coef: [f64; 6] = std::array::from_fn(|r| m[r][6] / m[r][r]);
    let f = |x: f64, y: f64| {
        let (u, v) = (x / h, y / h);
        let z = coef[0] + coef[1] * u + coef[2] * v + coef[3] * u * u + coef[4] * u * v + coef[5] * v * v;
        let fx = (coef[1] + 2.0 * coef[3] * u + coef[4] * v) / h;
        let fy = (coef[2] + coef[4] * u + 2.0 * coef[5] * v) / h;
        (z, fx, fy)
    };
    // Closest point on the graph: a few steps along the local normal.
    let lp = local(p);
    let mut q = lp;
    for _ in 0..4 {
        let (z, fx, fy) = f(q[0], q[1]);
        let g = [-fx, -fy, 1.0];
        let gn = norm(g);
        let g = scale(g, 1.0 / gn);
        let base = [q[0], q[1], z];
        q = sub(lp, scale(g, dot(sub(lp, base), g)));
    }
    let (z, _, _) = f(q[0], q[1]);
    if !z.is_finite() || (z - lp[2]).abs() > 4.0 * h {
        return None;
    }
    Some(add(o, add(add(scale(t1, q[0]), scale(t2, q[1])), scale(n, z))))
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

    // [Written by Claude]
    #[test]
    fn sampled_projection_follows_the_curvature() {
        // samples of a unit sphere cap: the quadric fit lands much closer than the tangent plane
        let mut pts = vec![];
        let mut nrm = vec![];
        for i in 0..20 {
            for j in 0..40 {
                let th = 0.05 + 1.0 * i as f64 / 19.0;
                let ph = 2.0 * std::f64::consts::PI * j as f64 / 40.0;
                let x = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
                pts.push(x);
                nrm.push(x);
            }
        }
        let s = Surface::Sampled { pts: pts.clone(), nrm: nrm.clone(), grid: (0, 0) };
        for p in [[0.31, 0.12, 0.9], [0.5, -0.4, 0.6], [0.05, 0.6, 0.85]] {
            let q = s.project(p);
            assert!((norm(q) - 1.0).abs() < 2e-4, "{p:?} -> {}", norm(q));
        }
        // the same samples as a (theta, phi) grid, closed in phi: bicubic interpolation
        let mut gp = vec![];
        for i in 0..20 {
            for j in 0..41 {
                let th = 0.05 + 1.0 * i as f64 / 19.0;
                let ph = 2.0 * std::f64::consts::PI * j as f64 / 40.0;
                gp.push([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()]);
            }
        }
        let s = Surface::Sampled { pts: gp.clone(), nrm: gp, grid: (20, 41) };
        for p in [[0.31, 0.12, 0.9], [0.5, -0.4, 0.6], [0.05, 0.6, 0.85], [0.6, -0.001, 0.7]] {
            let q = s.project(p);
            assert!((norm(q) - 1.0).abs() < 2e-5, "{p:?} -> {}", norm(q));
            // the closest point of a sphere is on the ray through the centre
            assert!(norm(cross(q, p)) / norm(p) < 1e-3, "{p:?} -> {q:?}");
        }
    }

    #[test]
    fn circle_projection() {
        let c = Curve::Circle { c: [0.0; 3], n: [0.0, 0.0, 1.0], r: 1.0 };
        let q = c.project([0.5, 0.5, 0.3]);
        assert!((q[2]).abs() < 1e-12);
        assert!((norm(q) - 1.0).abs() < 1e-12);
    }
}
