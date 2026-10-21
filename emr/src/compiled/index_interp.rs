//! Point location: finds the tetrahedron that contains each query point using a uniform spatial grid.
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

const EPS: f64 = 1e-8;
pub const GRID_RES: usize = 20;

/// Uniform grid over the query points in CSR form.
pub struct Grid {
    pub cell_offsets: Vec<i32>,
    pub cell_contents: Vec<i32>,
    pub xmin: f64,
    pub ymin: f64,
    pub zmin: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

/// Row-wise maximum of a (N, M) integer array.
pub fn matmax(a: &ndarray::ArrayView2<i64>) -> Array1<i64> {
    Array1::from_iter(a.outer_iter().map(|row| *row.iter().max().unwrap()))
}

fn cell_of(grid: &Grid, x: f64, y: f64, z: f64, res: usize) -> usize {
    let mut ix = ((x - grid.xmin) / grid.dx) as i64;
    let mut iy = ((y - grid.ymin) / grid.dy) as i64;
    let mut iz = ((z - grid.zmin) / grid.dz) as i64;
    let r = res as i64;
    if ix >= r {
        ix = r - 1;
    }
    if iy >= r {
        iy = r - 1;
    }
    if iz >= r {
        iz = r - 1;
    }
    (ix + iy * r + iz * r * r) as usize
}

/// Builds a CSR style res^3 grid of the query points (coords).
pub fn build_grid(coords: &ArrayView2<f64>, res: usize) -> Grid {
    let n_nodes = coords.shape()[1];
    let n_cells = res * res * res;
    let (mut xmin, mut xmax) = (coords[[0, 0]], coords[[0, 0]]);
    let (mut ymin, mut ymax) = (coords[[1, 0]], coords[[1, 0]]);
    let (mut zmin, mut zmax) = (coords[[2, 0]], coords[[2, 0]]);
    for j in 1..n_nodes {
        let (x, y, z) = (coords[[0, j]], coords[[1, j]], coords[[2, j]]);
        if x < xmin {
            xmin = x;
        }
        if x > xmax {
            xmax = x;
        }
        if y < ymin {
            ymin = y;
        }
        if y > ymax {
            ymax = y;
        }
        if z < zmin {
            zmin = z;
        }
        if z > zmax {
            zmax = z;
        }
    }
    xmin -= 1e-3;
    xmax += 1e-3;
    ymin -= 1e-3;
    ymax += 1e-3;
    zmin -= 1e-3;
    zmax += 1e-3;
    let mut grid = Grid {
        cell_offsets: vec![0; n_cells + 1],
        cell_contents: vec![0; n_nodes],
        xmin,
        ymin,
        zmin,
        dx: (xmax - xmin) / res as f64,
        dy: (ymax - ymin) / res as f64,
        dz: (zmax - zmin) / res as f64,
    };
    let mut counts = vec![0i32; n_cells];
    let cells: Vec<usize> = (0..n_nodes)
        .map(|j| cell_of(&grid, coords[[0, j]], coords[[1, j]], coords[[2, j]], res))
        .collect();
    for &c in &cells {
        counts[c] += 1;
    }
    for c in 0..n_cells {
        grid.cell_offsets[c + 1] = grid.cell_offsets[c] + counts[c];
    }
    let mut fill_pos: Vec<i32> = grid.cell_offsets[..n_cells].to_vec();
    for (j, &c) in cells.iter().enumerate() {
        grid.cell_contents[fill_pos[c] as usize] = j as i32;
        fill_pos[c] += 1;
    }
    grid
}

/// For every query point the (largest) index in tetids of a tet that contains it, -1 if none.
pub fn index_interp_inner(
    coords: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tetids: &ArrayView1<i64>,
    grid: &Grid,
    res: usize,
) -> Array1<i64> {
    let n_nodes = coords.shape()[1];
    let assigned: Vec<AtomicI64> = (0..n_nodes).map(|_| AtomicI64::new(-1)).collect();
    (0..tetids.len()).into_par_iter().for_each(|i_iter| {
        let itet = tetids[i_iter] as usize;
        let iv = [0, 1, 2, 3].map(|k| tets[[k, itet]] as usize);
        let (v1x, v1y, v1z) = (nodes[[0, iv[0]]], nodes[[1, iv[0]]], nodes[[2, iv[0]]]);
        let (m00, m10, m20) = (nodes[[0, iv[1]]] - v1x, nodes[[1, iv[1]]] - v1y, nodes[[2, iv[1]]] - v1z);
        let (m01, m11, m21) = (nodes[[0, iv[2]]] - v1x, nodes[[1, iv[2]]] - v1y, nodes[[2, iv[2]]] - v1z);
        let (m02, m12, m22) = (nodes[[0, iv[3]]] - v1x, nodes[[1, iv[3]]] - v1y, nodes[[2, iv[3]]] - v1z);
        let det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
        let inv_det = 1.0 / det;
        let b00 = (m11 * m22 - m12 * m21) * inv_det;
        let b01 = (m02 * m21 - m01 * m22) * inv_det;
        let b02 = (m01 * m12 - m02 * m11) * inv_det;
        let b10 = (m12 * m20 - m10 * m22) * inv_det;
        let b11 = (m00 * m22 - m02 * m20) * inv_det;
        let b12 = (m10 * m02 - m00 * m12) * inv_det;
        let b20 = (m10 * m21 - m11 * m20) * inv_det;
        let b21 = (m20 * m01 - m00 * m21) * inv_det;
        let b22 = (m00 * m11 - m10 * m01) * inv_det;

        let (mut bx0, mut bx1, mut by0, mut by1, mut bz0, mut bz1) = (v1x, v1x, v1y, v1y, v1z, v1z);
        for &ivk in &iv[1..] {
            let (nx, ny, nz) = (nodes[[0, ivk]], nodes[[1, ivk]], nodes[[2, ivk]]);
            bx0 = if nx < bx0 { nx } else { bx0 };
            bx1 = if nx > bx1 { nx } else { bx1 };
            by0 = if ny < by0 { ny } else { by0 };
            by1 = if ny > by1 { ny } else { by1 };
            bz0 = if nz < bz0 { nz } else { bz0 };
            bz1 = if nz > bz1 { nz } else { bz1 };
        }
        let r = res as i64;
        let ix0 = (((bx0 - grid.xmin) / grid.dx) as i64 - 1).max(0);
        let ix1 = (((bx1 - grid.xmin) / grid.dx) as i64 + 1).min(r - 1);
        let iy0 = (((by0 - grid.ymin) / grid.dy) as i64 - 1).max(0);
        let iy1 = (((by1 - grid.ymin) / grid.dy) as i64 + 1).min(r - 1);
        let iz0 = (((bz0 - grid.zmin) / grid.dz) as i64 - 1).max(0);
        let iz1 = (((bz1 - grid.zmin) / grid.dz) as i64 + 1).min(r - 1);
        for ix in ix0..=ix1 {
            for iy in iy0..=iy1 {
                for iz in iz0..=iz1 {
                    let cell_id = (ix + iy * r + iz * r * r) as usize;
                    for ci in grid.cell_offsets[cell_id]..grid.cell_offsets[cell_id + 1] {
                        let j = grid.cell_contents[ci as usize] as usize;
                        let (dx_, dy_, dz_) = (coords[[0, j]] - v1x, coords[[1, j]] - v1y, coords[[2, j]] - v1z);
                        let u = b00 * dx_ + b01 * dy_ + b02 * dz_;
                        let v = b10 * dx_ + b11 * dy_ + b12 * dz_;
                        let w = b20 * dx_ + b21 * dy_ + b22 * dz_;
                        if u >= -EPS && v >= -EPS && w >= -EPS && (u + v + w <= 1.0 + EPS) {
                            assigned[j].fetch_max(itet as i64, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    });
    Array1::from_iter(assigned.into_iter().map(|a| a.into_inner()))
}

/// Builds the spatial grid and returns the containing tet index of every query point (-1 if none).
pub fn index_interp(
    coords: &ArrayView2<f64>,
    tets: &ArrayView2<i64>,
    nodes: &ArrayView2<f64>,
    tetids: &ArrayView1<i64>,
    res: usize,
) -> Array1<i64> {
    if coords.shape()[1] == 0 {
        return Array1::zeros(0);
    }
    let grid = build_grid(coords, res);
    index_interp_inner(coords, tets, nodes, tetids, &grid, res)
}

#[pyfunction]
#[pyo3(name = "index_interp", signature = (coords, tets, nodes, tetids, res = GRID_RES))]
fn py_index_interp<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    tets: PyReadonlyArray2<i64>,
    nodes: PyReadonlyArray2<f64>,
    tetids: PyReadonlyArray1<i64>,
    res: usize,
) -> Bound<'py, PyArray1<i64>> {
    let (co, te, no, ti) = (coords.as_array(), tets.as_array(), nodes.as_array(), tetids.as_array());
    py.allow_threads(|| index_interp(&co, &te, &no, &ti, res)).into_pyarray_bound(py)
}

/// Registers the functions of this file on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_index_interp, m)?)?;
    Ok(())
}
