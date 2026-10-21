# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.


# Last Cleanup: 2025-01-01
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

def signed_area(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the signed area of a polygon defined by vertices (x, y)."""
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def points_equal(x1, y1, x2, y2, tol):
    return abs(x1 - x2) <= tol and abs(y1 - y2) <= tol

def check_min_vertices(x, y):
    if len(x) < 3:
        raise ValueError("Polygon has fewer than 3 vertices.")

def check_duplicate_consecutive(x, y, tol=1e-12):
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    if np.any(np.hypot(dx, dy) <= tol):
        raise ValueError("Duplicate consecutive points detected.")

def check_zero_length_edges(x, y, tol=1e-12):
    for i in range(len(x)):
        j = (i + 1) % len(x)
        if abs(x[i] - x[j]) <= tol and abs(y[i] - y[j]) <= tol:
            raise ValueError(f"Zero-length edge at index {i}.")

def check_nonzero_area(x, y, tol=1e-14):
    A = signed_area(x, y)
    if abs(A) <= tol:
        raise ValueError("Polygon has zero signed area.")

def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_intersect(a, b, c, d, tol=1e-12):
    o1 = _orient(*a, *b, *c)
    o2 = _orient(*a, *b, *d)
    o3 = _orient(*c, *d, *a)
    o4 = _orient(*c, *d, *b)

    return (
        o1 * o2 < -tol and
        o3 * o4 < -tol
    )

def check_self_intersections(x, y):
    n = len(x)
    pts = list(zip(x, y))

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]

        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue  # adjacent or closing edge

            c = pts[j]
            d = pts[(j + 1) % n]

            if segments_intersect(a, b, c, d):
                raise ValueError(
                    f"Self-intersection between edges {i}-{i+1} and {j}-{j+1}."
                )

def check_collinear_backtracking(x, y, tol=1e-12):
    n = len(x)
    for i in range(n):
        a = np.array([x[i - 1], y[i - 1]])
        b = np.array([x[i], y[i]])
        c = np.array([x[(i + 1) % n], y[(i + 1) % n]])

        ab = b - a
        bc = c - b

        cross = abs(ab[0] * bc[1] - ab[1] * bc[0])
        dot = np.dot(ab, bc)

        if cross <= tol and dot < 0:
            raise ValueError(
                f"Collinear backtracking at vertex {i}."
            )

def check_nonmanifold_vertices(x, y, tol=1e-12):
    pts = np.column_stack((x, y))
    used = []

    for p in pts:
        count = sum(
            abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol
            for q in used
        )
        if count >= 2:
            raise ValueError("Non-manifold vertex detected.")
        used.append(p)


############################################################
#                           PLOT                          #
############################################################

def plot_polygon_debug(
    x,
    y,
    title="Polygon debug view",
    show_indices=True,
    show_arrows=True,
    equal_aspect=True
):
    """
    Plot a polygon loop for debugging geometry failures.

    Parameters
    ----------
    x, y : array-like
        Polygon vertex coordinates (not necessarily valid).
    title : str
        Plot title.
    show_indices : bool
        Annotate vertex indices.
    show_arrows : bool
        Draw directed edges.
    equal_aspect : bool
        Use equal axis scaling.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n == 0:
        raise ValueError("Empty polygon.")

    # Close loop visually
    xp = np.append(x, x[0])
    yp = np.append(y, y[0])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xp, yp, "-k", lw=1.5)
    ax.plot(x, y, "ro", ms=4)

    if show_arrows:
        for i in range(n):
            dx = x[(i + 1) % n] - x[i]
            dy = y[(i + 1) % n] - y[i]
            ax.arrow(
                x[i],
                y[i],
                dx,
                dy,
                length_includes_head=True,
                head_width=0.02 * max(np.ptp(x), np.ptp(y), 1.0),
                head_length=0.03 * max(np.ptp(x), np.ptp(y), 1.0),
                fc="blue",
                ec="blue",
                alpha=0.6
            )

    if show_indices:
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(
                xi,
                yi,
                f"{i}",
                fontsize=9,
                color="darkred",
                ha="right",
                va="bottom"
            )

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.show()
    


class Loop:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        xs, ys = self.cleanup_exact_copies(xs, ys)
        self.x: np.ndarray = xs
        self.y: np.ndarray = ys

        if self.x[0] == self.x[-1] and self.y[0] == self.y[-1]:
            self.x = self.x[:-1]
            self.y = self.y[:-1]

        self.N = len(self.x)
        self.validate_polygon_loop()
    
    @staticmethod
    def cleanup_exact_copies(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = list(xs)
        y = list(ys)
        xn = [x[0],]
        yn = [y[0],]
        for x0, y0 in zip(x[1:], y[1:]):
            if x0 == xn[-1] and y0==yn[-1]:
                continue
            xn.append(x0)
            yn.append(y0)
        return np.array(xn).astype(np.float64), np.array(yn).astype(np.float64)
    
    def cleanup_close_points(self, min_dist_um: float = 100.0, *, closed: bool = True) -> int:
        """
        Remove consecutive points that are closer than `min_dist_um` (micrometers).

        Keeps the first point, then only keeps a point if its distance to the last
        kept point is >= threshold.

        Returns number of removed points.
        """
        
        if len(self.x) != len(self.y):
            raise ValueError("xs and ys must have the same length")

        # self.xs = [int(x*1000)/1000 for x in self.xs]
        # self.ys = [int(y*1000)/1000 for y in self.ys]
        n0 = len(self.x)
        if n0 <= 1:
            return 0

        min_dist = min_dist_um * 1e-6
        min_dist2 = min_dist * min_dist

        new_xs = [self.x[0]]
        new_ys = [self.y[0]]

        N = len(self.x[1:])-2
        for i, (x, y) in enumerate(zip(self.x[1:], self.y[1:])):
            dx = x - new_xs[-1]
            dy = y - new_ys[-1]
            if dx*dx + dy*dy >= min_dist2 or i>=N:
                new_xs.append(x)
                new_ys.append(y)

        # Optional: also clean up closure point if last is too close to first
        if closed and len(new_xs) > 2:
            dx = new_xs[-1] - new_xs[0]
            dy = new_ys[-1] - new_ys[0]
            if dx*dx + dy*dy < min_dist2:
                new_xs.pop()
                new_ys.pop()

        self.x = new_xs
        self.y = new_ys
        
        x = np.array(self.x)
        y = np.array(self.y)
        sA = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
        
        if sA > 0:
            self.x = self.x[::-1]
            self.y = self.y[::-1]

        return n0 - len(self.x)
    
    def validate_polygon_loop(self, tol=1e-12):
        try:
            check_min_vertices(self.x, self.y)
            check_duplicate_consecutive(self.x, self.y, tol)
            check_zero_length_edges(self.x, self.y, tol)
            check_nonzero_area(self.x, self.y)
            check_collinear_backtracking(self.x, self.y, tol)
            check_self_intersections(self.x, self.y)
            check_nonmanifold_vertices(self.x, self.y, tol)
        except ValueError as e:
            plot_polygon_debug(self.x, self.y, show_arrows=False)
            raise e
            

    def split(self, tol: float = 1e-9):
        """
        Split self-intersecting / toy-contour loops into
        add-loops and remove-loops suitable for GMSH.
        """

        active_x = []
        active_y = []
        extracted_loops = []

        # --- Phase 1: extract loops deterministically ---
        for x, y in zip(self.x, self.y):
            match_idx = None

            # backward scan
            for i in range(len(active_x) - 1, -1, -1):
                if points_equal(x, y, active_x[i], active_y[i], tol):
                    match_idx = i
                    break

            if match_idx is None:
                active_x.append(x)
                active_y.append(y)
            else:
                # form closed loop
                loop_x = active_x[match_idx:] + [x]
                loop_y = active_y[match_idx:] + [y]

                extracted_loops.append(
                    (np.array(loop_x), np.array(loop_y))
                )

                # collapse active path to loop start
                active_x = active_x[:match_idx + 1]
                active_y = active_y[:match_idx + 1]

        # add remaining active path as main loop
        if len(active_x) >= 3:
            extracted_loops.insert(
                0,
                (np.array(active_x), np.array(active_y))
            )

        if not extracted_loops:
            return [], []

        # --- Phase 2: area classification ---
        areas = np.array([
            signed_area(x, y) for x, y in extracted_loops
        ])

        # remove degenerate loops
        valid = np.abs(areas) > tol
        extracted_loops = [
            loop for loop, v in zip(extracted_loops, valid) if v
        ]
        areas = areas[valid]

        if not extracted_loops:
            return [], []

        # normalize orientation using main loop
        main_sign = np.sign(areas[0])
        
        if main_sign == 0:
            main_sign = 1.0

        
        add_loops = []
        remove_loops = []

        for (x, y), A in zip(extracted_loops, areas):
            A = main_sign*A

            if np.sign(A) > 0:
                add_loops.append((x, y))
            else:
                remove_loops.append((x, y))
                
        return add_loops, remove_loops