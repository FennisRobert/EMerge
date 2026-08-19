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

"""Finite-array tiling setup for domain-decomposition array simulation.

ArrayTiling turns a PeriodicCell's lattice definition (its `vectors`, and
the periodicity vectors yielded by `cell_data()`) into a finite set of cell
placements with an on/off inclusion mask, stable domain ids, physical (x, y)
offsets for mesh copying, and a neighbor-connectivity list -- everything the
CNOSDD assembler needs to build subdomains and interfaces for a finite array
without ever needing to detect adjacency from mesh geometry.

Typical usage:

    cell = RectCell(width=12e-3, height=12e-3)
    tiling = ArrayTiling(cell)
    tiling.add_rectangle(8, 8)          # 8x8 square array
    tiling.exclude(0, 0)                # remove one corner element
    tiling.exclude(7, 7)

    for idx, (x, y) in tiling.domain_positions().items():
        ...  # copy the cell mesh, offset by (x, y)

    for a, b, vec in tiling.neighbor_links():
        ...  # domain a and domain b share an interface along `vec`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ...periodic import PeriodicCell


CellKey = tuple[int, int]


############################################################
#                        DATA CONTAINER                    #
############################################################

@dataclass
class ArrayLayout:
    """A frozen snapshot of an ArrayTiling's active-cell layout.

    Produced by ArrayTiling.finalize() / freeze(). This is the object that
    should actually be handed to the assembler: it fixes domain ids once
    (subsequent include/exclude calls on the source ArrayTiling won't affect
    an already-produced ArrayLayout), so assembly can rely on stable
    ordering.
    """

    cell: PeriodicCell
    basis: tuple[np.ndarray, np.ndarray]
    ordered_keys: list[CellKey]
    positions: dict[int, tuple[float, float]]
    links: list[tuple[int, int, np.ndarray]]

    @property
    def n_domains(self) -> int:
        return len(self.ordered_keys)

    def key_of(self, domain_id: int) -> CellKey:
        return self.ordered_keys[domain_id]

    def domain_id(self, i: int, j: int) -> int | None:
        try:
            return self.ordered_keys.index((i, j))
        except ValueError:
            return None

    def domlink(self) -> dict[int, list[int]]:
        """Adjacency in the same shape the CNOSDD assembler already uses
        internally (domain id -> sorted list of linked domain ids)."""
        out: dict[int, list[int]] = {d: [] for d in range(self.n_domains)}
        for a, b, _vec in self.links:
            out[a].append(b)
            out[b].append(a)
        return {d: sorted(set(ls)) for d, ls in out.items()}

    def summary(self) -> str:
        lines = [f"ArrayLayout: {self.n_domains} active cells, {len(self.links)} interfaces"]
        for d, (i, j) in enumerate(self.ordered_keys):
            x, y = self.positions[d]
            lines.append(f"  domain {d}: cell ({i},{j}) at ({x * 1000:.2f}, {y * 1000:.2f}) mm")
        return "\n".join(lines)

    def plot(self, ax=None, show_ids: bool = True, show_links: bool = True):
        """Quick matplotlib visualization of the active layout, mainly to
        sanity-check castellated / triangular-derived footprints before
        running anything. Requires matplotlib; raises ImportError with a
        clear message if it isn't installed.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import RegularPolygon
        except ImportError as e:
            raise ImportError(
                "ArrayLayout.plot() requires matplotlib (pip install matplotlib)."
            ) from e

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots()

        for d, (i, j) in enumerate(self.ordered_keys):
            x, y = self.positions[d]
            ax.scatter([x], [y], s=30, color="tab:blue")
            if show_ids:
                ax.annotate(f"{d}\n({i},{j})", (x, y), fontsize=7, ha="center", va="center")

        if show_links:
            for a, b, _vec in self.links:
                xa, ya = self.positions[a]
                xb, yb = self.positions[b]
                ax.plot([xa, xb], [ya, yb], color="tab:gray", linewidth=0.75, zorder=0)

        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if own_fig:
            plt.show()
        return ax


############################################################
#                        ARRAY TILING                      #
############################################################

class ArrayTiling:
    """Defines a finite array footprint on top of a PeriodicCell's lattice.

    Cells are addressed by integer lattice indices (i, j) along the cell's
    first two lattice vectors. Inclusion is toggled per cell (include /
    exclude), which is what lets irregular or castellated footprints (e.g. a
    hex lattice trimmed to a rectangular outline) be built up from simple
    region-filling helpers plus manual corrections.
    """

    def __init__(self, cell: PeriodicCell):
        self.cell = cell
        self._basis = self._extract_basis(cell)
        self._directions = self._extract_directions(cell, self._basis)
        self._active: dict[CellKey, bool] = {}
        self._order: list[CellKey] = []

    # ------------------------------------------------------------------
    # Lattice geometry, derived once from the PeriodicCell
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_basis(cell: PeriodicCell) -> tuple[np.ndarray, np.ndarray]:
        if len(cell.vectors) < 2:
            raise ValueError(
                "PeriodicCell must define at least 2 lattice vectors to tile a plane."
            )
        v1 = np.asarray(cell.vectors[0].np, dtype=float)[:2]
        v2 = np.asarray(cell.vectors[1].np, dtype=float)[:2]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-12:
            raise ValueError(
                "The first two lattice vectors are collinear and cannot form a 2D basis."
            )
        return v1, v2

    @staticmethod
    def _extract_directions(
        cell: PeriodicCell, basis: tuple[np.ndarray, np.ndarray]
    ) -> list[tuple[int, int, np.ndarray]]:
        """Expresses every periodicity vector yielded by cell_data() as an
        integer (di, dj) step in the chosen basis, alongside the full 3D
        vector (kept for interface-normal bookkeeping downstream).

        A direction and its negation both resolve to valid neighbor steps;
        both are kept so neighbor_links() can walk outward from any cell in
        either sense along that lattice direction.
        """
        v1, v2 = basis
        M = np.array([v1, v2]).T
        Minv = np.linalg.inv(M)

        dirs: list[tuple[int, int, np.ndarray]] = []
        seen: set[tuple[int, int]] = set()
        for _f1, _f2, vec in cell.cell_data():
            vec3 = np.asarray(vec, dtype=float)
            dij = Minv @ vec3[:2]
            di, dj = int(round(dij[0])), int(round(dij[1]))
            if (di, dj) == (0, 0):
                continue
            if (di, dj) not in seen:
                seen.add((di, dj))
                dirs.append((di, dj, vec3))
            if (-di, -dj) not in seen:
                seen.add((-di, -dj))
                dirs.append((-di, -dj, -vec3))
        return dirs

    def position(self, i: int, j: int) -> tuple[float, float]:
        """Physical (x, y) offset of lattice cell (i, j)."""
        v1, v2 = self._basis
        p = i * v1 + j * v2
        return float(p[0]), float(p[1])

    # ------------------------------------------------------------------
    # Inclusion mask
    # ------------------------------------------------------------------

    def include(self, i: int, j: int) -> "ArrayTiling":
        key = (i, j)
        if key not in self._active:
            self._order.append(key)
        self._active[key] = True
        return self

    def exclude(self, i: int, j: int) -> "ArrayTiling":
        self._active[(i, j)] = False
        return self

    def is_active(self, i: int, j: int) -> bool:
        return self._active.get((i, j), False)

    def toggle(self, i: int, j: int) -> "ArrayTiling":
        if self.is_active(i, j):
            self.exclude(i, j)
        else:
            self.include(i, j)
        return self

    # ------------------------------------------------------------------
    # Region-filling helpers
    # ------------------------------------------------------------------

    def add_rectangle(self, nx: int, ny: int, i0: int = 0, j0: int = 0) -> "ArrayTiling":
        """Includes every cell in an nx-by-ny block of lattice indices,
        starting at (i0, j0). For a RectCell this is a literal rectangular
        array; for a HexCell it's a parallelogram-shaped block in (i, j)
        index space, not a physically rectangular footprint -- use
        add_predicate with a physical bounding box for a true rectangular
        outline on a hex lattice.
        """
        for i in range(i0, i0 + nx):
            for j in range(j0, j0 + ny):
                self.include(i, j)
        return self

    def add_indices(self, indices: Iterable[CellKey]) -> "ArrayTiling":
        for i, j in indices:
            self.include(i, j)
        return self

    def add_mask(self, mask: np.ndarray, i0: int = 0, j0: int = 0) -> "ArrayTiling":
        """Includes cells where a 2D boolean array is True. mask[a, b]
        corresponds to lattice cell (i0 + a, j0 + b).
        """
        for a in range(mask.shape[0]):
            for b in range(mask.shape[1]):
                if mask[a, b]:
                    self.include(i0 + a, j0 + b)
        return self

    def add_predicate(
        self,
        predicate: Callable[[float, float, int, int], bool],
        i_range: range,
        j_range: range,
    ) -> "ArrayTiling":
        """Includes every cell in the given index ranges for which
        predicate(x, y, i, j) is True, where (x, y) is the cell's physical
        position. This is the general-purpose tool for castellated /
        trimmed footprints -- e.g. a hex lattice cut to a rectangular or
        circular outline:

            W, H = 0.10, 0.08
            tiling.add_predicate(
                lambda x, y, i, j: abs(x) <= W / 2 and abs(y) <= H / 2,
                range(-20, 20), range(-20, 20),
            )
        """
        for i in i_range:
            for j in j_range:
                x, y = self.position(i, j)
                if predicate(x, y, i, j):
                    self.include(i, j)
        return self

    def remove_predicate(
        self,
        predicate: Callable[[float, float, int, int], bool],
    ) -> "ArrayTiling":
        """Excludes every currently-active cell for which
        predicate(x, y, i, j) is True. Useful for cutting notches/corners
        out of an already-filled region.
        """
        for (i, j) in list(self.cells):
            x, y = self.position(i, j)
            if predicate(x, y, i, j):
                self.exclude(i, j)
        return self

    # ------------------------------------------------------------------
    # Queries / finalization
    # ------------------------------------------------------------------

    @property
    def cells(self) -> list[CellKey]:
        """Active cell (i, j) keys, in stable insertion order. This order
        defines domain ids: cells[d] is the lattice key for domain d.
        """
        return [k for k in self._order if self._active.get(k, False)]

    def domain_positions(self) -> dict[int, tuple[float, float]]:
        """domain_id -> (x, y) physical offset, for copying the unit-cell
        mesh into each active cell's position."""
        return {d: self.position(i, j) for d, (i, j) in enumerate(self.cells)}

    def neighbor_links(self) -> list[tuple[int, int, np.ndarray]]:
        """(domain_a, domain_b, vector_ab) for every pair of active cells
        adjacent along one of the cell's lattice directions. vector_ab is
        the full 3D periodicity vector from cell a to cell b, matching what
        cell_data() yields -- reusable directly as the interface's
        translation vector without re-deriving it from mesh geometry.

        Each adjacent pair is returned exactly once (a < b).
        """
        cells = self.cells
        idmap = {k: d for d, k in enumerate(cells)}

        links: list[tuple[int, int, np.ndarray]] = []
        seen_pairs: set[tuple[int, int]] = set()

        for (i, j) in cells:
            for di, dj, vec in self._directions:
                neighbor_key = (i + di, j + dj)
                if neighbor_key not in idmap:
                    continue
                a, b = idmap[(i, j)], idmap[neighbor_key]
                pair = (min(a, b), max(a, b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                links.append((pair[0], pair[1], vec if a < b else -vec))

        return links

    def finalize(self) -> ArrayLayout:
        """Freezes the current inclusion state into an immutable ArrayLayout.
        Call this once the footprint is finished and hand the result to the
        assembler -- further include()/exclude() calls on this ArrayTiling
        will not retroactively change an already-finalized ArrayLayout.
        """
        cells = self.cells
        positions = self.domain_positions()
        links = self.neighbor_links()
        return ArrayLayout(
            cell=self.cell,
            basis=self._basis,
            ordered_keys=cells,
            positions=positions,
            links=links,
        )

    # Alias -- some callers may find "freeze" more natural than "finalize".
    freeze = finalize