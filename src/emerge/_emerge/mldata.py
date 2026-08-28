"""Training-data container for the ML preconditioner project.

Usage from wherever a problem gets assembled (e.g. inside an EMerge sweep):

    rec = MLPreconData(E, B, k0, b, solve_ids, mesh_nodes=nodes, mesh_edges=edges)
    ...
    rec.save("run1/snapshot_003.npz", solution=x)

One file == one (mesh, k0) instance. Mesh arrays are optional per-save (see
`include_mesh`) since they're identical across a frequency sweep and there's
no point writing them out 50 times.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, issparse

FORMAT_VERSION = 1


def _as_vector_or_matrix(arr: np.ndarray, n: int, argname: str) -> np.ndarray:
    """Accepts a single excitation `(n,)` or multiple excitations sharing one
    `A`, stacked as `(n, n_modes)` (e.g. one column per port mode)."""
    arr = np.asarray(arr)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{argname} must be 1-D (n,) or 2-D (n, n_modes), got shape {arr.shape}.")
    if arr.shape[0] != n:
        raise ValueError(f"{argname} has leading dimension {arr.shape[0]}, expected {n} (E.shape[0]).")
    return arr.astype(np.complex128)


class MLPreconData:
    """Bundles one (E, B, k0, b, solve_ids[, mesh]) training instance and
    knows how to serialize/deserialize itself losslessly to a single .npz.
    """

    def __init__(
        self,
        filename: str,
        E: csc_matrix,
        B: csc_matrix,
        k0: float,
        solve_ids: np.ndarray,
        mesh_nodes: np.ndarray | None = None,
        mesh_edges: np.ndarray | None = None,
        mesh_tris: np.ndarray | None = None,
        dof_types: np.ndarray | None = None,
        dof_coords: np.ndarray | None = None,
        name: str = "",
    ):
        self.filename = filename
        if not issparse(E) or not issparse(B):
            raise TypeError("E and B must be scipy.sparse matrices.")
        if E.shape != B.shape:
            raise ValueError(f"E.shape {E.shape} != B.shape {B.shape}.")
        if E.shape[0] != E.shape[1]:
            raise ValueError(f"E must be square, got {E.shape}.")

        n = E.shape[0]
        self.E: csc_matrix = E.astype(np.complex128).tocsc()
        self.B: csc_matrix = B.astype(np.complex128).tocsc()
        self.k0: float = float(k0)

        self.b: np.ndarray | None = None

        solve_ids = np.asarray(solve_ids).reshape(-1)
        if not np.issubdtype(solve_ids.dtype, np.integer):
            raise TypeError(f"solve_ids must be an integer array, got dtype {solve_ids.dtype}.")
        if solve_ids.min(initial=0) < 0 or solve_ids.max(initial=0) >= n:
            raise ValueError(f"solve_ids must index into [0, {n}), got range "
                              f"[{solve_ids.min()}, {solve_ids.max()}].")
        self.solve_ids: np.ndarray = solve_ids.astype(np.int64)

        self.mesh_nodes = None if mesh_nodes is None else np.asarray(mesh_nodes)
        self.mesh_edges = None if mesh_edges is None else np.asarray(mesh_edges)
        self.mesh_tris = None if mesh_tris is None else np.asarray(mesh_tris)

        if dof_types is not None:
            dof_types = np.asarray(dof_types).reshape(-1)
            if dof_types.shape[0] != n:
                raise ValueError(f"dof_types has length {dof_types.shape[0]}, expected {n}.")
        self.dof_types = dof_types

        if dof_coords is not None:
            dof_coords = np.asarray(dof_coords)
            if n not in dof_coords.shape:
                raise ValueError(f"dof_coords shape {dof_coords.shape} doesn't contain a "
                                  f"dimension of size {n} (E.shape[0]).")
        self.dof_coords = dof_coords

        self.name = name
        self.solution: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.E.shape[0]

    def __repr__(self) -> str:
        return (f"MLPreconData(name={self.name!r}, n={self.n}, "
                f"n_solve={self.solve_ids.shape[0]}, nnz(E)={self.E.nnz}, "
                f"k0={self.k0:.6g}, has_solution={self.solution is not None})")

    # ------------------------------------------------------------------ #
    #                              SAVE                                 #
    # ------------------------------------------------------------------ #

    def save(
        self,
        suffix: str,
        b: np.ndarray,
        solution: np.ndarray | None = None,
        include_mesh: bool = True,
        compressed: bool = True,
    ) -> None:
        """Write this instance to `filename` (`.npz` appended if missing).

        `solution`, if given, is stored alongside and validated against
        `solve_ids` the same way `b` is against `n`.
        `include_mesh=False` skips mesh/dof-metadata arrays even if they were
        provided at construction time -- use this for snapshot 2..N of a
        sweep sharing one mesh with snapshot 1, to avoid re-writing it.
        """
        self.b = _as_vector_or_matrix(b, self.E.shape[0], "b")
        filename = self.filename + suffix
        if solution is not None:
            solution = _as_vector_or_matrix(solution, self.n, "solution")
            if solution.shape != self.b.shape:
                raise ValueError(f"solution has shape {solution.shape}, but b has shape "
                                  f"{self.b.shape} -- each excitation column needs a matching "
                                  f"solution column.")
            self.solution = solution

        payload: dict[str, np.ndarray] = {
            "format_version": np.array(FORMAT_VERSION),
            "name": np.array(self.name),
            "n": np.array(self.n),
            "k0": np.array(self.k0),
            "b": self.b,
            "solve_ids": self.solve_ids,
            "E_data": self.E.data,
            "E_indices": self.E.indices,
            "E_indptr": self.E.indptr,
            "E_shape": np.array(self.E.shape),
            "B_data": self.B.data,
            "B_indices": self.B.indices,
            "B_indptr": self.B.indptr,
            "B_shape": np.array(self.B.shape),
        }

        if self.solution is not None:
            payload["solution"] = self.solution

        if include_mesh:
            if self.mesh_nodes is not None:
                payload["mesh_nodes"] = self.mesh_nodes
            if self.mesh_edges is not None:
                payload["mesh_edges"] = self.mesh_edges
            if self.mesh_tris is not None:
                payload["mesh_tris"] = self.mesh_tris
            if self.dof_types is not None:
                payload["dof_types"] = self.dof_types
            if self.dof_coords is not None:
                payload["dof_coords"] = self.dof_coords

        if not filename.endswith(".npz"):
            filename = filename + ".npz"

        writer = np.savez_compressed if compressed else np.savez
        writer(filename, **payload)

    # ------------------------------------------------------------------ #
    #                              LOAD                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, filename: str) -> "MLPreconData":
        with np.load(filename, allow_pickle=False) as f:
            version = int(f["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"{filename}: format_version {version} != "
                                  f"expected {FORMAT_VERSION}.")

            E = csc_matrix((f["E_data"], f["E_indices"], f["E_indptr"]),
                            shape=tuple(f["E_shape"]))
            B = csc_matrix((f["B_data"], f["B_indices"], f["B_indptr"]),
                            shape=tuple(f["B_shape"]))

            kwargs = dict(
                mesh_nodes=f["mesh_nodes"] if "mesh_nodes" in f else None,
                mesh_edges=f["mesh_edges"] if "mesh_edges" in f else None,
                mesh_tris=f["mesh_tris"] if "mesh_tris" in f else None,
                dof_types=f["dof_types"] if "dof_types" in f else None,
                dof_coords=f["dof_coords"] if "dof_coords" in f else None,
                name=str(f["name"]) if "name" in f else "",
            )

            rec = cls(E, B, float(f["k0"]), f["b"], f["solve_ids"], **kwargs)
            if "solution" in f:
                rec.solution = f["solution"]
            return rec
