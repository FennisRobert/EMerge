"""Auxiliary-space Maxwell (AMS) export operators.

SELF-CONTAINED AND EASILY REMOVABLE. Everything added to EMerge for the AMS
preconditioner work lives in this folder. To remove it completely:

  1. delete this folder
  2. in ../assembler.py, drop the `from .ams_export import ...` line and the
     `grad=`/`pi=`/`abc2_curl=`/`abc2_div=` keyword arguments in the
     `MLPreconData(...)` call
  3. in _emerge/mldata.py, drop the `grad`, `pi`, `abc2_curl` and `abc2_div`
     parameters and the `G_*`/`PI_*`/`ABC2C_*`/`ABC2D_*` payload entries

Nothing else in EMerge imports from here, and nothing here is used unless
`assembler.mldata_filename` is set.

What these operators are, and why they cannot be reconstructed downstream:
both depend on the finite element BASIS DEFINITIONS and on tet connectivity,
neither of which appears in the exported `.npz`. They are the standard inputs
every auxiliary-space Maxwell preconditioner requires from the assembler --
hypre takes them via `HYPRE_AMSSetDiscreteGradient` and
`HYPRE_AMSSetInterpolations`; MFEM and Palace assemble them from the FE space.

`assemble_abc2_terms` is of a different kind: not an AMS operator but the
second order ABC split out of the folded mass matrix, because its surface curl
term is the one part of Im(K) with the wrong sign for a shifted
preconditioner. See boundary_terms.py.
"""
from .discrete_gradient import assemble_discrete_gradient
from .interpolation import assemble_nedelec_interpolation
from .boundary_terms import assemble_abc2_terms

__all__ = ["assemble_discrete_gradient", "assemble_nedelec_interpolation",
           "assemble_abc2_terms"]
