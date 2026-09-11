"""Auxiliary-space Maxwell (AMS) export operators.

SELF-CONTAINED AND EASILY REMOVABLE. Everything added to EMerge for the AMS
preconditioner work lives in this folder. To remove it completely:

  1. delete this folder
  2. in ../assembler.py, drop the `from .ams_export import ...` line and the
     `grad=`/`pi=` keyword arguments in the `MLPreconData(...)` call
  3. in _emerge/mldata.py, drop the `grad` and `pi` parameters and the
     `G_*`/`PI_*` payload entries

Nothing else in EMerge imports from here, and nothing here is used unless
`assembler.mldata_filename` is set.

What these operators are, and why they cannot be reconstructed downstream:
both depend on the finite element BASIS DEFINITIONS and on tet connectivity,
neither of which appears in the exported `.npz`. They are the standard inputs
every auxiliary-space Maxwell preconditioner requires from the assembler --
hypre takes them via `HYPRE_AMSSetDiscreteGradient` and
`HYPRE_AMSSetInterpolations`; MFEM and Palace assemble them from the FE space.
"""
from .discrete_gradient import assemble_discrete_gradient
from .interpolation import assemble_nedelec_interpolation

__all__ = ["assemble_discrete_gradient", "assemble_nedelec_interpolation"]
