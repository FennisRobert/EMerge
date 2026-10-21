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

# Last Cleanup: 2026-10-17
"""EMR (EMerge Rust): compiled kernels of EMerge, built from the emr/ crate.

Importing the extension registers its submodules (emr.mth, emr.curlcurl, ...) in sys.modules.
"""
from ._emr import *  # noqa: F401,F403
from ._emr import __version__  # noqa: F401
