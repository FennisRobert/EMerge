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
import os
import sys
from typing import Tuple, Optional
from ._global import _GlobalHandler

def _is_microsoft_store_python() -> Tuple[bool, Optional[str]]:
    """
    Detect whether the running interpreter is the Microsoft Store (AppX) build.

    Returns
    -------
    (is_store, evidence)
        is_store : bool
            True if this looks like a Microsoft Store Python.
        evidence : str | None
            Short string explaining what matched, or None if not store.
    """
    if os.name != "nt":
        return False, None

    exe = (sys.executable or "").lower()
    prefix = (sys.prefix or "").lower()
    base_prefix = (getattr(sys, "base_prefix", "") or "").lower()
    exec_prefix = (sys.exec_prefix or "").lower()

    markers = [
        r"\windowsapps\pythonsoftwarefoundation.python.",  # typical Store location
        r"\appdata\local\packages\pythonsoftwarefoundation.python.",  # Store package data
    ]

    for m in markers:
        if m in exe:
            return True, f"sys.executable contains '{m}'"
        if m in prefix:
            return True, f"sys.prefix contains '{m}'"
        if m in base_prefix:
            return True, f"sys.base_prefix contains '{m}'"
        if m in exec_prefix:
            return True, f"sys.exec_prefix contains '{m}'"

    # Fallback: check sys.path entries for Store package layout
    for p in map(lambda s: (s or "").lower(), sys.path):
        if any(m in p for m in markers):
            return True, "a sys.path entry matches Microsoft Store layout"

    return False, None


def run_installation_checks():
    
    # Check 1.
    # Checks if the python installation is from the Microsoft App Store. This will lead to
    # Instabilities with Numba as it can't properly cache in the provided directories.
    
    is_store, why = _is_microsoft_store_python()
    if is_store:
        _GlobalHandler.active().debugcollector.add_report('A Microsoft Store Python installation is detected. This may cause errors when caching Numba code. \n' +
                                'If you run into issues, consider using Conda or UV to install Python on Windows.')