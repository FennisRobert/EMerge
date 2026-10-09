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

"""
Persistent global cache that survives between EMerge sessions.

The cache is a JSON file stored next to the installed package (like Numba's
__pycache__ binaries). If the install directory is not writable it falls back
to a per-user cache directory.

To add a new flag, add it to EMergeGlobalCache.DEFAULTS. Missing keys in an
existing cache file are filled in from DEFAULTS on load, so old cache files
keep working.
"""

from __future__ import annotations
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any
from loguru import logger

_CACHE_FILENAME = "emerge_global_cache.json"
_INSTALL_CACHE_DIR = Path(__file__).parent / "__emergecache__"


def _user_cache_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "emerge"


def _is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


class EMergeGlobalCache:
    """Manages persistent flags stored in a JSON file across EMerge sessions.

    Example:
    >>> cache = EMergeGlobalCache()
    >>> if not cache.user_warned("accelerate_installed_check"):
    ...     ...
    ...     cache.set_user_warned("accelerate_installed_check")
    """

    DEFAULTS: dict[str, dict[str, Any]] = {
        "_user_warned": {
            "welcome_message": False,
            "accelerate_installed_check": False,
        },
        "update_check": {
            "enabled": False,
            "last_checked": None,  # ISO date (YYYY-MM-DD) of the last PyPI query
            "notices": [],  # notices synced from notices.json that apply to this version
            "dismissed_notices": [],  # ids the user cleared; never synced again
        },
    }

    def __init__(self, path: Path | str | None = None):
        self.path: Path = Path(path) if path is not None else self._default_path()
        self._data: dict[str, dict[str, Any]] = copy.deepcopy(self.DEFAULTS)
        self.load()

    @staticmethod
    def _default_path() -> Path:
        for directory in (_INSTALL_CACHE_DIR, _user_cache_dir()):
            if _is_writable(directory):
                return directory / _CACHE_FILENAME
        return _INSTALL_CACHE_DIR / _CACHE_FILENAME

    ############################################################
    #                         FILE I/O                         #
    ############################################################

    def load(self) -> None:
        """Loads the cache file, merging it on top of the defaults."""
        if not self.path.exists():
            return
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read global cache {self.path}: {e}. Using defaults.")
            return
        for section, values in stored.items():
            if isinstance(values, dict):
                self._data.setdefault(section, {}).update(values)

    def save(self) -> bool:
        """Writes the cache to disk atomically. Returns False if it could not be written."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=4))
            os.replace(tmp, self.path)
            return True
        except OSError as e:
            logger.debug(f"Could not write global cache {self.path}: {e}")
            return False

    ############################################################
    #                      GENERIC ACCESS                      #
    ############################################################

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any, save: bool = True) -> None:
        self._data.setdefault(section, {})[key] = value
        if save:
            self.save()

    def reset(self) -> None:
        """Restores all flags to their defaults and saves."""
        self._data = copy.deepcopy(self.DEFAULTS)
        self.save()

    ############################################################
    #                       USER WARNINGS                      #
    ############################################################

    def user_warned(self, key: str) -> bool:
        return bool(self.get("_user_warned", key, False))

    def set_user_warned(self, key: str, value: bool = True) -> None:
        self.set("_user_warned", key, value)
