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
Opt-in check for newer EMerge releases on PyPI.

Disabled by default. Enable with:
    python -m emerge updates on
"""

from __future__ import annotations
import datetime
import json
import re
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Any
from loguru import logger
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from .global_cache import EMergeGlobalCache

PYPI_URL = "https://pypi.org/pypi/emerge/json"
NOTICES_URL = "https://raw.githubusercontent.com/FennisRobert/EMerge/main/notices.json"
TIMEOUT_S = 2.0
NOTICE_SEVERITIES = ("info", "warning", "critical")


def _installed_version() -> Version:
    from emerge import __version__
    return Version(__version__)


class _PipUnavailable(Exception):
    pass


def _latest_via_pip(include_prereleases: bool, timeout: float) -> Version | None:
    """Asks pip (`pip index versions emerge`) for the newest version. This respects the
    user's pip configuration (index URL, proxy, certificates).

    Raises _PipUnavailable if pip is not installed in this environment (e.g. uv venvs)."""
    cmd = [
        sys.executable, "-m", "pip", "index", "versions", "emerge",
        "--disable-pip-version-check", "--no-input", "--retries", "0",
        "--timeout", str(max(1, round(timeout))),
    ]
    if include_prereleases:
        cmd.append("--pre")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        logger.debug("Update check via pip timed out.")
        return None
    except OSError as e:
        raise _PipUnavailable(str(e)) from e

    if "No module named pip" in result.stderr:
        raise _PipUnavailable("pip is not installed in this environment")

    # First line of the output looks like: "emerge (3.0.0a22)"
    match = re.search(r"^emerge \(([^)]+)\)", result.stdout, re.MULTILINE)
    if result.returncode != 0 or match is None:
        logger.debug(f"Update check via pip failed: {result.stderr.strip()}")
        return None
    try:
        return Version(match.group(1))
    except InvalidVersion:
        return None


def _latest_via_pypi_json(include_prereleases: bool, timeout: float) -> Version | None:
    """Fallback when pip is not available: queries the PyPI JSON API directly."""
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=timeout) as response:
            data = json.load(response)
    except Exception as e:
        logger.debug(f"Update check failed: {e}")
        return None

    versions = []
    for ver_str, files in data.get("releases", {}).items():
        if not files or all(f.get("yanked", False) for f in files):
            continue
        try:
            ver = Version(ver_str)
        except InvalidVersion:
            continue
        if ver.is_prerelease and not include_prereleases:
            continue
        versions.append(ver)
    return max(versions, default=None)


def fetch_latest_version(include_prereleases: bool, timeout: float = TIMEOUT_S) -> Version | None:
    """Returns the newest EMerge version available, or None if it can't be determined.

    Uses pip when it is installed, otherwise falls back to the PyPI JSON API."""
    try:
        return _latest_via_pip(include_prereleases, timeout)
    except _PipUnavailable as e:
        logger.debug(f"pip unavailable ({e}), using the PyPI JSON API for the update check.")
        return _latest_via_pypi_json(include_prereleases, timeout)


def check_for_update(timeout: float = TIMEOUT_S) -> tuple[Version, Version | None]:
    """Returns (installed, newer) where newer is None if no newer version is available.

    Users on a pre-release are compared against pre-releases too; users on a stable
    release are only told about stable releases.
    """
    installed = _installed_version()
    latest = fetch_latest_version(installed.is_prerelease, timeout)
    if latest is not None and latest > installed:
        return installed, latest
    return installed, None


def _upgrade_hint(version: Version) -> str:
    pre = " --pre" if version.is_prerelease else ""
    return f"pip install --upgrade{pre} emerge"


def _check_is_due(cache: EMergeGlobalCache, today: datetime.date) -> bool:
    last = cache.get("update_check", "last_checked")
    try:
        return last is None or datetime.date.fromisoformat(last) < today
    except (TypeError, ValueError):
        return True


def daily_update_check(
    cache: EMergeGlobalCache,
    notices_source: str | Path = NOTICES_URL,
    include_tests: bool = False,
) -> None:
    """Runs on every launch if the user opted in.

    Once per day it queries PyPI for a newer version and syncs notices.json from GitHub.
    On every launch it shows the stored warning/critical notices.
    """
    if not cache.get("update_check", "enabled", False):
        return

    installed = _installed_version()
    today = datetime.date.today()
    new_info: list[dict[str, Any]] = []

    if _check_is_due(cache, today):
        # Record the attempt even when offline, so a missing connection doesn't stall every launch.
        cache.set("update_check", "last_checked", today.isoformat())

        logger.info("Checking PyPI for EMerge updates (once per day).")
        latest = fetch_latest_version(installed.is_prerelease)
        if latest is None:
            logger.info("Could not reach PyPI, skipping the update check until tomorrow.")
        else:
            if latest > installed:
                logger.warning(
                    f"A newer EMerge version is available: {latest} (installed: {installed}). "
                    f"Upgrade with: {_upgrade_hint(latest)}"
                )
            else:
                logger.info(f"No updates found, EMerge {installed} is up to date.")
            new_info = sync_notices(cache, installed, notices_source, include_tests)

    show_stored_notices(cache, installed, include_tests, new_info)


############################################################
#                          NOTICES                         #
############################################################
# Notices are short messages (known bugs, important changes) published in
# notices.json at the root of the EMerge repository. They are plain text only
# and are never executed. See the "_how_to" section in notices.json.
#
# Flow:
#   sync_notices          (daily) GitHub -> cache["update_check"]["notices"]
#   show_stored_notices   (every launch) shows stored warning/critical notices, plus
#                         info notices on the day they first arrive
#   clear_notices         (python -m emerge updates clear) removes stored notices
#                         and remembers their ids so a sync does not bring them back

CLEAR_HINT = (
    "Tip: hide these notices with: python -m emerge updates clear  "
    "(stop all update checks with: python -m emerge updates off)"
)
_REPEATING_SEVERITIES = ("warning", "critical")
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def fetch_notices(source: str | Path = NOTICES_URL, timeout: float = TIMEOUT_S) -> list[dict[str, Any]] | None:
    """Loads the notices list from a URL or a local file. Returns None if it can't be read."""
    try:
        if isinstance(source, Path) or not str(source).startswith("http"):
            data = json.loads(Path(source).read_text())
        else:
            with urllib.request.urlopen(source, timeout=timeout) as response:
                data = json.load(response)
    except Exception as e:
        logger.debug(f"Could not load notices from {source}: {e}")
        return None
    notices = data.get("notices", []) if isinstance(data, dict) else []
    return notices if isinstance(notices, list) else []


def validate_notice(notice: Any) -> list[str]:
    """Returns a list of problems with a notice entry (empty if it is valid)."""
    if not isinstance(notice, dict):
        return ["entry is not an object"]
    problems = []
    for field in ("id", "message"):
        if not isinstance(notice.get(field), str) or not notice[field].strip():
            problems.append(f"missing or empty '{field}'")
    if notice.get("severity", "warning") not in NOTICE_SEVERITIES:
        problems.append(f"'severity' must be one of {NOTICE_SEVERITIES}")
    try:
        SpecifierSet(notice.get("affects", ""))
    except (InvalidSpecifier, TypeError):
        problems.append(f"'affects' is not a valid version range: {notice.get('affects')!r}")
    for field in ("enabled", "test"):
        if not isinstance(notice.get(field, False), bool):
            problems.append(f"'{field}' must be true or false")
    return problems


def notice_status(notice: Any, installed: Version, include_tests: bool = False) -> str | None:
    """Returns None if the notice would be shown for the installed version, otherwise
    a short reason why it is not shown."""
    if validate_notice(notice):
        return "invalid"
    if notice.get("test", False) and not include_tests:
        reason = notice_status(notice, installed, include_tests=True)
        if reason is None:
            return "test notice: never shown to users, only in the demo script"
        return f"test notice, and {reason}"
    if not notice.get("enabled", True):
        return "disabled (enabled: false)"
    affects = notice.get("affects", "")
    if installed not in SpecifierSet(affects, prereleases=True):
        return f"not for this version (affects: {affects})"
    return None


def applicable_notices(
    notices: list[dict[str, Any]], installed: Version, include_tests: bool = False
) -> list[dict[str, Any]]:
    """Filters notices down to the valid, enabled, non-test ones that affect the installed
    version. Test notices are only included with include_tests=True (used by the demo)."""
    return [n for n in notices if notice_status(n, installed, include_tests) is None]


def describe_notices(notices: list[Any], installed: Version, source: str | Path) -> str:
    """Human readable overview of a notices list, used by `python -m emerge updates notices`."""
    width = 76
    lines = [f"Notices in {source}", f"Checked against EMerge {installed}", ""]
    counts = {"shown": 0, "hidden": 0, "invalid": 0}

    for i, notice in enumerate(notices):
        if not isinstance(notice, dict):
            notice = {"id": f"#{i}"}
        problems = validate_notice(notice)
        status = notice_status(notice, installed)
        test = "  (test)" if notice.get("test", False) else ""
        severity = str(notice.get("severity", "warning")).upper()

        if problems:
            counts["invalid"] += 1
            head = f"[INVALID]  {notice.get('id', f'#{i}')}"
        elif status is None:
            counts["shown"] += 1
            head = f"[SHOWN]    {notice['id']}  ({severity}){test}"
        else:
            counts["hidden"] += 1
            head = f"[hidden]   {notice['id']}  ({severity}){test}"

        lines.append(head)
        indent = " " * 11
        for problem in problems:
            lines.append(f"{indent}- {problem}")
        if status not in (None, "invalid"):
            lines.append(f"{indent}why: {status}")
        if not problems:
            lines.extend(textwrap.wrap(
                notice["message"].strip(), width, initial_indent=indent, subsequent_indent=indent
            ))
            if notice.get("url"):
                lines.append(f"{indent}more info: {notice['url']}")
        lines.append("")

    lines.append(
        f"{len(notices)} notice(s): {counts['shown']} shown, "
        f"{counts['hidden']} hidden, {counts['invalid']} invalid"
    )
    return "\n".join(lines)


def format_notice(notice: dict[str, Any]) -> str:
    severity = notice.get("severity", "warning")
    text = f"  [{severity.upper()}] {notice['message'].strip()}"
    if notice.get("url"):
        text += f" (more info: {notice['url']})"
    return text


def _log_notice(notice: dict[str, Any]) -> None:
    severity = notice.get("severity", "warning")
    if severity == "critical":
        logger.error(format_notice(notice))
    elif severity == "warning":
        logger.warning(format_notice(notice))
    else:
        logger.info(format_notice(notice))


def sync_notices(
    cache: EMergeGlobalCache,
    installed: Version,
    source: str | Path = NOTICES_URL,
    include_tests: bool = False,
) -> list[dict[str, Any]]:
    """Replaces the stored notices with the ones currently on GitHub that apply to this
    version and were not cleared by the user. Returns the 'info' notices that are new,
    since those are only shown once."""
    notices = fetch_notices(source)
    if notices is None:
        return []

    dismissed = set(cache.get("update_check", "dismissed_notices", []) or [])
    previous = {n.get("id") for n in cache.get("update_check", "notices", []) or []}
    current = [
        n for n in applicable_notices(notices, installed, include_tests) if n["id"] not in dismissed
    ]

    cache.set("update_check", "notices", current)
    return [n for n in current if n.get("severity") == "info" and n["id"] not in previous]


def stored_notices(
    cache: EMergeGlobalCache, installed: Version, include_tests: bool = False
) -> list[dict[str, Any]]:
    """Stored notices that still apply to the installed version (e.g. not after an upgrade)."""
    return applicable_notices(cache.get("update_check", "notices", []) or [], installed, include_tests)


def show_stored_notices(
    cache: EMergeGlobalCache,
    installed: Version,
    include_tests: bool = False,
    new_info: list[dict[str, Any]] = (),
) -> None:
    """Shows a header, the stored warning/critical notices plus any new info notices
    (most severe first), and a tip on how to turn them off."""
    notices = [
        n for n in stored_notices(cache, installed, include_tests)
        if n.get("severity", "warning") in _REPEATING_SEVERITIES
    ] + list(new_info)
    if not notices:
        return
    notices.sort(key=lambda n: _SEVERITY_ORDER.get(n.get("severity", "warning"), 1))

    header = f"EMerge has {len(notices)} notice(s) for version {installed}:"
    if any(n.get("severity", "warning") in _REPEATING_SEVERITIES for n in notices):
        logger.warning(header)
    else:
        logger.info(header)
    for notice in notices:
        _log_notice(notice)
    logger.info(CLEAR_HINT)


def clear_notices(cache: EMergeGlobalCache) -> int:
    """Removes all stored notices and remembers their ids. Returns how many were cleared."""
    stored = cache.get("update_check", "notices", []) or []
    dismissed = list(cache.get("update_check", "dismissed_notices", []) or [])
    for notice in stored:
        if notice.get("id") and notice["id"] not in dismissed:
            dismissed.append(notice["id"])
    cache.set("update_check", "dismissed_notices", dismissed, save=False)
    cache.set("update_check", "notices", [])
    return len(stored)
