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
One-time checks run when EMerge is imported. Each check uses the
EMergeGlobalCache so that the user is only notified once, ever.
"""

import importlib.util
import platform
import sys
from loguru import logger
from .global_cache import EMergeGlobalCache


def _is_interactive_terminal() -> bool:
    """True if we can prompt the user (not under pytest, piped input, subprocesses, etc.)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _prompt_once(message: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{message}\n\nThis message will not be shown again.\n{bar}")
    try:
        input("Press Enter to continue...")
    except (EOFError, KeyboardInterrupt):
        print()


############################################################
#                          CHECKS                          #
############################################################


def _welcome_message(cache: EMergeGlobalCache) -> None:
    key = "welcome_message"
    if cache.user_warned(key):
        return

    from emerge import __version__

    cache.set_user_warned(key)
    _prompt_once(
        f"Welcome to EMerge {__version__}! Thanks for installing!\n\n"
        "EMerge is an open-source project run by just one individual and is still\n"
        "worked on regularly. This means that over time, bugs may be discovered\n"
        "or new features will be implemented.\n\n"
        "By default EMerge will *NOT* check for updates. If you want to opt in to a\n"
        "once-per-day check for new versions on PyPI, run:\n\n"
        "    python -m emerge updates on\n\n"
        " ℹ The checker will check at most once each day to see if there is a new version on PyPI.\n"
        " ℹ The checker will also inspect the notices.json file in the Github repo to look for\n"
        "      additional information regarding critical bugs or other solver issues."
    )


def _check_accelerate_installed(cache: EMergeGlobalCache) -> None:
    key = "accelerate_installed_check"
    if cache.user_warned(key):
        return
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return
    if importlib.util.find_spec("emerge_aasds") is not None:
        return

    # Mark first so an interrupted prompt still counts as warned.
    cache.set_user_warned(key)
    _prompt_once(
        "Hey! I notice that you run Apple on ARM and you don't have Accelerate installed yet.\n"
        "I recommend you do! Just run:\n\n"
        "    pip install git+https://github.com/FennisRobert/emerge-aasds\n\n"
        "(or: emerge install-solver aasds)\n\n"
        "Accelerate is much faster than SuperLU."
    )


def run_launch_checks() -> None:
    # A failing check must never prevent EMerge from importing.
    try:
        cache = EMergeGlobalCache()

        # Prompts that wait for Enter only make sense in a real terminal.
        if _is_interactive_terminal():
            _welcome_message(cache)
            _check_accelerate_installed(cache)

        # Opt-in only; just logs, so it is safe without a terminal.
        from .update_check import daily_update_check
        daily_update_check(cache)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Launch checks failed: {e!r}")
