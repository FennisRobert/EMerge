import argparse
import platform
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod

from ._emerge.projects.generate_project import generate_project

REPO_URL = "https://github.com/FennisRobert/EMerge.git"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output to the terminal."""
    return subprocess.run(cmd, check=True, **kwargs)


def _pip(*args: str) -> None:
    _run([sys.executable, "-m", "pip", *args])


# ---------------------------------------------------------------------------
# Base command
# ---------------------------------------------------------------------------

class Command(ABC):
    """Base class for all CLI subcommands.

    Subclasses are automatically registered when they are defined.
    To add a new command, just subclass Command somewhere in this module
    (or import the subclass) and fill in the three required pieces:
        name        – the subcommand string (e.g. "compile")
        help_text   – one-line description shown in --help
        configure() – add arguments to the subparser
        execute()   – run the command
    """

    name: str = ""
    help_text: str = ""

    _registry: dict[str, type["Command"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            Command._registry[cls.name] = cls

    @abstractmethod
    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments to *parser*."""

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Run the command with parsed *args*."""

    # --- registry helpers ---------------------------------------------------

    @classmethod
    def registered(cls) -> dict[str, type["Command"]]:
        return dict(cls._registry)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

class NewCommand(Command):
    name = "new"
    help_text = "Create a new project"

    def configure(self, parser):
        parser.add_argument("projectname", type=str, help="Name of the project directory")
        parser.add_argument("filename", type=str, help="Base name for files")

    def execute(self, args):
        generate_project(args.projectname, args.filename)


class UpgradeCommand(Command):
    name = "upgrade"
    help_text = "Upgrade EMerge to the latest version from git"

    def configure(self, parser):
        parser.add_argument(
            "--branch", "-b",
            type=str,
            default=None,
            metavar="BRANCH",
            help="Git branch to install from (default: main)",
        )

    def execute(self, args):
        branch = args.branch or "main"
        url = f"git+{REPO_URL}@{branch}"
        print(f"Upgrading EMerge from branch '{branch}'...")
        _pip("install", "--upgrade", url)
        print("Upgrade complete.")

class CompileCommand(Command):
    name = "compile"
    help_text = "Precompiles all numba code for smooth execution"

    def execute(self, args):
        print('Compiling numba libraries. Please wait...')
        from ._emerge.compiled.base import interp
        from ._emerge import mesher
        from ._emerge.mth import csc_cast, csr_cast, integrals, optimized, pairing
        from ._emerge.physics.microwave import adaptive_mesh, sc
        from ._emerge.physics.microwave.assembly import curlcurl, generalized_eigen_hb, periodicbc, robin_abc_order2, robinbc
        from ._emerge.compiled.ccbf import _eval_curl_f_2d, _eval_curl_f_3d, _eval_div_f_2d, _eval_div_f_3d, _eval_f_2d, _eval_f_3d
        print('Compilation complete!')

    def configure(self, parser):
        pass
    
class InstallSolverCommand(Command):
    name = "install-solver"
    help_text = "Install an optional solver backend"

    SOLVERS = ["umfpack", "cudss", "dxf", "gerber", "aasds"]

    def configure(self, parser):
        parser.add_argument(
            "solver",
            choices=self.SOLVERS,
            help=(
                "Solver to install: "
                "umfpack (Linux/macOS via pip, Windows via conda), "
                "cudss / cudss12 (Windows + NVIDIA GPU), "
                "aasds (macOS Apple Silicon), "
                "dxf, gerber (file format extras)"
            ),
        )

    def execute(self, args):
        solver = args.solver
        system = platform.system()
        machine = platform.machine()

        installers = {
            "umfpack": lambda: self._install_umfpack(system),
            "cudss":   lambda: self._install_cudss(system),
            "cudss12": lambda: self._install_cudss(system),
            "aasds":   lambda: self._install_aasds(system, machine),
            "dxf":     lambda: self._install_extras("dxf"),
            "gerber":  lambda: self._install_extras("gerber"),
        }

        installer = installers.get(solver)
        if installer is None:
            print(f"Unknown solver '{solver}'. Available: {', '.join(self.SOLVERS)}")
            sys.exit(1)
        installer()

    # --- solver helpers -----------------------------------------------------

    @staticmethod
    def _install_umfpack(system: str) -> None:
        if system == "Windows":
            conda = shutil.which("conda")
            if conda is None:
                print(
                    "UMFPACK on Windows requires conda, but it was not found on PATH.\n"
                    "Please install Miniconda or Anaconda and try again, or run:\n\n"
                    "  conda install conda-forge::scikit-umfpack\n"
                )
                sys.exit(1)
            print("Installing scikit-umfpack via conda (conda-forge)...")
            _run([conda, "install", "--yes", "conda-forge::scikit-umfpack"])
        else:
            print("Installing scikit-umfpack via pip...")
            _pip("install", "scikit-umfpack")
        print("UMFPACK solver installed.")

    @staticmethod
    def _install_cudss(system: str) -> None:
        if system != "Windows":
            print(
                "Warning: cuDSS is primarily targeted at Windows + NVIDIA GPUs. "
                "Proceeding anyway..."
            )
        packages = [
            "nvidia-cudss-cu12==0.5.0.16",
            "nvmath-python[cu12]==0.5.0",
            "cupy-cuda12x",
        ]
        print("Installing cuDSS solver dependencies...")
        _pip("install", *packages)
        print("cuDSS solver installed.")

    @staticmethod
    def _install_aasds(system: str, machine: str) -> None:
        if system != "Darwin" or machine != "arm64":
            print(
                "The Apple Accelerate solver (emerge-aasds) is only supported on "
                "macOS with Apple Silicon (arm64).\n"
                f"Detected: {system} / {machine}"
            )
            sys.exit(1)
        print("Installing emerge-aasds (Apple Accelerate solver)...")
        _pip("install", "git+https://github.com/FennisRobert/emerge-aasds")
        print("Apple Accelerate solver installed.")

    @staticmethod
    def _install_extras(extra: str) -> None:
        package_map = {"dxf": "ezdxf", "gerber": "pygerber"}
        package = package_map[extra]
        print(f"Installing {extra} dependency ({package})...")
        _pip("install", package)
        print(f"{extra} dependency installed.")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMerge FEM Solver CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Build a subparser for every registered Command
    instances: dict[str, Command] = {}
    for name, cls in Command.registered().items():
        instance = cls()
        sub = subparsers.add_parser(name, help=instance.help_text)
        instance.configure(sub)
        instances[name] = instance

    args = parser.parse_args()

    if args.command in instances:
        instances[args.command].execute(args)
    else:
        parser.print_help()