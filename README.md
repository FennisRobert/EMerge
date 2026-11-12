<div align="center">

![LinkedInCover](https://github.com/user-attachments/assets/09142f09-8718-4943-a60a-7593ca738759)

![GitHub License](https://img.shields.io/github/license/FennisRobert/EMerge)
![GitHub Release](https://img.shields.io/github/v/release/FennisRobert/EMerge)
![PyPI - Downloads](https://img.shields.io/pypi/dw/emerge)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17025518-blue)](https://doi.org/10.5281/zenodo.17025518)

</div>

## Introduction

EMerge is a python based FEM EM library for the time harmonic helmholtz formulation. It is thus best suited for
Electromagnetic wave phenomenon. You can use it to simulate:

- RF Filters
- Signal propagation through PCBs
- Antennas
- Optycal systems
- Arrays and periodic structures
- Much more!

It is designed to be as easy to use and compatible as possible. It runs on all operating systems and on all hardware.

EMerge is designed to have your entire simulation start and finish in the same Python script (or more if you want).
You require no awkward configuration files, JSON's, external software to do modelling etc. It allows you to do everything
in Python:

- Geometry Creation/description |
- Material assignment |
- Meshing + mesh settings and adaptive mesh refinement |
- Boundary condition setup |
- Solving |
- Post processing and visualization |

> If you have questions, suggestions, bug reports or just want to hang out, feel free to join the discord!
>
> **[Discord Invitation](https://discord.gg/VMftDCZcNz)**

---

## How to install

You can now install the basic version of emerge from PyPi!

```bash
pip install emerge
```

---

## Direct solvers

EMerge solves all systems with direct solvers only. Some are faster than others. Depending on the operating
system and hardware you have, some might work better and/or are easier to install than others. EMerge picks a
solver automatically; `em.solvers.available()` lists the solvers installed on your system and
`model.set_solver(em.solvers.Sparta)` selects one yourself.

### SPARTA (recommended)

For the best performance, install SPARTA, the open source sparse direct solver made for EMerge. It is the fastest
CPU solver for EMerge, and it can also run on the GPU: Apple silicon (Metal) and AMD (Vulkan).

Install the wheel of the [latest SPARTA release](https://github.com/FennisRobert/SPARTA/releases/latest), with
the version of that release in the link. For example, for v0.3.2:

```bash
pip install sparta-solver --find-links https://github.com/FennisRobert/SPARTA/releases/expanded_assets/v0.3.2
```

Once installed, EMerge uses SPARTA on the CPU. Where PARDISO is available (x86) EMerge picks PARDISO for single
sweeps; select SPARTA with `model.set_solver(em.solvers.Sparta)`. To run SPARTA on the GPU, set:

```python
model.set_device("GPU")     # or "HYBRID" for a combined GPU + CPU factorization
```

### Threads

Set the number of CPU threads with:

```python
em.set_threads(4)
```

This applies to EMerge's compiled kernels (assembly, interpolation, far fields) and is the default thread count of
the solvers that support one (SPARTA, PARDISO). A run method can override it for the solver, for example
`model.mw.run_sweep(nthreads=4)`.

Settings that native libraries read when they load (the Accelerate, RSLab and MUMPS threads) must be set before
importing EMerge. The `emerge_config` module, installed with EMerge, does that:

```python
from emerge_config import config
config.set_acc_threads(4)   # Apple Accelerate
import emerge as em
```

### Other solvers

<details open>
<summary><b>MacOS (ARM)</b></summary>

<br>

Without SPARTA, EMerge uses the Apple Accelerate sparse solver. From 3.0 onwards it is installed automatically
with EMerge on macOS (the precompiled `emerge-aasds` package on PyPI), so no extra installation is needed.

The Accelerate solver does not take a thread count from `em.set_threads()`. EMerge limits it to a single thread
for frequency distributed sweeps. To allow more threads, use the `emerge_config` module before importing EMerge:

```python
from emerge_config import config
config.set_acc_threads(4)   # before importing emerge
import emerge as em
```

</details>

<details open>
<summary><b>Windows (x86)</b></summary>

<br>

Windows has easy access to the lightning fast PARDISO solver out of the box, no installation needed.

</details>

<details open>
<summary><b>GPU based CuDSS solver</b></summary>

<br>

If you have a new NVidia card you can try the first test implementation of the cuDSS solver. The dependencies can be installed through:

```bash
pip install emerge[cudss]
```

To use the GPU, set:

```python
model.set_device("GPU")
```

</details>

<details>
<summary><b>Legacy solvers</b></summary>

<br>

UMFPACK is still supported but is slow compared to the solvers above, so it is only used when none of them is
available. If you want it anyway, it installs through conda forge:

```bash
conda install conda-forge::scikit-umfpack
```

</details>

---

## Install GMSH Version 5.0.0 MacOS

```bash
pip install https://gmsh.info/python-packages-dev/gmsh/gmsh-5.0.0.dev1-py2.py3-none-macosx_12_0_arm64.whl --force-reinstall
```

---

## Dependencies


> EMerge ships compiled Rust kernels (the `emr` module, source in `emr/`). PyPI wheels include them; installing from source
> (`pip install /path/to/EMerge` or `pip install git+https://github.com/FennisRobert/EMerge.git`) compiles them automatically
> and requires a Rust toolchain (https://rustup.rs).

`pip install emerge` installs the required libraries automatically; you don't need to install them yourself.
The optional extras are only installed when you ask for them.

<table>
<tr>
<th align="left">Installed automatically</th>
<th align="left">Optional extras</th>
</tr>
<tr>
<td valign="top">

- numpy
- scipy
- gmsh
- loguru
- matplotlib (for the matplotlib base display)
- pyvista (for the PyVista base display)
- joblib
- msgpack, msgpack-numpy
- psutil
- emsutil
- mkl (x86 devices only)
- emerge-aasds (macOS only, the Apple Accelerate solver)

</td>
<td valign="top">

Installed with `pip install emerge[<extra>]`:

- `cudss`: the NVIDIA cuDSS GPU solver
- `dxf`: ezdxf
- `cad`: emcad and pygerber (Gerber and ODB++)
- `spyder`: trame-pyvista, nest_asyncio2, ipywidgets
- `umfpack`: scikit-umfpack (legacy)

Installed separately:

- SPARTA (recommended, see [SPARTA](#sparta-recommended))

</td>
</tr>
</table>

---

## PCB and CAD readers

You can read Gerber and ODB++ files using the optional `emcad` dependency which I also maintain. To get all the dependencies its easiest to do:

```bash
pip install emerge[cad]
```

---

## Resources / Manual

You can find the latest versions of the manual on: **https://www.emerge-software.com/resources/**
