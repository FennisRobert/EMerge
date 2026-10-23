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
system and hardware you have, some might work better and/or are easier to install than others.

<details open>
<summary><b>MacOS (ARM)</b></summary>

<br>

EMerge uses the Apple Accelerate sparse solver for all cases. From 3.0.0b5 onwards it is installed automatically
with EMerge on macOS (the precompiled `emerge-aasds` package on PyPI), so no extra installation is needed.

By default the Accelrate solver will be limited to a single thread for frequency distributed sweeps. To allow for more threads either set the environment variable manually:

```bash
VECLIB_MAXIMUM_THREADS = 4
```

Or in Python before loading EMerge:

```python
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
```

Of from 2.5.3 onwards also through the config sister module

```python
from emerge_config import config
config.set_acc_threads(4)
```

</details>

<details open>
<summary><b>MacOS (ARM, GPU)</b></summary>

<br>

To utilize the GPU on Apple silicon you can install the open source GPU based sparse direct solver SPARTA.

```bash
pip install git+https://github.com/FennisRobert/SPARTA
```

To activate the GPU, set:

```python
model.set_solver(em.EMSolver.SPARTA)
```

</details>

<details open>
<summary><b>Windows (x86)</b></summary>

<br>

Windows has easy access to the lightning fast PARDISO solver out of the box, no installation needed.
If you want to install the UMFPACK solver for distributed sweeps this distribution should work through conda forge:

```bash
conda install conda-forge::scikit-umfpack
```

Otherwise try the solution in the user manual.

https://www.emerge-software.com/resources

</details>

<details open>
<summary><b>GPU based CuDSS solver</b></summary>

<br>

If you have a new NVidia card you can try the first test implementation of the cuDSS solver. The dependencies can be installed through:

```bash
pip install emerge[cudss]
```

</details>

<details open>
<summary><b>GPU based SPARTA solver</b></summary>

<br>

You can utilize the GPU on AMD devices through Vulkan using the open source SPARTA solver.

```bash
pip install git+https://github.com/FennisRobert/SPARTA
```

To activate the GPU, set:

```python
model.set_solver(em.EMSolver.SPARTA)
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

<table>
<tr>
<th align="left">To run this FEM library you need the following libraries</th>
<th align="left">Optional:</th>
</tr>
<tr>
<td valign="top">

- numpy
- scipy
- gmsh
- loguru
- matplotlib (for the matplotlib base display)
- pyvista (for the PyVista base display)
- mkl (x86 devices only)
- emsutil

</td>
<td valign="top">

- scikit-umfpack
- cudss
- ezdxf
- emcad

</td>
</tr>
</table>

---

## PCB and CAD readers

You can read Gerber and ODB++ files using the optional `emcad` dependency which I also maintain. To get all the dependencies its easiest to do:

```bash
pip intall emerge[cad]
```

---

## Resources / Manual

You can find the latest versions of the manual on: **https://www.emerge-software.com/resources/**
