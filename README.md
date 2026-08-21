![GitHub License](https://img.shields.io/github/license/FennisRobert/EMerge)
![GitHub Release](https://img.shields.io/github/v/release/FennisRobert/EMerge)
![PyPI - Downloads](https://img.shields.io/pypi/dw/emerge)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17025518-blue)](https://doi.org/10.5281/zenodo.17025518)

## Introduction
![LinkedInCover](https://github.com/user-attachments/assets/09142f09-8718-4943-a60a-7593ca738759)


Hello everybody. Thanks for showing interest in my EM FEM library!

EMerge is a python based FEM EM library for the time harmonic helmholtz formulation. It is thus best suited for
Electromagnetic wave phenomenon. You can use it to simulate:
 - RF Filters
 - Signal propagation through PCBs
 - Antennas
 - Optycal systems
 - Arrays and periodic structures
 - Much more!

It is designed to be as easy to use and compatible as possible. It runs on all operating systems allthough some solvers
are a bit harder to make work on some systems than others.

EMerge is designed to have your entire simulation start and finish in the same Python script (or more if you want).
You require no awkward configuration files, JSON's, external software to do modelling etc. It allows you to do everything
in Python:
 * Geometry Creation/description
 * Material assignment
 * Meshing + mesh settings and adaptive mesh refinement
 * Boundary condition setup
 * Solving
 * Post processing and visualization

If you have questions, suggestions, bug reports or just want to hang out, feel free to join the discord!

**[Discord Invitation](https://discord.gg/VMftDCZcNz)**

## How to install

You can now install the basic version of emerge from PyPi!

```
pip install emerge
```

## Direct solvers
EMerge solves all systems with direct solvers only. Some are faster than others. Depending on the operating 
system and hardware you have, some might work better and/or are easier to install than others.

### MacOS (ARM)
It is advised to use the Apple Accelerate sparse solver for all cases. On all versions before 2.5.3 install through

```
pip install https://github.com/FennisRobert/emerge-aasds@v0.2.0
```

By default the Accelrate solver will be limited to a single thread for frequency distributed sweeps. To allow for more threads either set the environment variable manually:
```
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

#### Single threaded UMFPACK
There is also an UMFPACK interface but it is more difficult to install and not as fast as the Accelerate solver.
```
brew install cmake swig suite-sparse pkg-config #MacOS
sudo apt-get install libsuitesparse-dev #Linux
```
Then on MacOS do:
```
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export CFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"
export CFLAGS="-Wno-error=int-conversion"
```
Finally:
```
pip install meson-python ninja
pip install --no-build-isolation --no-binary=scikit-umfpack scikit-umfpack
```

**note**: If you have any corrections to these instructions (for any os) please let me know!

#### Multi threaded MUMPS
To install the MUMPS solver on MacOS, download the installer directory from my website and follow the instructions. Please note that Accelerate is also faster than MUMPS in multi-core performance.

https://www.emerge-software.com/resources

### Windows (x86)
Windows has easy access to the lightning fast PARDISO solver out of the box, no installation needed.
If you want to install the UMFPACK solver for distributed sweeps this distribution should work through conda forge:

```bash
conda install conda-forge::scikit-umfpack
```

Otherwise try the solution in the user manual.

https://www.emerge-software.com/resources

### GPU bsed CuDSS solver
If you have a new NVidia card you can try the first test implementation of the cuDSS solver. The dependencies can be installed through:
```
pip install emerge[cudss]
```
*Limitations: * Cupy is currently only supporting 32 bit integer address so large EM problems cannot be correctly solved currently. This is not something I can do anything about.*

## Install GMSH Version 5.0.0 MacOS
```bash
pip install https://gmsh.info/python-packages-dev/gmsh/gmsh-5.0.0.dev1-py2.py3-none-macosx_12_0_arm64.whl --force-reinstall
```
## Required libraries

To run this FEM library you need the following libraries

 - numpy
 - scipy
 - gmsh
 - loguru
 - numba
 - matplotlib (for the matplotlib base display)
 - pyvista (for the PyVista base display)
 - mkl (x86 devices only)
 - emsutil
 - 
Optional:
 - scikit-umfpack
 - cudss
 - ezdxf

## Resources / Manual

You can find the latest versions of the manual on: **https://www.emerge-software.com/resources/**

## Generative AI Policy
In this project, generative AI and LLM based AI tools are used for coding assistance, writing of code for quick tests and sometimes the implementations of functions.

This project does not or rarely uses functions that are completely written by generative AI.

This project does not accept completely vibe coded functionality without human supervision, interpretation and validation.

Because the repository has one active developer with multiple personal computers, the commit history is used to transfer code between computers that is not finished. The commit history may therefore contain code that is AI generated during development phases after which it is usually refactored/rewritten.

Some small trivial functions may be copied from LLM output but these are often so trivial that they are not copyrightable anyway. Examples are hard coded matrix multiplication for Numba compiled functions:
(this function is coincidentally not generated by LLM's but such functions might have been.)
```python
# Example of a function that could have been but in this case is not written by AI models.
@njit(f8[:, :](f8[:, :], f8[:, :]), cache=True, nogil=True)
def matmul(a, b):
    out = np.empty((3, b.shape[1]), dtype=np.float64)
    out[0, :] = a[0, 0] * b[0, :] + a[0, 1] * b[1, :] + a[0, 2] * b[2, :]
    out[1, :] = a[1, 0] * b[0, :] + a[1, 1] * b[1, :] + a[1, 2] * b[2, :]
    out[2, :] = a[2, 0] * b[0, :] + a[2, 1] * b[1, :] + a[2, 2] * b[2, :]
    return out
```
