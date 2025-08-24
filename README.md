## Introduction

Hello everybody. Thanks for showing interest in this repository.

Feel free to download your version of EMerge and start playing around with it!
If you have suggestions/changes/questions either use the Github issue system or join the Discord using the following link:

**https://discord.gg/7PF4WcS6uA**

## How to install

You can now install the basic version of emerge from PyPi!
```
pip install emerge
```
On MacOS and Linux you can install it with the very fast UMFPACK through scikit-umfpack

```
brew install swig suite-sparse #MacOS
sudo apt-get install libsuitesparse-dev #Linux
pip install emerge[umfpack]
```

### Experimental

If you have a new NVidia card you can try the first test implementation of the cuDSS solver. The dependencies can be installed through:
```
pip install emerge[cudss]
```
The `scikit-umfpack` solver can be installed on Windows as well from binaries with conda. This is a bit more complicated and is described in the installation guide.

## Compatibility

As far as I know, the library should work on all systems. PyPARDISO is not supported on ARM but the current SuperLU and UMFPACK solvers work on ARM as well. Both SuperLU and UMFPACK can run on multi-processing implementations as long as you do entry-point protection:
```
import emerge as em

def main():
    # setup simulation

    model.mw.run_sweep(True, ..., multi_processing=True)

if __name__ == "__main__":
    main()
```
Otherwise, the parallel solver will default to SuperLU which can be slower on larger problems with a very densely connected/compact matrix.

## Required libraries

To run this FEM library you need the following libraries

 - numpy
 - scipy
 - gmsh
 - loguru
 - numba
 - matplotlib (for the matplotlib base display)
 - pyvista (for the PyVista base display)
 - numba-progress
 - mkl (x86 devices only)

Optional:
 - scikit-umfpack
 - cudss

## NOTICE

First time runs will be very slow because Numba needs to generate local C-compiled functions of the assembler and other mathematical functions. These compilations are chached so this should only take time once.

## Third Party License Notice

“This package depends on Intel® Math Kernel Library (MKL), which is licensed separately under the Intel Simplified Software License (October 2022). Installing with pip will fetch the MKL wheel and prompt you to accept that licence.”
