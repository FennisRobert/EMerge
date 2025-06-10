## TO EVERYBODY, PLEASE READ

Hello everybody. Thanks for showing interest in this repository. Due to me using a free Github account managing a private repository got too frustrating. 

If all is well it should not be possible for others to push anything to this repository besides me. If you have suggestions/changes/questions either use the Github issue system or join the Discord using the following link:

**https://discord.gg/7PF4WcS6uA**

## Compatibility

As far as I know, the library should work on all systems (well) except for ARM processors like Apple Silicon. 
The only viable solver for reasonably sized problems that works currently is PyPARDISO which does not work on ARM architecture.

## Required libraries

To run this FEM library you need the following libraries

 - numpy
 - scipy
 - pypardiso
 - gmsh
 - loguru
 - numba
 - matplotlib (for the matplotlib base display)
 - pyvista (for the PyVista base display)

## NOTICE

First time runs will be very slow because Numba needs to generate local C-compiled functions of the assembler and other mathematical functions. These compilations are chached so this should only take time once.

## How to install

Just download the repository (local clone) and put the "fem" folder plus files in some of your directories.