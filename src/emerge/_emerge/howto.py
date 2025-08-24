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


class _HowtoClass:
    """The help class does nothing but offer docstring for
    function that have names related to potential activities.
    """
    def __init__(self):
        pass

    def start(self):
        """
        To start a simulation simply create a model object through:

        >>> model = emerge.Simulation('MyProjectName')

        Optionally, you can use a context manager for a more explicit handling of exiting the GMSH api and storing data after simulations.

        >>> with emerge.Simulation('MyProjectName') as model:

        """
        pass

    def make_geometry(self):
        """
        To make geometries in EMerge, you can use the various Geometry options in the .geo module.
        For example

        >>> box = emerge.geo.Box(...)
        >>> sphere = emerge.geo.Sphere(...)
        >>> pcb_layouter = emerge.geo.PCBLayout(...)
        >>> plate = emerge.geo.Plate(...)
        >>> cyl = emerge.geo.Cylinder(...)

        After making geometries, you should pass all of them to
        the simulation object
        >>> model.commit_geometry(box, sphere,...)

        You can also directly store geometries into the model class by treating
        it as a list (using __getitem__ and __setitem__)
        >>> model['box'] = emerge.geo.Box(...)

        In this case, all geometries will be automatically included. You shoul still call:
        >>> model.commit_geometry()

        """

        pass

    def select_face(self):
        """
        Faces can be selected in multiple ways. First, many native geometries have face definitions:
        >>> front_face = box.face('front')
        
        The naming convention is left(-X), right(+X), front(-Y), back(+Y), bottom(-Z), top(+Z)
        All outside faces can be selected using
        >>> outside = object.boundary()
        
        If objects are the results from operations, you can access the faces from the
        source objects using the optional tool argument
        >>> cutout = emerge.geo.subtract(sphere,box)
        >>> face = cutout.face('front', tool=box)
        
        Exclusions or specific isolations can be added with optional arguments.
        There is also a select object in your Simulation class that has various convenient selection options
        >>> faces = model.select.face.inlayer()
        >>> faces = model.select.inplane()
        >>> faces = model.select.face.near(x,y,z)
        
        """
        pass

    def boundary_conditions(self):
        """
        Boundary conditions can be added by first creating them and
        then passing them to the physics object. 
        >>> portbc = emerge.bc.ModalPort(...)
        >>> abc = emerge.bc.AbsorbingBoundary(...)
        >>> pec = emerge.bc.PEC(...)
        
        Then you should pass them to the physics object with comma separation
        and/or list comprehension (*bcs)
        >>> model.mw.assign(bc1, bc2, bc3, *bcs)

        """

    def run_sweep(self):
        """
        You can run a frequency domain study by simply calling:\
        
        >>> results = model.mw.run_sweep(...)

        You can distribute your frequency sweep across multiple threads using
        
        >>> results = model.mw.run_sweep(parallel=True, njobs=3)

        The frequency domain study will return an MWSimData object that contains all data.
        """
        pass

    def access_data(self):
        """
        Data from a frequency domain study is contained in the MWSimData object.
        Each simulation iteration is a separate MWDataSet object with all relevant parameters included.

        You can select an individual dataset based on the iteration number using:
        >>> dataset = results.item(3)
        
        This will select the 4th result. You can also select one by a specific value using
        >>> dataset = results(freq=3e9)
        
        The numbers must be exact. You can also approximately select a value using:
        >>> dataset = results.find(freq=3.9)
        
        Datasets contain various parameters such as the simulation frequency and associated k0:
        >>> k0 = dataset.k0
        >>> S11 = dataset.S(1,1)
        
        You can select all results along a specific axis using:
        >>> freq, S11 = results.ax('freq').S(1,1)
        
        If you ran a parameter sweep, this will select whatever value it finds.
        You can also select across multiple dimensions
        >>> freq, param, S11 = results.ax('freq','param').S(1,1)
        
        A dataset also contains all spatial data. You can probe the E-field or H-field using
        the interpolate method. This will compute the E and H fields and store them into the
        dataset after which they can be accessed
        >>> Ex, Ey, Ez = dataset(freq=1e9).interpolate(xs, ys, zs).E
        >>> Ex = dataset(freq=1e9).interpolate(xs, ys, zs).Ex
        
        You can automate the generation of sample coordinates and field values for
        specific plot instructions
        >>> X, Y, Z, Eyreal = dataset(freq=1e9).cutplane(ds=0.002, z=0.005).scalar('Ey','real')

        """
        pass

    def save_and_load(self):
        """
        You can save your project data by setting save_file to True:
        >>> model = emerge.Simulation(..., save_file=True)

        Whenever you want, you can save all data by calling the .save() method

        >>> model.save()

        If you run your simulation in a context manager, the data will be saved
        automatically when the context is done. By default, if your Python script crashes
        or ends, the data will also automatically be saved. Data is saved in a folder. 
        
        You can load the data from a simulation using:

        >>> model = emerge.Simulation(..., load_file=True)

        The data from a simulation can be found in:

        >>> results = model.data
        >>> results = model.mw.freq_data # the same


        """
        pass

    def parameter_sweep(self):
        """To run a parameter sweep, you can simply use the
        parameter_sweep iterator method of your model:
        
        >>>
        param1 = np.linspace(...)
        param2 = np.linspace(...)
        for p1, p2 in model.parameter_sweep(..., param_A=param1, param_B=param2):
            # run simulation
        
        The parameters will be automatically included in all simulation runs.
        The name will be the same as the keyword argument in the parameter sweep.
        You can extract the S-parameters for example using

        >>> freq, params, S11 = result.ax('freq','param_A').S(1,1)
        """
    
    def compute_farfield(self):
        """
        A farfield can be computed in any simulation but it only really
        represents something if the model has an absorbing boundary or PML.
        To compute the farfield on a single arc we will use the farfield_2d
        method of the MWDataSet class

        >>> theta, E, H = data.find(freq=1e9).farfield_2d(refdir, planedir, faces)

        The first argument should be the reference direction (angle = 0 degrees)
        The second argument is the plane normal. The arc is drawn as a semi-circle (or full)
        where the angle 0deg is in the reference direction.
        The faces argument should be a selection of faces to integrate the farfield over.
        
        Currently, EMerge only allows for Robin boundary conditions (ABC) on flat surfaces.

        """
        pass