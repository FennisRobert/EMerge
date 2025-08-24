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
from __future__ import annotations
import time
from ...mesh3d import Mesh3D
from ...geometry import GeoObject
from ...selection import FaceSelection, DomainSelection, EdgeSelection, Selection, encode_data
from ...physics.microwave.microwave_bc import PortBC, ModalPort
import numpy as np
import pyvista as pv
from typing import Iterable, Literal, Callable, Any
from ..display import BaseDisplay
from .display_settings import PVDisplaySettings
from matplotlib.colors import ListedColormap
from itertools import cycle
### Color scale

# Define the colors we want to use
col1 = np.array([57, 179, 227, 255])/255
col2 = np.array([22, 36, 125, 255])/255
col3 = np.array([33, 33, 33, 255])/255
col4 = np.array([173, 76, 7, 255])/255
col5 = np.array([250, 75, 148, 255])/255

cmap_names = Literal['bgy','bgyw','kbc','blues','bmw','bmy','kgy','gray','dimgray','fire','kb','kg','kr',
                     'bkr','bky','coolwarm','gwv','bjy','bwy','cwr','colorwheel','isolum','rainbow','fire',
                     'cet_fire','gouldian','kbgyw','cwr','CET_CBL1','CET_CBL3','CET_D1A']


def _gen_c_cycle():
    colors = [
        "#0000aa",
        "#aa0000",
        "#009900",
        "#990099",
        "#994400",
        "#005588"
    ]
    return cycle(colors)

C_CYCLE = _gen_c_cycle()

class _RunState:
    
    def __init__(self):
        self.state: bool = False
        self.ctr: int = 0
        
        
    def run(self):
        self.state = True
        self.ctr = 0
        
    def stop(self):
        self.state = False
        self.ctr = 0
        
    def step(self):
        self.ctr += 1
    
ANIM_STATE = _RunState()

def gen_cmap(mesh, N: int = 256):
    # build a linear grid of data‐values (not strictly needed for pure colormap)
    vmin, vmax = mesh['values'].min(), mesh['values'].max()
    mapping = np.linspace(vmin, vmax, N)
    
    # prepare output
    newcolors = np.empty((N, 4))
    
    # normalized positions of control points: start, middle, end
    control_pos = np.array([0.0, 0.25, 0.5, 0.75, 1]) * (vmax - vmin) + vmin
    # stack control colors
    controls = np.vstack([col1, col2, col3, col4, col5])
    
    # interp each RGBA channel independently
    for chan in range(4):
        newcolors[:, chan] = np.interp(mapping, control_pos, controls[:, chan])
    
    return ListedColormap(newcolors)

def setdefault(options: dict, **kwargs) -> dict:
    """Shorthand for overwriting non-existent keyword arguments with defaults

    Args:
        options (dict): The kwargs dict

    Returns:
        dict: the kwargs dict
    """
    for key in kwargs.keys():
        if options.get(key,None) is None:
            options[key] = kwargs[key]
    return options

def _logscale(dx, dy, dz):
    """
    Logarithmically scales vector magnitudes so that the largest remains unchanged
    and others are scaled down logarithmically.
    
    Parameters:
        dx, dy, dz (np.ndarray): Components of vectors.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Scaled dx, dy, dz arrays.
    """
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    dz = np.asarray(dz)

    # Compute original magnitudes
    mags = np.sqrt(dx**2 + dy**2 + dz**2)
    mags_nonzero = np.where(mags == 0, 1e-10, mags)  # avoid log(0)

    # Logarithmic scaling (scaled to max = original max)
    log_mags = np.log10(mags_nonzero)
    log_min = np.min(log_mags)
    log_max = np.max(log_mags)

    if log_max == log_min:
        # All vectors have the same length
        return dx, dy, dz

    # Normalize log magnitudes to [0, 1]
    log_scaled = (log_mags - log_min) / (log_max - log_min)

    # Scale back to original max magnitude
    max_mag = np.max(mags)
    new_mags = log_scaled * max_mag

    # Compute unit vectors
    unit_dx = dx / mags_nonzero
    unit_dy = dy / mags_nonzero
    unit_dz = dz / mags_nonzero

    # Apply scaled magnitudes
    scaled_dx = unit_dx * new_mags
    scaled_dy = unit_dy * new_mags
    scaled_dz = unit_dz * new_mags

    return scaled_dx, scaled_dy, scaled_dz

def _min_distance(xs, ys, zs):
    """
    Compute the minimum Euclidean distance between any two points
    defined by the 1D arrays xs, ys, zs.
    
    Parameters:
        xs (np.ndarray): x-coordinates of the points
        ys (np.ndarray): y-coordinates of the points
        zs (np.ndarray): z-coordinates of the points
    
    Returns:
        float: The minimum Euclidean distance between any two points
    """
    # Stack the coordinates into a (N, 3) array
    points = np.stack((xs, ys, zs), axis=-1)

    # Compute pairwise squared distances using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_squared = np.sum(diff ** 2, axis=-1)

    # Set diagonal to infinity to ignore zero distances to self
    np.fill_diagonal(dists_squared, np.inf)

    # Get the minimum distance
    min_dist = np.sqrt(np.min(dists_squared))
    return min_dist

def _norm(x, y, z):
    return np.sqrt(np.abs(x)**2 + np.abs(y)**2 + np.abs(z)**2)

def _select(obj: GeoObject | Selection) -> Selection:
    if isinstance(obj, GeoObject):
        return obj.select
    return obj

def _merge(lst: Iterable[GeoObject | Selection]) -> Selection:
    selections = [_select(item) for item in lst]
    dim = selections[0].dim
    all_tags = []
    for item in lst:
        all_tags.extend(_select(item).tags)
    
    if dim==1:
        return EdgeSelection(all_tags)
    elif dim==2:
        return FaceSelection(all_tags)
    elif dim==3:
        return DomainSelection(all_tags)
    else:
        return Selection(all_tags)

class _AnimObject:
    """ A private class containing the required information for plot items in a view
    that can be animated.
    """
    def __init__(self, 
                 field: np.ndarray,
                 T: Callable,
                 grid: pv.Grid,
                 actor: pv.Actor,
                 on_update: Callable):
        self.field: np.ndarray = field
        self.T: Callable = T
        self.grid: pv.Grid = grid
        self.actor: pv.Actor = actor
        self.on_update: Callable = on_update

    def update(self, phi: complex):
        self.on_update(self, phi)

class PVDisplay(BaseDisplay):

    def __init__(self, mesh: Mesh3D):
        self._mesh: Mesh3D = mesh
        self.set: PVDisplaySettings = PVDisplaySettings()
        
        # Animation options
        self._facetags: list[int] = []
        self._stop: bool = False
        self._objs: list[_AnimObject] = []
        self._do_animate: bool = False
        self._closed_via_x: bool = False
        self._Nsteps: int  = 0
        self._fps: int = 25
        self._ruler: ScreenRuler = ScreenRuler(self, 0.001)
        self._selector: ScreenSelector = ScreenSelector(self)
        self._stop = False
        self._objs = []

        self._plot = pv.Plotter()

        self._plot.add_key_event("m", self.activate_ruler) # type: ignore
        self._plot.add_key_event("f", self.activate_object) # type: ignore

        self._ctr: int = 0 
        
        self.camera_position = (1, -1, 1)     # +X, +Z, -Y
    
    def _wire_close_events(self):
        self._closed = False

        def mark_closed(*_):
            self._closed = True
            self._stop = True
        
        self._plot.add_key_event('q', lambda: mark_closed())
        
    def _update_camera(self):
        x,y,z = self._plot.camera.position
        d = (x**2+y**2+z**2)**(0.5)
        px, py, pz = self.camera_position
        dp = (px**2+py**2+pz**2)**(0.5)
        px, py, pz = px/dp, py/dp, pz/dp
        self._plot.camera.position = (d*px, d*py, d*pz)
        
    def activate_ruler(self):
        self._plot.disable_picking()
        self._selector.turn_off()
        self._ruler.toggle()

    def activate_object(self):
        self._plot.disable_picking()
        self._ruler.turn_off()
        self._selector.toggle()

    def show(self):
        """ Shows the Pyvista display. """
        self._ruler.min_length = max(1e-3, min(self._mesh.edge_lengths))
        self._update_camera()
        self._add_aux_items()
        if self._do_animate:
            self._wire_close_events()
            self.add_text('Press Q to close!',color='red', position='upper_left')
            self._plot.show(auto_close=False, interactive_update=True, before_close_callback=self._close_callback)
            self._animate()
            
            
        else:
            self._plot.show()
        self._reset()
    
    def set_mesh(self, mesh: Mesh3D):
        """Define the mesh to be used

        Args:
            mesh (Mesh3D): The mesh object
        """
        self._mesh = mesh

    def _reset(self):
        """ Resets key display parameters."""
        self._plot.close()
        self._plot = pv.Plotter()
        self._stop = False
        self._objs = []
        C_CYCLE = _gen_c_cycle()

    def _close_callback(self, arg):
        """The private callback function that stops the animation.
        """
        self._stop = True
        print('CLOSE!')

    def _animate(self) -> None:
        """Private function that starts the animation loop.
        """
        
        self._stop = False

        # guard values
        steps = max(1, int(self._Nsteps))
        fps   = max(1, int(self._fps))
        dt    = 1.0 / fps
        next_tick = time.perf_counter()
        step = 0

        while (not self._stop
                and not self._closed_via_x
                and self._plot.render_window is not None):
            # process window/UI events so close button works
            self._plot.update()

            now = time.perf_counter()
            if now >= next_tick:
                step = (step + 1) % steps
                phi = np.exp(1j * (step / steps) * 2*np.pi)

                # update all animated objects
                for aobj in self._objs:
                    aobj.update(phi)

                # draw one frame
                self._plot.render()

                # schedule next frame; catch up if we fell behind
                next_tick += dt
                if now > next_tick + dt:
                    next_tick = now + dt

            # be kind to the CPU
            time.sleep(0.001)

        # ensure cleanup pathway runs once
        self._close_callback(None)

    def animate(self, Nsteps: int = 35, fps: int = 25) -> PVDisplay:
        """ Turns on the animation mode with the specified number of steps and FPS.

        All subsequent plot calls will automatically be animated. This method can be
        method chained.
        
        Args:
            Nsteps (int, optional): The number of frames in the loop. Defaults to 35.
            fps (int, optional): The number of frames per seocond, Defaults to 25

        Returns:
            PVDisplay: The same PVDisplay object

        Example:
        >>> display.animate().surf(...)
        >>> display.show()
        """
        self._Nsteps = Nsteps
        self._fps = fps
        self._do_animate = True
        return self
    
    ## CUSTOM METHODS
    def mesh_volume(self, volume: DomainSelection) -> pv.UnstructuredGrid:
        tets = self._mesh.get_tetrahedra(volume.tags)
        ntets = tets.shape[0]
        cells = np.zeros((ntets,5), dtype=np.int64)
        cells[:,1:] = self._mesh.tets[:,tets].T
        cells[:,0] = 4
        celltypes = np.full(ntets, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def _volume_edges(self, obj: GeoObject | Selection) -> pv.UnstructuredGrid:
        """Adds the edges of objects

        Args:
            obj (DomainSelection | None, optional): _description_. Defaults to None.

        Returns:
            pv.UnstructuredGrid: The unstrutured grid object
        """
        edge_ids = self._mesh.domain_edges(obj.dimtags)
        nedges = edge_ids.shape[0]
        cells = np.zeros((nedges,3), dtype=np.int64)
        cells[:,1:] = self._mesh.edges[:,edge_ids].T
        cells[:,0] = 2
        celltypes = np.full(nedges, fill_value=pv.CellType.CUBIC_LINE, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh_surface(self, surface: FaceSelection) -> pv.UnstructuredGrid:
        tris = self._mesh.get_triangles(surface.tags)
        ntris = tris.shape[0]
        cells = np.zeros((ntris,4), dtype=np.int64)
        cells[:,1:] = self._mesh.tris[:,tris].T
        cells[:,0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        points[:,2] += self.set.z_boost
        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh(self, obj: GeoObject | Selection | Iterable) -> pv.UnstructuredGrid | None:
        if isinstance(obj, Iterable):
            obj = _merge(obj)
        else:
            obj = _select(obj)
        
        if isinstance(obj, DomainSelection):
            return self.mesh_volume(obj)
        elif isinstance(obj, FaceSelection):
            return self.mesh_surface(obj)
        else:
            return None

    ## OBLIGATORY METHODS
    def add_object(self, obj: GeoObject | Selection, mesh: bool = False, volume_mesh: bool = True, *args, **kwargs):
        
        show_edges = False
        opacity = obj.opacity
        line_width = 0.5
        color = obj.color_rgb
        style='surface'
        
        if mesh is True:
            show_edges = True
            opacity = 0.7
            line_width= 1
            style='wireframe'
            color=next(C_CYCLE)
        
        kwargs = setdefault(kwargs, color=color, opacity=opacity, line_width=line_width, show_edges=show_edges, pickable=True, style=style)
        mesh_obj = self.mesh(obj)
        
        if mesh is True and volume_mesh is True:
            mesh_obj = mesh_obj.extract_all_edges()
        
        actor = self._plot.add_mesh(mesh_obj, *args, **kwargs)
        self._plot.add_mesh(self._volume_edges(_select(obj)), color='#000000', line_width=2, show_edges=True)

    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        """Adds a scatter point cloud

        Args:
            xs (np.ndarray): The X-coordinate
            ys (np.ndarray): The Y-coordinate
            zs (np.ndarray): The Z-coordinate
        """
        cloud = pv.PolyData(np.array([xs,ys,zs]).T)
        self._plot.add_points(cloud)

    def add_portmode(self, port: PortBC, 
                     Npoints: int = 10, 
                     dv=(0,0,0), 
                     XYZ=None,
                     field: Literal['E','H'] = 'E', 
                     k0: float | None = None,
                     mode_number: int = 0) -> None:
        
        if XYZ:
            X,Y,Z = XYZ
        else:
            tris = self._mesh.get_triangles(port.selection.tags)
            X = self._mesh.tri_centers[0,tris]
            Y = self._mesh.tri_centers[1,tris]
            Z = self._mesh.tri_centers[2,tris]
        
        X = X+dv[0]
        Y = Y+dv[1]
        Z = Z+dv[2]
        xf = X.flatten()
        yf = Y.flatten()
        zf = Z.flatten()

        d = _min_distance(xf, yf, zf)

        if port.vintline is not None:
            xs, ys, zs = port.vintline.cpoint
            p_line = pv.Line(
                pointa=(xs[0], ys[0], zs[0]),
                pointb=(xs[-1], ys[-1], zs[-1]),
            )
            self._plot.add_mesh(
                p_line,
                color='red',
                pickable=False,
                line_width=3.0,
            )
        
        if k0 is None:
            if isinstance(port, ModalPort):
                k0 = port.get_mode(0).k0
            else:
                k0 = 1
        port.selected_mode = mode_number
        F = port.port_mode_3d_global(xf,yf,zf,k0, which=field)

        Fx = F[0,:].reshape(X.shape).T
        Fy = F[1,:].reshape(X.shape).T
        Fz = F[2,:].reshape(X.shape).T

        if field=='H':
            F = np.imag(F.T)
            Fnorm = np.sqrt(Fx.imag**2 + Fy.imag**2 + Fz.imag**2).T
        else:
            F = np.real(F.T)
            Fnorm = np.sqrt(Fx.real**2 + Fy.real**2 + Fz.real**2).T

        if XYZ is not None:
            grid = pv.StructuredGrid(X,Y,Z)
            self.add_surf(X,Y,Z,Fnorm, _fieldname = 'portfield')
            self._plot.add_mesh(grid, scalars = Fnorm.T, opacity=0.8, pickable=False)

        Emag = F/np.max(Fnorm.flatten())*d*3
        self._plot.add_arrows(np.array([xf,yf,zf]).T, Emag)

    def add_surf(self, 
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 field: np.ndarray,
                 scale: Literal['lin','log','symlog'] = 'lin',
                 cmap: cmap_names = 'coolwarm',
                 clim: tuple[float, float] | None = None,
                 opacity: float = 1.0,
                 symmetrize: bool = True,
                 _fieldname: str | None = None,
                 **kwargs,):
        """Add a surface plot to the display
        The X,Y,Z coordinates must be a 2D grid of data points. The field must be a real field with the same size.

        Args:
            x (np.ndarray): The X-grid array
            y (np.ndarray): The Y-grid array
            z (np.ndarray): The Z-grid array
            field (np.ndarray): The scalar field to display
            scale (Literal["lin","log","symlog"], optional): The colormap scaling¹. Defaults to 'lin'.
            cmap (cmap_names, optional): The colormap. Defaults to 'coolwarm'.
            clim (tuple[float, float], optional): Specific color limits (min, max). Defaults to None.
            opacity (float, optional): The opacity of the surface. Defaults to 1.0.
            symmetrize (bool, optional): Wether to force a symmetrical color limit (-A,A). Defaults to True.
        
        (¹): lin: f(x)=x, log: f(x)=log₁₀(|x|), symlog: f(x)=sgn(x)·log₁₀(1+|x·ln(10)|)
        """
        
        grid = pv.StructuredGrid(x,y,z)
        field_flat = field.flatten(order='F')


        if scale=='log':
            T = lambda x: np.log10(np.abs(x))
        elif scale=='symlog':
            T = lambda x: np.sign(x) * np.log10(1 + np.abs(x*np.log(10)))
        else:
            T = lambda x: x
        
        static_field = T(np.real(field_flat))
        if _fieldname is None:
            name = 'anim'+str(self._ctr)
        else:
            name = _fieldname
        self._ctr += 1
        grid[name] = static_field

        grid_no_nan = grid.ptc().threshold(scalars=name)

        # Determine color limits
        if clim is None:
            fmin = np.nanmin(static_field)
            fmax = np.nanmax(static_field)
            clim = (fmin, fmax)
        if symmetrize:
            lim = max(abs(clim[0]), abs(clim[1]))
            clim = (-lim, lim)

        kwargs = setdefault(kwargs, cmap=cmap, clim=clim, opacity=opacity, pickable=False, multi_colors=True)
        actor = self._plot.add_mesh(grid_no_nan, scalars=name, **kwargs)


        if self._do_animate:
            def on_update(obj: _AnimObject, phi: complex):
                field_anim = obj.T(np.real(obj.field * phi))
                obj.grid[name] = field_anim
            self._objs.append(_AnimObject(field_flat, T, grid_no_nan, actor, on_update))


    def add_title(self, title: str) -> None:
        """Adds a title

        Args:
            title (str): The title name
        """
        self._plot.add_text(
            title,
            position='upper_edge',
            font_size=18)

    def add_text(self, text: str, 
                 color: str = 'black', 
                 position: Literal['lower_left', 'lower_right', 'upper_left', 'upper_right', 'lower_edge', 'upper_edge', 'right_edge', 'left_edge']='upper_right',
                 abs_position: tuple[float, float, float] | None = None):
        viewport = False
        if abs_position is not None:
            final_position = abs_position
            viewport = True
        else:
            final_position = abs_position
        self._plot.add_text(
            text,
            position=final_position,
            color=color,
            font_size=18,
            viewport=viewport)
        
    def add_quiver(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              scale: float = 1,
              color: tuple[float, float, float] | None = None,
              scalemode: Literal['lin','log'] = 'lin'):
        """Add a quiver plot to the display

        Args:
            x (np.ndarray): The X-coordinates
            y (np.ndarray): The Y-coordinates
            z (np.ndarray): The Z-coordinates
            dx (np.ndarray): The arrow X-magnitude
            dy (np.ndarray): The arrow Y-magnitude
            dz (np.ndarray): The arrow Z-magnitude
            scale (float, optional): The arrow scale. Defaults to 1.
            scalemode (Literal['lin','log'], optional): Wether to scale lin or log. Defaults to 'lin'.
        """
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        dx = dx.flatten().real
        dy = dy.flatten().real
        dz = dz.flatten().real
        dmin = _min_distance(x,y,z)

        dmax = np.max(_norm(dx,dy,dz))
        
        Vec = scale * np.array([dx,dy,dz]).T / dmax * dmin 
        Coo = np.array([x,y,z]).T
        if scalemode=='log':
            dx, dy, dz = _logscale(Vec[:,0], Vec[:,1], Vec[:,2])
            Vec[:,0] = dx
            Vec[:,1] = dy
            Vec[:,2] = dz
        
        kwargs = dict()
        if color is not None:
            kwargs['color'] = color
        pl = self._plot.add_arrows(Coo, Vec, scalars=None, clim=None, cmap=None, **kwargs)

    def add_contour(self,
                     X: np.ndarray,
                     Y: np.ndarray,
                     Z: np.ndarray,
                     V: np.ndarray,
                     Nlevels: int = 5,
                     symmetrize: bool = True,
                     cmap: str = 'viridis'):
        """Adds a 3D volumetric contourplot based on a 3D grid of X,Y,Z and field values


        Args:
            X (np.ndarray): A 3D Grid of X-values
            Y (np.ndarray): A 3D Grid of Y-values
            Z (np.ndarray): A 3D Grid of Z-values
            V (np.ndarray): The scalar quantity to plot ()
            Nlevels (int, optional): The number of contour levels. Defaults to 5.
            symmetrize (bool, optional): Wether to symmetrize the countour levels (-V,V). Defaults to True.
            cmap (str, optional): The color map. Defaults to 'viridis'.
        """
        Vf = V.flatten()
        vmin = np.min(np.real(Vf))
        vmax = np.max(np.real(Vf))
        if symmetrize:
            level = max(np.abs(vmin),np.abs(vmax))
            vmin, vmax = (-level, level)
        grid = pv.StructuredGrid(X,Y,Z)
        field = V.flatten(order='F')
        grid['anim'] = np.real(field)
        levels = list(np.linspace(vmin, vmax, Nlevels))
        contour = grid.contour(isosurfaces=levels)
        actor = self._plot.add_mesh(contour, opacity=0.25, cmap=cmap, pickable=False)

        if self._do_animate:
            def on_update(obj: _AnimObject, phi: complex):
                new_vals = np.real(obj.field * phi)
                obj.grid['anim'] = new_vals
                new_contour = obj.grid.contour(isosurfaces=levels)
                obj.actor.GetMapper().SetInputData(new_contour) # type: ignore
            
            self._objs.append(_AnimObject(field, lambda x: x, grid, actor, on_update)) # type: ignore

    def _add_aux_items(self) -> None:
        saved_camera = {
            "position": self._plot.camera.position,
            "focal_point": self._plot.camera.focal_point,
            "view_up": self._plot.camera.up,
            "view_angle": self._plot.camera.view_angle,
            "clipping_range": self._plot.camera.clipping_range
        }
        #self._plot.add_logo_widget('src/_img/logo.jpeg',position=(0.89,0.89), size=(0.1,0.1))    
        bounds = self._plot.bounds
        max_size = max([abs(dim) for dim in [bounds.x_max, bounds.x_min, bounds.y_max, bounds.y_min, bounds.z_max, bounds.z_min]])
        length = self.set.plane_ratio*max_size*2
        if self.set.draw_xplane:
            plane = pv.Plane(
                center=(0, 0, 0),
                direction=(1, 0, 0),    # normal vector pointing along +X
                i_size=length, # type: ignore
                j_size=length, # type: ignore
                i_resolution=1,
                j_resolution=1
            )
            self._plot.add_mesh(
                plane,
                color='red',
                opacity=self.set.plane_opacity,
                show_edges=False,
                pickable=False,
            )
            self._plot.add_mesh(
                plane,
                edge_opacity=1.0,
                edge_color='red',
                color='red',
                line_width=self.set.plane_edge_width,
                style='wireframe',
                pickable=False,
            )
            
        if self.set.draw_yplane:
            plane = pv.Plane(
                center=(0, 0, 0),
                direction=(0, 1, 0),    # normal vector pointing along +X
                i_size=length, # type: ignore
                j_size=length, # type: ignore
                i_resolution=1,
                j_resolution=1
            )
            self._plot.add_mesh(
                plane,
                color='green',
                opacity=self.set.plane_opacity,
                show_edges=False,
                pickable=False,
            )
            self._plot.add_mesh(
                plane,
                edge_opacity=1.0,
                edge_color='green',
                color='green',
                line_width=self.set.plane_edge_width,
                style='wireframe',
                pickable=False,
            )
        if self.set.draw_zplane:
            plane = pv.Plane(
                center=(0, 0, 0),
                direction=(0, 0, 1),    # normal vector pointing along +X
                i_size=length, # type: ignore
                j_size=length, # type: ignore
                i_resolution=1,
                j_resolution=1
            )
            self._plot.add_mesh(
                plane,
                color='blue',
                opacity=self.set.plane_opacity,
                show_edges=False,
                pickable=False,
            )
            self._plot.add_mesh(
                plane,
                edge_opacity=1.0,
                edge_color='blue',
                color='blue',
                line_width=self.set.plane_edge_width,
                style='wireframe',
                pickable=False,
            )
        # Draw X-axis
        if getattr(self.set, 'draw_xax', False):
            x_line = pv.Line(
                pointa=(-length, 0, 0),
                pointb=(length, 0, 0),
            )
            self._plot.add_mesh(
                x_line,
                color='red',
                line_width=self.set.axis_line_width,
                pickable=False,
            )

        # Draw Y-axis
        if getattr(self.set, 'draw_yax', False):
            y_line = pv.Line(
                pointa=(0, -length, 0),
                pointb=(0, length, 0),
            )
            self._plot.add_mesh(
                y_line,
                color='green',
                line_width=self.set.axis_line_width,
                pickable=False,
            )

        # Draw Z-axis
        if getattr(self.set, 'draw_zax', False):
            z_line = pv.Line(
                pointa=(0, 0, -length),
                pointb=(0, 0, length),
            )
            self._plot.add_mesh(
                z_line,
                color='blue',
                line_width=self.set.axis_line_width,
                pickable=False,
            )

        exponent = np.floor(np.log10(length))
        gs = 10 ** exponent
        N = np.ceil(length/gs)
        if N < 5:
            gs = gs/10
        L = (2*np.ceil(length/(2*gs))+1)*gs

        # XY grid at Z=0
        if self.set.show_zgrid:
            x_vals = np.arange(-L, L+gs, gs)
            y_vals = np.arange(-L, L+gs, gs)

            # lines parallel to X
            for y in y_vals:
                line = pv.Line(
                    pointa=(-L, y, 0),
                    pointb=(L, y, 0)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5,pickable=False)

            # lines parallel to Y
            for x in x_vals:
                line = pv.Line(
                    pointa=(x, -L, 0),
                    pointb=(x, L, 0)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5,pickable=False)


        # YZ grid at X=0
        if self.set.show_xgrid:
            y_vals = np.arange(-L, L+gs, gs)
            z_vals = np.arange(-L, L+gs, gs)

            # lines parallel to Y
            for z in z_vals:
                line = pv.Line(
                    pointa=(0, -L, z),
                    pointb=(0, L, z)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)

            # lines parallel to Z
            for y in y_vals:
                line = pv.Line(
                    pointa=(0, y, -L),
                    pointb=(0, y, L)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)


        # XZ grid at Y=0
        if self.set.show_ygrid:
            x_vals = np.arange(-L, L+gs, gs)
            z_vals = np.arange(-L, L+gs, gs)

            # lines parallel to X
            for z in z_vals:
                line = pv.Line(
                    pointa=(-length, 0, z),
                    pointb=(length, 0, z)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)

            # lines parallel to Z
            for x in x_vals:
                line = pv.Line(
                    pointa=(x, 0, -length),
                    pointb=(x, 0, length)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)

        if self.set.add_light:
            light = pv.Light()
            light.set_direction_angle(*self.set.light_angle) # type: ignore
            self._plot.add_light(light)

        self._plot.set_background(self.set.background_bottom, top=self.set.background_top) # type: ignore
        self._plot.add_axes() # type: ignore

        self._plot.camera.position = saved_camera["position"]
        self._plot.camera.focal_point = saved_camera["focal_point"]
        self._plot.camera.up = saved_camera["view_up"]
        self._plot.camera.view_angle = saved_camera["view_angle"]
        self._plot.camera.clipping_range = saved_camera["clipping_range"]

        

def freeze(function):

    def new_function(self, *args, **kwargs):
        cam = self.disp._plot.camera_position[:]
        self.disp._plot.suppress_rendering = True
        function(self, *args, **kwargs)
        self.disp._plot.camera_position = cam
        self.disp._plot.suppress_rendering = False
        self.disp._plot.render()
    return new_function


class ScreenSelector:

    def __init__(self, display: PVDisplay):
        self.disp: PVDisplay = display
        self.original_actors: list[pv.Actor] = []
        self.select_actors: list[pv.Actor] = []
        self.grids: list[pv.UnstructuredGrid] = []
        self.surfs: dict[int, np.ndarray] = dict()
        self.state = False

    def toggle(self):
        if self.state:
            self.turn_off()
        else:
            self.activate()

    def activate(self):
        self.original_actors = list(self.disp._plot.actors.values())

        for actor in self.original_actors:
            if isinstance(actor, pv.Text):
                continue
            actor.pickable = False
        
        if len(self.grids) == 0:
            for key in self.disp._facetags:
                tris = self.disp._mesh.get_triangles(key)
                ntris = tris.shape[0]
                cells = np.zeros((ntris,4), dtype=np.int64)
                cells[:,1:] = self.disp._mesh.tris[:,tris].T
                cells[:,0] = 3
                nodes = np.unique(self.disp._mesh.tris[:,tris].flatten())
                celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
                points = self.disp._mesh.nodes.T
                grid = pv.UnstructuredGrid(cells, celltypes, points)
                grid._tag = key # type: ignore
                self.grids.append(grid)
                self.surfs[key] = points[nodes,:].T
        
        self.select_actors = []
        for grid in self.grids:
            actor = self.disp._plot.add_mesh(grid, opacity=0.001, color='red', pickable=True, name=f'FaceTag_{grid._tag}')
            self.select_actors.append(actor)

        def callback(actor: pv.Actor):
            key = int(actor.name.split('_')[1])
            points = self.surfs[key]
            xs = points[0,:]
            ys = points[1,:]
            zs = points[2,:]
            meanx = np.mean(xs)
            meany = np.mean(ys)
            meanz = np.mean(zs)
            data = (meanx, meany, meanz, min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
            encoded = encode_data(data) #type: ignore
            print(f'Face code key={key}: ', encoded)

        self.disp._plot.enable_mesh_picking(callback, style='surface', left_clicking=True, use_actor=True)
    
    def turn_off(self) -> None:
        for actor in self.select_actors:
            self.disp._plot.remove_actor(actor) # type: ignore
        self.select_actors = []
        for actor in self.original_actors:
            if isinstance(actor, pv.Text):
                continue
            actor.pickable = True

        
class ScreenRuler:

    def __init__(self, display: PVDisplay, min_length: float):
        self.disp: PVDisplay = display
        self.points: list[tuple] = [(0,0,0),(0,0,0)]
        self.text: pv.Text | None = None
        self.ruler: Any = None
        self.state: bool = False
        self.min_length: float = min_length
    
    @freeze
    def toggle(self):
        if not self.state:
            self.state = True
            self.disp._plot.enable_point_picking(self._add_point, left_clicking=True, tolerance=self.min_length)
        else:
            self.state = False
            self.disp._plot.disable_picking()

    @freeze
    def turn_off(self):
        self.state = False
        self.disp._plot.disable_picking()
    
    @property
    def dist(self) -> float:
        p1, p2 = self.points
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**(0.5)
    
    @property
    def middle(self) -> tuple[float, float, float]:
        p1, p2 = self.points
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
    
    @property
    def measurement_string(self) -> str:
        dist = self.dist
        p1, p2 = self.points
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        dz = p2[2]-p1[2]
        return f'{dist*1000:.2f}mm (dx={1000.*dx:.4f}mm, dy={1000.*dy:.4f}mm, dz={1000.*dz:.4f}mm)'
    
    def set_ruler(self) -> None:
        if self.ruler is None:
            self.ruler = self.disp._plot.add_ruler(self.points[0], self.points[1], title=f'{1000*self.dist:.2f}mm') # type: ignore
        else:
            p1 = self.ruler.GetPositionCoordinate()
            p2 = self.ruler.GetPosition2Coordinate()
            p1.SetValue(*self.points[0])
            p2.SetValue(*self.points[1])
            self.ruler.SetTitle(f'{1000*self.dist:.2f}mm')
    
    @freeze
    def _add_point(self, point: tuple[float, float, float]):
        self.points = [point,self.points[0]]
        self.text = self.disp._plot.add_text(self.measurement_string, self.middle, name='RulerText')
        self.set_ruler()