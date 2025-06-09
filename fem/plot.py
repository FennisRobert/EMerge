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

import numpy as np

# Function to animate a field oscillation with phase shift
def animate_field(X, Y, data: np.ndarray, Nsteps: int, vrange: float | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    # Compute the magnitude for consistent color scaling
    data_magnitude = np.abs(data)
    real_data_max = np.max(data_magnitude)
    if vrange is not None:
        real_data_max = vrange
    real_data_min = -real_data_max

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initial field with zero phase
    psi = np.exp(-1j * 2 * np.pi * 0 / Nsteps)
    real_data = np.real(data * psi)

    # Create initial contour plot
    levels = np.linspace(real_data_min, real_data_max, 100)
    contour = ax.contourf(X, Y, real_data, levels=levels, vmin=real_data_min, vmax=real_data_max)

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax, label="Field Intensity")

    # Set plot labels and title
    ax.set_title("Field Distribution")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.axis("equal")

    # Animation update function
    def update(frame):
        nonlocal contour
        # Remove old contours
        for c in contour.collections:
            c.remove()
        # Update the phase
        psi = np.exp(1j * 2 * np.pi * frame / Nsteps)
        real_data = np.real(data * psi)
        # Redraw contour plot
        contour = ax.contourf(X, Y, real_data, levels=levels, vmin=real_data_min, vmax=real_data_max)
        return contour.collections

    # Create animation
    anim = FuncAnimation(fig, update, frames=Nsteps, blit=False, interval=25, repeat=True)

    plt.show()

# def plot_field(dataset: Dataset2D, data: np.ndarray, cmap=EMERGE_Base) -> None:
#         """
#         Plot the field values on the 2D mesh.

#         Parameters:
#         field (np.ndarray): Field values at each vertex in the mesh.
#         """
#         # Extract vertices and triangles
#         vertices = dataset.vertices
#         triangles = (
#             dataset.triangles.T
#         )  # Transpose to match the (N, 3) format for Matplotlib

#         plt.figure(figsize=(8, 6))
#         plt.tricontourf(
#             vertices[0,:], vertices[1,:], triangles, data, levels=100, cmap=cmap
#         )
#         plt.colorbar(label="Field Intensity")
#         plt.title("Field Distribution")
#         plt.xlabel("X Coordinate")
#         plt.ylabel("Y Coordinate")
#         plt.axis("equal")
#         plt.show()

# def plot_field_tri(mesh: Mesh2D, data: np.ndarray, cmap=EMERGE_Base) -> None:
#         """
#         Plot the field values on the 2D mesh.

#         Parameters:
#         field (np.ndarray): Field values at each vertex in the mesh.
#         """
#         # Extract vertices and triangles
#         vertices = mesh.vertices
#         triangles = (
#             mesh.triangles.T
#         )  # Transpose to match the (N, 3) format for Matplotlib

#         plt.figure(figsize=(8, 6))
#         plt.tripcolor(
#             vertices[0,:], vertices[1,:], triangles, facecolors=data, cmap=cmap
#         )
#         plt.colorbar(label="Field Intensity")
#         plt.title("Field Distribution")
#         plt.xlabel("X Coordinate")
#         plt.ylabel("Y Coordinate")
#         plt.axis("equal")
#         plt.show()

# def radiation_plot(angles: np.ndarray, amplitude) -> None:
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

#     amax = np.max(np.abs(amplitude))

#     xs = amplitude*np.cos(angles)
#     ys = amplitude*np.sin(angles)


#     ax.plot(angles, amplitude, color='k')
#     #ax.set_xlim([-amax, amax])
#     #ax.set_ylim([-amax, amax])
#     ax.grid(True)


#     plt.show()

# def linplot(x, ys):
#     if not isinstance(ys, (list,tuple)):
#         ys = [ys,]
#     plt.figure(figsize=(8,6))
#     for y in ys:
#         plt.plot(x,y)
#     plt.grid(True)
#     plt.show()


# def plot_s_parameters(dataset: Dataset2D, sparam_labels=None, title="S-Parameter Plot"):
#     """
#     Plots S-parameters in dB (magnitude) and phase (degrees) across frequencies.

#     Parameters:
#     - datasets: List of Dataset2D objects, each representing a single frequency.
#                 Each dataset should contain:
#                     - dataset.frequency: The frequency at which the S-parameters are measured.
#                     - dataset.Sparam: A list of complex S-parameters (S11, S12, S21, etc.) at this frequency.
#     - sparam_labels: Optional list of labels for each S-parameter (e.g., ["S11", "S12", "S21", ...]).
#                      Defaults to S11, S12, ... based on the length of the first dataset's Sparam list.
#     - title: Title of the plot.

#     Returns:
#     - A matplotlib figure with two subplots:
#         - Top subplot: Magnitude in dB for each S-parameter across frequencies.
#         - Bottom subplot: Phase in degrees for each S-parameter across frequencies.
#     """
#     # Extract frequency and S-parameter data across all datasets
#     frequencies = dataset.freqs

#     # If no labels are provided, auto-generate based on the first dataset's S-parameter count
#     if sparam_labels is None:
#         sparam_labels = [f"S{i+1}{1}" for i in range(dataset.S.shape[1])]
    
#     sublines = [[],[]]

#     S = dataset.S
#     # Plot magnitude in dB for each S-parameter
#     for idx, label in enumerate(sparam_labels):
#         magnitudes_db = 20 * np.log10(np.abs(S[:,idx,0].squeeze()))
#         sublines[0].append(Line(frequencies, magnitudes_db, name=label, color='k'))
    

#     # Plot phase in degrees for each S-parameter
#     for idx, label in enumerate(sparam_labels):
#         phases_deg = np.angle(S[:,idx,0].squeeze(), deg=True)
#         sublines[1].append(Line(frequencies, phases_deg, name=label, color='k'))
    

#     with SubFigs(2,1) as axes:
#         eplot(sublines[0], axes=axes[0], linestyle_cycle=['-','--'],xlabel='Frequency [Hz]', ylabel='Sparam [dB]')
#         eplot(sublines[1], axes=axes[1], linestyle_cycle=['-','--'],xlabel='Frequency [Hz]', ylabel='Sparam [deg]')

# def plot_mesh(
#         vertices,
#         triangles,
#         highlight_vertices: list[int] = None,
#         highlight_triangles: list[int] = None,
#     ) -> None:
#     # Extract x and y coordinates of vertices
#     x = vertices[0, :]
#     y = vertices[1, :]

#     # Create the plot
#     plt.figure(figsize=(8, 6))
#     plt.triplot(
#         x, y, triangles.T, color="black", lw=0.5
#     )  # Use triplot for 2D triangular meshes
#     plt.scatter(x, y, color="red", s=10)  # Plot vertices for reference

#     # Set plot labels and title
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("2D Mesh Plot")
#     plt.gca().set_aspect("equal", adjustable="box")  # Set equal aspect ratio

#     if highlight_triangles:
#         for index in highlight_triangles:
#             plt.fill(
#                 x[triangles[:, index]],
#                 y[triangles[:, index]],
#                 alpha=0.3,
#                 color="blue",
#             )
#     if highlight_vertices:
#         ids = np.array(highlight_vertices)
#         x = vertices[0, ids]
#         y = vertices[1, ids]
#         plt.scatter(x, y)
#     # Show the plot
#     plt.show()