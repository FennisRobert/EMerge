import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

def generate_colormap(colors: list[tuple[int,int,int]]) -> LinearSegmentedColormap:
    colors = [(R/256, G/256, B/256) for R,G,B in colors]
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# Example list of RGB tuples
RGB_Base = [
    (35,13,86),
    (38,32,121),
    (83,157,196),
    (252,157,33),
    (244,45,80),
]
RGB_Wave = [
    (28,210,255),
    (45,32,224),
    (68,42,63),
    (244,45,80),
    (252,157,33),
]

# Create a continuously interpolated colormap from the given RGB tuples
EMERGE_Base = generate_colormap(RGB_Base)
EMERGE_Wave = generate_colormap(RGB_Wave)

# Define a constant for the output path (modify this as needed)
OUTPUT_PATH = 'images/'

class Line:
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        name: str = 'unnamed',
        marker: bool = False,
        msize: int = 7,
        msymbol: str = 'o',
        color: str = None,
        linestyle: str = None,
        linewidth: int = 1.5,
        mx: np.ndarray = None,
        my: np.ndarray = None
    ):
        """
        Represents a single line to be plotted.

        Parameters:
        - x_data: np.ndarray, x-coordinates of the line.
        - y_data: np.ndarray, y-coordinates of the line.
        - name: str, name of the line for the legend.
        - marker: bool, whether to display markers on the line.
        - color: str or None, color of the line. If None, uses the cycle.
        - linestyle: str or None, style of the line. If None, uses the cycle.
        - mx: np.ndarray or None, x-coordinates for additional markers.
        - my: np.ndarray or None, y-coordinates for additional markers.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.name = name
        self.msymbol = msymbol
        self.marker = marker
        self.msize = msize
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.mx = mx
        self.my = my

    def __add__(self, other):
        """
        Overloads the + operator to merge two Line objects.

        The resulting Line object takes all properties from self (line1)
        and merges the x_data and y_data from both self and other (line2).
        Additional markers (mx, my) are also merged if present in both.

        Parameters:
        - other: Line, the Line object to add.

        Returns:
        - Line: A new Line object with merged data.
        """
        if not isinstance(other, Line):
            return NotImplemented

        # Merge x_data and y_data
        merged_x = np.concatenate((self.x_data, other.x_data))
        merged_y = np.concatenate((self.y_data, other.y_data))

        # Merge additional markers
        if self.mx is not None and other.mx is not None:
            merged_mx = np.concatenate((self.mx, other.mx))
            merged_my = np.concatenate((self.my, other.my))
        elif self.mx is not None:
            merged_mx = self.mx.copy()
            merged_my = self.my.copy()
        elif other.mx is not None:
            merged_mx = other.mx.copy()
            merged_my = other.my.copy()
        else:
            merged_mx = None
            merged_my = None

        # Create the merged Line object
        merged_line = Line(
            x_data=merged_x,
            y_data=merged_y,
            name=self.name,  # Taking name from self
            marker=self.marker,  # Taking marker from self
            msymbol=self.msymbol,
            color=self.color,  # Taking color from self
            linestyle=self.linestyle,  # Taking linestyle from self
            mx=merged_mx,
            my=merged_my
        )

        return merged_line

# Assuming OUTPUT_PATH is already defined as in quick_plot
OUTPUT_PATH = ''

def eplot(
    lines: list[Line],
    grid: bool = False,
    linewidth: float = None,
    linestyle_cycle: list = ['-'],
    color_cycle: list = ['k', 'b','g', 'r', 'c', 'm', 'y'],
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    dpi: int = 300,
    black_borders: bool = True,
    ticks_inwards: bool = True,
    filename: str = None,
    output_path_constant: str = OUTPUT_PATH,
    show: bool = True,
    axes = None,
):
    """
    A plot function using Matplotlib that accepts a list of Line objects.

    Parameters:
    - lines: list of Line objects, each representing a dataset to plot.
    - grid: bool, default True, whether to display grid.
    - linewidth: float, default 1.5, thickness of plot lines.
    - linestyle_cycle: list of str, default ['-'], styles of lines.
    - color_cycle: list of str, default color list, colors of lines.
    - xticks: list or np.ndarray, custom x-axis ticks.
    - yticks: list or np.ndarray, custom y-axis ticks.
    - xlim: tuple, x-axis limits.
    - ylim: tuple, y-axis limits.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - title: str, title of the plot.
    - dpi: int, default 300, resolution of the output image.
    - black_borders: bool, default False, black borders on the plot edges.
    - ticks_inwards: bool, default False, ticks point inwards.
    - filename: str, the output filename.
    - output_path_constant: str, base path for output files.
    """
    if axes is None:
        # Create figure and axis with default aspect ratio 4:3
        fig, ax = plt.subplots(figsize=(6, 5))  # 4:3 aspect ratio
    else:
        ax = axes

    # Plot each Line object
    for idx, line in enumerate(lines):
        # Determine line style and color
        linestyle = line.linestyle if line.linestyle else linestyle_cycle[idx % len(linestyle_cycle)]
        color = line.color if line.color else color_cycle[idx % len(color_cycle)]
        plot_args = {
            'color': color,
            'label': line.name,
            'fillstyle': 'none'
        }

        if linewidth:
            plot_args['linewidth'] = linewidth
        if linestyle:
            plot_args['linestyle'] = linestyle
        if color:
            plot_args['color'] = color
        # Plot the main line
        
        if line.marker:
            plot_args['marker'] = line.msymbol
            plot_args['markersize'] = line.msize
        
        ax.plot(line.x_data, line.y_data, **plot_args)
        
        # Plot additional markers if provided
        if line.mx is not None and line.my is not None:
            ax.plot(line.mx, line.my, 'o', color=color)

    # Apply grid settings with black, dashed, thin lines
    if grid:
        ax.grid(grid, color='lightgrey', linestyle=':', linewidth=0.5)
        ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Set custom ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Set plot title
    if title is not None:
        ax.set_title(title)

    # Add legends
    ax.legend()

    # Apply black borders
    if black_borders:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    # Set ticks direction
    if ticks_inwards:
        ax.tick_params(direction='in', length=10, right=True, top=True)
        ax.tick_params(which='minor',direction='in', length=5, right=True, top=True)
    # Set aspect ratio to approximately 4:3
    #ax.set_aspect(4/3)

    if axes is not None:
        return 
    
    if filename is not None:
        # Construct the output file path
        if output_path_constant is not None:
            output_path = os.path.join(output_path_constant, filename)
        else:
            output_path = filename

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the figure
        plt.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    else:
        # Close the plot to free memory
        plt.close(fig)


def radiation_plot(angles: np.ndarray, amplitude) -> None:
    fig, ax = plt.subplot()

    amax = np.max(np.abs(amplitude))

    xs = amplitude*np.cos(angles)
    ys = amplitude*np.sin(angles)

    ax.plot(xs,ys)
    ax.set_xlim([-amax, amax])
    ax.set_ylim([-amax, amax])
    plt.show()



class SubFigs:

    def __init__(self, *shape: tuple[int]):
        self.shape = shape
        self.nfigs = self.shape[0]*self.shape[1]
        self.fig, self.axes = plt.subplots(self.shape[0],self.shape[1],figsize=(6, 5))

    def __enter__(self) -> tuple:
        return self.axes
    
    def __exit__(self, *args):
        plt.show()


def esubplot(
    sublines: list[list[Line]],
    grid: bool = False,
    linewidth: float = None,
    linestyle_cycle: list = ['-'],
    color_cycle: list = ['k', 'b','g', 'r', 'c', 'm', 'y'],
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    dpi: int = 300,
    black_borders: bool = True,
    ticks_inwards: bool = True,
    filename: str = None,
    output_path_constant: str = OUTPUT_PATH,
    show: bool = True,
    return_ax: bool = False
):
    """
    A plot function using Matplotlib that accepts a list of Line objects.

    Parameters:
    - lines: list of Line objects, each representing a dataset to plot.
    - grid: bool, default True, whether to display grid.
    - linewidth: float, default 1.5, thickness of plot lines.
    - linestyle_cycle: list of str, default ['-'], styles of lines.
    - color_cycle: list of str, default color list, colors of lines.
    - xticks: list or np.ndarray, custom x-axis ticks.
    - yticks: list or np.ndarray, custom y-axis ticks.
    - xlim: tuple, x-axis limits.
    - ylim: tuple, y-axis limits.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - title: str, title of the plot.
    - dpi: int, default 300, resolution of the output image.
    - black_borders: bool, default False, black borders on the plot edges.
    - ticks_inwards: bool, default False, ticks point inwards.
    - filename: str, the output filename.
    - output_path_constant: str, base path for output files.
    """

    Nsubfigs = len(sublines)

    # Create figure and axis with default aspect ratio 4:3
    fig, axes = plt.subplots(Nsubfigs,1,figsize=(6, 5))  # 4:3 aspect ratio
    
    for iax, lines in enumerate(sublines):
        ax = axes[iax]

        # Plot each Line object
        for idx, line in enumerate(lines):
            # Determine line style and color
            linestyle = line.linestyle if line.linestyle else linestyle_cycle[idx % len(linestyle_cycle)]
            color = line.color if line.color else color_cycle[idx % len(color_cycle)]
            plot_args = {
                'color': color,
                'label': line.name,
                'fillstyle': 'none'
            }

            if linewidth:
                plot_args['linewidth'] = linewidth
            if linestyle:
                plot_args['linestyle'] = linestyle
            if color:
                plot_args['color'] = color
            # Plot the main line
            
            if line.marker:
                plot_args['marker'] = line.msymbol
                plot_args['markersize'] = line.msize
            
            ax.plot(line.x_data, line.y_data, **plot_args)
            
            # Plot additional markers if provided
            if line.mx is not None and line.my is not None:
                ax.plot(line.mx, line.my, 'o', color=color)


        # Apply grid settings with black, dashed, thin lines
        if grid:
            ax.grid(grid, color='lightgrey', linestyle=':', linewidth=0.5)
            ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        # Set axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Set axis labels
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set custom ticks
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        # Set plot title
        if title is not None:
            ax.set_title(title)

        # Add legends
        ax.legend()

        # Apply black borders
        if black_borders:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

        # Set ticks direction
        if ticks_inwards:
            ax.tick_params(direction='in', length=10, right=True, top=True)
            ax.tick_params(which='minor',direction='in', length=5, right=True, top=True)
        # Set aspect ratio to approximately 4:3
        #ax.set_aspect(4/3)

    if filename is not None:
        # Construct the output file path
        if output_path_constant is not None:
            output_path = os.path.join(output_path_constant, filename)
        else:
            output_path = filename

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the figure
        plt.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    else:
        # Close the plot to free memory
        plt.close(fig)