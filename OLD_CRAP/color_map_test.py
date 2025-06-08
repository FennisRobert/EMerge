import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def generate_colormap(colors: list[tuple[int,int,int]]) -> LinearSegmentedColormap:
    colors = [(R/256, G/256, B/256) for R,G,B in colors]
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
# Example list of RGB tuples
RGB_Base = [
    (94,33,77),
    (35,13,86),
    (38,32,121),
    (83,157,196),
    (252,157,33),
    (244,45,80),
]
RGB_Wave = [
    (83,157,210),
    (38,32,121),
    (94,33,77),
    (244,45,80),
    (252,157,33),
]

# Create a continuously interpolated colormap from the given RGB tuples
EMERGE_Base = generate_colormap(RGB_Base)
EMERGE_Wave = generate_colormap(RGB_Wave)
# Define a coordinate grid
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Define some interesting function, for example: Z = sin(r) / r with r = sqrt(X²+Y²)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(3*R)  # adding a small epsilon to avoid division by zero

# Plotting the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Use the custom continuous colormap
surf = ax.plot_surface(X, Y, Z, cmap=EMERGE_Wave)

# Add a colorbar to show the mapping of values to colors
fig.colorbar(surf, shrink=0.5, aspect=10)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Custom Interpolated Colormap Surface Plot')

plt.show()
