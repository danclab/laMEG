import numpy as np
from matplotlib import cm, colors
import k3d
import warnings

"""
README:
install with:

pip install k3d

enable rendering:

jupyter nbextension install --py --user k3d
jupyter nbextension enable --py --user k3d

more details: https://k3d-jupyter.org/user/install.html

TO DO:
- printing points

"""


warnings.filterwarnings(
    "ignore", message="^.*A coerced copy has been created.*$"
)


def data_to_rgb(data, n_bins, cmap, vmin, vmax, vcenter=0.0, ret_map=False, norm="TS"):
    """
    Returns RGB values of a data mapped to the normalised matplotlib colormap.
    (optionally) Returns a colormap to use for e.g. a colorbar.
    
    Parameters:
    data (iterable): 1d numerical data
    n_bins (int): amount of bins in the histogram
    vmin (float): lowest value on the histogram range 
    vmax (float): highest value on the histogram range
    vcenter (float): centre of the histogram range (default=0 for zero-centred color mapping)
    ret_map (bool): return a colormap object
    norm (str): type of normalisation ("TS", "N", "LOG")
    
    Notes:
    - function creates a normalisation based on the "norm" argument
    - creates a suitable colormap
    - maps data values based on the histogram bins
    - returns RGB values for each data point
    """
    if norm == "TS":
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    elif norm == "N":
        divnorm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == "LOG":
        divnorm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    c = cm.ScalarMappable(divnorm, cmap=cmap)
    bins = np.histogram_bin_edges(data, bins=n_bins)
    bin_ranges = list(zip(bins[:-1], bins[1:]))
    color_mapped = np.zeros((data.shape[0], 4))
    for br_ix, br in enumerate(bin_ranges):
        map_c = (data >= br[0]) & (data <= br[1])
        color_mapped[map_c,:] = c.to_rgba(bins[1:][br_ix])
    
    if not ret_map:
        return color_mapped
    elif ret_map:
        return color_mapped, c


def rgbtoint(rgb):
    """
    Returns a 32bit representation of the color as an integer.
    
    Parameters:
    rgb (array): accepts decimal [R, G, B] array
    
    Notes:
    - function requires decimal RGB (values 0-255)
    """
    color = 0
    for c in rgb:
        color = (color<<8) + c
    return color


def color_map(data, cmap, vmin, vmax, n_bins=1000, vcenter=0, norm="TS"):
    """
    Returns a data mapped to the color map in the hexadecimal format,
    and a colormap to use for e.g. a colorbar.
    
    Parameters:
    data (iterable): 1d numerical data
    n_bins (int): amount of bins in the histogram
    vmin (float): lowest value on the histogram range 
    vmax (float): highest value on the histogram range
    vcenter (float): centre of the histogram range (default=0 for zero-centred color mapping)
    norm (str): type of normalisation ("TS", "N", "LOG")
    
    Notes:
    - function creates a normalisation based on the "norm" argument
    - creates a suitable colormap
    - maps data values based on the histogram bins
    - returns RGB values for each data point
    - converts percent based RGB to decimal
    - converts RGB to hexadecimal in a fromat appropriate for the visualisation function
    """
    colors, c_map = data_to_rgb(
        data, n_bins, cmap, vmin, vmax,
        vcenter=vcenter, ret_map=True, norm=norm
    )
    colors = colors[:,:3] * 255
    colors = colors.astype(int)
    colors = np.uint32([rgbtoint(i) for i in colors])
    return colors, c_map


def show_surface(surface, color=None, grid=False, menu=False, colors=None, info=False):
    """
    Renders a 3d surface with data (optional). Render is persistent even without a kernel.
    
    Parameters:
    surface (nibabel.gifti.GiftiImage): A Gifti surface mesh
    color (array): basic color of the surface, with absence of the data. Decimal RGB
    grid (bool): toggle grid rendering
    menu (bool): toggle menu with e.g. lighting options
    colors (array): array of vertices x hexadecimal 32bit color value to be rendered on the surface
    info (bool): prints info about the surface
    """
    if color is None:
        color = [166, 166, 166]
    color = rgbtoint(color)
    
    try:
        vertices, faces, _ = surface.agg_data()
    except:
        vertices, faces = surface.agg_data()
    
    mesh = k3d.mesh(vertices, faces, side="double", color=color)
    plot = k3d.plot(
        grid_visible=grid, camera_mode="orbit",
        menu_visibility=menu
    )
    plot += mesh
    if hasattr(colors, "__iter__"):
        mesh.colors = colors
    else:
        pass
    
    plot.display()
    if info:
        print("file:", path)
        print(vertices.shape[0], "vertices")