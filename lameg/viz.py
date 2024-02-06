import collections

import numpy as np
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import k3d
import warnings

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
    rgb (array): accepts integer [R, G, B] array
    
    Notes:
    - function requires integer RGB (values 0-255)
    """
    color = 0
    for c in rgb:
        color = (color << 8) + int(c)
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


def show_surface(surface, color=None, grid=False, menu=False, colors=None, info=False, camera_view=None, height=512,
                 opacity=1.0, coords=None, coord_size=1, coord_color=None):
    """
    Renders a 3D surface with optional data overlay. The rendering is persistent and does not require an active kernel.

    Parameters:
    - surface (nibabel.gifti.GiftiImage): The Gifti surface mesh to be rendered.
    - color (array, optional): Basic color of the surface in the absence of data. Specified as a decimal RGB array.
                               Default is [166, 166, 166].
    - grid (bool, optional): Toggles the rendering of a grid. Default is False.
    - menu (bool, optional): Toggles the display of a menu with options such as lighting adjustments. Default is False.
    - colors (array, optional): An array of vertex colors specified as hexadecimal 32-bit color values. Each color
                                corresponds to a vertex on the surface. Default is None.
    - info (bool, optional): If True, prints information about the surface, such as the number of vertices. Default is
                             False.
    - camera_view (array, optional): Specifies a camera view for the rendering. If None, an automatic camera view is
                                     set. Default is None.
    - height (int, optional): Height of the widget in pixels. Default is 512.
    - opacity (float, optional): Sets the opacity of the surface, with 1.0 being fully opaque and 0.0 being fully
                                 transparent. Default is 1.0.

    Returns:
    - plot: A k3d plot object containing the rendered surface.

    This function utilizes the k3d library for rendering the surface. It supports customization of surface color,
    opacity, and additional features like grid and menu display. The `colors` parameter allows for vertex-level color
    customization.
    """
    if color is None:
        color = [166, 166, 166]
    if coord_color is None:
        coord_color = [255, 0, 0]

    color = rgbtoint(color)
    coord_color = np.array(coord_color).reshape(-1, 3)
    
    try:
        vertices, faces, _ = surface.agg_data()
    except:
        vertices, faces = surface.agg_data()
    
    mesh = k3d.mesh(vertices, faces, side="double", color=color, opacity=opacity)
    cam_autofit = camera_view is None
    plot = k3d.plot(
        grid_visible=grid, menu_visibility=menu, camera_auto_fit=cam_autofit, height=height
    )
    plot += mesh
    if hasattr(colors, "__iter__"):
        mesh.colors = colors
    else:
        pass

    if coords is not None:
        coords = np.array(coords).reshape(-1,3)
        for c_idx in range(coords.shape[0]):
            coord = coords[c_idx,:]
            size = coord_size
            if isinstance(coord_size, collections.abc.Sequence):
                size = coord_size[c_idx]
            if coord_color.shape[0]>1:
                color = coord_color[c_idx,:]
            else:
                color = coord_color[0,:]
            color = list(map(int, color))
            pt = k3d.points(
                coord,
                point_size=size,
                color=rgbtoint(color)
            )
            plot += pt



    if camera_view is not None:
        plot.camera=camera_view
    
    plot.display()
    if info:
        print(vertices.shape[0], "vertices")

    return plot


def plot_csd(csd, times, ax, cb=True, cmap="RdBu_r", vmin_vmax=None, return_details=False):
    """
    Plot the computed Current Source Density (CSD) data.

    This function takes a CSD matrix and plots it over a specified time range. It offers options for
    color normalization, colormap selection, and including a colorbar. Optionally, it can return plot details.

    Parameters:
    csd (numpy.ndarray): The CSD matrix to be plotted, with dimensions corresponding to layers x time points.
    times (numpy.ndarray): A 1D array of time points corresponding to the columns of the CSD matrix.
    ax (matplotlib.axes.Axes): The matplotlib axes object where the CSD data will be plotted.
    cb (bool, optional): Flag to indicate whether a colorbar should be added to the plot. Default is True.
    cmap (str, optional): The colormap used for plotting the CSD data. Default is "RdBu_r".
    vmin_vmax (tuple or str, optional): A tuple specifying the (vmin, vmax) range for color normalization. If "norm",
                                        a standard normalization is used. If None, the range is set to the maximum
                                        absolute value in the CSD matrix. Default is None.
    return_details (bool, optional): If True, the function returns additional plot details. Default is False.

    Returns:
    tuple: If `return_details` is True, returns a tuple containing layer parameters and the imshow object of the plot.
           Otherwise, the function does not return anything.

    Notes:
    - This function requires 'numpy', 'matplotlib.colors', and 'matplotlib.pyplot' libraries.
    - The 'TwoSlopeNorm' from 'matplotlib.colors' is used for diverging color normalization.
    - The aspect ratio of the plot is automatically set to 'auto' for appropriate time-layer representation.
    - Layer labels are set from 1 to 11, assuming a total of 11 layers.
    """
    max_smooth = np.max(np.abs(csd))
    if vmin_vmax == None:
        divnorm = colors.TwoSlopeNorm(vmin=-max_smooth, vcenter=0, vmax=max_smooth)
    elif vmin_vmax == "norm":
        divnorm = colors.Normalize()
    else:
        divnorm = colors.TwoSlopeNorm(vmin=vmin_vmax[0], vcenter=0, vmax=vmin_vmax[1])
    extent = [times[0], times[-1], 0, 1]
    csd_imshow = ax.imshow(
        csd, norm=divnorm, origin="lower",
        aspect="auto", extent=extent,
        cmap=cmap, interpolation="none"
    )
    ax.set_ylim(1,0)
    ax.set_yticks(np.linspace(0,1, 11))
    ax.set_yticklabels(np.arange(1,12))
    layers_params = []
    if cb:
        clbr=plt.colorbar(csd_imshow, ax=ax)
        clbr.set_label('CSD')
    plt.tight_layout()
    if return_details:
        return layers_params, csd_imshow