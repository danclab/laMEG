import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt


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