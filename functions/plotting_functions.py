import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches

def plot_discrete_raster(
    fig, ax, arr, labels, bounds, cmap="cividis", norm=None, *args, **kwargs
):
    """
    Parameters
    ----------
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis

    arr: numpy.ndarray
        Contains raster data
    bounds: list
        [left, bottom, right, upper] 
    labels: list
        Supply unique string for each value
    cmap: string or matplotlib.colors.Colormap, default = 'cividis

    norm: matplotlib.colors.Normalize, optional
        normalisation function for colormapping
    Returns
    -------
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis

    """
    im = ax.imshow(
        arr,
        cmap=cmap,
        norm=norm,
        extent=np.array(bounds)[[0, 2, 1, 3]],
        interpolation="nearest",
        *args,
        **kwargs,
    )
    values = np.unique(arr)
    values = values[~np.isnan(values)]
    colours = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [
        mpatches.Patch(
            color=colours[i],
            label=labels[
                np.where(values[i] == np.arange(np.min(values), np.max(values) + 1))[0][
                    0
                ]
            ],
        )
        for i in range(len(values))
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    return fig, ax

def plot_continuous_raster(
    fig,
    ax,
    arr,
    bounds,
    cmap="cividis",
    hillshade = False,
    norm=None,
    colorbar=True,
    ticks=None,
    *args,
    **kwargs
):
    """
    Parameters
    ----------
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis

    arr: numpy.ndarray
        Contains raster data
    bounds: list
        [left, bottom, right, upper] as obtained from RasterProperties.bounds
        (See :class:`pycnws.geo.rasterproperties.RasterProperties`)
    cmap: string or matplotlib.colors.Colormap, default = 'cividis'
    hillshade: bool, default = False
        if True, adds a hillshade to the rasterplot
    norm: matplotlib.colors.Normalize, optional
        normalisation function for colormapping
    colorbar: bool, default = True
                Choice of displaying a colorbar
    ticks: list, optional
        Supply [min, 25th percentile, 50th percentile ,75th percentile, max]
        when norm is used!

    Returns
    -------
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis

    """
    if hillshade == False:
        im = ax.imshow(
            arr,
            cmap=cmap,
            norm=norm,
            extent=np.array(bounds)[[0, 2, 1, 3]],
            *args,
            **kwargs,
            interpolation="nearest",
        )
    else:
        ls = LightSource()
        im = ax.imshow(
            ls.hillshade(arr),
            cmap=cmap,
            norm=norm,
            extent=np.array(bounds)[[0, 2, 1, 3]],
            *args,
            **kwargs,
            interpolation="nearest",
        )

    if colorbar:
        fig.colorbar(im, orientation="vertical", shrink=0.5, ticks=ticks)
    else:  # allows custom colorbar
        fig = im
    return fig, ax

def plot_tf_history(history):
    """
    Function to plot the training and validation loss of training with tensorflow

    Inputs
    ------
    history: keras.callbacks.History
    
    Returns
    ------
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis

    
    """
    fig, ax = plt.subplots()
    ax.plot(range(len(history.history["loss"])),history.history["loss"],label = 'Training loss')
    ax.plot(range(len(history.history["loss"])),history.history["val_loss"], label = 'Validation loss')
    ax.legend()
    return fig, ax