import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
from statsmodels.distributions.empirical_distribution import ECDF


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
    ax.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.0)
    return fig, ax


def plot_continuous_raster(
    fig,
    ax,
    arr,
    bounds,
    cmap="cividis",
    hillshade=False,
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


def plot_tf_history(history, plot_object="loss"):
    """
    Function to plot the training and validation loss of training with tensorflow

    Inputs
    ------
    history: keras.callbacks.History

    plot_object: string
        To be plotted for both training and validation (if the latter exitst)

    Returns
    ------
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis


    """
    fig, ax = plt.subplots()
    metrics = history.history.keys()
    if plot_object in metrics:
        ax.plot(range(len(history.history[plot_object])),
                history.history[plot_object], label=f'Training {plot_object}')
    else:
        raise ValueError(
            f'The training metric {plot_object} is not defined for the training data')
    val_metric = 'val_' + plot_object
    if val_metric in metrics:
        ax.plot(range(len(history.history[val_metric])),
                history.history[val_metric], label=f'Validation {plot_object}')
    ax.legend()
    return fig, ax


def ensemble_plot(Cstar, tfull, out_dict, n_std=2, fig=None, ax=None):
    if (fig == None) and (ax == None):
        fig, ax = plt.subplots(figsize=(9, 7))
    Cstar[tfull].plot(ax=ax)
    ax.plot(out_dict['t_train'], out_dict['mean_y_train_hat'],  # type:ignore
            label='Mean training set')
    ax.plot(out_dict['t_test'], out_dict['mean_y_test_hat'],  # type:ignore
            label='Mean test set')
    ax.fill_between(out_dict['t_train'], out_dict['mean_y_train_hat'] - n_std * out_dict['sd_y_train_hat'],  # type:ignore
                    out_dict['mean_y_train_hat'] + n_std * out_dict['sd_y_train_hat'], color='grey', alpha=0.5)
    ax.fill_between(out_dict['t_test'], out_dict['mean_y_test_hat'] - n_std * out_dict['sd_y_test_hat'],  # type:ignore
                    out_dict['mean_y_test_hat'] + n_std * out_dict['sd_y_test_hat'], color='grey', alpha=0.5)
    ax.legend()  # type:ignore
    ax.set_title(r'$\mu \pm' + f'{n_std}' + r'\sigma$')  # type:ignore
    return fig, ax


def plot_FDC(Q, fig=None, ax=None, label=None, cutoff=0.02, cutoff_bool=True, *args, **kwargs):
    """
    Plotting the flow duration curve (flow exceedance probability curve) for a time series of flows (without NaNs)

    Parameters
    ----------
    Q: np.array
        array with flows values
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis

    Returns
    -------
    fig: matplotlib.figure.Figure

    ax: matplotlib.pyplot.axis
    """
    if not fig and not ax:
        fig, ax = plt.subplots()
    Q_nonan = Q[~np.isnan(Q)]
    Q_sorted = np.sort(Q_nonan)
    CDF_distribution = ECDF(Q_sorted)
    CDF = CDF_distribution(Q_sorted)
    ax.plot(1 - CDF, Q_sorted, label=label, *args, **kwargs)  # type:ignore
    if cutoff_bool:
        ax.vlines(cutoff, ax.get_ylim()[0], ax.get_ylim()[  # type:ignore
            1], color='black', label=f'$p$ = {cutoff}')
    ax.set_yscale('log')  # type:ignore
    ax.set_xlabel('Flow exceedance probability ($p$)')  # type:ignore
    ax.set_ylabel('$Q$ [m$^3$/s]')  # type:ignore
    ax.legend()  # type:ignore
    return fig, ax
