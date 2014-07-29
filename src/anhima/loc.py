"""
Utilities for locating samples, variants and genome positions.

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# standard library dependencies
import bisect


# third party dependencies
import numpy as np
import numexpr
import matplotlib.pyplot as plt
import scipy.stats


def take_samples(a, all_samples, selected_samples):
    """Extract columns from the array `a` corresponding to selected samples.

    Parameters
    ----------

    a : array_like
        An array with 2 or more dimensions, where the second dimension
        corresponds to samples.
    all_samples : sequence
        A sequence (e.g., list) of sample identifiers corresponding to the
        second dimension of `a`.
    selected_samples : sequence
        A sequence (e.g., list) of sample identifiers corresponding to the
        columns to be extracted from `a`.

    Returns
    -------

    b : ndarray
        An array obtained from `a` by taking columns corresponding to the
        selected samples.

    """

    # check a is an array of 2 or more dimensions
    assert a.ndim > 1

    # check length of samples dimension is as expected
    assert a.shape[1] == len(all_samples)

    # make sure it's a list
    all_samples = list(all_samples)

    # check selections are in all_samples
    assert all([s in all_samples for s in selected_samples])

    # determine indices for selected samples
    indices = [all_samples.index(s) for s in selected_samples]

    # take columns from the array
    b = np.take(a, indices, axis=1)

    return b


def take_sample(a, all_samples, selected_sample):
    """View a single column from the array `a` corresponding to a selected
    sample.

    Parameters
    ----------

    a : array_like
        An array with 2 or more dimensions, where the second dimension
        corresponds to samples.
    all_samples : sequence
        A sequence (e.g., list) of sample identifiers corresponding to the
        second dimension of `a`.
    selected_sample : string
        A sample identifiers corresponding to the columns to be extracted from
        `a`.

    Returns
    -------

    b : ndarray
        An array obtained from `a` by taking the column corresponding to the
        selected sample.

    """

    # check a is an array of 2 or more dimensions
    assert a.ndim > 1

    # check length of samples dimension is as expected
    assert a.shape[1] == len(all_samples)

    # make sure it's a list
    all_samples = list(all_samples)

    # check selection is in all_samples
    assert selected_sample in all_samples

    # determine indices for selected samples
    index = all_samples.index(selected_sample)

    # take column from the array
    b = a[:, index, ...]

    return b


def query_variants(expression, variants):
    """Evaluate `expression` with respect to the given `variants`.

    Parameters
    ----------

    expression : string
        The query expression to apply. The expression will be evaluated by
        :mod:`numexpr` against the provided `variants`.
    variants : dict-like
        The variables to include in scope for the expression evaluation.

    Returns
    -------

    result : ndarray
        The result of evaluating `expression` against `variants`.

    """

    result = numexpr.evaluate(expression, local_dict=variants)

    return result


def take_variants(a, selection):
    """Extract rows from the array `a` corresponding to a sparse `selection` of
    variants.

    Parameters
    ----------

    a :  array_like
        An array to extract rows from (e.g., genotypes).
    selection : array_like, bool
        A boolean array of the same length as the first dimension of `a`.

    Returns
    -------

    b : ndarray
        An array obtained from `a` by taking rows corresponding to the
        selected variants.

    """

    # check dimensions and sizes
    assert a.ndim >= 1
    assert selection.ndim == 1
    assert a.shape[0] == selection.shape[0]

    # take rows from the input array
    b = np.compress(selection, a, axis=0)

    return b


def locate_region(pos, start_position=0, stop_position=None):
    """Locate the start and stop indices within the `pos` array that include all
    positions within the `start_position` and `stop_position` range.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    start_position : int
        Start position of region.
    stop_position : int
        Stop position of region

    Returns
    -------

    loc : slice
        A slice object with the start and stop indices that include all
        positions within the region.

    """

    start_index = bisect.bisect_left(pos, start_position)
    stop_index = bisect.bisect_right(pos, stop_position) if stop_position is not None else None
    loc = slice(start_index, stop_index)
    return loc


def take_region(a, pos, start_position, stop_position):
    """View a contiguous slice along the first dimension of `a`
    corresponding to a genome region defined by `start_position` and
    `stop_position`.

    Parameters
    ----------

    a : array_like
        The array to extract from.
    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    start_position : int
        Start position of region.
    stop_position : int
        Stop position of region

    Returns
    -------

    b : ndarray
        A view of `a` obtained by slicing along the first dimension.

    """

    loc = locate_region(pos, start_position, stop_position)
    return a[loc, ...]


def plot_variant_locator(pos, step=1, ax=None, start_position=None,
                         stop_position=None, flip=False, line_args=None):
    """
    Plot lines indicating the physical genome location of variants. By
    default the top x axis is in variant index space, and the bottom x axis
    is in genome position space.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    step : int, optional
        Plot a line for every `step` variants.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    flip : bool, optional
        Flip the plot upside down.
    line_args : dict-like
        Additional keyword arguments passed through to `plt.Line2D`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 1))
        ax = fig.add_subplot(111)

    # determine x axis limits
    if start_position is None:
        start_position = np.min(pos)
    if stop_position is None:
        stop_position = np.max(pos)
    ax.set_xlim(start_position, stop_position)

    # plot the lines
    if line_args is None:
        line_args = dict()
    line_args.setdefault('linewidth', .5)
    n_variants = len(pos)
    for i, p in enumerate(pos[::step]):
        xfrom = p
        xto = (
            start_position +
            ((i * step / n_variants) * (stop_position-start_position))
        )
        l = plt.Line2D([xfrom, xto], [0, 1], **line_args)
        ax.add_line(l)

    # invert?
    if flip:
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    else:
        ax.xaxis.tick_bottom()

    # tidy up
    ax.set_yticks([])
    for l in 'left', 'right':
        ax.spines[l].set_visible(False)

    return ax


def windowed_variant_counts(pos, window_size, start_position=None,
                            stop_position=None):
    """Count variants in non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.

    Returns
    -------

    counts : ndarray, int
        The number of variants in each window.
    bin_edges : ndarray, int
        The edge positions of each window. Note that this has length
        `len(counts)+1`. To determine bin centers use
        `(bin_edges[:-1] + bin_edges[1:]) / 2`. To determine bin widths use
        `np.diff(bin_edges)`

    See Also
    --------

    windowed_variant_counts_plot, windowed_variant_density

    """

    # determine bins
    if stop_position is None:
        stop_position = np.max(pos)
    if start_position is None:
        start_position = np.min(pos)
    bin_edges = np.append(np.arange(start_position, stop_position, window_size),
                          stop_position)

    # make a histogram of positions
    counts, _ = np.histogram(pos, bins=bin_edges)

    return counts, bin_edges


def plot_windowed_variant_counts(pos, window_size, start_position=None,
                                 stop_position=None,
                                 ax=None, plot_kwargs=None):
    """Plot windowed variant counts.

    Parameters
    ----------

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    plot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.plot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    See Also
    --------

    windowed_variant_counts, windowed_variant_density_plot

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # count variants
    y, bin_edges = windowed_variant_counts(pos, window_size,
                                   start_position=start_position,
                                   stop_position=stop_position)

    # calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # plot data
    if plot_kwargs is None:
        plot_kwargs = dict()
    plot_kwargs.setdefault('linestyle', '-')
    plot_kwargs.setdefault('marker', None)
    ax.plot(bin_centers, y, **plot_kwargs)

    # tidy up
    ax.set_ylim(bottom=0)
    ax.set_xlabel('position')
    ax.set_ylabel('count')
    if start_position is None:
        start_position = np.min(pos)
    if stop_position is None:
        stop_position = np.max(pos)
    ax.set_xlim(start_position, stop_position)

    return ax


def windowed_variant_density(pos, window_size, start_position=None,
                             stop_position=None):
    """Calculate per-base-pair density of variants in non-overlapping windows
    over the genome.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.

    Returns
    -------

    density : ndarray, int
        The density of variants in each window.
    bin_edges : ndarray, int
        The edge positions of each window. Note that this has length
        `len(counts)+1`. To determine bin centers use
        `(bin_edges[:-1] + bin_edges[1:]) / 2`. To determine bin widths use
        `np.diff(bin_edges)`

    See Also
    --------

    windowed_variant_density_plot, windowed_variant_counts

    """

    # count variants in windows
    counts, bin_edges = windowed_variant_counts(pos, window_size,
                                                  start_position=start_position,
                                                  stop_position=stop_position)

    bin_widths = np.diff(bin_edges)
    
    # convert to per-base-pair density
    density = counts / bin_widths

    return density, bin_edges


def plot_windowed_variant_density(pos, window_size, start_position=None,
                                  stop_position=None,
                                  ax=None, plot_kwargs=None):
    """Plot windowed variant density.

    Parameters
    ----------

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    plot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.plot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    See Also
    --------

    windowed_variant_density, windowed_variant_counts_plot

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # count variants
    y, bin_edges = windowed_variant_density(pos, window_size,
                                    start_position=start_position,
                                    stop_position=stop_position)

    # calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # plot data
    if plot_kwargs is None:
        plot_kwargs = dict()
    plot_kwargs.setdefault('linestyle', '-')
    plot_kwargs.setdefault('marker', None)
    ax.plot(bin_centers, y, **plot_kwargs)

    # tidy up
    ax.set_ylim(bottom=0)
    ax.set_xlabel('position')
    ax.set_ylabel('density')
    if start_position is None:
        start_position = np.min(pos)
    if stop_position is None:
        stop_position = np.max(pos)
    ax.set_xlim(start_position, stop_position)

    return ax


def windowed_statistic(pos, values, window_size,
                       start_position=None,
                       stop_position=None,
                       statistic='mean'):
    """Calculate a statistic for `values` binned in non-overlapping windows
    over the genome.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    values : array_like
        A 1-D array of the same length as `pos`.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    statistic : string or function
        The function to apply to values in each bin.

    """

    # determine bins
    if stop_position is None:
        stop_position = np.max(pos)
    if start_position is None:
        start_position = np.min(pos)
    bin_edges = np.append(np.arange(start_position, stop_position, window_size),
                          stop_position)

    # compute binned statistic
    stats, _, _ = scipy.stats.binned_statistic(pos, values=values,
                                               statistic=statistic,
                                               bins=bin_edges)

    # # calculate bin centers
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return stats, bin_edges


def downsample_variants(a, target):
    """Downsample an array along the first dimension to the `target`
    length, assuming the first dimension corresponds to variants.

    Parameters
    ----------

    a : array_like
        The array to downsample.
    target : int
        The target number of variants.

    Returns
    -------

    b : array_like
        A downsampled view of `a`.

    """

    n_variants = a.shape[0]
    step = max(1, int(n_variants/target))
    b = a[::step, ...]
    return b


