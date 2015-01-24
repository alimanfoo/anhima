# -*- coding: utf-8 -*-
"""
Utilities for calculating and plotting linkage disequilbrium.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/ld.ipynb

"""  # noqa


from __future__ import division, print_function, absolute_import


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mpl_transforms
import scipy.stats as stats
import scipy.spatial.distance as distance


# internal dependencies
import anhima.loc
from anhima.opt.ld import ld_prune_pairwise_uint8 as _ld_prune_pairwise_uint8


def pairwise_genotype_ld(gn):
    """Given a set of genotypes at biallelic variants, calculate the
    square of the correlation coefficient between all distinct pairs
    of variants.

    Parameters
    ----------

    gn : array_like
        A 2-dimensional array of shape (`n_variants`, `n_samples`) where each
        element is a genotype call coded as a single integer counting the
        number of non-reference alleles.

    Returns
    -------

    r_squared : ndarray, float
        A 2-dimensional array of squared correlation coefficients between
        each pair of variants.

    """

    # check input array
    gn = np.asarray(gn)
    assert gn.ndim == 2

    # TODO deal with missing genotypes
    return np.corrcoef(gn) ** 2


def plot_pairwise_ld(r_squared, cmap='Greys', flip=True, ax=None):
    """Make a classic triangular linkage disequilibrium plot, given an
    array of pairwise correlation coefficients between variants.

    Parameters
    ----------

    r_squared : array_like
        A square 2-dimensional array of squared correlation coefficients
        between pairs of variants.
    cmap : color map, optional
        The color map to use when plotting. Defaults to 'Greys' (0=white,
        1=black).
    flip : bool, optional
        If True, draw the triangle upside down.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn

    """

    # check inputs
    r_squared = np.asarray(r_squared)
    assert r_squared.ndim == 2
    assert r_squared.shape[0] == r_squared.shape[1]

    # setup axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x, x//2))
        ax = fig.add_axes((0, 0, 1, 1))

    # define transformation to rotate the colormesh
    trans = mpl_transforms.Affine2D().rotate_deg_around(0, 0, -45)
    trans = trans + ax.transData

    # plot the data as a colormesh
    ax.pcolormesh(r_squared, cmap=cmap, vmin=0, vmax=1, transform=trans)

    # cut the plot in half so we see a triangle
    ax.set_ylim(bottom=0)

    # turn the triangle upside down
    if flip:
        ax.invert_yaxis()

    # remove axis lines
    ax.set_axis_off()

    return ax


def plot_windowed_ld(gn, pos, window_size, start_position=None,
                     stop_position=None, percentiles=(5, 95), ax=None,
                     median_plot_kwargs=None,
                     percentiles_plot_kwargs=None):
    """Plot average LD within non-overlapping genome windows.

    Parameters
    ----------

    gn : array_like
        A 2-dimensional array of shape (`n_variants`, `n_samples`) where each
        element is a genotype call coded as a single integer counting the
        number of non-reference alleles.
    pos : array_like
        A 1-dimensional array of genomic positions of variants.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    percentiles : sequence of integers, optional
        Percentiles to plot in addition to the median.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    median_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the median line.
    percentiles_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the percentiles.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check input array
    gn = np.asarray(gn)
    assert gn.ndim == 2

    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x, x//3))
        ax = fig.add_subplot(111)

    # determine bins
    if stop_position is None:
        stop_position = np.max(pos)
    if start_position is None:
        start_position = np.min(pos)
    bin_edges = np.arange(start_position, stop_position, window_size)
    n_bins = len(bin_edges) - 1

    # initialise plotting variables
    med = np.zeros((n_bins,), dtype='f4')
    if percentiles:
        pc = np.zeros((n_bins, len(percentiles)), dtype='f4')

    # iterate over bins
    for n in range(n_bins):

        # determine bin start and stop positions
        bin_start = bin_edges[n]
        bin_stop = bin_edges[n + 1]

        # map genome positions onto variant indices
        loc = anhima.loc.locate_interval(pos, bin_start, bin_stop)

        if loc.stop - loc.start > 0:

            # view genotypes for the current region
            gw = gn[loc, :]

            # calculate pairwise LD
            r_squared = pairwise_genotype_ld(gw)

            # convert to non-redundance form
            r_squared_nonredundant = distance.squareform(r_squared,
                                                         checks=False)

            # calculate median
            med[n] = np.median(r_squared_nonredundant)

            # calculate percentiles
            if percentiles:
                for i, p in enumerate(percentiles):
                    pc[n, i] = np.percentile(r_squared_nonredundant, p)

    # determine x coordinates for plotting, as bin centers
    x = (bin_edges[1:] + bin_edges[:-1]) / 2

    # plot median
    if median_plot_kwargs is None:
        median_plot_kwargs = dict()
    median_plot_kwargs.setdefault('linestyle', '-')
    median_plot_kwargs.setdefault('color', 'k')
    median_plot_kwargs.setdefault('linewidth', 2)
    ax.plot(x, med, **median_plot_kwargs)

    # plot percentiles
    if percentiles:
        if percentiles_plot_kwargs is None:
            percentiles_plot_kwargs = dict()
        percentiles_plot_kwargs.setdefault('linestyle', '--')
        percentiles_plot_kwargs.setdefault('color', 'k')
        percentiles_plot_kwargs.setdefault('linewidth', 1)
        for i, p in enumerate(percentiles):
            ax.plot(x, pc[:, i], **percentiles_plot_kwargs)

    # tidy up
    ax.set_xlabel('position')
    ax.set_ylabel('$r^2$', rotation=0)
    ax.grid(axis='y')

    return ax


def ld_prune_pairwise(gn, window_size=100, window_step=10, max_r_squared=.2):
    """Given a set of genotypes at biallelic variants, find a subset
    of the variants which are in approximate linkage equilibrium with
    each other.

    Parameters
    ----------

    gn : array_like
        A 2-dimensional array of shape (`n_variants`, `n_samples`) where each
        element is a genotype call coded as a single integer counting the
        number of non-reference alleles.
    window_size : int, optional
        The number of variants to work with at a time.
    window_step : int, optional
        The number of variants to shift the window by.
    max_r_squared : float, optional
        The maximum value of the genotype correlation coefficient, above which
        variants will be excluded.

    Returns
    -------

    included : ndarray, bool
        A boolean array of the same length as the number of variants,
        where a True value indicates the variant at the corresponding
        index is included, and a False value indicates the corresponding
        variant is excluded.

    Notes
    -----

    The algorithm is as follows. A window of `window_size` variants is
    taken from the beginning of the genotypes array. The genotype
    correlation coefficient is calculated between each pair of
    variants in the window. The first variant in the window is
    considered, and any other variants in the window with linkage
    above `max_r_squared` with respect to the first variant is
    excluded. The next non-excluded variant in the window is then
    considered, and so on. The window then shifts along by
    `window_step` variants, and the process is repeated.

    """

    # check input array
    gn = np.asarray(gn).astype('u1')
    assert gn.ndim == 2

    # use optimised implementation
    included = _ld_prune_pairwise_uint8(gn, window_size, window_step,
                                        max_r_squared)
    return included


def pairwise_ld_decay(r_squared, pos, step=1):
    """Compile data on linkage disequilibrium, separation (in number
    of variants), and physical distance between pairs of variants.

    Parameters
    ----------

    r_squared : array_like
        A square 2-dimensional array of squared correlation coefficients
        between pairs of variants.
    pos : array_like
        A 1-dimensional array of genomic positions of variants.
    step : int, optional
        When compiling the data, advance `step` variants.

    Returns
    -------

    cor : ndarray, float
        Each element in the array is the squared genotype correlation
        coefficient between a distinct pair of variants.
    sep : ndarray, int
        Each element in the array is the separation (in number of variants)
        between a distinct pair of variants.
    dist : ndarray, int
        Each element in the array is the physical distance between a distinct
        pair of variants.

    See Also
    --------

    windowed_ld_decay

    """

    # check inputs
    r_squared = np.asarray(r_squared)
    assert r_squared.ndim == 2
    assert r_squared.shape[0] == r_squared.shape[1]

    # determine the number of variants
    n_variants = r_squared.shape[0]

    # determine pairs of variants to use
    pairs = [(i, j)
             for i in range(0, n_variants, step)
             for j in range(i+1, n_variants)]

    # initialise output arrays
    cor = np.zeros((len(pairs),), dtype=np.float)
    sep = np.zeros((len(pairs),), dtype=np.int)
    dist = np.zeros((len(pairs),), dtype=np.int)

    # iterate over pairs
    for n, (i, j) in enumerate(pairs):
        cor[n] = r_squared[i, j]
        sep[n] = j - i
        dist[n] = np.abs(pos[j] - pos[i])

    return cor, sep, dist


def windowed_ld_decay(gn, pos, window_size, step=1):
    """Compile data on linkage disequilibrium, separation (in number
    of variants), and physical distance between pairs of variants.

    Parameters
    ----------

    gn : array_like
        A 2-dimensional array of shape (`n_variants`, `n_samples`) where each
        element is a genotype call coded as a single integer counting the
        number of non-reference alleles.
    pos : array_like
        A 1-dimensional array of genomic positions of variants.
    window_size : int, optional
        The number of variants to work with at a time.
    step : int, optional
        When compiling the data within each window, advance `step` variants.

    Returns
    -------

    cor : ndarray, float
        Each element in the array is the squared genotype correlation
        coefficient between a distinct pair of variants.
    sep : ndarray, int
        Each element in the array is the separation (in number of variants)
        between a distinct pair of variants.
    dist : ndarray, int
        Each element in the array is the physical distance between a distinct
        pair of variants.

    See Also
    --------

    pairwise_ld_decay

    Notes
    -----

    Similar to :func:`pairwise_ld_decay` except that not all pairs of
    variants are sampled to speed up computation and use less memory. Variants
    are divided into non-overlapping windows of size `window_size`. Genotype LD
    is calculated for all pairs within each window.

    """

    # check input array
    gn = np.asarray(gn)
    assert gn.ndim == 2

    # determine number of variants
    n_variants = gn.shape[0]

    # initialise output variables
    all_cor = list()
    all_sep = list()
    all_dist = list()

    # iterate over non-overlapping windows of variants
    for window_start in range(0, n_variants, window_size):

        # determine extent of current window
        window_stop = min(window_start + window_size, n_variants)

        # view genotypes for the current window
        gw = gn[window_start:window_stop, :]

        # calculate LD
        r_squared = pairwise_genotype_ld(gw)

        # compile data
        cor, sep, dist = pairwise_ld_decay(r_squared, pos, step=step)
        all_cor.append(cor)
        all_sep.append(sep)
        all_dist.append(dist)

    # concatenate results from each window
    all_cor = np.concatenate(all_cor)
    all_sep = np.concatenate(all_sep)
    all_dist = np.concatenate(all_dist)

    return all_cor, all_sep, all_dist


def plot_ld_decay_by_separation(cor, sep,
                                max_separation=100,
                                percentiles=(5, 95),
                                ax=None,
                                median_plot_kwargs=None,
                                percentiles_plot_kwargs=None):
    """Plot the decay of linkage disequilibrium with separation
    between variants.

    Parameters
    ----------

    cor : array_like
        A 1-dimensional array of squared correlation coefficients between
        pairs of variants.
    sep : array_like
        A 1-dimensional array of separations (in number of variants) between
        pairs of variants.
    max_separation : int, optional
        Maximum separation to consider.
    percentiles : sequence of integers, optional
        Percentiles to plot in addition to the median.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    median_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the median line.
    percentiles_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the percentiles.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check inputs
    cor = np.asarray(cor)
    sep = np.asarray(sep)
    assert cor.ndim == 1
    assert sep.ndim == 1
    assert cor.shape[0] == sep.shape[0]

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()

    # set up arrays for plotting
    cor_median = np.zeros((max_separation,), dtype='f4')
    if percentiles:
        cor_percentiles = np.zeros((max_separation, len(percentiles)),
                                   dtype='f4')

    # iterate over separations, compiling data
    for i in range(max_separation):

        # view correlations at the given separation
        c = cor[sep == i]

        # calculate median and percentiles
        if len(c) > 0:
            cor_median[i] = np.median(c)
            if percentiles:
                for n, p in enumerate(percentiles):
                    cor_percentiles[i, n] = np.percentile(c, p)

    # plot the median
    x = range(max_separation)
    y = cor_median
    if median_plot_kwargs is None:
        median_plot_kwargs = dict()
    median_plot_kwargs.setdefault('linestyle', '-')
    median_plot_kwargs.setdefault('color', 'k')
    median_plot_kwargs.setdefault('linewidth', 2)
    plt.plot(x, y, label='median', **median_plot_kwargs)

    # plot percentiles
    if percentiles:
        if percentiles_plot_kwargs is None:
            percentiles_plot_kwargs = dict()
        percentiles_plot_kwargs.setdefault('linestyle', '--')
        percentiles_plot_kwargs.setdefault('color', 'k')
        percentiles_plot_kwargs.setdefault('linewidth', 1)
        for n, p in enumerate(percentiles):
            y = cor_percentiles[:, n]
            plt.plot(x, y, label='%s%%' % p, **percentiles_plot_kwargs)

    # tidy up
    ax.set_xlim(left=1, right=max_separation)
    ax.set_ylim(0, 1)
    ax.set_xlabel('separation')
    ax.set_ylabel('$r^2$', rotation=0)
    ax.grid(axis='y')

    return ax


def plot_ld_decay_by_distance(cor, dist, bins,
                              percentiles=(5, 95),
                              ax=None,
                              median_plot_kwargs=None,
                              percentiles_plot_kwargs=None):
    """Plot the decay of linkage disequilibrium with physical distance
    between variants.

    Parameters
    ----------

    cor : array_like
        A 1-dimensional array of squared correlation coefficients between
        pairs of variants.
    dist : array_like
        A 1-dimensional array of physical distances between pairs of variants.
    bins : int or sequence of ints
        Number of bins or bin edges. Bins of distance to calculate LD within.
    percentiles : sequence of integers, optional
        Percentiles to plot in addition to the median.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    median_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the median line.
    percentiles_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the percentiles.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check inputs
    cor = np.asarray(cor)
    dist = np.asarray(dist)
    assert cor.ndim == 1
    assert dist.ndim == 1
    assert cor.shape[0] == dist.shape[0]

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()

    # calculate the median of correlation values within bins
    y, bin_edges, _ = stats.binned_statistic(dist, values=cor, bins=bins,
                                             statistic=np.median)

    # determine x axis variable as bin centers
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    # plot median
    if median_plot_kwargs is None:
        median_plot_kwargs = dict()
    median_plot_kwargs.setdefault('linestyle', '-')
    median_plot_kwargs.setdefault('color', 'k')
    median_plot_kwargs.setdefault('linewidth', 2)
    ax.plot(x, y, label='median', **median_plot_kwargs)

    # calculate and plot percentiles
    if percentiles:
        if percentiles_plot_kwargs is None:
            percentiles_plot_kwargs = dict()
        percentiles_plot_kwargs.setdefault('linestyle', '--')
        percentiles_plot_kwargs.setdefault('color', 'k')
        percentiles_plot_kwargs.setdefault('linewidth', 1)
        for p in percentiles:
            y, bin_edges, _ = stats.binned_statistic(
                dist,
                values=cor,
                bins=bins,
                statistic=lambda v: np.percentile(v, p)
            )
            ax.plot(x, y, label='%s%%' % p, **percentiles_plot_kwargs)

    # tidy up
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(0, 1)
    ax.set_xlabel('distance')
    ax.set_ylabel('$r^2$', rotation=0)
    ax.grid(axis='y')

    return ax
