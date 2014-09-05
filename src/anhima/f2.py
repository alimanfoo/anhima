"""
Doubleton sharing, a.k.a., analysis of f2 variants.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/f2.ipynb

"""


from __future__ import division, print_function, unicode_literals, \
    absolute_import


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# standard library dependencies
import itertools


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def count_shared_doubletons(subpops_ac):
    """Count subpopulation pairs sharing doubletons (where one allele is
    observed in each subpopulation).

    Parameters
    ----------

    subpops_ac : array_like, int
        An array of shape (n_variants, n_subpops) holding alternate allele
        counts for each subpopulation.

    Returns
    -------

    counts : ndarray, int or float
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.

    See Also
    --------

    normalise_doubleton_counts, plot_shared_doubletons,
    plot_total_doubletons, plot_f2_fig

    """

    # check input
    subpops_ac = np.asarray(subpops_ac)
    assert subpops_ac.ndim == 2

    # find doubletons in the total population
    is_total_doubleton = np.sum(subpops_ac, axis=1) == 2
    subpops_ac_doubletons = np.compress(is_total_doubleton, subpops_ac, axis=0)

    # count subpopulaton pairs sharing doubletons
    n_subpops = subpops_ac.shape[1]
    counts = np.zeros((n_subpops, n_subpops), dtype=np.int)
    for i in range(n_subpops):
        for j in range(i, n_subpops):
            if i == j:
                # count cases where doubleton is private to a subpopulation
                n = np.count_nonzero(subpops_ac_doubletons[:, i] == 2)
            else:
                # count cases where doubleton is shared between two
                # subpopulations
                n = np.count_nonzero((subpops_ac_doubletons[:, i] == 1)
                                     & (subpops_ac_doubletons[:, j] == 1))
            # fill both upper and lower triangles
            counts[i, j] = n
            counts[j, i] = n

    return counts


def normalise_doubleton_counts(counts, n_samples, ploidy=2):
    """Normalise doubleton counts by dividing by the number of distinct pairs of
    haplotypes in each population comparison.

    Parameters
    ----------

    counts : array_like, ints
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.
    n_samples : int or sequence of ints
        The number of samples in each sub-population.
    ploidy : int, optional
        The sample ploidy.

    Returns
    -------

    normed_counts : ndarray, float
        Normalised counts of shared doubletons.

    See Also
    --------

    count_shared_doubletons

    Notes
    -----

    This function corrects for the fact that there are fewer pairs of
    haplotypes when looking for doubletons within a single subpopulation of
    size n than there are when comparing two different subpopulations of size n.

    This function may also help to correct for the case where the number of
    samples from each subpopulation is not equal. However, note that if this
    is the case then there may still also be some bias in how doubletons have
    been ascertained.

    """

    # check inputs
    counts = np.asarray(counts)
    assert counts.ndim == 2
    assert counts.shape[0] == counts.shape[1]
    n_subpops = counts.shape[0]

    # deal with polymorphic input
    if isinstance(n_samples, int):
        n_samples = [n_samples] * n_subpops
    n_samples = np.asarray(n_samples)

    # normalise counts
    normed_counts = np.zeros((n_subpops, n_subpops), dtype=np.float)
    for i in range(n_subpops):
        for j in range(i, n_subpops):
            if i == j:
                # number of distinct pairs of haplotypes within a
                # subpopulation = number of haplotypes choose 2
                n_haplotypes = n_samples[i] * ploidy
                n_pairs = scipy.special.comb(n_haplotypes, 2)
            else:
                # number of distinct pairs of haplotypes between
                # subpopulations
                n_pairs = (n_samples[i] * ploidy) * (n_samples[j] * ploidy)
            # fill upper and lower triangles
            normed_counts[i, j] = counts[i, j] / n_pairs
            normed_counts[j, i] = counts[j, i] / n_pairs

    return normed_counts


def plot_shared_doubletons(counts, subpop_labels=None,
                           subpop_colors='bgrcmyk', axs=None,
                           figsize_factor=1, ylim=None, relative=False,
                           flip=False):
    """Plot counts of doubleton sharing between subpopulations as a bar chart.

    Parameters
    ----------

    counts : array_like, ints
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.
    subpop_labels : sequence of strings, optional
        Labels for the subpopulations.
    subpop_colors : sequence of colors, optional
        Colors for the subpopulations.
    axs : sequence of axes, optional
        The axes to use. If not provided, a new figure will be created.
    figsize_factor : float, optional
        Figure size in inches per subpopulation. Only used if `axs` is None.
    ylim : pair of ints or floats, optional
        Limits for the Y axes of all subplots.
    relative : bool, optional
        If True, normalise counts by dividing by the sum along each row.
    flip : bool, optional
        If True, invert the Y axis.

    Returns
    -------

    axs : sequence of axes
        The axes on which the plot was drawn.

    See Also
    --------

    count_shared_doubletons, plot_total_doubletons, plot_f2_fig

    """

    # check inputs
    counts = np.asarray(counts)
    assert counts.ndim == 2
    assert counts.shape[0] == counts.shape[1]
    n_subpops = counts.shape[0]

    # setup axes
    if axs is None:
        x = n_subpops * figsize_factor
        fig = plt.figure(figsize=(x, x))
        axs = [fig.add_subplot(n_subpops, 1, i+1) for i in range(n_subpops)]

    # ensure we have enough colors
    colors = list(itertools.islice(itertools.cycle(subpop_colors), n_subpops))

    # ensure we have subpopulation labels
    if subpop_labels is None:
        subpop_labels = range(n_subpops)

    # normalise by row
    if relative:
        # compute sum along each row
        row_sum = np.sum(counts, axis=1)
        # make sure we get the broadcasting right
        counts = counts / row_sum[:, np.newaxis]

    # determine global Y limits
    if ylim is None:
        ylim = (0, np.amax(counts))

    # plot main bar
    for i, color in zip(range(n_subpops), colors):

        if flip:
            # plot from the top down
            ax = axs[i]
        else:
            # plot from the bottom up
            ax = axs[n_subpops-i-1]

        # select data to plot
        data = counts[i, :]

        # make a bar
        ax.bar(range(n_subpops), data, width=1, color=colors)

        # tidy up
        ax.set_ylim(*ylim)
        ax.set_ylabel(subpop_labels[i], rotation=0, ha='right', va='center',
                      color=color)
        ax.tick_params(length=0)
        ax.set_yticks([])
        if (flip and i > 0) or (not flip and i < n_subpops - 1):
            ax.set_xticks([])
        else:
            ax.xaxis.tick_top()
            ax.set_xticks(np.arange(n_subpops) + .5)
            ax.set_xticklabels(subpop_labels, rotation=90)
        for s in 'top', 'left', 'right':
            ax.spines[s].set_visible(False)

    return axs


def plot_total_doubletons(counts, subpop_labels=None,
                          width=.8, orientation='vertical',
                          ax=None, bar_kwargs=None):
    """Plot total counts of doubletons per subpopulations as a bar chart.

    Parameters
    ----------

    counts : array_like, ints
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.
    subpop_labels : sequence of strings, optional
        Labels for the subpopulations.
    width : float, optional
        The relative width of each bar.
    orientation : {'vertical', 'horizontal'}
        The bar orientation.
    ax : axes, optional
        The axes on which to plot. If not provided, a new figure will be
        created.
    bar_kwargs : dict, optional
        Keyword arguments passed through to ax.bar().

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    See Also
    --------

    count_shared_doubletons, plot_shared_doubletons, plot_total_doubletons,
    plot_f2_fig

    """

    # check inputs
    counts = np.asarray(counts)
    assert counts.ndim == 2
    assert counts.shape[0] == counts.shape[1]
    n_subpops = counts.shape[0]

    # setup axes
    if ax is None:
        fig, ax = plt.subplots()

    # sum rows
    y = np.sum(counts, axis=1)

    # plot bar
    x = np.arange(n_subpops) + .5
    if bar_kwargs is None:
        bar_kwargs = dict()
    bar_kwargs.setdefault('color', 'gray')
    bar_kwargs.setdefault('align', 'center')
    bar_kwargs.setdefault('linewidth', 0)
    if orientation == 'vertical':
        ax.bar(x, y, width, **bar_kwargs)
    else:
        ax.barh(x, y, width, **bar_kwargs)

    # ensure we have subpopulation labels
    if subpop_labels is None:
        subpop_labels = range(n_subpops)

    # tidy up
    ax.tick_params(length=0)
    if orientation == 'vertical':
        for s in 'top', 'right', 'left':
            ax.spines[s].set_visible(False)
        ax.set_xticks(range(n_subpops))
        ax.set_xticklabels(subpop_labels, rotation=90)
        ax.set_yticks([])
        ax.xaxis.tick_bottom()
        ax.set_ylabel('doubletons')
        ax.set_xlim(0, n_subpops)
    else:
        for s in 'top', 'right', 'bottom':
            ax.spines[s].set_visible(False)
        ax.set_yticks(range(n_subpops))
        ax.set_yticklabels(subpop_labels, rotation=0)
        ax.set_xticks([])
        ax.yaxis.tick_left()
        ax.set_xlabel('doubletons')
        ax.xaxis.set_label_position('top')
        ax.set_ylim(0, n_subpops)

    return ax


def plot_f2_fig(counts, subpop_labels=None, subpop_colors='bgrcmyk', fig=None,
                figsize_factor=1, relative=False, normed=False, n_samples=None,
                ploidy=2):
    """Plot a combined figure of shared doubleton counts and total counts per
    subpopulation.

    Parameters
    ----------

    counts : array_like, ints
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.
    subpop_labels : sequence of strings, optional
        Labels for the subpopulations.
    subpop_colors : sequence of colors, optional
        Colors for the subpopulations.
    fig : figure, optional
        The figure to use. If not provided, a new figure will be created.
    figsize_factor : float, optional
        Figure size in inches per subpopulation. Only used if `fig` is None.
    relative : bool, optional
        If True, plot counts relative to the sum along each row.
    normed : bool, optional
        If True, normalise counts by dividing by the number of possible 
        pairs of haplotypes.
    n_samples : int or sequence of ints
        The number of samples in each sub-population. (Only applies if `normed` 
        is True.)
    ploidy : int, optional
        The sample ploidy. (Only applies if `normed` is True.)

    Returns
    -------

    fig : figure
        The figure on which the plot was drawn.

    See Also
    --------

    count_shared_doubletons, plot_shared_doubletons, plot_total_doubletons

    """

    # check inputs
    counts = np.asarray(counts)
    assert counts.ndim == 2
    assert counts.shape[0] == counts.shape[1]
    n_subpops = counts.shape[0]

    # setup figure
    if fig is None:
        width = (n_subpops + 1) * figsize_factor
        height = n_subpops * figsize_factor
        fig = plt.figure(figsize=(width, height))

    # plot main bar
    main_axs = [plt.subplot2grid((n_subpops, n_subpops+1), (i, 0), rowspan=1,
                                 colspan=n_subpops)
                for i in range(n_subpops)]
    if normed:
        main_counts = normalise_doubleton_counts(counts, 
                                                 n_samples=n_samples, 
                                                 ploidy=ploidy)
    else:
        main_counts = counts
    plot_shared_doubletons(main_counts,
                           subpop_labels=subpop_labels,
                           subpop_colors=subpop_colors,
                           relative=relative,
                           axs=main_axs)

    # plot totals bar
    tot_ax = plt.subplot2grid((n_subpops, n_subpops+1), (0, n_subpops),
                              rowspan=n_subpops, colspan=1)

    plot_total_doubletons(counts, subpop_labels=subpop_labels,
                          orientation='horizontal', ax=tot_ax)
    tot_ax.set_yticks([])

    return fig
