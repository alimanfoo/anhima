"""
Genetic distance calculations.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/dist.ipynb

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt


def pairwise_distance(gn, metric='euclidean'):
    """Compute pairwise distance between samples.

    Parameters
    ----------

    gn : array_like
        A 2-dimensional array of shape (`n_variants`, `n_samples`) where each
        element is a genotype call coded as a single integer counting the
        number of non-reference alleles.
    metric : string or function, optional
        The distance metric to use. See documentation for the function
        :func:`scipy.spatial.distance.pdist` for a list of supported
        distance metrics.

    Returns
    -------

    dist : ndarray, float
        The distance matrix in compact form.
    dist_square : ndarray, float
        The distance matrix in square form.

    """

    # normalise inputs
    gn = np.asarray(gn)
    assert gn.ndim == 2

    # transpose genotypes as pdist expects (m, n) for m observations in an
    # n-dimensional space
    x = gn.T

    # compute the distance matrix
    dist = scipy.spatial.distance.pdist(x, metric=metric)

    # convert to square form for easy plotting
    dist_square = scipy.spatial.distance.squareform(dist)

    return dist, dist_square


def plot_pairwise_distance(dist_square, labels=None,
                           colorbar=True, ax=None,
                           vmin=None, vmax=None, cmap='jet',
                           imshow_kwargs=None):
    """Plot pairwise distances.

    Parameters
    ----------

    dist_square : array_like
        The distance matrix in square form.
    labels : sequence of strings, optional
        Sample labels for the axes.
    colorbar : bool, optional
        If True, add a colorbar to the current figure.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    vmin : float, optional
        The minimum distance value for normalisation.
    vmax : float, optional
        The maximum distance value for normalisation.
    cmap : string, optional
        The color map for the image.
    imshow_kwargs : dict-like, optional
        Additional keyword arguments passed through to `plt.imshow`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn

    """

    # normalise inputs
    dist_square = np.asarray(dist_square)
    assert dist_square.ndim == 2

    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))

    # set up normalisation
    dist = scipy.spatial.distance.squareform(dist_square)
    if vmin is None:
        vmin = np.min(dist)
    if vmax is None:
        vmax = np.max(dist)

    # plot as image
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    im = ax.imshow(dist_square, interpolation='none', cmap=cmap,
                   vmin=vmin, vmax=vmax, **imshow_kwargs)

    # tidy up
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
    if colorbar:
        plt.gcf().colorbar(im)

    return ax


# TODO add color option to distance plot instead of labels
# TODO clustering prior to distance plots
# TODO tree with distance plot