"""TODO

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt


def pairwise_distance(gn, metric='euclidean'):
    """TODO

    """

    # transpose genotypes as pdist expects (m, n) for m observations in an
    # n-dimensional space
    x = gn.T

    # compute the distance matrix
    dist = scipy.spatial.distance.pdist(x, metric=metric)

    # normalise by number of variants for easier comparison
    n_variants = gn.shape[0]
    dist = dist / n_variants

    # convert to square form for easy plotting
    dist_square = scipy.spatial.distance.squareform(dist)

    return dist_square


def pairwise_distance_plot(dist_square, metric='euclidean', labels=None,
                           colorbar=True, ax=None,
                           vmin=None, vmax=None, cmap='jet',
                           imshow_kwargs=None):
    """TODO

    """

    # set up axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

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
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
    if colorbar:
        plt.gcf().colorbar(im)

