"""
Utility functions for multidimensional scaling.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/mds.ipynb

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt


def mds(dist_square, **kwargs):
    """
    Multidimensional scaling using the SMACOF (Scaling by Majorizing a
    Complicated Function) algorithm.

    Parameters
    ----------

    dist_square : array_like, shape (n_samples, n_samples)
        A distance matrix in square form.
    kwargs : additional keyword arguments
        Additional keyword arguments are passed through to
        :func:`sklearn.manifold.MDS`.

    Returns
    -------

    model : ``sklearn.manifold.MDS``
        The fitted model.
    coords : ndarray, shape (n_samples, n_components)
        The result of fitting the model with `dist_square` and applying
        dimensionality reduction.

    See Also
    --------

    sklearn.manifold.MDS, anhima.pca.pca

    """

    # dependencies
    import sklearn.manifold

    # setup model
    model = sklearn.manifold.MDS(dissimilarity=b'precomputed',
                                 **kwargs)

    # fit model and get transformed coordinates
    coords = model.fit(dist_square).embedding_

    return model, coords


def plot_coords(coords, dimx=1, dimy=2, ax=None, colors='b', sizes=20,
                labels=None, scatter_kwargs=None, annotate_kwargs=None):
    """Scatter plot of transformed coordinates from multidimensional scaling.

    Parameters
    ----------

    coords : ndarray, shape (`n_samples`, `n_components`)
        The transformed coordinates.
    dimx : int, optional
        The dimension to plot on the X axis. N.B., this is
        one-based, so `1` is the first dimension, `2` is the second
        dimension, etc.
    dimy : int, optional
        The dimension to plot on the Y axis. N.B., this is
        one-based, so `1` is the first dimension, `2` is the second
        dimension, etc.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    colors : color or sequence of color, optional
        Can be a single color format string, or a sequence of color
        specifications of length `n_samples`.
    sizes : scalar or array_like, shape (`n_samples`), optional
        Size in points^2.
    labels : sequence of strings
        If provided, will be used to label points in the plot.
    scatter_kwargs : dict-like
        Additional keyword arguments passed through to `plt.scatter`.
    annotate_kwargs : dict-like
        Additional keyword arguments passed through to `plt.annotate` when
        labelling points.


    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))

    # obtain X and Y data, N.B., `pcx` and `pcy` are 1-based
    x = coords[:, dimx-1]
    y = coords[:, dimy-1]

    # plot points
    if scatter_kwargs is None:
        scatter_kwargs = dict()
    ax.scatter(x, y, c=colors, s=sizes, **scatter_kwargs)

    # label points
    if labels is not None:
        if annotate_kwargs is None:
            annotate_kwargs = dict()
        annotate_kwargs.setdefault('xycoords', 'data')
        annotate_kwargs.setdefault('xytext', (3, 3))
        annotate_kwargs.setdefault('textcoords', 'offset points')
        for l, lx, ly in zip(labels, x, y):
            if l is not None:
                ax.annotate(str(l), xy=(lx, ly), **annotate_kwargs)

    # tidy up
    ax.set_xlabel('dimension %s' % dimx)
    ax.set_ylabel('dimension %s' % dimy)

    return ax


# TODO cmdscale via R