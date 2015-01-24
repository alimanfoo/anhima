# -*- coding: utf-8 -*-
"""
Utility functions for multidimensional scaling.

R must be installed, and the Python package ``rpy2`` must be installed, e.g.::

    $ apt-get install r-base
    $ pip install rpy2

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/mds.ipynb

"""  # noqa


from __future__ import division, print_function, absolute_import


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold


_r_initialised = False
rpy2 = None
ro = None
r = None


def _init_r():
    """Private function to initialise R, only executed when needed.

    """

    global _r_initialised
    global rpy2
    global ro
    global r

    if not _r_initialised:
        import rpy2  # noqa
        import rpy2.robjects as ro  # noqa
        from rpy2.robjects import r  # noqa
        import rpy2.robjects.numpy2ri as numpy2ri
        numpy2ri.activate()
        _r_initialised = True


def smacof(dist_square, **kwargs):
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

    coords : ndarray, shape (n_samples, n_components)
        An array whose rows give the coordinates of the points chosen to
        represent the dissimilarities.

    See Also
    --------

    anhima.dist.pairwise_distance, anhima.mds.classical, sklearn.manifold.MDS,
    anhima.pca.pca

    """

    # normalise inputs
    dist_square = np.asarray(dist_square)
    assert dist_square.ndim == 2
    assert dist_square.shape[0] == dist_square.shape[1]

    # setup model
    model = sklearn.manifold.MDS(dissimilarity='precomputed',
                                 **kwargs)

    # fit model and get transformed coordinates
    coords = model.fit(dist_square).embedding_

    return coords


def classical(dist_square, k=2):
    """
    Classical multidimensional scaling via the R ``cmdscale`` function.

    Parameters
    ----------

    dist_square : array_like, shape (n_samples, n_samples)
        A distance matrix in square form.
    k : integer, optional
        The maximum dimension of the space which the data are to be represented
        in; must be in {1, 2, ..., n-1}.

    Returns
    -------

    coords : ndarray, shape (n_samples, k)
        An array whose rows give the coordinates of the points chosen to
        represent the dissimilarities.

    See Also
    --------

    anhima.dist.pairwise_distance, anhima.mds.smacof, anhima.pca.pca

    """

    # setup R
    _init_r()

    # normalise inputs
    dist_square = np.asarray(dist_square)
    assert dist_square.ndim == 2
    assert dist_square.shape[0] == dist_square.shape[1]

    # convert distance matrix to R
    m = ro.vectors.Matrix(dist_square)

    # apply MDS
    coords = r['cmdscale'](m, k=k)

    return np.asarray(coords)


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

    See Also
    --------

    anhima.mds.smacof, anhima.mds.classical

    """

    # normalise inputs
    coords = np.asarray(coords)
    assert coords.ndim == 2

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
