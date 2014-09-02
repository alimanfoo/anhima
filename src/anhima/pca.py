"""
Utility functions for running principal components analysis and plotting the
results.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/pca.ipynb

"""


from __future__ import division, print_function, unicode_literals, \
    absolute_import


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition


def pca(gn, n_components=10, whiten=False):
    """Perform a principal components analysis of genotypes, treating each
    variant as a feature.

    Parameters
    ----------

    gn : array_like, shape (`n_variants`, `n_samples`)
        A 2-dimensional array where each element is a genotype call coded as
        a single integer counting the number of non-reference alleles.
    n_components : int, None or string
        Number of components to keep. If `n_components` is None all
        components are kept: ``n_components == min(n_samples, n_features)``. If
        `n_components` == 'mle', Minka's MLE is used to guess the dimension. If
        0 < `n_components` < 1, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.
    whiten : bool
        When True (False by default) the components vectors are divided by
        n_samples times singular values to ensure uncorrelated outputs with unit
        component-wise variances.

    Returns
    -------

    model : ``sklearn.decomposition.PCA``
        The fitted model.
    coords : ndarray, shape (`n_samples`, `n_components`)
        The result of fitting the model with `genotypes` and applying
        dimensionality reduction to `genotypes`.

    See Also
    --------

    sklearn.decomposition.PCA, anhima.ld.ld_prune_pairwise

    Notes
    -----

    The :func:`anhima.ld.ld_prune_pairwise` can be used to obtain a set of
    variants in approximate linkage equilibrium prior to running PCA.

    """

    # check inputs
    gn = np.asarray(gn)
    assert gn.ndim == 2

    # transpose because sklearn expects data as (n_samples, n_features)
    m = gn.T

    # set up PCA
    model = sklearn.decomposition.PCA(n_components=n_components, whiten=whiten,
                                      copy=True)

    # fit the model and apply dimensionality reduction
    coords = model.fit_transform(m)

    return model, coords


def plot_coords(model, coords, pcx=1, pcy=2, ax=None, colors='b', sizes=20,
                labels=None, scatter_kwargs=None, annotate_kwargs=None):
    """Scatter plot of transformed coordinates from principal components
    analysis.

    Parameters
    ----------

    model : ``sklearn.decomposition.PCA``
        The fitted model.
    coords : ndarray, shape (`n_samples`, `n_components`)
        The transformed coordinates.
    pcx : int, optional
        The principal component to plot on the X axis. N.B., this is
        one-based, so `1` is the first principal component, `2` is the second
        component, etc.
    pcy : int, optional
        The principal component to plot on the Y axis. N.B., this is
        one-based, so `1` is the first principal component, `2` is the second
        component, etc.
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

    # check inputs
    coords = np.asarray(coords)
    assert coords.ndim == 2

    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))

    # obtain X and Y data, N.B., `pcx` and `pcy` are 1-based
    x = coords[:, pcx-1]
    y = coords[:, pcy-1]

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
    ax.set_xlabel('PC%s (%.2f%%)' %
                  (pcx, model.explained_variance_ratio_[pcx-1] * 100))
    ax.set_ylabel('PC%s (%.2f%%)' %
                  (pcy, model.explained_variance_ratio_[pcy-1] * 100))

    return ax


def plot_variance_explained(model, bar_kwargs=None, ax=None):
    """

    Parameters
    ----------

    model : ``sklearn.decomposition.PCA``
        The fitted model.
    bar_kwargs : dict-like, optional
        Additional keyword arguments passed through to ``ax.bar()``.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()

    # how many components are available

    # coordinates for bar
    y = model.explained_variance_ratio_ * 100  # express as percent
    n = len(y)
    x = np.arange(n)

    # plot bar
    if bar_kwargs is None:
        bar_kwargs = dict()
    bar_kwargs.setdefault('width', 1)
    ax.bar(x, y, **bar_kwargs)

    # tidy up
    ax.set_xticks(x+.5)
    ax.set_xticklabels(range(1, n+1))
    ax.set_xlabel('principal component')
    ax.set_ylabel('% variance explained')

    return ax


def plot_loadings(model, pc=1, pos=None, plot_kwargs=None, ax=None):
    """
    Plot loadings for the given principal component.

    Parameters
    ----------

    model : ``sklearn.decomposition.PCA``
        The fitted model.
    pc : int, optional
        The principal component to plot loadings for. N.B., this is
        one-based, so `1` is the first principal component, `2` is the second
        component, etc.
    pos : array_like, int, optional
        An array of variant positions to use for the X axis, If not given,
        variant index will be used for the X axis.
    plot_kwargs : dict-like, optional
        Additional keyword arguments passed through to ``ax.plot()``.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x, x//3))
        ax = fig.add_subplot(111)

    # obtain loadings
    y = model.components_[pc-1, :]

    # plot them
    if plot_kwargs is None:
        plot_kwargs = dict()
    if pos is not None:
        assert len(y) == len(pos)
        ax.plot(pos, y, **plot_kwargs)
        ax.set_xlabel('position')
        ax.set_xlim(min(pos), max(pos))
    else:
        ax.plot(y, **plot_kwargs)
        ax.set_xlabel('variant')
        ax.set_xlim(0, len(y))

    return ax
