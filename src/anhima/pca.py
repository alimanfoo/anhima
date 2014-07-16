"""Utilities for principal components analysis of genotypes.

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import matplotlib.pyplot as plt
import sklearn.decomposition


def pca(gn, n_components=None, whiten=False):
    """Perform a principal components analysis of the genotypes, treating each
    variant as a feature.

    Parameters
    ----------

    gn : array_like, shape (`n_variants`, `n_samples`)
        A 2-dimensional array where each element is a genotype call coded as
        a single integer counting the number of non-reference alleles.
    n_components : int, None or string
        Number of components to keep. If `n_components` is not set all
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
    trans : ndarray, shape (`n_samples`, `n_components`)
        The result of fitting the model with `genotypes` and applying
        dimensionality reduction to `genotypes`.

    See Also
    --------

    anhima.ld.ld_prune_pairwise, sklearn.decomposition.PCA

    Notes
    -----

    The :func:`anhima.ld.ld_prune_pairwise` can be used to obtain a set of
    variants in approximate linkage equilibrium prior to running PCA.

    """

    # set up PCA
    model = sklearn.decomposition.PCA(n_components=n_components, whiten=whiten,
                                      copy=True)

    # transpose because sklearn expects data as (n_samples, n_features)
    x = gn.T

    # fit the model and apply dimensionality reduction
    trans = model.fit_transform(x)

    return model, trans


def plot_components(model, trans, pcx=1, pcy=2, ax=None, colors='b', sizes=20,
                    labels=None, scatter_kwargs=None, annotate_kwargs=None):
    """Scatter plot of principal components.

    Parameters
    ----------

    model : ``sklearn.decomposition.PCA``
        The fitted model.
    trans : ndarray, shape (`n_samples`, `n_components`)
        The result of fitting the model with `genotypes` and applying
        dimensionality reduction to `genotypes`.
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

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()

    # obtain X and Y data, N.B., `pcx` and `pcy` are 1-based
    x = trans[:, pcx-1]
    y = trans[:, pcy-1]

    # plot points
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
    ax.set_xlabel('PC%s (%.2f%%)' %
                  (pcy, model.explained_variance_ratio_[pcy-1] * 100))

    return ax


# TODO loadings plot
# TODO variance explained plot