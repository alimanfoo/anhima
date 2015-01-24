# -*- coding: utf-8 -*-
"""
Site frequency spectra.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/sf.ipynb

"""  # noqa


from __future__ import division, print_function, absolute_import


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def site_frequency_spectrum(derived_ac):
    """Calculate the site frequency spectrum, given derived allele counts for a
    set of biallelic variant sites.

    Parameters
    ----------

    derived_ac : array_like, int
        A 1-dimensional array of shape (n_variants,) where each array
        element holds the count of derived alleles found for a single variant
        across some set of samples.

    Returns
    -------

    sfs : ndarray, int
        An array of integers where the value of the kth element is the
        number of variant sites with k derived alleles.

    See Also
    --------

    site_frequency_spectrum_scaled, site_frequency_spectrum_folded,
    site_frequency_spectrum_folded_scaled, plot_site_frequency_spectrum

    """

    # check input
    derived_ac = np.asarray(derived_ac)
    assert derived_ac.ndim == 1

    # calculate frequency spectrum
    sfs = np.bincount(derived_ac)

    return sfs


def site_frequency_spectrum_folded(biallelic_ac):
    """Calculate the folded site frequency spectrum, given reference and
    alternate allele counts for a set of biallelic variants.

    Parameters
    ----------

    biallelic_ac : array_like int
        A 2-dimensional array of shape (n_variants, 2), where each row
        holds the reference and alternate allele counts for a single
        biallelic variant across some set of samples.

    Returns
    -------

    sfs_folded : ndarray, int
        An array of integers where the value of the kth element is the
        number of variant sites with k observations of the minor allele.

    See Also
    --------

    site_frequency_spectrum, site_frequency_spectrum_scaled,
    site_frequency_spectrum_folded_scaled, plot_site_frequency_spectrum

    """

    # check input
    biallelic_ac = np.asarray(biallelic_ac)
    assert biallelic_ac.ndim == 2
    assert biallelic_ac.shape[1] == 2

    # calculate minor allele counts
    minor_ac = np.amin(biallelic_ac, axis=1)

    # calculate frequency spectrum
    sfs_folded = np.bincount(minor_ac)

    return sfs_folded


def site_frequency_spectrum_scaled(derived_ac):
    """Calculate the site frequency spectrum, scaled such that a constant value
    is expected across the spectrum for neutral variation and a population at
    constant size.

    Parameters
    ----------

    derived_ac : array_like, int
        A 1-dimensional array of shape (n_variants,) where each array
        element holds the count of derived alleles found for a single variant
        across some set of samples.

    Returns
    -------

    sfs_scaled : ndarray, int
        An array of integers where the value of the kth element is the
        number of variant sites with k derived alleles, multiplied by k.

    Notes
    -----

    Under neutrality and constant population size, site frequency
    is expected to be constant across the spectrum, and to approximate
    the value of the population-scaled mutation rate theta.

    See Also
    --------

    site_frequency_spectrum, site_frequency_spectrum_folded,
    site_frequency_spectrum_folded_scaled, plot_site_frequency_spectrum

    """

    # calculate frequency spectrum
    sfs = site_frequency_spectrum(derived_ac)

    # scaling
    k = np.arange(sfs.size)
    sfs_scaled = sfs * k

    return sfs_scaled


def site_frequency_spectrum_folded_scaled(biallelic_ac, m=None):
    """Calculate the folded site frequency spectrum, scaled such that a
    constant value is expected across the spectrum for neutral variation and
    a population at constant size.

    Parameters
    ----------

    biallelic_ac : array_like int
        A 2-dimensional array of shape (n_variants, 2), where each row
        holds the reference and alternate allele counts for a single
        biallelic variant across some set of samples.
    m : int, optional
        The total number of alleles observed at each variant site. Equal to
        the number of samples multiplied by the ploidy. If not provided,
        will be inferred to be the maximum value of the sum of reference and
        alternate allele counts present in `biallelic_ac`.

    Returns
    -------

    sfs_folded_scaled : ndarray, int
        An array of integers where the value of the kth element is the
        number of variant sites with k observations of the minor allele,
        multiplied by the scaling factor (k * (m - k) / m).

    Notes
    -----

    Under neutrality and constant population size, site frequency
    is expected to be constant across the spectrum, and to approximate
    the value of the population-scaled mutation rate theta.

    This function is useful where the ancestral and derived status of alleles
    is unknown.

    See Also
    --------

    site_frequency_spectrum, site_frequency_spectrum_scaled,
    site_frequency_spectrum_folded, plot_site_frequency_spectrum

    """

    # calculate the folded site frequency spectrum
    sfs_folded = site_frequency_spectrum_folded(biallelic_ac)

    # determine the total number of alleles per variant
    if m is None:
        m = np.amax(np.sum(biallelic_ac, axis=1))

    # scaling
    k = np.arange(sfs_folded.size)
    sfs_folded_scaled = sfs_folded * k * (m - k) / m

    return sfs_folded_scaled


def plot_site_frequency_spectrum(sfs, bins=None, m=None,
                                 clip_endpoints=True, ax=None, label=None,
                                 plot_kwargs=None):
    """Plot a site frequency spectrum.

    Parameters
    ----------

    sfs : array_like, int
        Site frequency spectrum. Can be folded or unfolded, scaled or
        unscaled.
    bins : int or sequence of ints, optional
        Number of bins or bin edges to aggregate frequencies. If not given,
        no binning will be applied.
    m : int, optional
        The total number of alleles observed at each variant site. Equal to
        the number of samples multiplied by the ploidy. If given, will be
        used to scale the X axis as allele frequency instead of allele count.
        used to scale the X axis as allele frequency instead of allele count.
    clip_endpoints : bool, optional
        If True, remove the first and last values from the site frequency
        spectrum.
    ax : axes, optional
        The axes on which to plot. If not given, a new figure will be created.
    label : string, optional
        Label for this data series.
    plot_kwargs : dict, optional
        Passed through to ax.plot().

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    See Also
    --------

    site_frequency_spectrum, site_frequency_spectrum_folded,
    site_frequency_spectrum_scaled, site_frequency_spectrum_folded_scaled

    """

    if ax is None:
        fig, ax = plt.subplots()

    if bins is None:
        # no binning
        if clip_endpoints:
            x = np.arange(1, sfs.size-1)
            y = sfs[1:-1]
        else:
            x = np.arange(sfs.size)
            y = sfs

    else:
        # bin the frequencies
        if clip_endpoints:
            y, b, _ = scipy.stats.binned_statistic(np.arange(1, sfs.size-1),
                                                   values=sfs[1:-1],
                                                   bins=bins,
                                                   statistic='mean')
        else:
            y, b, _ = scipy.stats.binned_statistic(np.arange(sfs.size),
                                                   values=sfs,
                                                   bins=bins,
                                                   statistic='mean')
        # use bin midpoints for plotting
        x = (b[:-1] + b[1:]) / 2

    if m is not None:
        # convert allele counts to allele frequencies
        x = x / m
        ax.set_xlabel('allele frequency')
    else:
        ax.set_xlabel('allele count')

    # plotting
    if plot_kwargs is None:
        plot_kwargs = dict()
    ax.plot(x, y, label=label, **plot_kwargs)

    # tidy up
    ax.set_ylabel('site frequency')

    return ax
