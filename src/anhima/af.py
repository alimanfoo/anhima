"""
Allele frequency calculations, site frequency spectra, doubleton sharing.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/af.ipynb

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# standard library dependencies
import itertools


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy


def _check_genotypes(genotypes):
    """
    Internal function to check the genotypes input argument meets
    expectations.

    """

    genotypes = np.asarray(genotypes)
    assert genotypes.ndim >= 2

    if genotypes.ndim == 2:
        # assume haploid, add ploidy dimension
        genotypes = genotypes[..., np.newaxis]

    return genotypes


def is_variant(genotypes):
    """Find variants with at least one non-reference allele observation.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_variant : ndarray, bool
        An array of shape (n_variants,) where an element is True if there
        are at least `min_ac` non-reference alleles found for the corresponding
        variant.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    out = np.sum(genotypes > 0, axis=(1, 2)) >= 1

    return out


def count_variant(genotypes):
    """Count variants with at least one non-reference allele observed.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    min_ac : int, optional
        The minimum number of non-reference alleles required to consider
        variant.

    Returns
    -------

    n : int
        The number of variants.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    return np.count_nonzero(is_variant(genotypes))


def is_non_variant(genotypes):
    """Find variants with no non-reference alleles.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_non_variant : ndarray, bool
        An array of shape (n_variants,) where an element is True if there
        are no non-reference alleles found for the corresponding variant.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    out = np.all(genotypes <= 0, axis=(1, 2))

    return out


def count_non_variant(genotypes):
    """Count variants with no non-reference alleles.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    n : int
        The number of variants.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    return np.count_nonzero(is_non_variant(genotypes))


def is_singleton(genotypes, allele=1):
    """Find variants with only a single instance of `allele` observed.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to find singletons of.

    Returns
    -------

    is_singleton : ndarray, bool
        An array of shape (n_variants,) where an element is True if there
        is a single instance of `allele` observed.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note this function checks for a
    specific `allele`.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    out = np.sum(genotypes == allele, axis=(1, 2)) == 1

    return out


def count_singletons(genotypes, allele=1):
    """Count variants with only a single instance of `allele` observed.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to find singletons of.

    Returns
    -------

    n : int
        The number of variants.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note this function checks for a
    specific `allele`.

    """

    return np.count_nonzero(is_singleton(genotypes, allele))


def is_doubleton(genotypes, allele=1):
    """Find variants with only two instances of `allele` observed.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to find doubletons of.

    Returns
    -------

    is_doubleton : ndarray, bool
        An array of shape (n_variants,) where an element is True if there
        are exactly two instances of `allele` observed.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note this function checks for a
    specific `allele`.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    out = np.sum(genotypes == allele, axis=(1, 2)) == 2

    return out


def count_doubletons(genotypes, allele=1):
    """Count variants with only two instances of `allele` observed.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to find doubletons of.

    Returns
    -------

    n : int
        The number of variants.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note this function checks for a
    specific `allele`.

    """

    return np.count_nonzero(is_doubleton(genotypes, allele))


def allele_number(genotypes):
    """Count the number of non-missing allele calls per variant.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    an : ndarray, int
        An array of shape (n_variants,) counting the total number of
        non-missing alleles observed.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    an = np.sum(genotypes >= 0, axis=(1, 2))

    return an


def allele_count(genotypes, allele=1):
    """Calculate number of observations of the given allele per variant.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to count.

    Returns
    -------

    ac : ndarray, int
        An array of shape (n_variants,) counting the number of
        times the given `allele` was observed.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note that this function
    calculates the frequency of a specific `allele`.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    ac = np.sum(genotypes == allele, axis=(1, 2))

    return ac


def allele_frequency(genotypes, allele=1):
    """Calculate frequency of the given allele per variant.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to calculate the frequency of.

    Returns
    -------

    an : ndarray, int
        An array of shape (n_variants,) counting the total number of
        non-missing alleles observed.
    ac : ndarray, int
        An array of shape (n_variants,) counting the number of
        times the given `allele` was observed.
    af : ndarray, float
        An array of shape (n_variants,) containing the allele frequency.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note that this function
    calculates the frequency of a specific `allele`.

    """

    # count non-missing alleles
    an = allele_number(genotypes)

    # count alleles
    ac = allele_count(genotypes, allele=allele)

    # calculate allele frequencies
    af = ac / an

    return an, ac, af


def max_allele(genotypes, axis=None):
    """
    Return the highest allele index.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    axis : int, optional
        The axis along which to determine the maximum. If not given, return 
        the highest overall.

    Returns
    -------

    n : int
        The value of the highest allele index present in the genotypes array.

    """
    
    return np.amax(genotypes, axis=axis)


def allele_counts(genotypes, alleles=None):
    """Calculate allele counts per variant.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    alleles : sequence of ints, optional
        The alleles to calculate the frequency of. If not specified, all 
        alleles will be counted.

    Returns
    -------

    ac : ndarray, int
        An array of shape (n_variants, n_alleles) counting the number of
        times the given `alleles` were observed.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input
    genotypes = _check_genotypes(genotypes)

    # determine number of variants
    n_variants = genotypes.shape[0]

    # if alleles not specified, count all alleles
    if alleles is None:
        m = max_allele(genotypes)
        alleles = range(m+1)

    # count alleles
    ac = np.zeros((n_variants, len(alleles)), dtype='i4')
    for i, a in enumerate(alleles):
        ac[:, i] = allele_count(genotypes, allele=a)

    return ac
    

def allele_frequencies(genotypes, alleles=None):
    """Calculate allele frequencies per variant.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    alleles : sequence of ints, optional
        The alleles to calculate the frequency of. If not specified, all 
        alleles will be counted.

    Returns
    -------

    an : ndarray, int
        An array of shape (n_variants,) counting the total number of
        non-missing alleles observed.
    ac : ndarray, int
        An array of shape (n_variants, n_alleles) counting the number of
        times the given `alleles` were observed.
    af : ndarray, float
        An array of shape (n_variants, n_alleles) containing the allele
        frequencies.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # count non-missing alleles
    an = allele_number(genotypes)

    # count alleles
    ac = allele_counts(genotypes, alleles=alleles)

    # calculate allele frequencies
    af = ac / an[..., np.newaxis]

    return an, ac, af


def allelism(genotypes):
    """Determine the number of distinct alleles found for each variant.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    n : ndarray, int
        An array of shape (n_variants,) where an element holds the allelism 
        of the corresponding variant.

    See Also
    --------

    max_allele

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """
    
    # check inputs
    genotypes = _check_genotypes(genotypes)

    # calculate allele counts
    ac = allele_counts(genotypes)

    # count alleles present
    n = np.sum(ac > 0, axis=1)

    return n


def is_non_segregating(genotypes, allele=None):
    """Find non-segregating variants (fixed for a single allele).

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        If given, find variants fixed with respect to `allele`. Otherwise
        find variants fixed for any allele.

    Returns
    -------

    is_non_segregating : ndarray, bool
        An array of shape (n_variants,) where an element is True if all
        genotype calls for the corresponding variant are either missing or
        equal to the same allele.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    if allele is None:

        # count distinct alleles
        n_alleles = allelism(genotypes)

        # find fixed variants
        out = n_alleles == 1

    else:

        # find fixed variants with respect to a specific allele
        out = np.all((genotypes < 0) | (genotypes == allele), axis=(1, 2))

    return out


def count_non_segregating(genotypes, allele=None):
    """Count non-segregating variants.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        If given, find variants fixed with respect to `allele`.

    Returns
    -------

    n : int
        The number of variants.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    return np.count_nonzero(is_non_segregating(genotypes, allele=allele))


def is_segregating(genotypes):
    """Find segregating variants (where more than one allele is observed).

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_segregating : ndarray, bool
        An array of shape (n_variants,) where an element is True if more 
        than one allele is found for the given variant.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # count distinct alleles
    n_alleles = allelism(genotypes)

    # find segregating variants
    out = n_alleles > 1

    return out


def count_segregating(genotypes):
    """Count segregating variants.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (n_variants, n_samples, ploidy) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    n : int
        The number of variants.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    return np.count_nonzero(is_segregating(genotypes))


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

    This function is useful where the ancestral and derived status of alleles is
    unknown.

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
                                                   statistic=b'mean')
        else:
            y, b, _ = scipy.stats.binned_statistic(np.arange(sfs.size),
                                                   values=sfs,
                                                   bins=bins,
                                                   statistic=b'mean')
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

    counts : ndarray, int
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.

    See Also
    --------

    plot_shared_doubletons_heatmap, plot_shared_doubletons_bar

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
            counts[i, j] = n
            counts[j, i] = n

    return counts


def plot_shared_doubletons_heatmap(counts, subpop_labels=None, ax=None,
                                   color_diagonal=False,
                                   pcolormesh_kwargs=None, text_kwargs=None):
    """Plot counts of doubleton sharing between subpopulations as a heatmap.

    Parameters
    ----------

    counts : array_like, ints
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.
    subpop_labels : sequence of strings, optional
        Labels for the subpopulations.
    color_diagonal : bool, optional
        If True, color the diagonal, otherwise leave it white and use it to
        annotate the counts.
    ax : axes, optional
        The axes on which to plot. If not provided, a new figure will be
        created.
    pcolormesh_kwargs : dict, optional
        Additional keyword arguments passed through to ax.pcolormesh().
    text_kwargs : dict, optional
        Additional keyword arguments passed through when annotating the axes
        with the counts.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    See Also
    --------

    count_shared_doubletons, plot_shared_doubletons_bar

    """

    # check inputs
    counts = np.asarray(counts)
    assert counts.ndim == 2
    assert counts.shape[0] == counts.shape[1]
    n_subpops = counts.shape[0]

    # whiten the upper triangle so we can plot numbers
    if color_diagonal:
        k = 0
    else:
        k = 1
    counts_triu = np.triu(counts, k)

    # setup axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))

    # plot a colormesh
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = dict()
    pcolormesh_kwargs.setdefault('cmap', 'Greys')
    pcolormesh_kwargs.setdefault('edgecolor', 'None')
    ax.pcolormesh(counts_triu, **pcolormesh_kwargs)

    # add counts as text
    if text_kwargs is None:
        text_kwargs = dict()
    text_kwargs.setdefault('color', 'k')
    text_kwargs.setdefault('ha', 'center')
    text_kwargs.setdefault('va', 'center')
    for i in range(n_subpops):
        for j in range(i, n_subpops):
            if i != j or not color_diagonal:
                ax.text(i+.5, j+.5, counts[i, j], **text_kwargs)

    # tidy up
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(n_subpops) + .5)
    ax.set_yticks(np.arange(n_subpops) + .5)
    if subpop_labels is None:
        subpop_labels = range(n_subpops)
    ax.set_xticklabels(subpop_labels, rotation=90)
    ax.set_yticklabels(subpop_labels, rotation=0)
    for s in 'top', 'right', 'bottom', 'left':
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

    return ax


def plot_shared_doubletons_bar(counts, figsize_factor=1, subpop_labels=None,
                               subpop_colors='bgrcmyk'):
    """Plot counts of doubleton sharing between subpopulations as a bar chart.

    Parameters
    ----------

    counts : array_like, ints
        A square matrix of shape (n_subpops, n_subpops) where the array
        element at index (i, j) holds the count of shared doubletons between
        the ith and jth subpopulations.
    figsize_factor : float, optional
        Figure size in inches per subpopulation.
    subpop_labels : sequence of strings, optional
        Labels for the subpopulations.
    subpop_colors : sequence of colors
        Colors for the subpopulations.

    Returns
    -------

    fig : figure
        The figure on which the plot was drawn.

    See Also
    --------

    count_shared_doubletons, plot_shared_doubletons_heatmap

    """

    # check inputs
    counts = np.asarray(counts)
    assert counts.ndim == 2
    assert counts.shape[0] == counts.shape[1]
    n_subpops = counts.shape[0]

    # setup figure
    height = n_subpops * figsize_factor
    width = (n_subpops + 1) * figsize_factor
    fig = plt.figure(figsize=(width, height))

    # ensure we have enough colors
    colors = list(itertools.islice(itertools.cycle(subpop_colors), n_subpops))

    # ensure we have subpopulation labels
    if subpop_labels is None:
        subpop_labels = range(n_subpops)

    # plot main bar
    for i, color in zip(range(n_subpops), colors):

        # select axes
        # N.B., plot from the bottom upwards
        ax = plt.subplot2grid((n_subpops, n_subpops+1),
                              (n_subpops - i - 1, 0),
                              rowspan=1,
                              colspan=n_subpops)

        # select data to plot
        data = counts[i, :]

        # make a bar
        ax.bar(range(n_subpops), data, width=1, color=colors)

        # tidy up
        ax.set_ylabel(subpop_labels[i], rotation=0, ha='right', va='center',
                      color=color)
        ax.tick_params(length=0)
        ax.set_yticks([])
        if i < n_subpops-1:
            ax.set_xticks([])
        else:
            ax.xaxis.tick_top()
            ax.set_xticks(np.arange(n_subpops) + .5)
            ax.set_xticklabels(subpop_labels, rotation=90)
        for s in 'top', 'left', 'right':
            ax.spines[s].set_visible(False)

    # plot marginal bar
    ax = plt.subplot2grid((n_subpops, n_subpops+1),
                          (0, n_subpops),
                          rowspan=n_subpops,
                          colspan=1)
    data = np.sum(counts, axis=1)
    ax.barh(range(n_subpops, 0, -1), data, color='gray', height=.8,
            align='center', lw=0)

    # tidy up
    for s in 'top', 'right', 'bottom':
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('doubletons', rotation=0, ha='center')
    ax.xaxis.set_label_position('top')

    return fig
