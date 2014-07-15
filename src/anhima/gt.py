"""
Utilities for working with genotype data.

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# standard library dependencies
import random


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


# internal dependencies
import anhima.loc


def simulate_biallelic_genotypes(n_variants, n_samples, af_dist,
                                 missingness=.1,
                                 ploidy=2):
    """Simulate genotypes at biallelic variants for a population in
    Hardy-Weinberg equilibrium

    Parameters
    ----------

    n_variants : int
        The number of variants.
    n_samples : int
        The number of samples.
    af_dist : frozen continuous random variable
        The distribution of allele frequencies.
    missingness : float, optional
        The fraction of missing genotype calls.
    ploidy : int, optional
        The sample ploidy.

    Returns
    -------

    genotypes : ndarray, int8
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = alternate allele).

    """

    # initialise output array
    genotypes = np.empty((n_variants, n_samples, ploidy), dtype='i1')

    # generate allele frequencies under the given distribution
    af = af_dist.rvs(n_variants)

    # iterate over variants
    for i, p in zip(range(n_variants), af):

        # randomly generate alleles under the given allele frequency
        alleles = scipy.stats.bernoulli.rvs(p, size=n_samples*ploidy)

        # reshape alleles as genotypes under the given ploidy
        genotypes[i] = alleles.reshape(n_samples, ploidy)

        # simulate some missingness
        missing_indices = random.sample(range(n_samples),
                                        int(missingness*n_samples))
        genotypes[i, missing_indices] = (-1,) * ploidy

    return genotypes


def is_called(genotypes):
    """Find non-missing genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_called : ndarray, bool
        An array where elements are True if the genotype call is non-missing.

    See Also
    --------

    is_missing, is_hom_ref, is_het_diploid, is_hom_alt_diploid

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # determine output array
    out = np.all(genotypes >= 0, axis=dim_ploidy)

    return out


def count_called(genotypes, axis=None):
    """Count non-missing genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    axis : int, optional
        The axis along which to count.

    Returns
    -------

    n : int or array
        If `axis` is None, returns the number of called (i.e., non-missing)
        genotypes. If `axis` is specified, returns the sum along the given
        `axis`.

    See Also
    --------
    is_called

    """

    n = np.sum(is_called(genotypes), axis=axis)
    return n


def is_missing(genotypes):
    """Find missing genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_missing: ndarray, bool
        An array where elements are True if the genotype call is missing.

    See Also
    --------

    is_called, is_hom_ref, is_het_diploid, is_hom_alt_diploid

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # determine output array
    out = np.any(genotypes < 0, axis=dim_ploidy)

    return out


def count_missing(genotypes, axis=None):
    """Count non-missing genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    axis : int, optional
        The axis along which to count.

    Returns
    -------

    n : int or array
        If `axis` is None, returns the number of missing genotypes. If `axis`
        is specified, returns the sum along the given `axis`.

    See Also
    --------
    is_missing

    """

    n = np.sum(is_missing(genotypes), axis=axis)
    return n


def is_hom_ref(genotypes):
    """Find homozygous reference genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_hom_ref : ndarray, bool
        An array where elements are True if the genotype call is homozygous
        reference.

    See Also
    --------
    is_called, is_missing, is_het_diploid, is_hom_alt_diploid

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # determine output array
    out = np.all(genotypes == 0, axis=dim_ploidy)

    return out


def count_hom_ref(genotypes, axis=None):
    """Count homozygous reference genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    axis : int, optional
        The axis along which to count.

    Returns
    -------

    n : int or array
        If `axis` is None, returns the number of homozygous
        reference genotypes. If `axis` is specified, returns the sum along
        the given `axis`.

    See Also
    --------
    is_hom_ref

    """

    n = np.sum(is_hom_ref(genotypes), axis=axis)
    return n


def is_het_diploid(genotypes):
    """Find heterozygous genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_het : ndarray, bool
        An array where elements are True if the genotype call is heterozygous.

    See Also
    --------

    is_called, is_hom_ref, is_hom_alt_diploid

    Notes
    -----

    **Not** applicable to polyploid genotype calls, diploids only.

    Applicable to multiallelic variants, although note that the return value
    will be true in any case where the two alleles in a genotype are
    different, e.g., (0, 1), (0, 2), (1, 2), etc.

    """

    # check input array has 2 or more dimensions
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # check diploid
    assert genotypes.shape[dim_ploidy] == 2

    # find hets
    allele1 = genotypes[..., 0]
    allele2 = genotypes[..., 1]
    is_het = allele1 != allele2

    return is_het


def count_het_diploid(genotypes, axis=None):
    """Count heterozygous genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    axis : int, optional
        The axis along which to count.

    Returns
    -------

    n : int or array
        If `axis` is None, returns the number of heterozygous genotypes. If
        `axis` is specified, returns the sum along the given `axis`.

    See Also
    --------
    is_het_diploid

    """

    n = np.sum(is_het_diploid(genotypes), axis=axis)
    return n


def is_hom_alt_diploid(genotypes):
    """Find homozygous non-reference genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_hom_alt : ndarray, bool
        An array where elements are True if the genotype call is homozygous
        non-reference.

    See Also
    --------

    is_called, is_hom_ref, is_het_diploid

    Notes
    -----

    **Not** applicable to polyploid genotype calls, diploids only.

    Applicable to multiallelic variants, although note that the return value
    will be true in any case where the two alleles in a genotype are
    the same and non-reference, e.g., (1, 1), (2, 2), etc.

    """

    # check input array has 2 or more dimensions
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # check diploid
    assert genotypes.shape[dim_ploidy] == 2

    # find homozygotes
    allele1 = genotypes[..., 0]
    allele2 = genotypes[..., 1]
    is_hom_alt = (allele1 > 0) & (allele1 == allele2)

    return is_hom_alt


def count_hom_alt_diploid(genotypes, axis=None):
    """Count homozygous non-reference genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    axis : int, optional
        The axis along which to count.

    Returns
    -------

    n : int or array
        If `axis` is None, returns the number of homozygous non-reference
        genotypes. If `axis` is specified, returns the sum along the given
        `axis`.

    See Also
    --------
    is_hom_alt_diploid

    """

    n = np.sum(is_hom_alt_diploid(genotypes), axis=axis)
    return n


def as_alleles(genotypes):
    """Reshape an array of genotypes as an array of alleles, collapsing the
    ploidy dimension.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    alleles : ndarray
        An array where the third (ploidy) dimension has been collapsed.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array
    assert genotypes.ndim == 3

    # reshape, preserving size of first dimension
    newshape = (genotypes.shape[0], -1)
    alleles = np.reshape(genotypes, newshape)

    return alleles


def as_n_alt(genotypes):
    """Transform an array of genotypes as the number of non-reference alleles.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    gn : ndarray, int8
        An array where each genotype is coded as a single integer counting
        the number of alternate alleles.

    See Also
    --------

    as_diploid_012

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, although this function simply
    counts the number of non-reference alleles, it makes no distinction
    between different non-reference alleles.

    Note that this function returns 0 for missing genotype calls **and** for
    homozygous reference genotype calls, because in both cases the number of
    non-reference alleles is zero.

    """

    # check input array
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # count number of alternate alleles
    gn = np.empty(genotypes.shape[:-1], dtype='i1')
    np.sum(genotypes > 0, axis=dim_ploidy, out=gn)

    return gn


def as_diploid_012(genotypes, fill=-1):
    """Transform an array of genotypes recoding homozygous reference calls a
    0, heterozygous calls as 1, homozygous non-reference calls as 2, and
    missing calls as -1.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or ('n_samples', 'ploidy'), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    fill : int, optional
        Default value for missing calls.

    Returns
    -------

    gn : ndarray, int8
        An array where each genotype is coded as a single integer as
        described above.

    See Also
    --------

    as_nalt

    Notes
    -----

    **Not** applicable to polyploid genotype calls, diploids only.

    Applicable to multiallelic variants, although note the following.
    All heterozygous genotypes, e.g., (0, 1), (0, 2), (1, 2), ..., will be
    coded as 1. All homozygous non-reference genotypes, e.g., (1, 1), (2, 2),
    ..., will be coded as 2.

    """

    # check input array
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # check diploid
    assert genotypes.shape[dim_ploidy] == 2

    # set up output array
    gn = np.empty(genotypes.shape[:-1], dtype='i1')
    gn.fill(fill)

    # determine genotypes
    gn[is_hom_ref(genotypes)] = 0
    gn[is_het_diploid(genotypes)] = 1
    gn[is_hom_alt_diploid(genotypes)] = 2

    return gn


def windowed_genotype_counts(pos, gn, t, window_size, start_position=None,
                             stop_position=None):
    """Count genotype calls of a given type for a single sample in
    non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like
        A 1-D array of genotypes for a single sample, where each genotype is
        coded as a single integer.
    t : int
        The genotype to count.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.

    Returns
    -------

    counts : ndarray, int
        Genotype counts for each window.
    bin_centers : ndarray, float
        The central position of each window.

    See Also
    --------

    as_diploid_012, as_n_alt, windowed_genotype_density, windowed_genotype_rate

    """

    # check input array
    assert gn.ndim == 1

    # find matching genotypes
    values = gn == t

    # computed binned statistic
    counts, bin_centers = anhima.loc.windowed_statistic(
        pos, values=values, statistic=b'sum', window_size=window_size,
        start_position=start_position, stop_position=stop_position
    )

    return counts, bin_centers


def windowed_genotype_density(pos, gn, t, window_size, start_position=None,
                              stop_position=None):
    """As :func:`windowed_genotype_counts` but returns per-base-pair density
    instead of counts.

    """

    counts, bin_centers = windowed_genotype_counts(pos, gn, t,
                                                   window_size=window_size,
                                                   start_position=start_position,
                                                   stop_position=stop_position)
    density = counts / window_size
    return density, bin_centers


def windowed_genotype_rate(pos, gn, t, window_size, start_position=None,
                           stop_position=None):
    """As :func:`windowed_genotype_counts` but returns the per-variant rate
    instead of counts."""

    variant_counts, _ = anhima.loc.windowed_variant_counts(
        pos, window_size, start_position=start_position,
        stop_position=stop_position
    )
    counts, bin_centers = windowed_genotype_counts(
        pos, gn, t, window_size=window_size, start_position=start_position,
        stop_position=stop_position
    )
    rate = counts / variant_counts
    return rate, bin_centers


def windowed_genotype_counts_plot(pos, gn, t, window_size, start_position=None,
                                  stop_position=None, ax=None,
                                  plot_kwargs=None):
    """Plots counts of genotype calls of a given type for a single sample in
    non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like
        A 1-D array of genotypes for a single sample, where each genotype is
        coded as a single integer.
    t : int
        The genotype to count.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    plot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.plot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # count genotypes
    y, x = windowed_genotype_counts(pos, gn, t, window_size,
                                    start_position=start_position,
                                    stop_position=stop_position)

    # plot data
    if plot_kwargs is None:
        plot_kwargs = dict()
    plot_kwargs.setdefault('linestyle', '-')
    plot_kwargs.setdefault('marker', None)
    ax.plot(x, y, label=t, **plot_kwargs)

    # tidy up
    ax.set_ylim(bottom=0)
    ax.set_xlabel('position')
    ax.set_ylabel('counts')
    if start_position is None:
        start_position = np.min(pos)
    if stop_position is None:
        stop_position = np.max(pos)
    ax.set_xlim(start_position, stop_position)

    return ax


def windowed_genotype_density_plot(pos, gn, t, window_size,
                                   start_position=None,
                                   stop_position=None, ax=None,
                                   plot_kwargs=None):
    """Plots per-base-pair density of genotype calls of a given type for a
    single sample in non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like
        A 1-D array of genotypes for a single sample, where each genotype is
        coded as a single integer.
    t : int
        The genotype to count.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    plot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.plot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # count genotypes
    y, x = windowed_genotype_density(pos, gn, t, window_size,
                                     start_position=start_position,
                                     stop_position=stop_position)

    # plot data
    if plot_kwargs is None:
        plot_kwargs = dict()
    plot_kwargs.setdefault('linestyle', '-')
    plot_kwargs.setdefault('marker', None)
    ax.plot(x, y, label=t, **plot_kwargs)

    # tidy up
    ax.set_ylim(bottom=0)
    ax.set_xlabel('position')
    ax.set_ylabel('density')
    if start_position is None:
        start_position = np.min(pos)
    if stop_position is None:
        stop_position = np.max(pos)
    ax.set_xlim(start_position, stop_position)

    return ax


def windowed_genotype_rate_plot(pos, gn, t, window_size,
                                start_position=None,
                                stop_position=None, ax=None,
                                plot_kwargs=None):
    """Plots per-variant rate of genotype calls of a given type for a
    single sample in non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like
        A 1-D array of genotypes for a single sample, where each genotype is
        coded as a single integer.
    t : int
        The genotype to count.
    window_size : int
        The size in base-pairs of the windows.
    start_position : int, optional
        The start position for the region over which to work.
    stop_position : int, optional
        The stop position for the region over which to work.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    plot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.plot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # count genotypes
    y, x = windowed_genotype_rate(pos, gn, t, window_size,
                                  start_position=start_position,
                                  stop_position=stop_position)

    # plot data
    if plot_kwargs is None:
        plot_kwargs = dict()
    plot_kwargs.setdefault('linestyle', '-')
    plot_kwargs.setdefault('marker', None)
    ax.plot(x, y, label=t, **plot_kwargs)

    # tidy up
    ax.set_ylim(bottom=0)
    ax.set_xlabel('position')
    ax.set_ylabel('per variant rate')
    if start_position is None:
        start_position = np.min(pos)
    if stop_position is None:
        stop_position = np.max(pos)
    ax.set_xlim(start_position, stop_position)

    return ax


# TODO plot genotype counts by sample


# TODO plot genotypes (colormesh)
# plot_discrete_calldata
# plot_continuous_calldata

