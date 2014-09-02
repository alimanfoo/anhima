"""
Utility functions for working with genotype data.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/gt.ipynb

"""


from __future__ import division, print_function, unicode_literals, \
    absolute_import


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# internal dependencies
import anhima.loc


def is_called(genotypes):
    """Find non-missing genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_called : ndarray, bool
        An array where elements are True if the genotype call is non-missing.

    See Also
    --------

    is_missing, is_hom_ref, is_het, is_hom_alt

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    genotypes = np.asarray(genotypes)
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

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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

    # count genotypes
    n = np.sum(is_called(genotypes), axis=axis)

    return n


def is_missing(genotypes):
    """Find missing genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_missing: ndarray, bool
        An array where elements are True if the genotype call is missing.

    See Also
    --------

    is_called, is_hom_ref, is_het, is_hom_alt

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    genotypes = np.asarray(genotypes)
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

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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

    # count genotypes
    n = np.sum(is_missing(genotypes), axis=axis)

    return n


def is_hom(genotypes):
    """Find homozygous genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_hom : ndarray, bool
        An array where elements are True if the genotype call is homozygous.

    See Also
    --------

    is_called, is_missing, is_hom_ref, is_hom_alt

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # find hets
    allele1 = genotypes[..., 0, np.newaxis]
    other_alleles = genotypes[..., 1:]
    is_hom = np.any((allele1 >= 0) & (allele1 == other_alleles),
                    axis=dim_ploidy)

    return is_hom


def count_hom(genotypes, axis=None):
    """Count homozygous genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    axis : int, optional
        The axis along which to count.

    Returns
    -------

    n : int or array
        If `axis` is None, returns the number of homozygous genotypes. If
        `axis` is specified, returns the sum along the given `axis`.

    See Also
    --------
    is_hom

    """

    # count genotypes
    n = np.sum(is_hom(genotypes), axis=axis)

    return n


def is_het(genotypes):
    """Find heterozygous genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_het : ndarray, bool
        An array where elements are True if the genotype call is heterozygous.

    See Also
    --------

    is_called, is_missing, is_hom_ref, is_hom_alt

    Notes
    -----

    Applicable to polyploid genotype calls, although note that all
    types of heterozygous genotype (i.e., anything not completely
    homozygous) will give an element value of True.

    Applicable to multiallelic variants, although note that the element value
    will be True in any case where the two alleles in a genotype are
    different, e.g., (0, 1), (0, 2), (1, 2), etc.

    """

    # check input array has 2 or more dimensions
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # find hets
    allele1 = genotypes[..., 0, np.newaxis]
    other_alleles = genotypes[..., 1:]
    is_het = np.any(allele1 != other_alleles, axis=dim_ploidy)

    return is_het


def count_het(genotypes, axis=None):
    """Count heterozygous genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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
    is_het

    """

    # count genotypes
    n = np.sum(is_het(genotypes), axis=axis)

    return n


def is_hom_ref(genotypes):
    """Find homozygous reference genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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
    is_called, is_missing, is_het, is_hom_alt

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    genotypes = np.asarray(genotypes)
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

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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

    # count genotypes
    n = np.sum(is_hom_ref(genotypes), axis=axis)

    return n


def is_hom_alt(genotypes):
    """Find homozygous non-reference genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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

    is_called, is_missing, is_hom_ref, is_het

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array has 2 or more dimensions
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # find hets
    allele1 = genotypes[..., 0, np.newaxis]
    other_alleles = genotypes[..., 1:]
    is_hom_alt = np.all((allele1 > 0) & (allele1 == other_alleles),
                        axis=dim_ploidy)

    return is_hom_alt


def count_hom_alt(genotypes, axis=None):
    """Count homozygous non-reference genotype calls.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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
    is_hom_alt

    """

    # count genotypes
    n = np.sum(is_hom_alt(genotypes), axis=axis)

    return n


def as_haplotypes(genotypes):
    """Reshape an array of genotypes to view it as haplotypes by dropping the
    ploidy dimension.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    haplotypes : ndarray
        An array of shape (`n_variants`, `n_samples` * `ploidy`).

    Notes
    -----

    Note that if genotype calls are unphased, the haplotypes returned by this
    function will bear no resemblance to the true haplotypes.

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim == 3

    # reshape, preserving size of first dimension
    newshape = (genotypes.shape[0], -1)
    haplotypes = np.reshape(genotypes, newshape)

    return haplotypes


def as_n_alt(genotypes):
    """Transform genotypes as the number of non-reference alleles.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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

    as_012

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
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim > 1

    # assume ploidy is fastest changing dimension
    dim_ploidy = genotypes.ndim - 1

    # count number of alternate alleles
    gn = np.empty(genotypes.shape[:-1], dtype='i1')
    np.sum(genotypes > 0, axis=dim_ploidy, out=gn)

    return gn


def as_012(genotypes, fill=-1):
    """Transform genotypes recoding homozygous reference calls a
    0, heterozygous calls as 1, homozygous non-reference calls as 2, and
    missing calls as -1.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
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

    Applicable to polyploid genotype calls, although note that all
    types of heterozygous genotype (i.e., anything not completely
    homozygous) will be coded as 1.

    Applicable to multiallelic variants, although note the following.
    All heterozygous genotypes, e.g., (0, 1), (0, 2), (1, 2), ..., will be
    coded as 1. All homozygous non-reference genotypes, e.g., (1, 1), (2, 2),
    ..., will be coded as 2.

    """

    # check input array
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim > 1

    # set up output array
    gn = np.empty(genotypes.shape[:-1], dtype='i1')
    gn.fill(fill)

    # determine genotypes
    gn[is_hom_ref(genotypes)] = 0
    gn[is_het(genotypes)] = 1
    gn[is_hom_alt(genotypes)] = 2

    return gn


def pack_diploid(genotypes):
    """
    Pack diploid genotypes into a single byte for each genotype,
    using the left-most 4 bits for the first allele and the right-most 4 bits
    for the second allele. Allows single byte encoding of diploid genotypes
    for variants with up to 15 alleles.

    Parameters
    ----------

    genotypes : array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    packed : ndarray, int8
        An array of genotypes where the `ploidy` dimension has been collapsed
        by bit packing the two alleles for each genotype into a single byte.

    See Also
    --------

    unpack_diploid_genotypes

    """

    # normalise inputs
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim > 1

    # add 1 to handle missing alleles coded as -1
    genotypes = genotypes + 1

    # left shift first allele by 4 bits
    a1 = np.left_shift(genotypes[..., 0], 4)

    # mask left-most 4 bits to ensure second allele doesn't clash with first
    # allele
    a2 = np.bitwise_and(genotypes[..., 1], 15)

    # pack them
    packed = np.bitwise_or(a1, a2)

    return packed


def unpack_diploid(packed):
    """
    Unpack an array of diploid genotypes that have been bit packed into
    single bytes.

    Parameters
    ----------

    packed : ndarray, int8
        An array of genotypes where the `ploidy` dimension has been collapsed
        by bit packing the two alleles for each genotype into a single byte.

    Returns
    -------

    genotypes : ndarray, int8
        An array of genotypes where the ploidy dimension has been restored by
        unpacking the input array.

    See Also
    --------

    pack_diploid_genotypes

    """

    # check input array
    assert 1 <= packed.ndim <= 2

    # right shift 4 bits to extract first allele
    a1 = np.right_shift(packed, 4)

    # mask left-most 4 bits to extract second allele
    a2 = np.bitwise_and(packed, 15)

    # stack to restore ploidy dimension
    if packed.ndim == 2:
        genotypes = np.dstack((a1, a2))
    elif packed.ndim == 1:
        genotypes = np.column_stack((a1, a2))

    # subtract 1 to restore coding of missing alleles as -1
    genotypes = genotypes - 1

    return genotypes


# packed representation of some common diploid genotypes
BMISSING = 0
BHOM00 = 17
BHET01 = 18
BHOM11 = 34


def count_genotypes(gn, t, axis=None):
    """Count genotypes of a given type.

    Parameters
    ----------

    gn : array_like, int
        An array of shape (`n_variants`, `n_samples`) or (`n_variants`,) or
        (`n_samples`,) where each element is a genotype called coded as a
        single integer.
    t : int
        The genotype to count.
    axis : int, optional
        The axis along which to count.

    Returns
    -------

    n : int or array
        If `axis` is None, returns the total number of matching genotypes. If
        `axis` is specified, returns the sum along the given `axis`.

    """

    # normalise inputs
    gn = np.asarray(gn)

    # count genotypes
    n = np.sum(gn == t, axis=axis)

    return n


def windowed_genotype_counts(pos, gn, t, window_size, start_position=None,
                             stop_position=None):
    """Count genotype calls of a given type for a single sample in
    non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like, int
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like, int
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
    bin_edges : ndarray, float
        The edges of the windows.

    See Also
    --------

    as_diploid_012, as_n_alt, windowed_genotype_density, windowed_genotype_rate

    """

    # check input array
    gn = np.asarray(gn)
    assert gn.ndim == 1

    # find matching genotypes
    values = gn == t

    # computed binned statistic
    counts, bin_edges = anhima.loc.windowed_statistic(
        pos, values=values, statistic=b'sum', window_size=window_size,
        start_position=start_position, stop_position=stop_position
    )

    return counts, bin_edges


def windowed_genotype_density(pos, gn, t, window_size, start_position=None,
                              stop_position=None):
    """Compute per-base-pair density of genotype calls of a given type for a
    single sample in non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like, int
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like, int
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

    density : ndarray, float
        Genotype density for each window.
    bin_edges : ndarray, float
        The edges of the windows.

    See Also
    --------

    as_diploid_012, as_n_alt, windowed_genotype_counts, windowed_genotype_rate

    """

    counts, bin_edges = windowed_genotype_counts(pos, gn, t,
                                                 window_size=window_size,
                                                 start_position=start_position,
                                                 stop_position=stop_position)
    density = counts / np.diff(bin_edges)
    return density, bin_edges


def windowed_genotype_rate(pos, gn, t, window_size, start_position=None,
                           stop_position=None):
    """Compute per-variant rate of genotype calls of a given type for a
    single sample in non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like, int
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like, int
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

    rate : ndarray, float
        Per-variant rate for each window.
    bin_edges : ndarray, float
        The edges of the windows.

    See Also
    --------

    as_diploid_012, as_n_alt, windowed_genotype_counts,
    windowed_genotype_density

    """

    variant_counts, _ = anhima.loc.windowed_variant_counts(
        pos, window_size, start_position=start_position,
        stop_position=stop_position
    )
    counts, bin_edges = windowed_genotype_counts(
        pos, gn, t, window_size=window_size, start_position=start_position,
        stop_position=stop_position
    )
    rate = counts / variant_counts
    return rate, bin_edges


def plot_windowed_genotype_counts(pos, gn, t, window_size, start_position=None,
                                  stop_position=None, ax=None,
                                  plot_kwargs=None):
    """Plots counts of genotype calls of a given type for a single sample in
    non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like, int
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like, int
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
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x, x//3))
        ax = fig.add_subplot(111)

    # count genotypes
    y, bin_edges = windowed_genotype_counts(pos, gn, t, window_size,
                                            start_position=start_position,
                                            stop_position=stop_position)
    x = (bin_edges[1:] + bin_edges[:-1])/2

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


def plot_windowed_genotype_density(pos, gn, t, window_size,
                                   start_position=None,
                                   stop_position=None, ax=None,
                                   plot_kwargs=None):
    """Plots per-base-pair density of genotype calls of a given type for a
    single sample in non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like, int
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like, int
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
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x, x//3))
        ax = fig.add_subplot(111)

    # count genotypes
    y, bin_edges = windowed_genotype_density(pos, gn, t, window_size,
                                             start_position=start_position,
                                             stop_position=stop_position)
    x = (bin_edges[1:] + bin_edges[:-1])/2

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


def plot_windowed_genotype_rate(pos, gn, t, window_size,
                                start_position=None,
                                stop_position=None, ax=None,
                                plot_kwargs=None):
    """Plots per-variant rate of genotype calls of a given type for a
    single sample in non-overlapping windows over the genome.

    Parameters
    ----------

    pos : array_like, int
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    gn : array_like, int
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
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x, x//3))
        ax = fig.add_subplot(111)

    # count genotypes
    y, bin_edges = windowed_genotype_rate(pos, gn, t, window_size,
                                          start_position=start_position,
                                          stop_position=stop_position)
    x = (bin_edges[1:] + bin_edges[:-1])/2

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


def plot_discrete_calldata(a, labels=None, colors='wbgrcmyk', states=None,
                           ax=None, pcolormesh_kwargs=None):
    """
    Plot a color grid from discrete calldata (e.g., genotypes).

    Parameters
    ----------

    a : array_like, int, shape (`n_variants`, `n_samples`)
        2-dimensional array of integers containing the call data to plot.
    labels : sequence of strings, optional
        Axis labels (e.g., sample IDs).
    colors : sequence, optional
        Colors to use for different values of the array.
    states : sequence, optional
        Manually specify discrete calldata states (if not given will be
        determined from the data).
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    pcolormesh_kwargs : dict-like, optional
        Additional keyword arguments passed through to `plt.pcolormesh`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check input array
    a = np.asarray(a)
    assert a.ndim == 2

    # set up axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # determine discrete states
    if states is None:
        states = np.unique(a)

    # determine colors for states
    colors = colors[:np.max(states)-np.min(states)+1]

    # plotting defaults
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = dict()
    pcolormesh_kwargs.setdefault('cmap', mpl.colors.ListedColormap(colors))
    pcolormesh_kwargs.setdefault(
        'norm', plt.Normalize(np.min(states), np.max(states)+1)
    )

    # plot the colormesh
    ax.pcolormesh(a.T, **pcolormesh_kwargs)

    # tidy up
    ax.set_xlim(0, a.shape[0])
    ax.set_ylim(0, a.shape[1])
    ax.set_xticks([])
    if labels is not None:
        ax.set_yticks(np.arange(a.shape[1]) + .5)
        ax.set_yticklabels(labels, rotation=0)
    else:
        ax.set_yticks([])

    return ax


def plot_continuous_calldata(a, labels=None, ax=None, pcolormesh_kwargs=None):
    """
    Plot a color grid from continuous calldata (e.g., DP).

    Parameters
    ----------

    a : array_like, shape (`n_variants`, `n_samples`)
        2-dimensional array of integers or floats containing the call data to
        plot.
    labels : sequence of strings, optional
        Axis labels (e.g., sample IDs).
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    pcolormesh_kwargs : dict-like, optional
        Additional keyword arguments passed through to `plt.pcolormesh`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check input array
    a = np.asarray(a)
    assert a.ndim == 2

    # set up axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # plotting defaults
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = dict()
    pcolormesh_kwargs.setdefault('cmap', 'jet')

    # plot the color mesh
    ax.pcolormesh(a.T, **pcolormesh_kwargs)

    # tidy up
    ax.set_xlim(0, a.shape[0])
    ax.set_ylim(0, a.shape[1])
    ax.set_xticks([])
    if labels is not None:
        ax.set_yticks(np.arange(a.shape[1]) + .5)
        ax.set_yticklabels(labels, rotation=0)
    else:
        ax.set_yticks([])

    return ax


def plot_diploid_genotypes(gn,
                           labels=None,
                           colors='wbgr',
                           states=(-1, 0, 1, 2),
                           ax=None,
                           colormesh_kwargs=None):
    """Plot diploid genotypes as a color grid.

    Parameters
    ----------

    gn : array_like, int, shape (`n_variants`, `n_samples`)
        An array where each genotype is coded as a single integer as
        described above.
    labels : sequence of strings, optional
        Axis labels (e.g., sample IDs).
    colors : sequence, optional
        Colors to use for different values of the array.
    states : sequence, optional
        Manually specify discrete calldata states (if not given will be
        determined from the data).
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    colormesh_kwargs : dict-like
        Additional keyword arguments passed through to `plt.pcolormesh`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    return plot_discrete_calldata(gn, labels=labels, colors=colors,
                                  states=states, ax=ax,
                                  pcolormesh_kwargs=colormesh_kwargs)


def plot_genotype_counts_by_sample(gn, states=(-1, 0, 1, 2),
                                   colors='wbgr', labels=None,
                                   ax=None, width=1, orientation='vertical',
                                   bar_kwargs=None):
    """Plot a bar graph of genotype counts by sample.

    Parameters
    ----------

    gn : array_like, int, shape (`n_variants`, `n_samples`)
        An array where each genotype is coded as a single integer as
        described above.
    states : sequence, optional
        The genotype states to count.
    colors : sequence, optional
        Colors to use for corresponding states.
    labels : sequence of strings, optional
        Axis labels (e.g., sample IDs).
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    width : float, optional
        Width of the bars (will be used as height if `orientation` ==
        'horizontal').
    orientation : {'horizontal', 'vertical'}
        Which type of bar to plot.
    bar_kwargs : dict-like
        Additional keyword arguments passed through to `plt.bar`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check input array
    gn = np.asarray(gn)
    assert gn.ndim == 2
    n_variants = gn.shape[0]
    n_samples = gn.shape[1]

    # check orientation
    assert orientation in ('vertical', 'horizontal')

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()

    # determine bar positions
    x = np.arange(n_samples)

    # plot bars for each type
    if bar_kwargs is None:
        bar_kwargs = dict()
    bar_kwargs.setdefault('linewidth', 0)
    yc = None
    for t, c in zip(states, colors):

        # count genotypes
        y = count_genotypes(gn, t, axis=0)

        # plot as bar
        if orientation == 'vertical':
            ax.bar(x, y, width=width, bottom=yc, color=c, label=t, **bar_kwargs)
        else:
            ax.barh(x, y, height=width, left=yc, color=c, label=t, **bar_kwargs)

        # keep cumulative count
        if yc is None:
            yc = y
        else:
            yc += y

    # tidy up
    # TODO code smells

    # set plot limits
    if orientation == 'vertical':
        ax.set_ylim(0, n_variants)
        ax.set_xlim(0, n_samples)
    else:
        ax.set_xlim(0, n_variants)
        ax.set_ylim(0, n_samples)

    # determine tick labels
    if labels:
        if orientation == 'vertical':
            ax.set_xticks(range(n_samples))
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(range(n_samples))
            ax.set_yticklabels(labels)
    else:
        if orientation == 'vertical':
            ax.set_xticks([])
        else:
            ax.set_yticks([])

    return ax


def plot_genotype_counts_by_variant(gn, states=(-1, 0, 1, 2),
                                    colors='wbgr', ax=None, width=1,
                                    orientation='vertical',
                                    bar_kwargs=None):
    """Plot a bar graph of genotype counts by variant.

    Parameters
    ----------

    gn : array_like, int, shape (`n_variants`, `n_samples`)
        An array where each genotype is coded as a single integer as
        described above.
    states : sequence, optional
        The genotype states to count.
    colors : sequence, optional
        Colors to use for corresponding states.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    width : float, optional
        Width of the bars (will be used as height if `orientation` ==
        'horizontal').
    orientation : {'horizontal', 'vertical'}
        Which type of bar to plot.
    bar_kwargs : dict-like
        Additional keyword arguments passed through to `plt.bar`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check input array
    gn = np.asarray(gn)
    assert gn.ndim == 2
    n_variants = gn.shape[0]
    n_samples = gn.shape[1]

    # check orientation
    assert orientation in ('vertical', 'horizontal')

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()

    # determine bar positions
    x = np.arange(n_variants)

    # plot bars for each type
    if bar_kwargs is None:
        bar_kwargs = dict()
    bar_kwargs.setdefault('linewidth', 0)
    yc = None
    for t, c in zip(states, colors):

        # count genotypes
        y = count_genotypes(gn, t, axis=1)

        # plot as bar
        if orientation == 'vertical':
            ax.bar(x, y, width=width, bottom=yc, color=c, label=t, **bar_kwargs)
        else:
            ax.barh(x, y, height=width, left=yc, color=c, label=t, **bar_kwargs)

        # keep cumulative count
        if yc is None:
            yc = y
        else:
            yc += y

    # tidy up
    if orientation == 'vertical':
        ax.set_xticks([])
        ax.set_xlim(0, n_variants)
        ax.set_ylim(0, n_samples)
    else:
        ax.set_yticks([])
        ax.set_ylim(0, n_variants)
        ax.set_xlim(0, n_samples)

    return ax


def plot_continuous_calldata_by_sample(a, labels=None,
                                       ax=None,
                                       orientation='vertical',
                                       boxplot_kwargs=None):
    """Plot a boxplot of continuous call data (e.g., DP) by sample.

    Parameters
    ----------

    a : array_like, shape (`n_variants`, `n_samples`)
        2-dimensional array of integers or floats containing the call data to
        plot.
    labels : sequence of strings, optional
        Axis labels (e.g., sample IDs).
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    orientation : {'horizontal', 'vertical'}
        Which type of bar to plot.
    boxplot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.boxplot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check input array
    a = np.asarray(a)
    assert a.ndim == 2

    # check orientation
    assert orientation in ('vertical', 'horizontal')
    vert = orientation == 'vertical'

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()

    # plot
    if boxplot_kwargs is None:
        boxplot_kwargs = dict()
    ax.boxplot(a, vert=vert, **boxplot_kwargs)

    # tidy up
    n_samples = a.shape[1]
    if labels:
        if orientation == 'vertical':
            ax.set_xticks(range(n_samples))
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(range(n_samples))
            ax.set_yticklabels(labels)
    else:
        if orientation == 'vertical':
            ax.set_xticks([])
        else:
            ax.set_yticks([])

    return ax
