"""
Utilities for working with genotype data.

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np


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
        An array  where each genotype is coded as a single integer as
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


