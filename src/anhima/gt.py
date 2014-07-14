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
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_called : ndarray, bool
        An array of shape (`n_variants`, `n_samples`) where elements are True
        if the genotype call is non-missing.

    See Also
    --------

    is_hom_ref, is_het_diploid, is_hom_alt_diploid

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array
    assert hasattr(genotypes, 'ndim')
    assert genotypes.ndim == 3

    # determine output array
    out = np.all(genotypes >= 0, axis=2)

    return out


def is_hom_ref(genotypes):
    """Find homozygous reference genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_hom_ref : ndarray, bool
        An array of shape (`n_variants`, `n_samples`) where elements are True
        if the genotype call is homozygous reference.

    See Also
    --------
    is_called, is_het_diploid, is_hom_alt_diploid

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array
    assert hasattr(genotypes, 'ndim')
    assert genotypes.ndim == 3

    # determine output array
    out = np.all(genotypes == 0, axis=2)

    return out


def is_het_diploid(genotypes):
    """Find diploid heterozygous genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_het : ndarray, bool
        An array of shape (`n_variants`, `n_samples`) where elements are True
        if the genotype call is heterozygous.

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

    # check input array
    assert hasattr(genotypes, 'ndim')
    assert genotypes.ndim == 3
    assert genotypes.shape[2] == 2

    # find hets
    allele1 = genotypes[:, :, 0]
    allele2 = genotypes[:, :, 1]
    is_het = allele1 != allele2

    return is_het


def is_hom_alt_diploid(genotypes):
    """Find diploid homozygous non-reference genotype calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_hom_alt : ndarray, int8
        An array of shape (`n_variants`, `n_samples`) where elements are
        non-zero if the genotype call is homozygous non-reference. The actual
        value of the element will be the non-reference allele index.

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

    # check input array
    assert hasattr(genotypes, 'ndim')
    assert genotypes.ndim == 3
    assert genotypes.shape[2] == 2

    # find homozygotes
    n_variants = genotypes.shape[0]
    n_samples = genotypes.shape[1]
    out = np.zeros((n_variants, n_samples), dtype='i1')
    allele1 = genotypes[:, :, 0]
    allele2 = genotypes[:, :, 1]
    is_hom_alt = (allele1 > 0) & (allele1 == allele2)
    out[is_hom_alt] = allele1

    return out


def as_alleles(genotypes):
    """Reshape an array of genotypes as an array of alleles.

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
        An array of shape (`n_variants`, `n_samples` * `ploidy`) where the
        third dimension has been collapsed.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # check input array
    assert hasattr(genotypes, 'ndim')
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
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    gn : ndarray, int8
        An array of shape (`n_variants`, `n_samples`) where each genotype is
        coded as a single integer counting the number of alternate alleles.

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
    assert hasattr(genotypes, 'ndim')
    assert genotypes.ndim == 3

    # count number of alternate alleles
    n_variants = genotypes.shape[0]
    n_samples = genotypes.shape[1]
    gn = np.empty((n_variants, n_samples), dtype='i1')
    np.sum(genotypes > 0, axis=2, out=gn)

    return gn


def as_diploid_012(genotypes):
    """Transform an array of genotypes recoding homozygous reference calls a
    0, heterozygous calls as 1, homozygous non-reference calls as 2, and
    missing calls as -1.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    gn : ndarray, int8
        An array of shape (`n_variants`, `n_samples`) where each genotype is
        coded as a single integer as described above.

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
    assert hasattr(genotypes, 'ndim')
    assert genotypes.ndim == 3

    # set up output array
    n_variants = genotypes.shape[0]
    n_samples = genotypes.shape[1]
    gn = np.empty((n_variants, n_samples), dtype='i1')
    # fill with -1 to start with
    gn.fill(-1)

    # determine genotypes
    gn[is_hom_ref(genotypes)] = 0
    gn[is_het_diploid(genotypes)] = 1
    gn[is_hom_alt_diploid(genotypes)] = 2

    return gn


