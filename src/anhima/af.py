"""
Allele frequency calculations.

"""


from __future__ import division, print_function


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np


# internal dependencies
import anhima.gt


def is_variant(genotypes, min_ac=1):
    """Find variants with at least `min_ac` non-reference alleles.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    min_ac : int, optional
        The minimum number of non-reference alleles required to consider
        variant.

    Returns
    -------

    is_variant : ndarray, bool
        An array of shape (`n_variants`,) where an element is True if there
        are at least `min_ac` non-reference alleles found for the corresponding
        variant.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # reshape as alleles
    a = anhima.gt.as_alleles(genotypes)

    # determine output
    out = np.sum(a > 0, axis=1) >= min_ac

    return out


def is_non_variant(genotypes):
    """Find variants with no non-reference alleles.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_non_variant : ndarray, bool
        An array of shape (`n_variants`,) where an element is True if there
        are no non-reference alleles found for the corresponding variant.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # reshape as alleles
    a = anhima.gt.as_alleles(genotypes)

    # determine output
    out = np.all(a <= 0, axis=1)

    return out


def is_non_segregating(genotypes, allele=1):
    """Find non-segregating variants for the given allele.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to test for fixation.

    Returns
    -------

    is_non_segregating : ndarray, bool
        An array of shape (`n_variants`,) where an element is True if all
        genotype calls for the corresponding variant are either missing or
        equal to `allele`.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, although note that this function
    tests for variants non-segregating with respect to the given `allele`,
    it does not find non-segregating variants in general.

    """

    # reshape as alleles
    a = anhima.gt.as_alleles(genotypes)

    # determine output
    out = np.all((a < 0) | (a == allele), axis=1)

    return out


def is_singleton(genotypes, allele=1):
    """Find variants with only a single instance of `allele` called.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to find singletons of.

    Returns
    -------

    is_singleton : ndarray, bool
        An array of shape (`n_variants`,) where an element is True if there
        is a single instance of `allele` called.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note this function checks for a
    specific `allele`.

    """

    # reshape as alleles
    a = anhima.gt.as_alleles(genotypes)

    # determine output
    out = np.sum(a == allele, axis=1) == 1

    return out


def is_doubleton(genotypes, allele=1):
    """Find variants with only two instances of `allele` called.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to find doubletons of.

    Returns
    -------

    is_doubleton : ndarray, bool
        An array of shape (`n_variants`,) where an element is True if there
        are exactly two instances of `allele` called.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note this function checks for a
    specific `allele`.

    """

    # reshape as alleles
    a = anhima.gt.as_alleles(genotypes)

    # determine output
    out = np.sum(a == allele, axis=1) == 2

    return out


def allele_number(genotypes):
    """Count the number of non-missing allele calls.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    an : ndarray, int
        An array of shape (`n_variants`,) counting the total number of
        non-missing alleles called.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    # reshape as alleles
    a = anhima.gt.as_alleles(genotypes)

    # count non-missing alleles
    an = np.sum(a >= 0, axis=1)

    return an


def allele_count(genotypes, allele=1):
    """Calculate number of instances of the given allele.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to count.

    Returns
    -------

    ac : ndarray, int
        An array of shape (`n_variants`,) counting the number of
        times the given `allele` was called.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note that this function
    calculates the frequency of a specific `allele`.

    """

    # reshape as alleles
    a = anhima.gt.as_alleles(genotypes)

    # count alleles
    ac = np.sum(a == allele, axis=1)

    return ac


def allele_frequency(genotypes, allele=1):
    """Calculate frequency of the given allele.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        The allele to calculate the frequency of.

    Returns
    -------

    an : ndarray, int
        An array of shape (`n_variants`,) counting the total number of
        non-missing alleles called.
    ac : ndarray, int
        An array of shape (`n_variants`,) counting the number of
        times the given `allele` was called.
    af : ndarray, float
        An array of shape (`n_variants`,) containing the allele frequency.

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


def allele_frequencies(genotypes, alleles=(0, 1)):
    """Calculate frequencies of the given alleles.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    alleles : sequence of ints, optional
        The alleles to calculate the frequency of.

    Returns
    -------

    an : ndarray, int
        An array of shape (`n_variants`,) counting the total number of
        non-missing alleles called.
    ac : ndarray, int
        An array of shape (`n_variants`, `len(alleles)`) counting the number of
        times the given `alleles` were called.
    af : ndarray, float
        An array of shape (`n_variants`, `len(alleles)`) containing the allele
        frequencies.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants, but note that this function
    calculates the frequency of a specific `alleles`.

    """

    # set up output arrays
    n_variants = genotypes.shape[0]
    n_alleles = len(alleles)
    ac = np.empty((n_variants, n_alleles), dtype='i4')
    af = np.empty((n_variants, n_alleles), dtype='f4')

    # count non-missing alleles
    an = allele_number(genotypes)

    # loop over alleles
    for n, allele in enumerate(alleles):
        ac[:, n] = allele_count(genotypes, allele=allele)
        af[:, n] = ac[:, n] / an

    return an, ac, af


