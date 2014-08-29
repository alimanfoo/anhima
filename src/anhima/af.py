"""
Allele frequency calculations.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/af.ipynb

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np


# internal dependencies
import anhima.gt


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

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    out = np.sum(genotypes > 0, axis=(1, 2)) >= min_ac

    return out


def count_variant(genotypes, min_ac=1):
    """Count variants with at least `min_ac` non-reference alleles.

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

    n : int
        The number of variants.

    Notes
    -----

    Applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    """

    return np.count_nonzero(is_variant(genotypes, min_ac))


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
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
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

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    out = np.sum(genotypes == allele, axis=(1, 2)) == 1

    return out


def count_singletons(genotypes, allele=1):
    """Count variants with only a single instance of `allele` called.

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

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    out = np.sum(genotypes == allele, axis=(1, 2)) == 2

    return out


def count_doubletons(genotypes, allele=1):
    """Count variants with only two instances of `allele` called.

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

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    an = np.sum(genotypes >= 0, axis=(1, 2))

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

    # check inputs
    genotypes = _check_genotypes(genotypes)

    # aggregate over samples and ploidy dimensions
    ac = np.sum(genotypes == allele, axis=(1, 2))

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


def max_allele(genotypes, axis=None):
    """
    Return the highest allele index.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
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
    """Calculate allele counts.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    alleles : sequence of ints, optional
        The alleles to calculate the frequency of. If not specified, all 
        alleles will be counted.

    Returns
    -------

    ac : ndarray, int
        An array of shape (`n_variants`, `n_alleles`) counting the number of
        times the given `alleles` were called.

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
    """Calculate allele frequencies.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    alleles : sequence of ints, optional
        The alleles to calculate the frequency of. If not specified, all 
        alleles will be counted.

    Returns
    -------

    an : ndarray, int
        An array of shape (`n_variants`,) counting the total number of
        non-missing alleles called.
    ac : ndarray, int
        An array of shape (`n_variants`, `n_alleles`) counting the number of
        times the given `alleles` were called.
    af : ndarray, float
        An array of shape (`n_variants`, `n_alleles`) containing the allele
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
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    n : ndarray, int
        An array of shape (`n_variants`,) where an element holds the allelism 
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
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    allele : int, optional
        If given, find variants fixed with respect to `allele`.

    Returns
    -------

    is_non_segregating : ndarray, bool
        An array of shape (`n_variants`,) where an element is True if all
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
        out = np.all((genotypes < 0) | (genotypes == allele), axis=1)

    return out


def count_non_segregating(genotypes, allele=None):
    """Count non-segregating variants.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
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
    """Find segregating variants (where more than one allele is found).

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_segregating : ndarray, bool
        An array of shape (`n_variants`,) where an element is True if more 
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
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
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
    """TODO

    """

    # check input
    derived_ac = np.asarray(derived_ac)
    assert derived_ac.ndim == 1

    # calculate frequency spectrum
    sfs = np.bincount(derived_ac)

    return sfs


def site_frequency_spectrum_folded(biallelic_ac):
    """TODO

    """

    # check input
    biallelic_ac = np.asarray(biallelic_ac)
    assert biallelic_ac.ndim == 2
    assert biallelic_ac.shape[1] == 2

    # calculate minor allele counts
    minor_ac = np.amin(biallelic_ac, axis=1)

    # calcate frequency spectrum
    sfs_folded = np.bincount(minor_ac)

    return sfs_folded


def site_frequency_spectrum_scaled(derived_ac):
    """TODO

    """

    # calculate frequency spectrum
    sfs = site_frequency_spectrum()

    # scaling
    k = np.arange(sfs.size + 1)
    sfs_scaled = sfs * k

    return sfs_scaled


def site_frequency_spectrum_folded_scaled(biallelic_ac, an=None):
    """TODO

    """

    # TODO
    pass