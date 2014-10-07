"""
Allele frequency calculations.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/af.ipynb

"""  # noqa


from __future__ import division, print_function, unicode_literals, \
    absolute_import


# third party dependencies
import numpy as np


# internal dependencies
import anhima


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

    # calculate allele frequency, accounting for missingness
    err = np.seterr(invalid='ignore')
    af = np.where(an > 0, ac / an, 0)
    np.seterr(**err)

    return an, ac, af


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
        The alleles to count. If not specified, all alleles will be counted.

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
        m = np.amax(genotypes)
        alleles = range(m+1)

    # count alleles
    ac = np.zeros((n_variants, len(alleles)), dtype='i4')
    for i, allele in enumerate(alleles):
        np.sum(genotypes == allele, axis=(1, 2), out=ac[:, i])

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
    an = allele_number(genotypes)[:, np.newaxis]

    # count alleles
    ac = allele_counts(genotypes, alleles=alleles)

    # calculate allele frequencies, accounting for missingness
    err = np.seterr(invalid='ignore')
    af = np.where(an > 0, ac / an, 0)
    np.seterr(**err)

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


def maximum_likelihood_ancestry(genotypes, qa, qb, filter_size=0):
    """Given alternate allele frequencies in two populations `qa` and `qb`,
    predict the ancestry for a set of `genotypes`.

    Parameters
    ----------

    genotypes : array_like
        An array of diploid genotype calls of shape (n_variants, n_samples,
        2) where each element of the array is an integer corresponding to an
        allele index (-1 = missing, 0 = reference allele, 1 = first alternate
        allele, 2 = second alternate allele, etc.).
    qa : array_like, float
        A 1-dimensional array of shape (n_variants, ) containing alternate
        allele frequencies for population A.
    qb : array_like, float
        A 1-dimensional array of shape (n_variants, ) containing alternate
        allele frequencies for population B.
    filter_size : int, optional
        Sum likelihoods in a moving window of size `filter_size`.

    Returns
    -------

    ancestry : ndarray, int, shape (n_variants, n_samples)
        An array containing the ancestry predictions, where 0 = AA (both
        alleles derive from population A), 1 = AB (hybrid ancestry) and 2 =
        BB (both alleles derive from population B), and -1 = ambiguous (models
        are equally likely).
    confidence : ndarray, float, shape (n_variants, n_samples)
        The confidence in the ancestry prediction (natural logarithm of the
        likelihood ratio for the two most likely models).

    Notes
    -----

    Where allele frequencies are similar between populations A and B,
    ancestry predictions will have low confidence, because different ancestry
    models will have similar likelihoods. Greater confidence will be obtained by
    filtering variants to select those where the difference in allele
    frequencies is greater. E.g.::

        >>> flt = np.abs(qa - qb) > .5
        >>> genotypes_flt = genotypes[flt]
        >>> qa_flt = qa[flt]
        >>> qb_flt = qb[flt]
        >>> ancestry, confidence = maximum_likelihood_ancestry(genotypes_flt, qa_flt, qb_flt)

    """  # noqa

    # check inputs
    genotypes = _check_genotypes(genotypes)
    # require biallelic genotypes
    assert np.amax(genotypes) < 2
    n_variants, n_samples, ploidy = genotypes.shape
    # require diploid genotypes
    assert ploidy == 2
    qa = np.asarray(qa)
    qb = np.asarray(qb)
    assert qa.ndim == qb.ndim == 1
    assert n_variants == qa.shape[0] == qb.shape[0]

    # calculate reference allele frequencies, assuming biallelic variants
    pa = 1 - qa
    pb = 1 - qb

    # work around zero frequencies which cause problems when calculating logs
    pa[pa == 0] = np.exp(-250)
    qa[qa == 0] = np.exp(-250)
    pb[pb == 0] = np.exp(-250)
    qb[qb == 0] = np.exp(-250)

    # calculate likelihoods
    logpa = np.log(pa)
    logqa = np.log(qa)
    logpb = np.log(pb)
    logqb = np.log(qb)

    # set up likelihoods array
    n_models = 3
    n_gn_states = 3
    log_likelihoods = np.empty((n_variants, n_samples, n_models, n_gn_states),
                               dtype='f8')

    # probability of genotype (e.g., 0 = hom ref) given model (e.g., 0 = aa)
    log_likelihoods[:, :, 0, 0] = (2 * logpa)[:, np.newaxis]
    log_likelihoods[:, :, 1, 0] = (np.log(2) + logpa + logqa)[:, np.newaxis]
    log_likelihoods[:, :, 2, 0] = (2 * logqa)[:, np.newaxis]
    log_likelihoods[:, :, 0, 1] = (logpa + logpb)[:, np.newaxis]
    log_likelihoods[:, :, 1, 1] = (np.logaddexp(logpa + logqb,
                                                logqa + logpb)[:, np.newaxis])
    log_likelihoods[:, :, 2, 1] = (logqa + logqb)[:, np.newaxis]
    log_likelihoods[:, :, 0, 2] = (2 * logpb)[:, np.newaxis]
    log_likelihoods[:, :, 1, 2] = (np.log(2) + logpb + logqb)[:, np.newaxis]
    log_likelihoods[:, :, 2, 2] = (2 * logqb)[:, np.newaxis]

    # transform genotypes for convenience
    gn = anhima.gt.as_012(genotypes)

    # calculate actual model likelihoods for each genotype call
    model_likelihoods = np.empty((n_variants, n_samples, n_models), dtype='f8')
    model_likelihoods.fill(-250)
    for model in 0, 1, 2:
        for gn_state in 0, 1, 2:
            model_likelihoods[:, :, model][gn == gn_state] = \
                log_likelihoods[:, :, model, gn_state][gn == gn_state]

    # optionally combine likelihoods in a moving window
    if filter_size:
        model_likelihoods = np.apply_along_axis(np.convolve,
                                                0,
                                                model_likelihoods,
                                                np.ones((filter_size,)))
        # remove edges
        model_likelihoods = \
            model_likelihoods[filter_size//2:-1*(filter_size//2), ...]

    # predict ancestry as model with highest likelihood
    ancestry = np.argmax(model_likelihoods, axis=2)

    # calculate confidence by comparing first and second most likely models
    model_likelihoods.sort(axis=2)
    confidence = model_likelihoods[:, :, 2] - model_likelihoods[:, :, 1]

    # recind prediction where confidence is zero (models are equally likely)
    ancestry[confidence == 0] = -1

    return ancestry, confidence
