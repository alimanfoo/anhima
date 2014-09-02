"""
Extremely naive simulation functions to generate genotype data for
illustration of other features in the ``anhima`` package.

"""


from __future__ import division, print_function, unicode_literals, \
    absolute_import


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# python standard library dependencies
import random


# third party dependencies
import numpy as np
import scipy


def simulate_biallelic_genotypes(n_variants, n_samples, af_dist,
                                 p_missing=.1,
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
    p_missing : float, optional
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

    # freeze binomial distribution to model missingness
    miss_dist = scipy.stats.binom(p=p_missing, n=n_samples)

    # iterate over variants
    for i, p in zip(range(n_variants), af):

        # randomly generate alleles under the given allele frequency
        # ensure p is valid probability
        p = min(p, 1)
        alleles = scipy.stats.bernoulli.rvs(p, size=n_samples*ploidy)

        # reshape alleles as genotypes under the given ploidy
        genotypes[i] = alleles.reshape(n_samples, ploidy)

        # simulate some missingness
        n_missing = miss_dist.rvs()
        missing_indices = random.sample(range(n_samples),
                                        n_missing)
        genotypes[i, missing_indices] = (-1,) * ploidy

    return genotypes


def simulate_genotypes_with_ld(n_variants, n_samples, correlation=0.2):
    """A very simple function to simulate a set of genotypes, where
    variants are in some degree of linkage disequilibrium with their
    neighbours.

    Parameters
    ----------

    n_variants : int
        The number of variants to simulate data for.
    n_samples : int
        The number of individuals to simulate data for.
    correlation : float, optional
        The fraction of samples to copy genotypes between neighbouring
        variants.

    Returns
    -------

    gn : ndarray, int8
        A 2-dimensional array of shape (`n_variants`, `n_samples`) where each
        element is a genotype call coded as a single integer counting the
        number of non-reference alleles.

    """

    # initialise an array of random genotypes
    gn = np.random.randint(size=(n_variants, n_samples), low=0, high=3)
    gn = gn.astype('i1')

    # determine the number of samples to copy genotypes for
    n_copy = int(correlation * n_samples)

    # introduce linkage disequilibrium by copying genotypes from one sample to
    # the next
    for i in range(1, n_variants):

        # randomly pick the samples to copy from
        sample_indices = random.sample(range(n_samples), n_copy)

        # view genotypes from the previous variant for the selected samples
        c = gn[i-1, sample_indices]

        # randomly choose whether to invert the correlation
        inv = random.randint(0, 1)
        if inv:
            c = 2-c

        # copy across genotypes
        gn[i, sample_indices] = c

    return gn


def simulate_relatedness(genotypes, relatedness=.5, n_iter=1000, copy=True):
    """
    Simulate relatedness by randomly copying genotypes between individuals.

    Parameters
    ----------

    genotypes : array_like
        An array of shape (`n_variants`, `n_samples`, `ploidy`) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    relatedness : float, optional
        Fraction of variants to copy genotypes for.
    n_iter : int, optional
        Number of times to randomly copy genotypes between individuals.
    copy : bool, optional
        If False, modify `genotypes` in place.

    Returns
    -------

    genotypes : ndarray, shape (n_variants, n_samples, ploidy)
        The input genotype array but with relatedness simulated.

    """

    # check genotypes array
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim >= 2
    n_variants = genotypes.shape[0]
    n_samples = genotypes.shape[1]

    # copy input array
    if copy:
        genotypes = genotypes.copy()
    else:
        # modify in place
        pass

    # determine the number of variants to copy genotypes for
    n_copy = int(relatedness * n_variants)

    # iteratively introduce relatedness
    for i in xrange(n_iter):

        # randomly choose donor and recipient
        donor_index = random.randint(0, n_samples-1)
        donor = genotypes[:, donor_index]
        recip_index = random.randint(0, n_samples-1)
        recip = genotypes[:, recip_index]

        # randomly pick a set of variants to copy
        variant_indices = random.sample(range(n_variants), n_copy)

        # copy across genotypes
        recip[variant_indices] = donor[variant_indices]

    return genotypes

