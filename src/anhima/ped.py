"""
Utilities for working with related individuals (crosses, families, etc.).

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/ped.ipynb

"""


from __future__ import division, print_function, unicode_literals, \
    absolute_import


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# third party dependencies
import numpy as np
import numexpr as ne

# internal dependencies
import anhima.gt

# constants to represent inheritance states
INHERIT_PARENT1 = 1
INHERIT_PARENT2 = 2
INHERIT_NONSEG_REF = 3
INHERIT_NONSEG_ALT = 4
INHERIT_NONPARENTAL = 5
INHERIT_PARENT_MISSING = 6
INHERIT_MISSING = 7


def diploid_inheritance(parent_diplotype, gamete_haplotypes):
    """
    Determine the transmission of parental alleles to a set of gametes.

    Parameters
    ----------

    parent_diplotype : array_like, shape (n_variants, 2)
        An array of phased genotypes for a single diploid individual, where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).
    gamete_haplotypes : array_like, shape (n_variants, n_gametes)
        An array of haplotypes for a set of gametes derived from the given 
        parent, where each element of the array is an integer corresponding
        to an allele index (-1 = missing, 0 = reference allele, 1 = first
        alternate allele, 2 = second alternate allele, etc.).
        
    Returns
    -------
    
    inheritance : ndarray, uint8, shape (n_variants, n_gametes)
        An array of integers coding the allelic inheritance, where 1 = 
        inheritance from first parental haplotype, 2 = inheritance from second 
        parental haplotype, 3 = inheritance of reference allele from parent 
        that is homozygous for the reference allele, 4 = inheritance of 
        alternate allele from parent that is homozygous for the alternate 
        allele, 5 = non-parental allele, 6 = parental genotype is missing, 
        7 = gamete allele is missing. 

    """

    # normalise inputs
    parent_diplotype = np.asarray(parent_diplotype)
    assert parent_diplotype.ndim == 2
    assert parent_diplotype.shape[1] == 2
    gamete_haplotypes = np.asarray(gamete_haplotypes)
    assert gamete_haplotypes.ndim == 2

    # convenience variables
    parent1 = parent_diplotype[:, 0, np.newaxis]
    parent2 = parent_diplotype[:, 1, np.newaxis]
    gamete_is_missing = gamete_haplotypes < 0
    parent_is_missing = np.any(parent_diplotype < 0, axis=1)
    parent_is_hom_ref = anhima.gt.is_hom_ref(parent_diplotype)[:, np.newaxis]
    parent_is_het = anhima.gt.is_het(parent_diplotype)[:, np.newaxis]
    parent_is_hom_alt = anhima.gt.is_hom_alt(parent_diplotype)[:, np.newaxis]

    # need this for broadcasting, but also need to retain original for later
    parent_is_missing_bc = parent_is_missing[:, np.newaxis]
    
    # N.B., use numexpr below where possible to avoid temporary arrays

    # utility variable, identify allele calls where inheritance can be
    # determined
    callable = ne.evaluate('~gamete_is_missing & ~parent_is_missing_bc')
    callable_seg = ne.evaluate('callable & parent_is_het')

    # main inheritance states
    inherit_parent1 = ne.evaluate(
        'callable_seg & (gamete_haplotypes == parent1)'
    )
    inherit_parent2 = ne.evaluate(
        'callable_seg & (gamete_haplotypes == parent2)'
    )
    nonseg_ref = ne.evaluate(
        'callable & parent_is_hom_ref & (gamete_haplotypes == parent1)'
    )
    nonseg_alt = ne.evaluate(
        'callable & parent_is_hom_alt & (gamete_haplotypes == parent1)'
    )
    nonparental = ne.evaluate(
        'callable & (gamete_haplotypes != parent1)'
        ' & (gamete_haplotypes != parent2)'
    )

    # record inheritance states
    # N.B., order in which these are set matters
    inheritance = np.zeros_like(gamete_haplotypes, dtype='u1')
    inheritance[inherit_parent1] = INHERIT_PARENT1
    inheritance[inherit_parent2] = INHERIT_PARENT2
    inheritance[nonseg_ref] = INHERIT_NONSEG_REF
    inheritance[nonseg_alt] = INHERIT_NONSEG_ALT
    inheritance[nonparental] = INHERIT_NONPARENTAL
    inheritance[parent_is_missing] = INHERIT_PARENT_MISSING
    inheritance[gamete_is_missing] = INHERIT_MISSING

    return inheritance


def is_non_mendelian_diploid(parental_genotypes, progeny_genotypes):

    """Find `impossible` genotypes according to mendelian inheritance laws

    Parameters
    ----------

    parental_genotypes, progeny_genotypes: array_like, int
        An array of shape (`n_variants`, `n_samples`, `ploidy`) or
        (`n_variants`, `ploidy`) or (`n_samples`, `ploidy`), where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele,
        2 = second alternate allele, etc.).

    Returns
    -------

    is_non_mendelian : ndarray, bool
        An array where elements are True if the genotype call is non-mendelian

    See Also
    --------

    count_non_mendelian

    Notes
    -----

    Not applicable to polyploid genotype calls, or multiallelic variants.

    Does not handle phased genotypes.

    Missing parental genotypes will always result in ME of offspring being
    recorded as F. Missing
    GTs in offspring result in an entry of F for Mendelian Error.

    """
    # check input array has 2 or more dimensions
    assert parental_genotypes.ndim > 1 and progeny_genotypes.ndim > 1

    # check that parent gts match the progeny in dimensions, 0 and 2.
    assert parental_genotypes.shape[0] == progeny_genotypes.shape[0]
    assert parental_genotypes.shape[2] == progeny_genotypes.shape[2]

    # check there are no multiallelic sites
    assert np.all(parental_genotypes != 2) and np.all(progeny_genotypes != 2)

    # sum across the ploidy dimension
    parental_genotypes_012 = anhima.gt.as_012(parental_genotypes)
    progeny_genotypes_012 = anhima.gt.as_012(progeny_genotypes)

    count_mendelian_diploid = np.zeros(progeny_genotypes_012.shape)

    # build 6 classifications of parental gts. Calling ref/het/alt
    parent1_ref = 0 == parental_genotypes_012[:, 0]
    parent2_ref = 0 == parental_genotypes_012[:, 1]

    parent1_het = 1 == parental_genotypes_012[:, 0]
    parent2_het = 1 == parental_genotypes_012[:, 1]

    parent1_alt = 2 == parental_genotypes_012[:, 0]
    parent2_alt = 2 == parental_genotypes_012[:, 1]

    # hom ref X hom ref case:
    count_mendelian_diploid[parent1_ref & parent2_ref] = \
        progeny_genotypes_012[parent1_ref & parent2_ref]

    # hom alt x hom alt case:
    count_mendelian_diploid[parent1_alt & parent2_alt] = \
        2 - progeny_genotypes_012[parent1_alt & parent2_alt]

    # het vs het case
    # not needed as all ok

    # hom ref vs hom alt
    # both 0 and 2 unacceptable
    hom_ref_alt = (parent1_alt & parent2_ref) | (parent1_ref & parent2_alt)
    count_mendelian_diploid[hom_ref_alt] = np.abs(progeny_genotypes_012 - 1)

    # now het vs ref
    # only a '2' is unacceptable
    het_homref = (parent1_ref & parent2_het) | (parent1_het & parent2_ref)
    count_mendelian_diploid[het_homref] = progeny_genotypes_012[het_homref]//2

    # now het vs alt
    # only a '0' is unacceptable. Implicitly convert from bool to int
    het_homalt = (parent1_alt & parent2_het) | (parent1_het & parent2_alt)
    count_mendelian_diploid[het_homalt] = (0 == progeny_genotypes_012[
        het_homalt])

    # set all missings to 0
    count_mendelian_diploid[progeny_genotypes_012 == -1] = 0

    return count_mendelian_diploid


def is_variant_non_mendelian(args):

    """Internal function that determines whether an ME has occurred for a
    single variant

    Parameters
    ----------

    A tuple which is internally unpacked to:
      missing_parent:
      classification:
      parental_gt:
      progeny_gt:

    Returns
    -------

    n : boolean array of size progeny as above

    See Also
    --------
    is_non_mendelian_diploid

    """

    missing_parent, classification, parental_gt, progeny_gt = args
    # classification = 2 * either_parent_het + same_genotype
    if missing_parent:
        return np.zeros(progeny_gt.size, dtype='bool')
    elif classification == 0:  # parents different homozygotes
        return np.array(progeny_gt != 1)  # must be het
    elif classification == 1:  # parents hom same
        return np.array(progeny_gt != parental_gt[0])
    elif classification == 2:  # parents different, 1 is het
        is_alt = np.any(parental_gt == 2)
        allowed = [1, int(is_alt)*2]
        return np.array([p not in allowed for p in progeny_gt])
    else:              # parents both het
        return np.zeros(progeny_gt.size, dtype='bool')  # anything goes


def count_non_mendelian_diploid(parental_genotypes,
                                progeny_genotypes,
                                axis=None):

    """Count `impossible` genotypes according to mendelian inheritance laws

    Parameters
    ----------

    parental_genotypes, progeny_genotypes: array_like, int
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
    is_non_mendelian_diploid

    """

    # deal with axis argument
    if axis == 'variants':
        axis = 0
    if axis == 'samples':
        axis = 1

    # count errors
    n = np.sum(is_non_mendelian_diploid(parental_genotypes,
                                        progeny_genotypes), axis=axis)

    return n
