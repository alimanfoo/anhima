"""
Utilities for working with related individuals (crosses, families, etc.).

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/ped.ipynb

"""


from __future__ import division, print_function, unicode_literals


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
