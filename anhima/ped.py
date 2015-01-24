# -*- coding: utf-8 -*-
"""
Utilities for working with related individuals (crosses, families, etc.).

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/ped.ipynb

"""  # noqa


from __future__ import division, print_function, absolute_import


# third party dependencies
import numpy as np
import numexpr as ne
import pandas

# internal dependencies
import anhima.gt

# constants to represent inheritance states
INHERIT_UNDETERMINED = 0
INHERIT_PARENT1 = 1
INHERIT_PARENT2 = 2
INHERIT_NONSEG_REF = 3
INHERIT_NONSEG_ALT = 4
INHERIT_NONPARENTAL = 5
INHERIT_PARENT_MISSING = 6
INHERIT_MISSING = 7
INHERITANCE_STATES = range(8)
INHERITANCE_LABELS = ('undetermined', 'parent1', 'parent2', 'non-seg ref',
                      'non-seg alt', 'non-parental', 'parent missing',
                      'missing')


def diploid_inheritance(parent_diplotype, gamete_haplotypes):
    """
    Determine the transmission of parental alleles to a set of gametes.

    Parameters
    ----------

    parent_diplotype : array_like, shape (n_variants, 2)
        An array of phased genotypes for a single diploid individual, where
        each element of the array is an integer corresponding to an allele
        index (-1 = missing, 0 = reference allele, 1 = first alternate allele,
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
    parent1 = parent_diplotype[:, 0, np.newaxis]  # noqa
    parent2 = parent_diplotype[:, 1, np.newaxis]  # noqa
    gamete_is_missing = gamete_haplotypes < 0
    parent_is_missing = np.any(parent_diplotype < 0, axis=1)
    parent_is_hom_ref = anhima.gt.is_hom_ref(parent_diplotype)[:, np.newaxis]  # noqa
    parent_is_het = anhima.gt.is_het(parent_diplotype)[:, np.newaxis]  # noqa
    parent_is_hom_alt = anhima.gt.is_hom_alt(parent_diplotype)[:, np.newaxis]  # noqa

    # need this for broadcasting, but also need to retain original for later
    parent_is_missing_bc = parent_is_missing[:, np.newaxis]  # noqa

    # N.B., use numexpr below where possible to avoid temporary arrays

    # utility variable, identify allele calls where inheritance can be
    # determined
    callable = ne.evaluate('~gamete_is_missing & ~parent_is_missing_bc')  # noqa
    callable_seg = ne.evaluate('callable & parent_is_het')  # noqa

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


def diploid_mendelian_error_biallelic(parental_genotypes, progeny_genotypes):
    """Implementation of function to find Mendelian errors optimised for
    biallelic variants.

    """

    # recode genotypes for convenience
    parental_genotypes_012 = anhima.gt.as_012(parental_genotypes)
    progeny_genotypes_012 = anhima.gt.as_012(progeny_genotypes)

    # convenience variables
    p1 = parental_genotypes_012[:, 0, np.newaxis]  # noqa
    p2 = parental_genotypes_012[:, 1, np.newaxis]  # noqa
    o = progeny_genotypes_012  # noqa

    # enumerate all possible combinations of Mendel error genotypes
    ex = '((p1 == 0) & (p2 == 0) & (o == 1))' \
        ' + ((p1 == 0) & (p2 == 0) & (o == 2)) * 2' \
        ' + ((p1 == 2) & (p2 == 2) & (o == 1))' \
        ' + ((p1 == 2) & (p2 == 2) & (o == 0)) * 2' \
        ' + ((p1 == 0) & (p2 == 2) & (o == 0))' \
        ' + ((p1 == 0) & (p2 == 2) & (o == 2))' \
        ' + ((p1 == 2) & (p2 == 0) & (o == 0))' \
        ' + ((p1 == 2) & (p2 == 0) & (o == 2))' \
        ' + ((p1 == 0) & (p2 == 1) & (o == 2))' \
        ' + ((p1 == 1) & (p2 == 0) & (o == 2))' \
        ' + ((p1 == 2) & (p2 == 1) & (o == 0))' \
        ' + ((p1 == 1) & (p2 == 2) & (o == 0))'
    errors = ne.evaluate(ex).astype('u1')

    return errors


def diploid_mendelian_error_multiallelic(parental_genotypes,
                                          progeny_genotypes,
                                          max_allele):
    """Implementation of function to find Mendelian errors generalised for
    multiallelic variants.

    """

    # transform genotypes into per-call allele counts
    alleles = range(max_allele + 1)
    p = anhima.gt.as_allele_counts(parental_genotypes, alleles=alleles)
    o = anhima.gt.as_allele_counts(progeny_genotypes, alleles=alleles)

    # detect nonparental and hemiparental inheritance by comparing allele
    # counts between parents and progeny
    ps = p.sum(axis=1)[:, np.newaxis]  # add allele counts for both parents
    ac_diff = (o - ps).astype('i1')
    ac_diff[ac_diff < 0] = 0
    # sum over all alleles
    errors = np.sum(ac_diff, axis=2).astype('u1')

    # detect uniparental inheritance by finding cases where no alleles are
    # shared between parents, then comparing progeny allele counts to each
    # parent
    pc1 = p[:, 0, np.newaxis, :]
    pc2 = p[:, 1, np.newaxis, :]
    # find variants where parents don't share any alleles
    is_shared_allele = (pc1 > 0) & (pc2 > 0)
    no_shared_alleles = ~np.any(is_shared_allele, axis=2)
    # find calls where progeny genotype is identical to one or the other parent
    errors[
        no_shared_alleles
        & (np.all(o == pc1, axis=2)
           | np.all(o == pc2, axis=2))
    ] = 1

    # retrofit where either or both parent has a missing call
    is_parent_missing = anhima.gt.is_missing(parental_genotypes)
    errors[np.any(is_parent_missing, axis=1)] = 0

    return errors


def diploid_mendelian_error(parental_genotypes, progeny_genotypes):
    """Find impossible genotypes according to Mendelian inheritance laws.

    Parameters
    ----------

    parental_genotypes : array_like, int
        An array of shape (n_variants, 2, 2) where each element of the array
        is an integer corresponding to an allele index (-1 = missing,
        0 = reference allele, 1 = first alternate allele, 2 = second
        alternate allele, etc.).
    progeny_genotypes : array_like, int
        An array of shape (n_variants, n_progeny, 2) where each element of the
        array is an integer corresponding to an allele index (-1 = missing,
        0 = reference allele, 1 = first alternate allele, 2 = second
        alternate allele, etc.).

    Returns
    -------

    errors : ndarray, uint8
        An array of shape (n_variants, n_progeny) where each element counts
        the number of non-Mendelian alleles in a progeny genotype call.

    See Also
    --------

    count_diploid_mendelian_error

    Notes
    -----

    Not applicable to polyploid genotype calls.

    Applicable to multiallelic variants.

    Assumes that genotypes are unphased.

    """

    # check inputs
    parental_genotypes = np.asarray(parental_genotypes)
    progeny_genotypes = np.asarray(progeny_genotypes)
    assert parental_genotypes.ndim == 3
    assert progeny_genotypes.ndim == 3

    # check the number of variants is equal in parents and progeny
    assert parental_genotypes.shape[0] == progeny_genotypes.shape[0]

    # check the number of parents
    assert parental_genotypes.shape[1] == 2

    # check the ploidy
    assert parental_genotypes.shape[2] == progeny_genotypes.shape[2] == 2

    # determine which implementation to use
    max_allele = max(np.amax(parental_genotypes), np.amax(progeny_genotypes))
    if max_allele < 2:
        errors = diploid_mendelian_error_biallelic(parental_genotypes,
                                                    progeny_genotypes)
    else:
        errors = diploid_mendelian_error_multiallelic(parental_genotypes,
                                                       progeny_genotypes,
                                                       max_allele)

    return errors


def count_diploid_mendelian_error(parental_genotypes,
                                  progeny_genotypes,
                                  axis=None):
    """Count impossible genotypes according to Mendelian inheritance laws,
    summed over all progeny genotypes, or summed along variants or samples.

    Parameters
    ----------

    parental_genotypes : array_like, int
        An array of shape (n_variants, 2, 2) where each element of the array
        is an integer corresponding to an allele index (-1 = missing,
        0 = reference allele, 1 = first alternate allele, 2 = second
        alternate allele, etc.).
    progeny_genotypes : array_like, int
        An array of shape (n_variants, n_progeny, 2) where each element of the
        array is an integer corresponding to an allele index (-1 = missing,
        0 = reference allele, 1 = first alternate allele, 2 = second
        alternate allele, etc.).
    axis : int, optional
        The axis along which to count (0 = variants, 1 = samples).

    Returns
    -------

    n : int or array
        If `axis` is None, returns the total number of Mendelian errors. If
        `axis` is specified, returns the sum along the given `axis`.

    See Also
    --------

    diploid_mendelian_error

    """

    # sum errors
    n = np.sum(diploid_mendelian_error(parental_genotypes,
                                       progeny_genotypes),
               axis=axis)

    return n


def impute_inheritance_nearest(inheritance, pos, pos_impute):
    """Impute inheritance at unknown positions, by copying from
    nearest neighbouring position where inheritance is known.

    Parameters
    ----------

    inheritance : array_like, int, shape (n_variants, n_gametes)
        An array of integers coding the allelic inheritance state at the
        known positions.
    pos : array_like, int, shape (n_variants,)
        Array of genomic positions at which `inheritance` was determined.
    pos_impute : array_like, int
        Array of positions at which to impute inheritance.

    Returns
    -------

    imputed_inheritance : ndarray, int
        An array of integers coding the imputed allelic inheritance.

    """

    # check inputs
    inheritance = np.asarray(inheritance)
    assert inheritance.ndim == 2
    pos = np.asarray(pos)
    assert pos.ndim == 1
    pos_impute = np.asarray(pos_impute)
    assert pos_impute.ndim == 1
    n_variants = pos.size
    assert inheritance.shape[0] == n_variants

    # find indices of neighbouring variants
    indices_left = np.clip(np.searchsorted(pos, pos_impute), 0, n_variants - 1)
    indices_right = np.clip(indices_left + 1, 0, n_variants - 1)
    inh_left = np.take(inheritance, indices_left, axis=0)
    inh_right = np.take(inheritance, indices_right, axis=0)

    # find positions of neighbouring variants
    pos_left = np.take(pos, indices_left)
    pos_right = np.take(pos, indices_right)

    # compute distance to neighbours
    dist_left = np.abs(pos_impute - pos_left)
    dist_right = np.abs(pos_right - pos_impute)

    # build output
    out = np.zeros_like(inh_left)
    out[dist_left < dist_right] = inh_left[dist_left < dist_right]
    out[dist_left > dist_right] = inh_right[dist_left > dist_right]

    # # use neighbour from other side where missing
    # override_left = ((dist_left < dist_right)[:, np.newaxis]
    #                  & (out == INHERIT_MISSING))
    # out[override_left] = inh_right[override_left]
    # override_right = ((dist_left > dist_right)[:, np.newaxis]
    #                   & (out == INHERIT_MISSING))
    # out[override_right] = inh_left[override_right]

    return out


def tabulate_inheritance_switches(inheritance, pos, gametes=None):
    """Tabulate switches in inheritance.

    Parameters
    ----------

    inheritance : array_like, int, shape (n_variants, n_gametes)
        An array of integers coding the allelic inheritance state at the
        known positions.
    pos : array_like, int, shape (n_variants,)
        Array of genomic positions at which `inheritance` was determined.
    gametes : sequence, length (n_gametes), optional

    Returns
    -------

    df : DataFrame
        A table of all inheritance switches observed in the gametes,
        where each row corresponds to a switch from one parental allele to
        another. The table has a hierarchical index, where the first level
        corresponds to the gamete.

    See Also
    --------

    tabulate_inheritance_blocks

    """

    # check inputs
    inheritance = np.asarray(inheritance)
    assert inheritance.ndim == 2
    n_variants, n_gametes = inheritance.shape
    pos = np.asarray(pos)
    assert pos.ndim == 1
    assert pos.size == n_variants
    if gametes is None:
        gametes = np.arange(n_gametes)
    else:
        gametes = np.asarray(gametes)
        assert gametes.ndim == 1
        assert gametes.size == n_gametes

    states = INHERIT_PARENT1, INHERIT_PARENT2
    dfs = [anhima.util.tabulate_state_transitions(inheritance[:, i],
                                                  states,
                                                  pos)
           for i in range(n_gametes)]
    df = pandas.concat(dfs, keys=gametes)
    return df


def tabulate_inheritance_blocks(inheritance, pos, gametes=None):
    """Tabulate inheritance blocks.

    Parameters
    ----------

    inheritance : array_like, int, shape (n_variants, n_gametes)
        An array of integers coding the allelic inheritance state at the
        known positions.
    pos : array_like, int, shape (n_variants,)
        Array of genomic positions at which `inheritance` was determined.
    gametes : sequence, length (n_gametes), optional

    Returns
    -------

    df : DataFrame
        A table of all inheritance blocks observed in the gametes,
        where each row corresponds to a block of inheritance from a single
        parent. The table has a hierarchical index, where the first level
        corresponds to the gamete.

    See Also
    --------

    tabulate_inheritance_switches

    """

    # check inputs
    inheritance = np.asarray(inheritance)
    assert inheritance.ndim == 2
    n_variants, n_gametes = inheritance.shape
    pos = np.asarray(pos)
    assert pos.ndim == 1
    assert pos.size == n_variants
    if gametes is None:
        gametes = np.arange(n_gametes)
    else:
        gametes = np.asarray(gametes)
        assert gametes.ndim == 1
        assert gametes.size == n_gametes

    states = INHERIT_PARENT1, INHERIT_PARENT2
    dfs = [anhima.util.tabulate_state_blocks(inheritance[:, i],
                                             states,
                                             pos)
           for i in range(n_gametes)]
    df = pandas.concat(dfs, keys=gametes)
    return df


def inheritance_block_masks(inheritance, pos):
    """Construct arrays suitable for masking the original inheritance array
    using the attributes of the inheritance blocks.

    Parameters
    ----------

    inheritance : array_like, int, shape (n_variants, n_gametes)
        An array of integers coding the allelic inheritance state at the
        known positions.
    pos : array_like, int, shape (n_variants,)
        Array of genomic positions at which `inheritance` was determined.

    Returns
    -------

    block_support : ndarray, int, shape (n_variants, n_gametes)
        An array where each item has the number of variants supporting the
        containing inheritance block.
    block_length_min : ndarray, int, shape (n_variants, n_gametes)
        An array where each item has the minimal length of the containing
        inheritance block.

    See Also
    --------

    tabulate_inheritance_blocks

    """

    # check inputs
    inheritance = np.asarray(inheritance)
    assert inheritance.ndim == 2
    n_variants, n_gametes = inheritance.shape
    pos = np.asarray(pos)
    assert pos.ndim == 1
    assert pos.size == n_variants

    df_all_blocks = tabulate_inheritance_blocks(inheritance, pos)
    block_support = np.empty(inheritance.shape, dtype=int)
    block_support.fill(-1)
    block_length_min = np.empty(inheritance.shape, dtype=pos.dtype)
    block_support.fill(-1)

    for ig in range(n_gametes):

        df_blocks = df_all_blocks.loc[ig]

        for ib, block in df_blocks.iterrows():

            block_support[block.start_max_idx:block.stop_max_idx, ig] = \
                block.support
            block_length_min[block.start_max_idx:block.stop_max_idx, ig] = \
                block.length_min

    return block_support, block_length_min
