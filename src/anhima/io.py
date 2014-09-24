"""
Input/output utilities.

"""


from __future__ import division, print_function, unicode_literals, \
    absolute_import


# third party dependencies
import numpy as np
import itertools


def save_tped(path, genotypes, ref, alt, pos,
              chromosome='0',
              identifier=None,
              genetic_distance=None):
    """Write biallelic diploid genotype data to a file using the Plink
    transposed format (TPED).

    Parameters
    ----------

    path : string or file-like
        Path of file to write, or file-like object to write to.
    genotypes : array_like, int
        An array of shape (n_variants, n_samples, 2) where each
        element of the array is an integer corresponding to an allele index
        (-1 = missing, 0 = reference allele, 1 = first alternate allele, etc.).
    ref : array_like, string
        A 1-dimensional array of single character strings encoding the
        reference nucleotide.
    alt : array_like, string
        A 1-dimensional array of single character strings encoding the
        alternate nucleotide.
    pos : array_like, int
        A 1-dimensional array of genomic positions.
    chromosome : string or array_like, string, optional
        Either a single string (if all variants are from the same
        chromosome/contig) or an array of strings with the chromosome of each
        variant.
    identifier : array_like, string, optional
        An array of SNP identifiers. If not provided, identifiers will be
        created based on the variant index, e.g., 'snp1', 'snp2', etc.
    genetic_distance : array_like, float
        An array of genetic distances. If not provided, a zero value ('0') will
        be written for all variants.

    """

    # check genotypes
    genotypes = np.asarray(genotypes)
    assert genotypes.ndim == 3
    assert genotypes.shape[2] == 2, 'genotypes must be diploid'
    assert np.amax(genotypes) < 2, 'genotypes must be biallelic'
    n_variants = genotypes.shape[0]

    # check ref
    ref = np.asarray(ref)
    assert ref.ndim == 1
    assert ref.shape[0] == n_variants

    # check alt
    alt = np.asarray(alt)
    assert alt.ndim == 1
    assert alt.shape[0] == n_variants

    # check pos
    pos = np.asarray(pos)
    assert pos.ndim == 1
    assert pos.shape[0] == n_variants

    # check chromosome
    if isinstance(chromosome, basestring):
        chromosome = np.array([chromosome] * n_variants)
    else:
        chromosome = np.asarray(chromosome)
        assert chromosome.ndim == 1
        assert chromosome.shape[0] == n_variants

    # check identifier
    if identifier is None:
        identifier = np.array(['snp%s' % i for i in pos])
    else:
        identifier = np.asarray(identifier)
        assert identifier.shape[0] == n_variants

    # check genetic distance
    if genetic_distance is None:
        genetic_distance = np.zeros((n_variants,))
    else:
        genetic_distance = np.asarray(genetic_distance)
        assert genetic_distance.shape[0] == n_variants

    # setup output file
    tped_needs_closing = False
    if isinstance(path, basestring):
        tped_file = open(path, 'w')
        tped_needs_closing = True
    else:
        # assume file-like
        tped_file = path

    try:
        for row_data in itertools.izip(genotypes, ref, alt, pos, chromosome,
                                       identifier, genetic_distance):
            out_string = _get_tped_row(*row_data)
            tped_file.write(out_string + '\n')

    finally:
        if tped_needs_closing:
            tped_file.close()


def _convert_gts_to_strings(genotypes, ref, alt):

    lu = {-1: '0', 0: ref, 1: alt}
    return [lu[a] + ' ' + lu[b] for a, b in genotypes]


def _get_tped_row(gt_data, reference, alternate, position, contig, iden,
                  genetic_dist):

    str_gts = _convert_gts_to_strings(gt_data, reference, alternate)
    return "\t".join([contig,
                      iden,
                      str(genetic_dist),
                      str(position)] + str_gts)
