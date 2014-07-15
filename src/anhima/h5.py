"""HDF5 utilities.

"""


from __future__ import division, print_function, unicode_literals


__author__ = "Alistair Miles <alimanfoo@googlemail.com>"


# internal dependencies
import anhima.loc


def load_region(callset, chrom, start_position, stop_position,
                variants_fields=None, 
                calldata_fields=None):
    """Load data into memory from `callset` for the given region.

    Parameters
    ----------

    callset : HDF5 file or group
        A file or group containing a variant call set.
    chrom : string
        The chromosome to extract data for.
    start_position : int
        The start position for the region to extract data for.
    stop_position : int
        The stop position for the region to extract data for.
    variants_fields : sequence of strings
        Names of the variants datasets to extract.
    calldata_fields : sequence of strings
        Names of the calldata datasets to extract.

    Returns
    -------

    variants : dict
        A dictionary mapping dataset identifiers to ndarrays.
    calldata : dict
        A dictionary mapping dataset identifiers to ndarrays.

    """

    # obtain chromosome group
    grp_chrom = callset[chrom]

    # setup output variables
    variants = dict()
    calldata = dict()

    # obtain variant positions
    pos = grp_chrom['variants']['POS']

    # locate region
    loc = anhima.loc.locate_region(pos, start_position, stop_position)

    # extract variants data
    if variants_fields:
        for f in variants_fields:
            variants[f] = grp_chrom['variants'][f][loc, ...]

    # extract calldata
    if calldata_fields:
        for f in calldata_fields:
            calldata[f] = grp_chrom['calldata'][f][loc, ...]

    return variants, calldata


# TODO map_chunks