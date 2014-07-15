"""TODO

"""


from __future__ import division, print_function, unicode_literals


__author__ = "Alistair Miles <alimanfoo@googlemail.com>"


# internal dependencies
import anhima.loc


def load_region(callset, chrom, start_position, stop_position, 
                variants_fields=None, 
                calldata_fields=None):
    """TODO

    """

    # obtain chromosome group
    grp_chrom = callset[chrom]

    # setup output variables
    variants = dict()
    calldata = dict()

    # obtain variant positions
    pos = grp_chrom['variants']['POS']

    # extract variants data
    if variants_fields:
        for f in variants_fields:
            variants[f] = anhima.loc.take_region(grp_chrom['variants'][f],
                                                 pos,
                                                 start_position,
                                                 stop_position)

    # extract calldata
    if calldata_fields:
        for f in calldata_fields:
            calldata[f] = anhima.loc.take_region(grp_chrom['calldata'][f],
                                                 pos,
                                                 start_position,
                                                 stop_position)

    return variants, calldata

