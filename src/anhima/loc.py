"""
Utilities for working with genomic locations.

"""


from __future__ import division, print_function


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# standard library dependencies
import bisect


def locate_region(pos, start, stop):
    """Locate the start and stop indices within the `pos` array that
    include all positions within the `start` and `stop` range.

    Parameters
    ----------

    pos : array_like
        A 1-dimensional array of genomic positions.
    start : int
        Start position of region.
    stop : int
        Stop position of region

    Returns
    -------

    loc : slice
        A slice object with the start and stop indices that capture all
        positions within the region.

    """

    start_index = bisect.bisect_left(pos, start)
    stop_index = bisect.bisect_right(pos, stop)
    loc = slice(start_index, stop_index)
    return loc


