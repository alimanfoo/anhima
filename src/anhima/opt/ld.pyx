# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False


from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport sqrt


cdef inline double corrcoef_uint8(np.uint8_t[:] x, np.uint8_t[:] y):
    cdef int n = x.size
    cdef int i
    cdef int xtot = 0
    cdef int ytot = 0
    for i in range(n):
        xtot += x[i]
        ytot += y[i]
    cdef double ex = xtot/n
    cdef double ey = ytot/n
    cdef double num = 0
    cdef double xsqsumdev = 0
    cdef double ysqsumdev = 0
    for i in range(n):
        num += (x[i] - ex) * (y[i] - ey)
        xsqsumdev += (x[i] - ex)**2
        ysqsumdev += (y[i] - ey)**2
    cdef double den = sqrt(xsqsumdev) * sqrt(ysqsumdev)
    return num/den


def ld_prune_pairwise_uint8(np.uint8_t[:, :] gn,
                            int window_size=100,
                            int window_step=10,
                            float max_r_squared=.2):
    """Optimised function for pruning variants in approximate LD."""

    cdef int window_start, window_stop, n_variants, i, j
    cdef np.uint8_t[:] included, x, y
    cdef double[:, :] r_squared

    # set up output array
    n_variants = gn.shape[0]
    included = np.ones((n_variants,), dtype='u1')

    # outer loop - iterate over windows
    for window_start in range(0, n_variants, window_step):

        # determine extent of the current window
        window_stop = min(window_start + window_size, n_variants)

        # calculate pairwise genotype correlation
        if window_start == 0:
            # initialise correlation coefficients
            r_squared = np.corrcoef(gn[window_start:window_stop, :])**2
        else:
            # move up data from previous window
            r_squared[:window_size-window_step, :window_size-window_step] = \
                r_squared[window_step:, window_step:]

        # inner loop - iterate over variants within the window
        for i in range(window_stop - window_start):

            # check to see if the variant was previously excluded
            if included[window_start + i]:

                # look for linkage with other variants in window
                for j in range(i + 1, window_stop - window_start):

                    # check to see if the variant was previously excluded
                    if included[window_start + j]:

                        # do we need to calculate r_squared?
                        if j >= (window_size - window_step):
                            x = gn[window_start + i, :]
                            y = gn[window_start + j, :]
                            r_squared[i, j] = corrcoef_uint8(x, y)**2

                        if r_squared[i, j] > max_r_squared:
                            # threshold exceeded, exclude the variant
                            included[window_start + j] = 0

    return np.asarray(included).astype('b1')

