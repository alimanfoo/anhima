"""
Miscellaneous utilities.

"""

from __future__ import division, print_function, unicode_literals, \
    absolute_import


# third party dependencies
import numpy as np


def block_take2d(dataset, row_indices, col_indices=None, block_size=None):
    """Select rows and optionally columns from a Numpy array or HDF5
    dataset with 2 or more dimensions.

    Parameters
    ----------

    dataset : array_like or HDF5 dataset
        The input dataset.
    row_indices : sequence of ints
        The indices of the selected rows. N.B., will be sorted in ascending
        order.
    col_indices : sequence of ints, optional
        The indices of the selected columns. If not provided, all columns
        will be returned.
    block_size : int, optional
        The size (in number of rows) of the block of data to process at a time.

    Returns
    -------

    out : ndarray
        An array containing the selected rows and columns.

    See Also
    --------

    anhima.util.block_compress2d, anhima.h5.take2d_pointsel

    Notes
    -----

    This function is mainly a work-around for the fact that fancy indexing via
    h5py is currently slow, and fancy indexing along more than one axis is not
    supported. The function works by reading the entire dataset in blocks of
    `block_size` rows, and processing each block in memory using numpy.

    """

    # N.B., make sure row_indices are sorted
    row_indices = np.asarray(row_indices)
    row_indices.sort()

    # how many rows are we selecting?
    n_rows_in = dataset.shape[0]
    n_rows_out = len(row_indices)

    # how many columns are we selecting?
    n_cols_in = dataset.shape[1]
    if col_indices:
        n_cols_out = len(col_indices)
    else:
        n_cols_out = n_cols_in

    # setup output array
    out_shape = (n_rows_out, n_cols_out) + dataset.shape[2:]
    out = np.empty(out_shape, dtype=dataset.dtype)

    # determine block size
    if block_size is None:
        if hasattr(dataset, 'chunks') and dataset.chunks is not None:
            # use dataset chunk height
            block_size = dataset.chunks[0]
        else:
            # use arbitrary number
            block_size = 1000

    # iterate block-wise
    offset = 0
    for block_start in xrange(0, n_rows_in, block_size):
        block_stop = min(block_start+block_size, n_rows_in)

        # how many indices to process in this block?
        i = np.searchsorted(row_indices, block_start)
        j = np.searchsorted(row_indices, block_stop)
        n = j-i
        ridx = row_indices[i:j]

        # only do anything if there are indices for this block
        if n:

            # load data for this block
            a = dataset[block_start:block_stop]

            # take rows
            b = np.take(a, ridx-block_start, axis=0)

            # take columns
            if col_indices:
                b = np.take(b, col_indices, axis=1)

            # store output
            out[offset:offset+n, ...] = b

            # keep track of offset
            offset += n

    return out


def block_compress2d(dataset, row_condition, col_condition=None,
                     block_size=None):
    """Select rows and optionally columns from a Numpy array or HDF5
    dataset with 2 or more dimensions.

    Parameters
    ----------

    dataset : array_like or HDF5 dataset
        The input dataset.
    row_condition : array_like, bool
        A boolean array indicating the selected rows.
    col_indices : array_like, bool, optonal
        A boolean array indicated the selected columns. If not provided,
        all columns will be returned.
    block_size : int, optional
        The size (in number of rows) of the block of data to process at a time.

    Returns
    -------

    out : ndarray
        An array containing the selected rows and columns.

    See Also
    --------

    anhima.util.block_take2d, anhima.h5.take2d_pointsel

    Notes
    -----

    This function is mainly a work-around for the fact that fancy indexing via
    h5py is currently slow, and fancy indexing along more than one axis is not
    supported. The function works by reading the entire dataset in blocks of
    `block_size` rows, and processing each block in memory using numpy.

    """

    row_indices = np.nonzero(row_condition)[0]
    col_indices = np.nonzero(col_condition)[0] if col_condition is not None \
        else None
    return block_take2d(dataset, row_indices, col_indices,
                        block_size=block_size)


def block_apply(f, dataset, block_size=None, out=None):
    """Apply function `f` to `dataset` split along the first axis into
    contiguous slices of `block_size`. The result should be equivalent to
    calling ``f(dataset)`` directly, however may require less total memory,
    especially if `dataset` is an HDF5 dataset.

    Parameters
    ----------

    f : function
        The function to apply.
    dataset : array_like or HDF5 dataset
        The input dataset.
    block_size : int, optional
        The size (in number of items along `axis`) of the blocks passed to `f`.
    out : array_like or HDF5 dataset, optional
        If given, used to store the output.

    Returns
    -------

    out : ndarray
        The result of applying `f` to `dataset` blockwise.

    """

    # determine block size
    if block_size is None:
        if hasattr(dataset, 'chunks') and dataset.chunks is not None:
            # use dataset chunk size along slice axis
            block_size = dataset.chunks[0]
        else:
            # use arbitrary number
            block_size = 1000

    # determine total size along slice axis
    dim_size = dataset.shape[0]

    # iterate over blocks
    for block_start in xrange(0, dim_size, block_size):
        block_stop = min(block_start + block_size, dim_size)

        # load input block
        x = dataset[block_start:block_stop, ...]

        # compute output block
        y = f(x)

        if out is None:
            # initialise output array
            out_shape = list(y.shape)
            out_shape[0] = dim_size
            out = np.empty(out_shape, y.dtype)

        # store output block
        out[block_start:block_stop, ...] = y

    return out


