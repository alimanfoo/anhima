# -*- coding: utf-8 -*-
"""
Miscellaneous utilities.

"""


from __future__ import division, print_function, absolute_import
from anhima.compat import range


# third party dependencies
import numpy as np
import pandas


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
    for block_start in range(0, n_rows_in, block_size):
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
    for block_start in range(0, dim_size, block_size):
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


def state_transitions(x, states):
    """Find state transitions.

    Parameters
    ----------

    x : array_like
        One-dimensional array of state values.
    states : set-like
        Set of states to consider (other state values are ignored).

    Returns
    -------

    switch_points : ndarray
        Two-dimensional array of switch points, where each row stores a pair
        of indices corresponding to the indices either side of a switch.
    transitions : ndarray
        Two-dimensional array of transitions, where each row stores a pair
        of state values corresponding to states either side of a switch.

    """

    # check inputs
    x = np.asarray(x)
    assert x.ndim == 1

    # setup output variables
    switch_points = list()
    transitions = list()

    # utility variables
    prv = None
    prv_idx = None

    # iterate over state values
    for cur_idx, cur in enumerate(x):

        if cur not in states:
            # ignore
            pass

        else:

            if prv is None:
                # first informative state
                pass

            elif cur != prv:
                # record a state transition
                switch = prv_idx, cur_idx
                switch_points.append(switch)
                transition = prv, cur
                transitions.append(transition)

            # advance
            prv = cur
            prv_idx = cur_idx

    return np.array(switch_points), np.array(transitions)


def tabulate_state_transitions(x, states, pos):
    """TODO

    """

    # check inputs
    x = np.asarray(x)
    assert x.ndim == 1
    pos = np.asarray(pos)
    assert pos.ndim == 1
    assert x.size == pos.size

    switch_points, transitions = state_transitions(x, states)
    switch_positions = np.take(pos, switch_points)
    data = [('lidx', switch_points[:, 0]),
            ('ridx', switch_points[:, 1]),
            ('lpos', switch_positions[:, 0]),
            ('rpos', switch_positions[:, 1]),
            ('lval', transitions[:, 0]),
            ('rval', transitions[:, 1])]
    df = pandas.DataFrame.from_items(data)
    return df


def tabulate_state_blocks(x, states, pos):
    """TODO

    """

    # check inputs
    x = np.asarray(x)
    assert x.ndim == 1
    n = x.size
    pos = np.asarray(pos)
    assert pos.ndim == 1
    assert pos.size == n

    blocks = list()

    df_switches = tabulate_state_transitions(x, states, pos)
    for i, switch in enumerate(df_switches.values):
        lidx, ridx, lpos, rpos, lval, rval = switch
        block_stop_min_idx = lidx
        block_stop_max_idx = ridx
        block_stop_min_pos = lpos
        block_stop_max_pos = rpos
        block_state = lval

        if i == 0:
            # special case the first switch
            block_start_min_idx = 0
            block_start_max_idx = 0
            block_start_min_pos = pos[0]
            block_start_max_pos = pos[0]
            block_is_marginal = True
            block_size_min = block_stop_min_idx
            block_size_max = block_stop_max_idx - 1
        else:
            previous_switch = df_switches.iloc[i-1]
            block_start_min_idx = previous_switch.lidx
            block_start_max_idx = previous_switch.ridx
            block_start_min_pos = previous_switch.lpos
            block_start_max_pos = previous_switch.rpos
            block_is_marginal = False
            block_size_min = block_stop_min_idx - block_start_max_idx + 1
            block_size_max = block_stop_max_idx - block_start_min_idx - 1

        y = x[block_start_max_idx:block_stop_min_idx+1]
        block_support = np.count_nonzero(y == block_state)
        block_length_min = block_stop_min_pos - block_start_max_pos
        block_length_max = block_stop_max_pos - block_start_min_pos

        block = (block_start_min_idx, block_start_max_idx,
                 block_stop_min_idx, block_stop_max_idx,
                 block_start_min_pos, block_start_max_pos,
                 block_stop_min_pos, block_stop_max_pos,
                 block_state, block_support,
                 block_size_min, block_size_max,
                 block_length_min, block_length_max, block_is_marginal)
        blocks.append(block)

    # special case the last block
    previous_switch = df_switches.iloc[-1]
    block_start_min_idx = previous_switch.lidx
    block_start_max_idx = previous_switch.ridx
    block_start_min_pos = previous_switch.lpos
    block_start_max_pos = previous_switch.rpos
    block_is_marginal = True
    block_stop_min_idx = n-1
    block_stop_max_idx = n-1
    block_stop_min_pos = pos[-1]
    block_stop_max_pos = pos[-1]
    block_state = previous_switch.rval
    y = x[block_start_max_idx:block_stop_min_idx+1]
    block_support = np.count_nonzero(y == block_state)
    block_size_min = block_stop_min_idx - block_start_max_idx + 1
    block_size_max = block_stop_max_idx - block_start_min_idx
    block_length_min = block_stop_min_pos - block_start_max_pos
    block_length_max = block_stop_max_pos - block_start_min_pos

    block = (block_start_min_idx, block_start_max_idx,
             block_stop_min_idx, block_stop_max_idx,
             block_start_min_pos, block_start_max_pos,
             block_stop_min_pos, block_stop_max_pos,
             block_state, block_support,
             block_size_min, block_size_max,
             block_length_min, block_length_max, block_is_marginal)
    blocks.append(block)

    columns = ('start_min_idx', 'start_max_idx',
               'stop_min_idx', 'stop_max_idx',
               'start_min_pos', 'start_max_pos',
               'stop_min_pos', 'stop_max_pos',
               'state', 'support',
               'size_min', 'size_max',
               'length_min', 'length_max', 'is_marginal')
    df = pandas.DataFrame.from_records(blocks, columns=columns)
    return df
