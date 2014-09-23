"""
Utility functions for working with data stored in HDF5 files.

**HDF5 file convention for variant call sets**

Note that this module assumes that data for a variant call set has been
organised in an HDF5 file following a particular convention. Briefly,
the convention is as follows.

An HDF5 file may contain one or more call sets. Each call set is stored
within a separate group. A call set may be stored within the root group.

Within each call set, data are grouped by chromosome.

Within each chromosome group, there are two subgroups,
named `variants` and `calldata`.

The `variants` group contains one or more datasets holding data on the
variants in the call set. The first dimension of all `variants` datasets
must have the same length, being the number of variants on the chromosome.

The `variants` group must contain a `POS` dataset holding the genome
positions of the variants.

The `calldata` group contains one or more datasets holding data relating to
genotype calls. The first dimension of all `calldata` datasets must have
the same length, being the number of variants on the chromosome. The second
dimension of all `calldata` datasets must have the same length,
being the number of samples in the cohort.

So, for example, an HDF5 file containing a SNP call set for a cohort of
*Anopheles gambiae* samples with chromosomes (2R, 2L, 3R, 3L, X)
might be organised as follows::

    / [callset group]
    /2L [chromosome group]
    /2L/variants [variants group]
    /2L/variants/POS [dataset, shape (n_variants,), dtype int32]
    /2L/variants/REF [dataset, shape (n_variants,), dtype S1]
    /2L/variants/ALT [dataset, shape (n_variants, 3), dtype S1]
    /2L/variants/MQ [dataset, shape (n_variants,), dtype f4]
    /2L/variants/...
    /2L/calldata [calldata group]
    /2L/calldata/genotype [dataset, shape (n_variants, n_samples, ploidy), dtype int8]
    /2L/calldata/DP [dataset, shape (n_variants, n_samples), dtype=int32]
    /2L/calldata/...
    /3L/variants/...
    /3L/calldata/...
    /...

"""


from __future__ import division, print_function, unicode_literals, \
    absolute_import


# standard library dependencies
import bisect
import itertools


# third party dependencies
import numpy as np
import h5py
import numexpr


# internal dependencies
import anhima.loc


def load_region(callset, chrom, start_position=0, stop_position=None,
                variants_fields=None, 
                calldata_fields=None,
                variants_query=None):
    """Load data into memory from `callset` for the given region.

    Parameters
    ----------

    callset : HDF5 file or group
        A file or group containing a variant call set.
    chrom : string
        The chromosome to extract data for.
    start_position : int, optional
        The start position for the region to extract data for.
    stop_position : int, optional
        The stop position for the region to extract data for.
    variants_fields : sequence of strings, optional
        Names of the variants datasets to extract.
    calldata_fields : sequence of strings, optional
        Names of the calldata datasets to extract.
    variants_query : string, optional
        A query to filter variants. Note that this query is applied 
        after data for the region has been loaded, so any fields 
        referenced in this query need to be included in `variants_fields`.

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
            
    # select variants
    if variants_query is not None:
        condition = numexpr.evaluate(variants_query, local_dict=variants)
        for f in variants:
            variants[f] = np.compress(condition, variants[f], axis=0)
        for f in calldata:
            calldata[f] = np.compress(condition, calldata[f], axis=0)

    return variants, calldata


def take2d(dataset, row_indices, col_indices=None, block_size=None):
    """
    Load selected rows and optionally columns from an HDF5 dataset with 2 or
    more dimensions.

    Parameters
    ----------

    dataset : HDF5 dataset
        The dataset to load data from.
    row_indices : sequence of ints
        The indices of the selected rows.
    col_indices : sequence of ints, optional
        The indices of the selected columns. If not provided, all columns
        will be returned.
    block_size : int, optional
        The size (in number of rows) of the block of data to load and process at
        a time.

    Returns
    -------

    out : ndarray
        An array containing the selected rows and columns.

    See Also
    --------

    take2d_points

    Notes
    -----

    This function is a work-around for the fact that fancy indexing via h5py
    is currently slow, and fancy indexing along more than one axis is not
    supported. The function works by reading the entire dataset in blocks of
    `block_size` rows, and processing each block in memory using numpy.

    """

    # make sure row_indices are sorted array
    row_indices = np.array(sorted(row_indices))

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
        if dataset.chunks is not None:
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
        i = bisect.bisect_left(row_indices, block_start)
        j = bisect.bisect_left(row_indices, block_stop)
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


def take2d_points(dataset, row_indices=None, col_indices=None,
                  block_size=1000):
    """
    Load selected rows and optionally columns from an HDF5 dataset with 2 or
    more dimensions, using HDF5 point selections.

    Parameters
    ----------

    dataset : HDF5 dataset
        The dataset to load data from.
    row_indices : sequence of ints, optional
        The indices of the selected rows. If not provided, all rows will be
        returned.
    col_indices : sequence of ints, optional
        The indices of the selected columns. If not provided, all columns
        will be returned.
    block_size : int, optional
        The size (in number of points) of the block of data to load and
        process at a time.

    Returns
    -------

    out : ndarray
        An array containing the selected rows and columns.

    See Also
    --------

    take2d

    Notes
    -----

    This function is similar to :func:`take2d` but uses an HDF5 point
    selection under the hood. Performance characteristics will be different
    to :func:`take2d`, and may be much better or much worse, depending on the
    size, shape and configuration of the dataset, and depending on the number of
    points to be selected.

    """

    n_rows_in = dataset.shape[0]
    if row_indices:
        row_indices = sorted(row_indices)
        n_rows_out = len(row_indices)
    else:
        # select all rows
        row_indices = xrange(n_rows_in)
        n_rows_out = n_rows_in

    n_cols_in = dataset.shape[1]
    if col_indices:
        # select all columns
        col_indices = sorted(col_indices)
        n_cols_out = len(col_indices)
    else:
        col_indices = xrange(n_cols_in)
        n_cols_out = n_cols_in

    n_items_out = n_rows_out * n_cols_out

    # initialise output array
    out = np.empty((n_items_out,), dtype=dataset.dtype)

    # convert indices into coordinates
    coords = itertools.product(row_indices, col_indices)

    # set up selection
    sel = h5py._hl.selections.PointSelection(dataset.shape)
    typ = h5py.h5t.py_create(dataset.dtype)

    # process blocks at a time
    for block_start in xrange(0, n_items_out, block_size):

        # materialise a block of coordinates
        selection = np.asarray(list(itertools.islice(coords, block_size)))

        # set selection
        sel.set(selection)

        # read data
        block_stop = block_start + len(selection)
        space = h5py.h5s.create_simple(sel.mshape)
        dataset.id.read(space,
                        sel._id,
                        out[block_start:block_stop],
                        typ)

    # reshape output array
    out = out.reshape(n_rows_out, n_cols_out)

    return out


def save_tped(path, callset, chrom,
              start_position=0,
              stop_position=None,
              samples=None):

    """Save genotype data from an HDF5 callset to a Plink transposed format
    file (TPED).

    Parameters
    ----------

    path : string or file-like
        Path of file to write, or file-like object to write to.
    callset : HDF5 file or group
        A file or group containing a variant call set.
    chrom : string
        The chromosome to extract data for.
    start_position : int, optional
        The start position for the region to extract data for.
    stop_position : int, optional
        The stop position for the region to extract data for.
    samples : sequence of strings, optional
        Selection of samples to extract genotypes for, defaults to all samples.

    """

    variants, calldata = load_region(callset,
                                     chrom,
                                     start_position,
                                     stop_position,
                                     variants_fields=['POS', 'REF', 'ALT'],
                                     calldata_fields=['genotype'])

    # determine samples that we will use
    if samples is None:
        genotypes = calldata['genotype']
    else:
        h5_samples = callset['chrom']['samples'][:].tolist()
        genotypes = np.take(
            calldata['genotype'],
            [h5_samples.index(s) for s in samples],
            axis=1)

    ref = variants['REF']

    alt = variants['ALT']
    if alt.ndim > 1:
        alt = alt[:, 0]

    pos = variants['POS']

    anhima.io.save_tped(path,
                        genotypes=genotypes,
                        ref=ref,
                        alt=alt,
                        pos=pos,
                        chromosome=chrom,
                        identifier=None,
                        genetic_distance=None)