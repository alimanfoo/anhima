"""
Utilities for calculating and plotting linkage disequilbrium.

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import itertools


def simulate_genotypes_with_ld(n_variants, n_samples, correlation=0):
    """
    Simulate a set of genotypes, where variants are in some degree of
    linkage disequilibrium with their neighbours.
    
    Parameters
    ----------
    
    n_variants : int
        The number of variants to simulate data for.
    n_samples : int
        The number of individuals to simulate data for.
    correlation : float
        The fraction of samples to copy genotypes between neighbouring 
        variants.
        
    Returns
    -------

    array
        A 2-dimensional array of genotypes, where the first dimension is
        variants, the second dimension is samples, and each genotype call
        is coded as a single integer counting the number of non-reference
        alleles (0 is homozygous reference, 1 is heterozygous, and 2 is
        homozygous alternate).
    
    """
    
    # initialise an array of random genotypes
    g = np.random.randint(size=(n_variants, n_samples), low=0, high=3)
    g = g.astype('i1')

    # determine the number of samples to copy genotypes for
    n_copy = int(correlation*n_samples)
    
    # introduce linkage disequilibrium by copying genotypes from one sample to
    # the next
    for i in range(1, n_variants):
        
        # randomly pick the samples to copy from
        sample_indices = random.sample(range(n_samples), n_copy)
        
        # view genotypes from the previous variant for the selected samples
        c = g[i-1, sample_indices]
        
        # randomly choose whether to invert the correlation
        inv = random.randint(0, 1)
        if inv:
            c = 2-c
            
        # copy across genotypes
        g[i, sample_indices] = c
        
    return g


def pairwise_genotype_ld(g):
    """
    Given a set of genotypes at biallelic variants, calculate the
    square of the correlation coefficient between each pair of
    variants.
    
    Parameters
    ----------
    
    g : array_like
        A 2-dimensional array of genotypes, where the first dimension is
        variants, the second dimension is samples, and each genotype call
        is coded as a single integer counting the number of non-reference
        alleles (for diploids, 0 is homozygous reference, 1 is heterozygous,
        and 2 is homozygous alternate). A missing genotype should be coded as 
        a negative number.
        
    Returns
    -------

    array
        A 2-dimensional array of squared correlation coefficients between
        each pair of variants.
    
    """
    
    # TODO deal with missing genotypes
    return np.power(np.corrcoef(g), 2)


def plot_ld(r_squared, cmap='Greys', flip=True, ax=None):
    """
    Make a classic triangular linkage disequilibrium plot, given an
    array of pairwise correlation coefficients between variants.
    
    Parameters
    ----------
    
    r_squared : array_like
        A square 2-dimensional array of squared correlation coefficients between 
        pairs of variants.
    cmap : color map, optional
        The color map to use when plotting. Defaults to 'Greys' (0=white,
        1=black).
    flip : bool, optional
        If True, draw the triangle upside down.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
        
    Returns
    -------

    axes
        The axes on which the plot was drawn
    
    """

    # setup axes
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes((0, 0, 1, 1))
        
    # define transformation to rotate the colormesh
    trans = mpl.transforms.Affine2D().rotate_deg_around(0, 0, -45)
    trans = trans + ax.transData 
    
    # plot the data as a colormesh
    ax.pcolormesh(r_squared, cmap=cmap, vmin=0, vmax=1, transform=trans)
    
    # cut the plot in half so we see a triangle
    ax.set_ylim(bottom=0)
    
    if flip:
        ax.invert_yaxis()
        
    # remove axis lines
    ax.set_axis_off()
    
    return ax
    

def ld_prune_pairwise(g, window_size=100, window_step=10, max_r_squared=.2):
    """
    Given a set of genotypes at biallelic variants, find a subset of
    the variants which are in approximate linkage equilibrium with
    each other.
    
    The algorithm is as follows. A window of `window_size` variants is
    taken from the beginning of the genotypes array. The genotype
    correlation coefficient is calculated between each pair of
    variants in the window. The first variant in the window is
    considered, and any other variants in the window with linkage
    above `max_r_squared` with respect to the first variant is
    excluded. The next non-excluded variant in the window is then
    considered, and so on. The window then shifts along by
    `window_step` variants, and the process is repeated.
    
    Parameters
    ----------
    
    g : array_like
        A 2-dimensional array of genotypes, where the first dimension is
        variants, the second dimension is samples, and each genotype call
        is coded as a single integer counting the number of non-reference
        alleles (for diploids, 0 is homozygous reference, 1 is heterozygous,
        and 2 is homozygous alternate). A missing genotype should be coded as 
        a negative number.
    window_size : int, optional
        The number of variants to work with at a time.
    window_step : int, optional
        The number of variants to shift the window by.
    max_r_squared : float, optional
        The maximum value of the genotype correlation coefficient, above which
        variants will be excluded.
        
    Returns
    -------

    array
        A boolean array of the same length as the number of variants,
        where a True value indicates the variant at the corresponding
        index is included, and a False value indicates the corresponding
        variant is excluded.
        
    """
    
    # set up output array
    n_variants = g.shape[0]
    included = np.ones((n_variants,), dtype=np.bool)

    # outer loop - iterate over windows
    for window_start in range(0, n_variants-window_size+1, window_step):
        window_stop = window_start + window_size
        
        # view genotypes for current window
        gw = g[window_start:window_stop, :]
        
        # calculate pairwise genotype correlation
        # TODO deal with missing genotypes
        r_squared = np.power(np.corrcoef(gw), 2)
        
        # inner loop - iterate over variants within the window
        for i in range(window_size):
            
            if included[window_start + i]:
                # look for linkage with other variants in window
                for j in range(i+1, window_size):
                    if r_squared[i, j] > max_r_squared:
                        # threshold exceeded, exclude the variant
                        included[window_start + j] = False
                    else:
                        # below threshold, leave included
                        pass

            else:
                # don't bother to look at variants previously excluded
                pass   
            
    return included


def ld_separation(r_squared):
    """
    Compile data on linkage disequilibrium and separation between
    pairs of variants.
    
    Parameters
    ----------
    
    r_squared : array_like
        A square 2-dimensional array of squared correlation coefficients between 
        pairs of variants.

    Returns
    -------

    sep : array
        Each element in the array is the separation (in number of variants)
        between a distinct pair of variants.
    cor : array
        Each element in the array is the squared genotype correlation
        coefficient between a distinct pair of variants.

    """
    
    # determine the number of variants
    n_variants = r_squared.shape[0]
    
    # determine all distinct pairs of variants
    pairs = list(itertools.combinations(range(n_variants), 2))
    
    # initialise output arrays
    sep = np.zeros((len(pairs),), dtype=np.int)
    cor = np.zeros((len(pairs),), dtype=np.float)
    
    # iterate over pairs
    for n, (i, j) in enumerate(pairs):
        sep[n] = j-i
        cor[n] = r_squared[i, j]
        
    return sep, cor


def plot_ld_separation(r_squared, 
                       max_separation=100, 
                       percentiles=(5, 95), 
                       ax=None,
                       median_plot_kwargs=dict(),
                       percentiles_plot_kwargs=dict()):
    """
    Plot the decay of linkage disequilibrium with separation between
    variants.
    
    Parameters
    ----------
    
    r_squared : array_like
        A square 2-dimensional array of squared correlation coefficients between 
        pairs of variants.
    max_separation : int, optional
        Maximum separation to consider.
    percentiles : sequence of integers, optional
        Percentiles to plot in addition to the median.
    ax : axes, optional
        Axes on which to draw.
    median_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the median line.
    percentiles_plot_kwargs : dict, optional
        Keyword arguments to pass through when plotting the percentiles.
        
    Returns
    -------

    axes
        The axes on which the plot was drawn.
    
    """

    # set up axes
    if ax is None:
        fig, ax = plt.subplots()
        
    # compile data on separations
    sep, cor = ld_separation(r_squared)
    
    # set up arrays for plotting
    cor_median = np.zeros((max_separation,), dtype='f4')
    if len(percentiles) > 0:
        cor_percentiles = np.zeros((max_separation, len(percentiles)),
                                   dtype='f4')

    # iterate over separations, compiling data
    for i in range(max_separation):
        
        # view correlations at the given separation
        c = cor[sep == i]
        
        # calculate median and percentiles
        if len(c) > 0:
            cor_median[i] = np.median(c)
            for n, p in enumerate(percentiles):
                cor_percentiles[i, n] = np.percentile(c, p)

    # plot the median
    x = range(max_separation)
    y = cor_median
    median_plot_kwargs.setdefault('linestyle', '-')
    median_plot_kwargs.setdefault('color', 'k')
    median_plot_kwargs.setdefault('linewidth', 2)
    plt.plot(x, y, label='median', **median_plot_kwargs)

    # plot percentiles
    percentiles_plot_kwargs.setdefault('linestyle', '--')
    median_plot_kwargs.setdefault('color', 'k')
    percentiles_plot_kwargs.setdefault('linewidth', 1)
    for n, p in enumerate(percentiles):
        y = cor_percentiles[:, n]
        plt.plot(x, y, label='%s%%' % p, **percentiles_plot_kwargs)
    
    # tidy up
    ax.set_xlim(left=1, right=max_separation)
    ax.set_ylim(0, 1)
    ax.set_xlabel('separation')
    ax.set_ylabel('$r^2$', rotation=0)
    
    return ax
