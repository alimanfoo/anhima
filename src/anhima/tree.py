"""
This module provides some facilities for constructing and plotting trees. It
is mostly a wrapper around a very limited subset of functions from the R
`ape` package (Analyses of Phylogenetics and Evolution).

R must be installed, the `ape` R package must be installed, and the Python
package ``rpy2`` must be installed, e.g.::

    $ apt-get install r-base
    $ pip install rpy2
    $ R
    > install.packages("ape")

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/tree.ipynb

"""  # noqa


from __future__ import division, print_function, unicode_literals, \
    absolute_import


# standard library dependencies
import tempfile


# third party dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


_r_initialised = False
r = None
ro = None
grdevices = None
ape = None


def _init_r():
    """Private function to initialise R, only executed when needed.

    """

    global _r_initialised
    global r
    global ro
    global grdevices
    global ape

    if not _r_initialised:

        import rpy2.robjects as ro
        from rpy2.robjects import r
        from rpy2.robjects.numpy2ri import numpy2ri
        from rpy2.robjects.packages import importr
        ro.conversion.py2ri = numpy2ri
        grdevices = importr(b'grDevices')
        ape = importr(
            b'ape',
            robject_translations={
                'delta.plot': 'delta_dot_plot',
                'dist.dna': 'dist_dot_dna',
                'dist.nodes': 'dist_dot_nodes',
                'node.depth': 'node_dot_depth',
                'node.depth.edgelength': 'node_dot_depth_dot_edgelength',
                'node.height': 'node_dot_height',
                'node.height.clado': 'node_dot_height_dot_clado',
                'prop.part': 'prop_dot_part',
            }
        )

        # Define custom R functions to help with coloring tree edges by
        # population. These functions were written by Jacob Almagro-Garcia
        # <jg10@sanger.ac.uk> at the Wellcome Trust Sanger Institute.
        r("""
library(ape)


######################################################################################################################
#' Computes the number of leaves of each group that hang from each branch.
#' @param phylotree A tree of class phylo.
#' @param labelgroups A vector with the group of the tip labels (named with the labels).
#' @return A named matrix with the membership counts for each interior edge of the tree.
######################################################################################################################

computeEdgeGroupCounts <- function(phylotree, labelgroups) {

  labels <- phylotree$tip.label
  num_tips <- length(labels)
  edge_names <- unique(sort(c(phylotree$edge)))

  # This matrix will keep track of the group counts for each edge.
  edge_group_counts <- matrix(0, nrow=length(edge_names), ncol=length(unique(sort(labelgroups))))
  rownames(edge_group_counts) <- edge_names
  colnames(edge_group_counts) <- unique(labelgroups)

  # Init the leaf branches.
  sapply(1:num_tips, function(l) {
    edge_group_counts[as.character(l), as.character(labelgroups[phylotree$tip.label[l]])] <<- 1
  })

  # Sort edges by the value of the descendent
  # The first segment will contain the leaves whereas the second the branches (closer to leaves first).
  # We need to do this because leaves are numbered 1:num_tips and the branches CLOSER to the leaves
  # with higher numbers.
  edges <- phylotree$edge[order(phylotree$edge[,2]),]
  branches <- edges[num_tips:nrow(edges),]
  edges[num_tips:nrow(edges),] <- branches[order(branches[,1],decreasing=T),]
  invisible(apply(edges, 1, function(edge) {
    # Check if we are connecting a leaf.
    if(edge[2] <= num_tips) {
      e <- as.character(edge[1])
      g <- as.character(labelgroups[phylotree$tip.label[edge[2]]])
      edge_group_counts[e,g] <<- edge_group_counts[e,g] + 1
    }
    else {
      e1 <- as.character(edge[1])
      e2 <- as.character(edge[2])
      edge_group_counts[e1,] <<- edge_group_counts[e1,] + edge_group_counts[e2,]
    }
  }))
  return(edge_group_counts)
}


######################################################################################################################
#' Assigns the color of the majority group (hanging from) each branch.
#' @param phylotree A tree of class phylo.
#' @param edge_group_counts A named matrix with the group counts for each branch.
#' @param groupcolors A named vector with the color of each group.
#' @param equality_color The color to be used if there is no majority group.
#' @return A vector with the colors to be used with the tree branches.
######################################################################################################################

assignMajorityGroupColorToEdges <- function(phylotree, edge_group_counts, groupcolors, equality_color="gray") {
  edge_colors <- apply(phylotree$edge, 1, function(branch) {
    e <- as.character(branch[2])
    major_group_index <- which.max(edge_group_counts[e,])
    if(all(edge_group_counts[e,] == edge_group_counts[e,major_group_index]))
      return(equality_color)
    else
      return(groupcolors[colnames(edge_group_counts)[major_group_index]])
  })
  return(edge_colors)
}
""")  # noqa

        _r_initialised = True


def nj(dist_square, labels=None):
    """Wrapper for the `ape` ``nj`` function, which performs the
    neighbor-joining tree estimation of Saitou and Nei (1987).

    Parameters
    ----------

    dist_square : array_like, shape (`n_samples`, `n_samples`)
        A pairwise distance matrix in square form.
    labels : sequence of strings, optional
        A sequence of strings to label the tips of the tree. Must be in the
        same order as rows of the distance matrix.

    Returns
    -------

    An R object of class "phylo".

    See Also
    --------

    anhima.dist.pairwise_distance

    """

    # setup R
    _init_r()

    # normalise inputs
    dist_square = np.asarray(dist_square)
    assert dist_square.ndim == 2
    assert dist_square.shape[0] == dist_square.shape[1]

    # convert distance matrix to R
    m = ro.vectors.Matrix(dist_square)

    # assign row and column labels
    if labels:
        # map all strings to str
        labels = [str(l) for l in labels]
        s = ro.StrVector(labels)
        m.rownames = s
        m.colnames = s

    # build the tree
    tree = ape.nj(m)

    return tree


def bionj(dist_square, labels=None):
    """Wrapper for the `ape` ``bionj`` function, which performs the BIONJ
    algorithm of Gascuel (1997).

    Parameters
    ----------

    dist_square : array_like, shape (`n_samples`, `n_samples`)
        A pairwise distance matrix in square form.
    labels : sequence of strings, optional
        A sequence of strings to label the tips of the tree. Must be in the
        same order as rows of the distance matrix.

    Returns
    -------

    An R object of class "phylo".

    See Also
    --------

    anhima.dist.pairwise_distance

    """

    # setup R
    _init_r()

    # normalise inputs
    dist_square = np.asarray(dist_square)
    assert dist_square.ndim == 2
    assert dist_square.shape[0] == dist_square.shape[1]

    # convert distance matrix to R
    m = ro.vectors.Matrix(dist_square)

    # assign row and column labels
    if labels:
        # map all strings to str
        labels = [str(l) for l in labels]
        s = ro.StrVector(labels)
        m.rownames = s
        m.colnames = s

    # build the tree
    tree = ape.bionj(m)

    return tree


def plot_phylo(tree, plot_kwargs=None, add_scale_bar=None,
               filename=None, width=None, height=None, units=None, res=None,
               pointsize=None, bg=None, ax=None, imshow_kwargs=None):
    """Wrapper for the `ape` ``plot.phylo`` function, which plots phylogenetic
    trees. Plotting will use the R `png` graphics device.

    Parameters
    ----------

    tree : R object of class "phylo"
        The tree to plot.
    plot_kwargs : dict-like, optional
        A dictionary of keyword arguments that will be passed through to the
        `ape` function ``plot.phylo()``. See the documentation for the `ape`
        package for a full list of supported arguments.
    add_scale_bar : dict-like, optional
        A dictionary of keyword arguments that will be passed through to the
        `ape` function ``add.scale.bar()``. See the documentation for the
        `ape` package for a full list of supported arguments.
    filename : string, optional
        File path for the generated PNG image. If None, a temporary file will
        be used.
    width : int or float, optional
        Width of the plot in `units`.
    height : int or float, optional
        Height of the plot in `units`.
    units : {'px', 'in', 'cm', 'mm'}, optional
        The units in which 'height' and 'width' are given. Can be 'px' (pixels,
        the default), 'in' (inches), 'cm' or 'mm'.
    res : int, optional
        The nominal resolution in ppi which will be recorded in the bitmap
        file, if a positive integer.  Also used for 'units' other than the
        default, and to convert points to pixels.
    pointsize : float, optional
        The default pointsize of plotted text, interpreted as big points (
        1/72 inch) at 'res' ppi.
    bg : color, optional
        The background color.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    imshow_kwargs : dict-like
        Additional keyword arguments passed through to `imshow()`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # setup R
    _init_r()

    # setup image file
    if filename is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.png')
        filename = tmp.name

    # initialise PNG device
    png_arg_names = 'width', 'height', 'units', 'res', 'pointsize', 'bg'
    png_args = dict()
    for n in png_arg_names:
        v = locals()[n]
        if v is not None:
            png_args[n] = v
    grdevices.png(filename, **png_args)

    # plot
    if plot_kwargs is None:
        plot_kwargs = dict()
    # adapt values for certain properties
    for k in 'tip.color', 'edge.color':
        if k in plot_kwargs:
            v = plot_kwargs[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                plot_kwargs[k] = ro.StrVector(v)
    ape.plot_phylo(tree, **plot_kwargs)

    # add scale bar
    if add_scale_bar is not None:
        ape.add_scale_bar(**add_scale_bar)

    # finalise PNG device
    grdevices.dev_off()

    # read in PNG for matplotlib plotting
    png = mpimg.imread(filename)

    # set up axes for matplotlib plotting
    if ax is None:
        # try to make the figure exactly the right size for image native
        # resolution
        pxw, pxh = png.shape[:2]
        dpi = plt.rcParams['savefig.dpi']
        w, h = pxw/dpi, pxh/dpi
        fig, ax = plt.subplots(figsize=(w, h))
        # no margin
        fig.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0)
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('aspect', 'equal')
    imshow_kwargs.setdefault('interpolation', 'none')
    ax.imshow(png, **imshow_kwargs)
    ax.set_axis_off()

    return ax


def write_tree(tree, filename=None, **kwargs):
    """
    Wrapper for the `ape` ``write.tree`` function, which writes in a file a
    tree in parenthetic format using the Newick (also known as New Hampshire)
    format.

    Parameters
    ----------

    tree : R object of class "phylo"
        The tree to be written.
    filename : string, optional
        The name of the file to write to. If ommitted, write the file to a
        string and return it.
    **kwargs : keyword arguments
        All further keyword arguments are passed through to ``write.tree``.

    Returns
    -------

    result : string
        A string if `filename` is None, otherwise no return value.

    """

    # setup R
    _init_r()

    # write the file
    if filename is None:
        kwargs['file'] = b''
    else:
        kwargs['file'] = filename
    result = ape.write_tree(tree, **kwargs)

    # handle the case where tree is written to stdout
    if filename is None:
        return result[0]


def read_tree(filename, **kwargs):
    """
    Wrapper for the `ape` ``read.tree`` function, which reads a file which
    contains one or several trees in parenthetic format known as the Newick
    or New Hampshire format.

    Parameters
    ----------

    filename : string
        Name of the file to read.
    **kwargs : keyword arguments
        All further keyword arguments are passed through to ``read.tree``.

    Returns
    -------

    tree : R object of class "phylo"
        If several trees are read in the file, the returned object is of
        class "multiPhylo", and is a list of objects of class "phylo". The name
        of each tree can be specified by tree.names, or can be read from the
        file (see details).

    """

    # setup R
    _init_r()

    kwargs['file'] = filename
    return ape.read_tree(**kwargs)


def color_edges_by_group_majority(tree, labels, groups,
                                  colors,
                                  equality_color=b'gray'):
    """
    Color the edges of a tree according to the majority group membership of
    the descendant tips.

    Parameters
    ----------

    tree : R object of class "phylo"
        The tree containing the edges to be colored.
    labels : sequence of strings
        The tip labels.
    groups : sequence of strings
        A sequence of strings of the same length as `labels`, where each item
        is an identifier for the group to which the corresponding tip belongs.
    colors : dict-like
        A dictionary mapping groups to colors.
    equality_color : string, optional
        The color to use in the event of a tie.

    Returns
    -------

    edge_colors : list of strings
        A list of colors for the edges of the tree, to be passed into
        :func:`plot_phylo`.

    """

    # setup R
    _init_r()

    r_groups = ro.StrVector([str(g) for g in groups])
    r_groups.names = ro.StrVector([str(l) for l in labels])
    counts = r.computeEdgeGroupCounts(tree, r_groups)

    r_colors = ro.StrVector([str(v) for v in colors.values()])
    r_colors.names = ro.StrVector([str(k) for k in colors.keys()])
    edge_colors = r.assignMajorityGroupColorToEdges(
        tree, counts, groupcolors=r_colors, equality_color=equality_color
    )

    return list(edge_colors)
