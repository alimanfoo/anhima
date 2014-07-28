"""
This module provides some facilities for constructing and plotting trees. It
is mostly a wrapper around a very limited subset of functions from the R
``ape`` package (Analyses of Phylogenetics and Evolution).

R must be installed, the ``ape`` R package must be installed, and the Python
package ``rpy2`` must be installed, e.g.::

    $ apt-get install r-base
    $ pip install rpy2
    $ R
    > install.packages("ape")

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/tree.ipynb

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# standard library dependencies
import tempfile


# third party dependencies
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri
from rpy2.robjects.packages import importr
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


def nj(dist_square, labels=None):
    """Wrapper for the ``ape.nj`` function, which performs the neighbor-joining
    tree estimation of Saitou and Nei (1987).

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

    """

    # convert distance matrix to R
    m = numpy2ri(dist_square)

    # assign row and column labels
    if labels:
        s = ro.StrVector(labels)
        m.rownames = s
        m.colnames = s

    # build the tree
    tree = ape.nj(m)

    return tree


def bionj(dist_square, labels=None):
    """Wrapper for the ``ape.bionj`` function, which performs the BIONJ
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

    """

    # convert distance matrix to R
    m = numpy2ri(dist_square)

    # assign row and column labels
    if labels:
        s = ro.StrVector(labels)
        m.rownames = s
        m.colnames = s

    # build the tree
    tree = ape.bionj(m)

    return tree


def plot_phylo(tree, plot_kwargs=None,
               display=True, filename=None, width=None, height=None,
               units=None, res=None, pointsize=None, bg=None):
    """Wrapper for the ``ape.plot.phylo`` function, which plots phylogenetic
    trees. Plotting will use the R ``png`` graphics device.

    Parameters
    ----------

    tree : R object of class "phylo"
        The tree to plot.
    plot_kwargs : dict-like, optional
        A dictionary of keyword arguments that will be passed through to
        ``ape.plot.phylo()``. See the documentation for the ``ape`` package
        for a full list of supported arguments.
    display : bool, optional
        If True, assume that the function is being called from within an
        IPython notebook and attempt to publish the generated PNG image.
    filename : string, optional
        File path for the generated PNG image. If None, a temporary file will be
        used.
    width : int or float, optional
        Width of the plot in `units`.
    height : int or float, optional
        Height of the plot in `units`.
    units : {'px', 'in', 'cm', 'mm'}, optional
        The units in which 'height' and 'width' are given. Can be 'px' (pixels,
        the default), 'in' (inches), 'cm' or 'mm'.
    res : int
        The nominal resolution in ppi which will be recorded in the bitmap
        file, if a positive integer.  Also used for 'units' other than the
        default, and to convert points to pixels.
    pointsize : float
        The default pointsize of plotted text, interpreted as big points (
        1/72 inch) at 'res' ppi.

    """

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

    # finalise PNG device
    grdevices.dev_off()

    if display:
        # display in IPython notebook
        from IPython.core.displaypub import publish_display_data
        with open(filename, 'rb') as f:
            display_data = f.read()
            publish_display_data(source='anhima', data={'image/png':
                                                        display_data})


# Define custom R functions to help with coloring tree edges by population.
# These functions were written by Jacob Almagro-Garcia at the University of
# Oxford (@@TODO email)
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
""")


def color_edges_by_group_majority(tree, labels, groups,
                                  colors,
                                  equality_color=b'gray'):

    r_groups = ro.StrVector(groups)
    r_groups.names = ro.StrVector(labels)
    counts = r.computeEdgeGroupCounts(tree, r_groups)

    r_colors = ro.StrVector(colors.values())
    r_colors.names = ro.StrVector(colors.keys())
    edge_colors = r.assignMajorityGroupColorToEdges(
        tree, counts, groupcolors=r_colors, equality_color=equality_color
    )

    return edge_colors