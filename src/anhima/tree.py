"""
TODO

"""


from __future__ import division, print_function, unicode_literals


__author__ = 'Alistair Miles <alimanfoo@googlemail.com>'


# standard library dependencies
import tempfile


# third party dependencies
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


# define custom R functions
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


def plot_phylo(tree, type=b'phylogram', tip_color=None,
               display=True, filename=None, width=None, height=None,
               units=None, res=None, pointsize=None, bg=None):

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
    pltargs = dict()
    if type:
        pltargs[b'type'] = type
    if tip_color:
        pltargs[b'tip.color'] = ro.StrVector(tip_color)
    ape.plot_phylo(tree, **pltargs)

    # finalise PNG device
    grdevices.dev_off()

    if display:
        # display in IPython notebook
        from IPython.core.displaypub import publish_display_data
        with open(filename, 'rb') as f:
            display_data = f.read()
            publish_display_data(source='anhima', data={'image/png':
                                                        display_data})


def nj(dist_square, labels=None):
    """
    TODO

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



