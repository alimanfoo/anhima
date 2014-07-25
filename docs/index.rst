.. module:: anhima
.. moduleauthor:: Alistair Miles <alimanfoo@googlemail.com>

``anhima`` - Exploration and analysis of genetic variation data
===============================================================

N.B., this package is in a very early stage of development. Please report any
bugs to the `GitHub issue tracker <https://github
.com/alimanfoo/anhima/issues>`_.

Please note that this package is mostly an extremely thin wrapper around
:mod:`numpy`, :mod:`scipy`, :mod:`numexpr`, :mod:`matplotlib`,
:mod:`sklearn` and other generic scientific libraries. This package is
intended to provide convenience for those working with genetic variation
data who need quick access to some simple analysis and plotting functions.
Viewing the source code is recommended, as this may suggest ways that
generic libraries like :mod:`numpy` could be used or adapted for other
purposes beyond the limited set of functionalities supported here.

- Source: http://github.com/alimanfoo/anhima
- Documentation: http://anhima.readthedocs.org
- Examples: http://nbviewer.ipython.org/github/alimanfoo/anhima/tree/master/examples/

**Installation**

Install from GitHub::

    git clone https://github.com/alimanfoo/anhima.git
    cd anhima
    python setup.py install

**Contents**

.. toctree::
    :maxdepth: 2

    loc
    gt
    af
    ld
    dist
    pca
    h5
    sim

**Indices and tables**

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
