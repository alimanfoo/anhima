.. module:: anhima
.. moduleauthor:: Alistair Miles <alimanfoo@googlemail.com>

``anhima`` - Exploration and analysis of genetic variation data
===============================================================

**This package is no longer under active development. Existing functionality
is gradually being migrated into the `scikit-allel`_ package, where development
efforts are now focused.**

- Documentation: http://anhima.readthedocs.org
- Examples: http://nbviewer.ipython.org/github/alimanfoo/anhima/tree/master/examples/
- Source: http://github.com/alimanfoo/anhima
- Mailing list: https://groups.google.com/forum/#!forum/anhima
- Release notes: https://github.com/alimanfoo/anhima/releases

**Installation**

Install latest stable release from PyPI::

    pip install -U anhima

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
    sf
    f2
    ld
    dist
    pca
    mds
    tree
    ped
    io
    h5
    util
    sim

**Indices and tables**

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _scikit-allele: http://scikit-allel.readthedocs.org/en/latest/index.html
