`anhima` - Information for developers
======================================

Setting up the development environment
--------------------------------------

Recommended to [install and setup virtualenvwrapper]
(http://virtualenvwrapper.readthedocs.org/en/latest/install.html).

Create a virtual environment:

    $ mkvirtualenv anhima

Fork then clone the repo:

    $ git clone git@github.com:alimanfoo/anhima.git
    $ cd anhima

Install main dependencies:

    $ export C_INCLUDE_PATH=/usr/lib/openmpi/include
    $ pip install -r requirements.txt

Install development dependencies:

    $ pip install -r dev_requirements.txt

Building Cython extensions
--------------------------

To build Cython extensions in-place (required for running unit tests):

    $ python setup.py build_ext --inplace

If you are developing a Cython extension module, run the following before
building:

    $ cythonize src/anhima/opt/*.pyx

Running unit tests
------------------

From the root directory of the repo:

    $ nosetests -v

Please run unit tests and make sure they pass before submitting a PR.

Running example notebooks
-------------------------

To run all notebooks:

    $ python setup.py install
    $ cd examples
    $ ./runall.sh

Please run all example notebooks and make sure they execute successfully
before submitting a PR.

Checking code syntax (PyFlakes) and style (PEP8)
------------------------------------------------

From the root directory of the repo:

    $ flake8 src

Please fix all errors before submitting a PR, but use your good judgment
about when not to conform to PEP8. Mark any line you want to be excluded
from [flake8](http://flake8.readthedocs.org/) checks with the comment '# noqa'.

Developing Cython extensions
----------------------------

If you add a new Cython extension, ensure that the Cythonized C files are
added to the Git repository.