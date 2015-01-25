#!/bin/bash

# set -x
set -o pipefail
set -e
set -u

echo prepare extensions
cythonize src/anhima/opt/*.pyx

echo build locally
python setup.py build_ext --inplace

echo run syntax checks
tox -e flake8

echo run unit tests
tox -e py27
tox -e py34
# TODO tox -e doctests

echo build documentation
tox -e docs

echo run examples
tox -e examples

echo execute release

echo remove .dev0 from anhima/__init__.py
sed -i -e 's/.dev0//' anhima/__init__.py
version=`grep __version__ anhima/__init__.py | sed -e "s/.*__version__[ ]=[ ]'\(.*\)'/\1/"`
echo $version

echo git commit and push
git commit -a -m v$version
git push

echo git tag and push
git tag -a v$version -m v$version
git push --tags

echo update pypi
python setup.py register sdist upload
