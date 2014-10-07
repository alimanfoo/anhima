#!/bin/bash

set -x
set -o pipefail
set -e
set -u

run_examples=0
run_release=0
while getopts "er" o; do
    case $o in
        e)
            run_examples=1
            ;;
        r)
            run_release=1
            ;;
        *)
            exit
            ;;
    esac
done
shift $((OPTIND-1))

echo prepare extensions
cythonize src/anhima/opt/*.pyx

echo build locally
python setup.py build_ext --inplace

echo run syntax checks
flake8 src

echo run unit tests
nosetests -v

echo install locally
python setup.py install

echo build documentation
cd docs
make clean
make html
cd ..

if [ $run_examples -eq  1 ] ; then

    echo run examples
    cd examples
    ./runall.sh
    cd ..

else

    echo skip running examples

fi

if [ $run_release -eq  1 ] ; then

    echo execute release

    echo remove -SNAPSHOT from src/petl/__init__.py
    sed -i -e 's/-SNAPSHOT//' src/anhima/__init__.py
    version=`grep __version__ src/anhima/__init__.py | sed -e "s/.*__version__[ ]=[ ]'\(.*\)'/\1/"`
    echo $version

    echo git commit and push
    git commit -a -m v$version
    git push

    echo git tag and push
    git tag -a v$version -m v$version
    git push --tags

    echo update pypi
    python setup.py register sdist upload

else

    echo skip release

fi
