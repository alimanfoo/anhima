#!/bin/bash

set -x
set -o pipefail
set -e
set -u

t=''
while getopts ":t:" o; do
    case $o in
        t)
            t='skip'
            echo skipping tests $t
            ;;
        *)
            exit
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z $t ] ; then
    # run examples unless told not to
    python setup.py install
    cd examples
    ./runall.sh
    cd ..
fi

# remove -SNAPSHOT from src/petl/__init__.py
sed -i -e 's/-SNAPSHOT//' src/anhima/__init__.py
version=`grep __version__ src/anhima/__init__.py | sed -e "s/.*__version__[ ]=[ ]'\(.*\)'/\1/"`
echo $version

# git commit and push
git commit -a -m v$version
git push

# git tag and push
git tag -a v$version -m v$version
git push --tags

# update pypi
python setup.py register sdist upload
