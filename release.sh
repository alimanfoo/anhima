#!/bin/bash

set -x
set -o pipefail
set -e
set -u

skip_examples=0
while getopts "e" o; do
    case $o in
        e)
            skip_examples=1
            ;;
        *)
            exit
            ;;
    esac
done
shift $((OPTIND-1))

python setup.py install

if [ $skip_examples -eq  0 ] ; then
    echo run examples
    cd examples
    ./runall.sh
    cd ..
else
    echo skip running examples
fi

echo executing release

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
