#!/bin/bash

set -x
set -o pipefail
set -e
set -u

cd examples
for f in *.ipynb
do
    runipy -o $f
    # sanitise notebooks so reruns dont arbitrarily create new git commits
    cat $f | sed 's/Figure at 0x[a-f0-9]*>/Figure at 0xFFFFFFFFF>/' | sed 's/}\n\r/}/' > ${f}.sanitised
    mv ${f}.sanitised $f
done
cd ..
