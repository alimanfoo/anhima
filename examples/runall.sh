#!/bin/bash

# debug
set -x

# bail on first error
set -e

for f in *.ipynb
do
    runipy -o $f
    cat $f | sed 's/Figure at 0x[a-f0-9]*>/Figure at 0xFFFFFFFFF>/' | sed s/}\n\r/}/ > ${f}.sanitised
    mv ${f}.sanitised $f
done
