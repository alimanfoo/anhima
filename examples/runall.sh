#!/bin/bash

# debug
set -x

# bail on first error
set -e

for f in *.ipynb
do
    runipy -o $f
done
