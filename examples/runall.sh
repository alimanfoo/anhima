#!/bin/bash

for f in *.ipynb
do
    runipy -o $f
done
