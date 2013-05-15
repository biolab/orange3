#!/bin/bash

currentdir=`pwd`
tempdir=`pwd`/`mktemp -d canvas_merge_XXXXXX`

cd "$tempdir"

wget https://bitbucket.org/biolab/orange/get/tip.tar.gz
#cp ../tip.tar.gz tip.tar.gz
tar -xzf tip.tar.gz
mv biolab-orange-*/Orange/OrangeCanvas canvas
rm canvas/orng*
mv canvas/main.py canvas/__main__.py
2to3 -n -W --no-diffs canvas
echo biolab-orange-* | awk -F"-" '{print $3}' > canvas/source-revision.txt
echo
echo "Converted canvas sources are in directory canvas."
echo "Please apply the following patches:"
echo
for p in ../patches/*; do
    echo patch -p3 \< $p
done
echo "When all the conflicts have been fixed, type"
echo "ready"

function ready {
    rm -rf ../../Orange/canvas
    mv canvas ../../Orange
    cd "$currentdir"
    rm -rf "$tempdir"    
}

