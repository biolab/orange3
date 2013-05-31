#!/bin/bash

git diff-index --quiet --cached HEAD || git diff-files --quiet || { echo "Please, commit your changes or stash them before updating canvas."; echo "Aborting"; exit 1; }

if [ ! -f tip.tar.gz ]; then
  wget https://bitbucket.org/biolab/orange/get/tip.tar.gz
fi
rm -rf biolab-orange-*
tar -xzf tip.tar.gz
mv biolab-orange-*/Orange/OrangeCanvas canvas
rm canvas/orng*
mv canvas/main.py canvas/__main__.py
2to3 -n -W --no-diffs canvas
echo biolab-orange-* | awk -F"-" '{print $3}' > canvas/source-revision.txt
rev=`cat canvas/source-revision.txt`
rm -rf ../Orange/canvas
mv canvas ../Orange/
rm -rf biolab-orange-*
git checkout -b merge-canvas-$rev || exit 1
git add ../Orange/canvas
git commit -am "Updated canvas to $rev"

echo "Please apply the following patches:"
echo
echo cd ../Orange 
for p in ../scripts/patches/*; do
    echo patch -p2 \< $p
done
