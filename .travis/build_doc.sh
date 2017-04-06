#!/bin/bash

set -o pipefail
set -o errexit

cd "$TRAVIS_BUILD_DIR"

# Ensure new images have indexed palettes
images="$(git diff --name-only origin/master..HEAD |
          grep -E '\bdoc/' | grep -iE '\.(png|jpg)$' || true )"
echo "Checking if images are indexed:"
while read image; do
    [ -f "$image" ] || continue
    imtype=$(identify -verbose "$image" | awk '/^ *Type: /{ print $2 }')
    echo "$image  $imtype"
    if ! echo "$imtype" | grep -Eq '(Palette|Grayscale)'; then
        echo "Error: image '$image' is not indexed or grayscale" >&2
        not_ok=1
    fi
done < <(echo "$images")
[ "$not_ok" ] && false
echo -e 'all ok\n'

# build Orange inplace (needed for docs to build)
python setup.py build_ext --inplace

cd $TRAVIS_BUILD_DIR/doc/development
make html
cd $TRAVIS_BUILD_DIR/doc/data-mining-library
make html
cd $TRAVIS_BUILD_DIR/doc/visual-programming
make html
./build_widget_catalog.py --input build/html/index.html --output build/html/widgets.json --url-prefix "http://docs.orange.biolab.si/3/visual-programming/"
