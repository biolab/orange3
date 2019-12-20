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
python setup.py egg_info build_ext --inplace

# build docs for upload to the old web page
cd $TRAVIS_BUILD_DIR/doc/development
make html
cd $TRAVIS_BUILD_DIR/doc/data-mining-library
make html
cd $TRAVIS_BUILD_DIR/doc/visual-programming
make html

# create widget catalog for the old (django) Orange webpage
export PYTHONPATH=$TRAVIS_BUILD_DIR:$PYTHONPATH
# Screen must be 24bpp lest pyqt5 crashes, see pytest-dev/pytest-qt/35
XVFBARGS="-screen 0 1280x1024x24"
catchsegv xvfb-run -a -s "$XVFBARGS" \
    python $TRAVIS_BUILD_DIR/scripts/create_widget_catalog.py \
        --output build/html/ \
        --url-prefix "http://docs.biolab.si/3/visual-programming/"

# check if the widget catalog in the repository (for orange-hugo is up to date
cd $TRAVIS_BUILD_DIR/doc/
wget_command="wget -N https://raw.githubusercontent.com/biolab/orange-hugo/master/scripts/create_widget_catalog.py"
run_command="python create_widget_catalog.py --categories Data,Visualize,Model,Evaluate,Unsupervised --doc visual-programming/source/"
eval "$wget_command"
eval "catchsegv xvfb-run -a -s "\"$XVFBARGS"\" $run_command"
diff=$(git diff -- widgets.json)
echo $diff
if [ ! -z "$diff" ]
then
  echo "Widget catalog is stale. Rebuild it with:"
  echo "cd doc"
  echo "$wget_command"
  echo "$run_command"
  echo
  echo "$diff"
  exit 1
fi
