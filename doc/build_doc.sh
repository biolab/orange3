#!/bin/bash

set -o pipefail
set -o errexit

# Ensure new images have indexed palettes
images="$(git diff --name-only origin/master..HEAD |
          grep -E '\bdoc/' | grep -iE '\.(png|jpg)$' || true )"
echo "Checking if images are indexed:"
while read image; do
    [ -f "$image" ] || continue
    if  [[ "$image" == *"_unindexed"* ]]; then
      continue
    fi
    imtype=$(identify -verbose "$image" | awk '/^ *Type: /{ print $2 }')
    echo "$image  $imtype"
    if ! echo "$imtype" | grep -Eq '(Palette|Grayscale)'; then
        echo "Error: image '$image' is not indexed or grayscale" >&2
        not_ok=1
    fi
done < <(echo "$images")
[ "$not_ok" ] && false
echo -e 'all ok\n'

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
make html --directory "$SCRIPT_DIR"/development
make html --directory "$SCRIPT_DIR"/data-mining-library
make html --directory "$SCRIPT_DIR"/visual-programming

# check if the widget catalog in the repository (for orange-hugo is up to date
cd "$SCRIPT_DIR"
wget_command="wget -N https://raw.githubusercontent.com/biolab/orange-hugo/master/scripts/create_widget_catalog.py"
run_command="python create_widget_catalog.py --categories Data,Transform,Visualize,Model,Evaluate,Unsupervised --doc visual-programming/source/"
eval "$wget_command"
eval "$run_command"
diff=$(git diff -- widgets.json)
echo "$diff"
if [ -n "$diff" ]
then
  echo "Widget catalog is stale. Rebuild it with:"
  echo "cd doc"
  echo "$wget_command"
  echo "$run_command"
  echo
  echo "$diff"
  exit 1
fi
