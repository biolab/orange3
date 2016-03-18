pip install -U setuptools pip

# Install dependencies sequentially
cat requirements-core.txt \
    requirements-dev.txt \
    requirements-doc.txt |
    while read dep; do
        dep="${dep%%#*}"  # Strip the comment
        [ "$dep" ] &&
            pip install $dep
    done

# Create a source tarball from the git checkout
python setup.py sdist
# Create a binary wheel from the packed source
pip wheel --no-deps -w dist dist/Orange-*.tar.gz
# Install into a testing folder
ORANGE_DIR="$(pwd)"/build/travis-test
mkdir -p "$ORANGE_DIR"
pip install --no-deps --target "$ORANGE_DIR"  dist/Orange-*.whl

cd $TRAVIS_BUILD_DIR
