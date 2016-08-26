foldable pip install -U setuptools pip codecov

# Install dependencies sequentially
cat requirements-core.txt \
    requirements-dev.txt \
    requirements-doc.txt |
    while read dep; do
        # Strip the comment, but check for "egg" after # to allow "-e git://...#egg=.."
        dep="${dep%%#!(egg)*}"
        [ "$dep" ] &&
            foldable pip install $dep
    done

# Create a source tarball from the git checkout
foldable python setup.py sdist
# Create a binary wheel from the packed source
foldable pip wheel --no-deps -w dist dist/Orange3-*.tar.gz
# Install into a testing folder
ORANGE_DIR="$(pwd)"/build/travis-test
mkdir -p "$ORANGE_DIR"
pip install --no-deps --target "$ORANGE_DIR"  dist/Orange3-*.whl

cd $TRAVIS_BUILD_DIR
