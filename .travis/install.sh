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

# Create a source tarball, unpack it, and run its tests
python setup.py sdist
cd dist
tar xzf Orange-*.tar.gz
cd Orange-*
python setup.py build_ext -i
cd $TRAVIS_BUILD_DIR
