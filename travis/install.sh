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

python setup.py build_ext -i
