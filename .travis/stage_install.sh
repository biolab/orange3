
[ "$RUN_PYLINT" ] && return 0   # Nothing to do

# pandas requires cython to install from source, cython recommends --no-cython-compile for CI builds
# also modify the pandas dependency in requirements-core.txt dependency to the git version
[ "$PANDAS_GIT" ] && \
    pip install Cython --install-option="--no-cython-compile" && \
    sed -Eibak "s,^pandas.*,git+git://github.com/pydata/pandas.git," "$TRAVIS_BUILD_DIR/requirements-core.txt"

for script in \
    install_orange.sh    \
    install_postgres.sh  \
    install_pyqt.sh      \
    build_doc.sh
do
    foldable source $TRAVIS_BUILD_DIR/.travis/$script
done
