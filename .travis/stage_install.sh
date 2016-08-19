
[ "$RUN_PYLINT" ] && return 0   # Nothing to do

# pandas requires cython to install from source, cython recommends --no-cython-compile for CI builds
# install pandas manually, then remove the version requirement from requirements-core.txt
# to avoid pip complaints
[ "$PANDAS_GIT" ] && \
    pip install Cython --install-option="--no-cython-compile" && \
    pip install git+git://github.com/pydata/pandas.git && \
    sed -Eibak "s/^pandas.*/pandas/" "$TRAVIS_BUILD_DIR/requirements-core.txt"

for script in \
    install_orange.sh    \
    install_postgres.sh  \
    install_pyqt.sh      \
    build_doc.sh
do
    foldable source $TRAVIS_BUILD_DIR/.travis/$script
done
