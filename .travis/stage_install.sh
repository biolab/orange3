
[ "$RUN_PYLINT" ] && return 0   # Nothing to do

for script in \
    install_orange.sh    \
    install_postgres.sh  \
    install_pyqt.sh      \
    build_doc.sh
do
    source $TRAVIS_BUILD_DIR/.travis/$script
done

pip install codecov;
