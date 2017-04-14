
[ "$RUN_PYLINT" ] && return 0  # Nothing to do

if [ "$BUILD_DOCS" ] &&
   [ $TRAVIS_REPO_SLUG = biolab/orange3 ] &&
   [ $TRAVIS_PULL_REQUEST = false ]; then
        source $TRAVIS_BUILD_DIR/.travis/upload_doc.sh
        return 0
fi

if [ "$UPLOAD_COVERAGE" ]; then
    codecov
fi

cd $TRAVIS_BUILD_DIR
