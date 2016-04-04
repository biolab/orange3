
[ "$RUN_PYLINT" ] && return 0  # Nothing to do

codecov
cd $TRAVIS_BUILD_DIR
if [ $TRAVIS_REPO_SLUG = biolab/orange3 ] && [ $TRAVIS_PULL_REQUEST = false ]; then
    source $TRAVIS_BUILD_DIR/.travis/upload_doc.sh
fi
