
if [ "$RUN_PYLINT" ]; then
    cd $TRAVIS_BUILD_DIR
    foldable pip install -r requirements-dev.txt
    cp pylintrc ~/.pylintrc
    .travis/check_pylint_diff
    exit $?
fi

cd "$ORANGE_DIR"
python -c "from Orange.tests import *"
cp "$TRAVIS_BUILD_DIR"/.coveragerc ./  # for covereage and codecov
export PYTHONPATH="$ORANGE_DIR" PYTHONUNBUFFERED=x
catchsegv xvfb-run coverage run --source=Orange -m unittest -v Orange.tests
