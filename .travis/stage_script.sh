
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

# Screen must be 24bpp lest pyqt5 crashes, see pytest-dev/pytest-qt/35
XVFBARGS="-screen 0 1280x1024x24"

catchsegv xvfb-run -a -s "$XVFBARGS" \
    coverage run -m unittest -v \
        Orange.tests \
        Orange.widgets.tests \
        Orange.canvas.report.tests
