cd "$TRAVIS_BUILD_DIR"

# build Orange inplace (needed for docs to build)
python setup.py build_ext --inplace

cd $TRAVIS_BUILD_DIR/doc/development
make html
cd $TRAVIS_BUILD_DIR/doc/data-mining-library
make html
cd $TRAVIS_BUILD_DIR/doc/visual-programming
make html
./build_widget_catalog.py --input build/html/index.html --output build/html/widgets.json --url-prefix "http://docs.orange.biolab.si/3/visual-programming/"
