cd $TRAVIS_BUILD_DIR/dist/Orange-*
( cd doc/development
make html )
( cd doc/data-mining-library
make html )
( cd doc/visual-programming
make html )
./build_widget_catalog.py --input build/html/index.html --output build/html/widgets.json --url-prefix "http://docs.orange.biolab.si/3/visual-programming/"
