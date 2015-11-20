cd $TRAVIS_BUILD_DIR/doc
make html
./build_widget_catalog.py --input build/html/index.html --output build/html/widgets.json --url-prefix "http://docs.orange.biolab.si/3/"
