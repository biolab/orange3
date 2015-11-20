cd $TRAVIS_BUILD_DIR/doc
make html
./build_widget_catalog.py "http://docs.orange.biolab.si/3/"
