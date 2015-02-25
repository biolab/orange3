# PyQt requirements
ln -s $TRAVIS_BUILD_DIR/../../biolab $TRAVIS_BUILD_DIR/../../astaric
cd $TRAVIS_BUILD_DIR/wheelhouse/pyqt/sip-4.16.5
sudo make install
cd $TRAVIS_BUILD_DIR/wheelhouse/pyqt/PyQt-x11-gpl-4.11.3
sudo make install
