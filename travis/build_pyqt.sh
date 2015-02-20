# PyQt requirements
mkdir -p doc/build/html
ln -s `pwd`/wheelhouse/pyqt/sip-4.16.5 doc/build/html/
ln -s `pwd`/wheelhouse/pyqt/PyQt-x11-gpl-4.11.3 doc/build/html/sip-4.16.5/
cd wheelhouse/pyqt/sip-4.16.5
python configure.py
make
sudo make install
cd ../PyQt-x11-gpl-4.11.3
python configure.py --confirm-license
make -j 2
sudo make install
cd ..
git config user.email "travis@travis-ci.org"
git config user.name "Travis"
git config --global push.default simple
git add .
git commit -m "compile pyqt on travis."
git push
cd ../..
