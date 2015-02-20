sudo apt-get update
sudo apt-get install -qq libblas-dev liblapack-dev postgresql-server-dev-9.1 libqt4-dev

pip install -U setuptools pip wheel pgxnclient
git clone -b pyqt --depth=1 https://github.com/astaric/orange3-requirements wheelhouse

pip install wheelhouse/*.whl
pip install -r requirements.txt

python setup.py build_ext -i

# Sql requirements
pip install wheelhouse/sql/*.whl
psql -c 'create database test;' -U postgres
cd wheelhouse/sql/quantile-1.1.3
sudo make install
cd ../../..
cd wheelhouse/sql/binning
sudo make install
cd ../../..
psql test -c 'CREATE EXTENSION quantile;' -U postgres
psql test -c 'CREATE EXTENSION binning;' -U postgres
