sudo apt-get install -qq libblas-dev liblapack-dev postgresql-server-dev-9.1
pip install -U setuptools pip wheel pgxnclient
git clone https://github.com/astaric/orange3-requirements wheelhouse
pip install wheelhouse/*.whl

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
