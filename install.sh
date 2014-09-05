sudo apt-get install -qq libblas-dev liblapack-dev
pip install -U setuptools pip wheel
git clone https://github.com/astaric/orange3-requirements wheelhouse
pip install wheelhouse/*.whl
python setup.py build_ext -i
