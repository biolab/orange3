pip install -U setuptools pip wheel
if [ ! "$(ls wheelhouse)" ]; then
    git clone -b pyqt --depth=1 https://github.com/astaric/orange3-requirements wheelhouse
else
    echo 'Using cached wheelhouse.';
fi

pip install wheelhouse/*.whl
pip install -r requirements.txt

python setup.py build_ext -i
