#!/bin/bash

p=$PWD
sudo apt-get update
sudo apt-get -y install git python-pip python-virtualenv python3-dev python3-numpy python3-scipy python3-pyqt4 python-qt4-dev python3-sip-dev libqt4-dev

virtualenv -p python3 --system-site-packages orange3env
source orange3env/bin/activate

echo "/usr/lib/python3/dist-packages/" > "orange3env/lib/python3.4/site-packages/0.pth"
pip install --upgrade numpy

git clone https://github.com/biolab/orange3
cd orange3
pip install -r requirements-core.txt  # For Orange Python library
pip install -r requirements-gui.txt   # For Orange GUI
pip install -r requirements-sql.txt   # To use SQL support
pip install -r requirements-opt.txt   # Optional dependencies, may fail
pip install Orange-Bioinformatics
python setup.py develop