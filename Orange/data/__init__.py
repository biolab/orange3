from .variable import *
from .domain import *
from .table.impl import *
from .io import *


def get_sample_datasets_dir():
    orange_data_table = os.path.dirname(__file__)
    dataset_dir = os.path.join(orange_data_table, '..', 'datasets')
    return os.path.realpath(dataset_dir)

dataset_dirs = ['', get_sample_datasets_dir()]
