import os
import json

import Orange
from Orange.data import DiscreteVariable, ContinuousVariable

external_datasets = [
    ("iris_url", "https://raw.githubusercontent.com/biolab/orange3/master/Orange/datasets/iris.tab"),
]


def data_info(name, location):
    print(location)
    data = Orange.data.Table(location)
    attr = data.domain.attributes
    class_var = data.domain.class_var
    return {
        'name': name,
        'location': location,
        'rows': len(data),
        'features': {
            'discrete': sum(isinstance(x, DiscreteVariable) for x in attr),
            'continuous': sum(isinstance(x, ContinuousVariable) for x in attr),
            'meta': len(data.domain.metas)
        },
        'missing': bool(data.has_missing()),
        'target': {
            'type': 'discrete' if isinstance(class_var, DiscreteVariable) else 'continuous',
            'values': len(class_var.values) if isinstance(class_var, DiscreteVariable) else None
        }
    }

if __name__ == "__main__":
    info = dict()

    for name, location in external_datasets:
        info[name] = data_info(name, location)

    for fname in os.listdir('.'):
        if not os.path.isfile(fname):
            continue
        name, ext = os.path.splitext(fname)
        if ext != '.tab':
            continue
        info[name] = data_info(name, fname)

    with open('datasets.info', 'w') as f:
        json.dump(info, f, indent=4, sort_keys=True)
