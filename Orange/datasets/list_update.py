import os
import json

import Orange

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
            'discrete': sum(a.is_discrete for a in attr),
            'continuous': sum(a.is_continuous for a in attr),
            'meta': len(data.domain.metas),
        },
        'missing': bool(data.has_missing()),
        'target': {
            'type': 'discrete' if data.domain.has_discrete_class else 'continuous',
            'values': len(class_var.values) if data.domain.has_discrete_class else None,
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
