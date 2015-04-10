import json
import os


class _DatasetInfo(dict):
    def __init__(self):
        super().__init__(self)
        datasets_folder = os.path.join(os.path.dirname(__file__),
                                       '../datasets')
        with open(os.path.join(datasets_folder, 'datasets.info'), 'r') as f:
            info = json.load(f)
        self.update(info)
        self.__dict__.update(info)
