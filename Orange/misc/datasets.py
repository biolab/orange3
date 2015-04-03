import json, os

class DatasetInfo(dict):
    def __init__(self):
        super().__init__(self)
        datasets_folder = '../datasets'
        f = open(os.path.join(datasets_folder,'datasets.info'), 'r')
        info = json.load(f)
        f.close()
        self.update(info)

    def __getattr__(self, item):
        return self[item]
