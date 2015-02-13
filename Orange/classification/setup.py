import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('classification', parent_package, top_path)
    config.add_extension('_simple_tree',
                         sources=['_simple_tree.c'],
                         include_dirs=[],
                         libraries=libraries,
                         export_symbols=[
                             "build_tree", "destroy_tree", "new_node",
                             "predict_classification", "predict_regression"]
                         )
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
