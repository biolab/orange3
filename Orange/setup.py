import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('Orange', parent_package, top_path)
    config.add_subpackage('classification')
    config.add_subpackage('data')
    config.add_subpackage('evaluation')
    config.add_subpackage('feature')
    config.add_subpackage('misc')
    config.add_subpackage('statistics')
    config.add_subpackage('testing')
    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
