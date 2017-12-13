import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('modelling', parent_package, top_path)
    config.add_extension('_scite',
                         language='c++',
                         sources=['_scite.cpp'],
                         include_dirs=[],
                         libraries=libraries,
                         export_symbols=["scite_export"]
                         )
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())