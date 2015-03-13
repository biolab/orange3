# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.
import os

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('data', parent_package, top_path)
    config.add_extension('_valuecount',
                         sources=['_valuecount.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)
    config.add_extension('_contingency',
                         sources=['_contingency.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)
    config.add_extension('_io',
                         sources=['_io.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)
    config.add_extension('_variable',
                         sources=['_variable.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
