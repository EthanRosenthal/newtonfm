import os.path

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('newtonfm', parent_package, top_path)

    config.add_extension('newtonfm_fast', sources=['newtonfm_fast.cpp'],
                         include_dirs=[numpy.get_include()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())