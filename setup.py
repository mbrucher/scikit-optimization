#!/usr/bin/env python
# -*- coding: utf-8 -*-

descr   = """
"""

from os.path import join
import os
import sys
DISTNAME            = 'scikits.optimization'
DESCRIPTION         = 'A python module for numerical optimization'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Matthieu Brucher'
MAINTAINER_EMAIL    = 'matthieu.brucher@gmail.com'
URL                 = 'http://projects.scipy.org/scipy/scikits'
LICENSE             = 'new BSD'

optimization_version = 0.2

#DOWNLOAD_URL        = 'http://scipy.org/scipy/scikits/attachment/wiki/OpenOptInstall/openopt' + str(openopt_version) + '.tar.bz2'

import setuptools, string, shutil
from distutils.errors import DistutilsError
from numpy.distutils.system_info import system_info, NotFoundError, dict_append, so_ext
from numpy.distutils.core import setup, Extension
import os, sys

DOC_FILES = []

def configuration(parent_package='',top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')
    pkg_prefix_dir = os.path.join('scikits', 'optimization')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name,parent_package,top_path,
        version     = optimization_version,
        maintainer  = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
#        download_url = DOWNLOAD_URL,
        long_description = LONG_DESCRIPTION)

    return config

if __name__ == "__main__":
    setup(configuration = configuration,
        install_requires='numpy', # can also add version specifiers
        namespace_packages=['scikits'],
        packages=setuptools.find_packages(),
        include_package_data = True,
        test_suite='',#"scikits.openopt.tests", # for python setup.py test
        zip_safe=False, # the package can run out of an .egg file
        classifiers =
            [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering']
    )
