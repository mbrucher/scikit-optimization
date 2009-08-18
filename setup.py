#! /usr/bin/env python

descr   = """
"""

from os.path import join
import os
import sys, compileall

DISTNAME            = 'scikits.openopt'
DESCRIPTION         = 'A python module for numerical optimization'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'mainteiner of OpenOpt is Dmitrey Kroshko, mainteiner of GenericOpt is Matthieu Brucher',
MAINTAINER_EMAIL    = 'dmitrey.kroshko@scipy.org',
URL                 = 'http://projects.scipy.org/scipy/scikits',
LICENSE             = 'new BSD'

openopt_version = 0.19

DOWNLOAD_URL        = 'http://scipy.org/scipy/scikits/attachment/wiki/OpenOptInstall/openopt' + str(openopt_version) + '.tar.bz2'

import setuptools, string, shutil
from distutils.errors import DistutilsError
from numpy.distutils.system_info import system_info, NotFoundError, dict_append, so_ext
from numpy.distutils.core import setup, Extension
import os, sys

DOC_FILES = []
##from scikits import openopt
##from openopt.info import __version__ as openopt_version
#from scikits.openopt.info import __version__ as openopt_version
from shutil import copytree, rmtree


def configuration(parent_package='',top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')
    pkg_prefix_dir = os.path.join('scikits', 'openopt')

    # Get the version


    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name,parent_package,top_path,
        version     = openopt_version,
        maintainer  = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
        download_url = DOWNLOAD_URL,
        long_description = LONG_DESCRIPTION)
    #config.add_data_dir(('src', 'scikits'))
    #config.add_data_dir(('openopt', 'scikits'))


    # XXX: once in SVN, should add svn version...
    #print config.make_svn_version_py()

    # package_data does not work with sdist for setuptools 0.5 (setuptools bug),
    # so we need to add them here while the bug is not solved...
    #config.add_data_files(('docs', ['scikits/openopt/docs/' + i for i in DOC_FILES]))

    #config.add_data_dir(('examples', 'scikits/openopt/docs/examples'))

    return config


if __name__ == "__main__":
    solverPaths = {}
    #File = string.join(__file__.split(os.sep)[:-1], os.sep)
    for root, dirs, files in os.walk('scikits'+os.sep+'openopt'+os.sep +'solvers'):
        #for root, dirs, files in os.walk(os.path.dirname(file)+os.sep+'solvers'):
        rd = root.split(os.sep)
        if '.svn' in rd: continue
        rd = rd[rd.index('solvers')+1:]
        for file in files:
            if len(file)>6 and file[-6:] == '_oo.py':
                solverPaths[file[:-6]] = 'scikits.openopt.solvers.' + string.join(rd,'.') + '.'+file[:-3]
    f = open('solverPaths.py', 'w')
    f.write('solverPaths = ' + str(solverPaths))
    f.close()
    shutil.move('solverPaths.py', 'scikits' + os.sep + 'openopt' + os.sep +'Kernel' + os.sep + 'solverPaths.py')



    # setuptools version of config script

    # package_data does not work with sdist for setuptools 0.5 (setuptools bug)
    # So we cannot add data files via setuptools yet.

    #data_files = ['test_data/' + i for i in TEST_DATA_FILES]
    #data_files.extend(['docs/' + i for i in doc_files])
    setup(configuration = configuration,
        install_requires='numpy', # can also add version specifiers
        namespace_packages=['scikits'],
        packages=setuptools.find_packages(),
        include_package_data = True,
        #package_data = '*.txt',
        test_suite='',#"scikits.openopt.tests", # for python setup.py test
        zip_safe=False, # the package can run out of an .egg file
        #FIXME url, download_url, ext_modules
        classifiers =
            [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering']
    )
