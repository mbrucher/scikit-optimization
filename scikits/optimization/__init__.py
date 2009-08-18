
"""
Optimization module
"""

import defaults
import criterion
import line_search
import optimizer
import step
import helpers

__all__= ['defaults', 'criterion', 'line_search', 'optimizer', 'step', 'helpers']

def test(level = -5, verbosity = 1):
  from numpy.testing import NumpyTest
  return NumpyTest().test(level, verbosity)

def testall(level = -5, verbosity = 1):
  from numpy.testing import NumpyTest
  return NumpyTest().testall(level, verbosity)
