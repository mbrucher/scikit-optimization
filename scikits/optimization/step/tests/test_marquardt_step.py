#/usr/bin/env python

import unittest
import numpy
from numpy.testing import *
from .. import MarquardtStep

class Function(object):
  def __call__(self, x):
    return (x[0] - 2.) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2. * (x[0] - 2), 4 * (2 * x[1] + 4)))

  def hessian(self, x):
    return numpy.diag((2., 8.))

class test_MarquardtStep(unittest.TestCase):
  def test_call(self):
    step = MarquardtStep(gamma = 10.)
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    assert(state['gamma'] == 5.)
    assert(function(numpy.zeros((2))) > function(direction))

if __name__ == "__main__":
  unittest.main()