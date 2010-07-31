#/usr/bin/env python

import unittest
import numpy
from numpy.testing import *
from .. import SimpleLineSearch, FixedLastStepModifier

from .function import Function

class test_FixedLastStepModifier(unittest.TestCase):

  def test_call(self):
    lineSearch = FixedLastStepModifier(SimpleLineSearch())
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.ones((2))}
    function = Function()
    assert_array_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.ones((2)))
    assert(state['alpha_step'] == 1.)

  def test_call_twice(self):
    lineSearch = FixedLastStepModifier(SimpleLineSearch())
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.array((4., -8.)), 'alpha_step' : 0.5}
    function = Function()
    assert_array_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.array((4., -8.)))
    assert(state['alpha_step'] == 1.)

if __name__ == "__main__":
  unittest.main()
