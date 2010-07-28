# -*- coding: utf-8 -*-

import unittest
import numpy
from numpy.testing import *
from .. import FRConjugateGradientStep, RestartPeriodicallyConjugateGradientStep, RestartNotOrthogonalConjugateGradientStep

class Function(object):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2 * (x[0] - 2), 4 * (2 * x[1] + 4)))

class test_RestartPeriodicallyConjugateGradientStep(unittest.TestCase):
  def test_call(self):
    step = FRConjugateGradientStep()
    step_decorated = RestartPeriodicallyConjugateGradientStep(step, 10)
    state = {'iteration' : 0}
    function = Function()
    assert_equal(step_decorated(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_restart(self):
    step = FRConjugateGradientStep()
    step_decorated = RestartPeriodicallyConjugateGradientStep(step, 10)
    state = {'iteration' : 10, 'direction' : None}
    function = Function()
    assert_equal(step_decorated(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

class test_RestartNotOrthogonalConjugateGradientStep(unittest.TestCase):
  def test_call(self):
    step = FRConjugateGradientStep()
    step_decorated = RestartNotOrthogonalConjugateGradientStep(step, 0.1)
    state = {'iteration' : 0}
    function = Function()
    assert_equal(step_decorated(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_restart(self):
    step = FRConjugateGradientStep()
    step_decorated = RestartNotOrthogonalConjugateGradientStep(step, 0.1)
    state = {'iteration' : 10, 'gradient' : numpy.array((4., -16.)), 'direction' : numpy.array((4., -16.))}
    function = Function()
    assert_equal(step_decorated(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

if __name__ == "__main__":
  unittest.main()
