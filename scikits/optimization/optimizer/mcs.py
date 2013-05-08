#!/usr/bin/env python

import numpy as np

from scikits.optimization import *
from scikits.optimization.optimizer import optimizer

class Quadratic(object):
  """ A MCS specific quadratif function """
  def __init__(self, x, f):
    """ Initialize the class with coeffs based on values and associated cost """
    f23 = (f[2] - f[1])/(x[2] - x[1])
    self.coeff = f[0], (f[1] - f[0])/(x[1] - x[0]), (f23 - d[1])/(x[2] - x[0]);
    self.x = x
  
  def __call__(self, x):
    """ Computes the quadratic function at a given point """
    return self.coeff[0] + self.coeff[1] * (x - self.x[0]) + self.coeff[2] * (x - self.x[0]) * (x - self.x[1]);

  def find_min(self, a, b):
    """ Finds the minimum value in a given interval """
    return self.__find(self, a, b, self.coeff)

  def find_max(self, a, b):
    """ Finds the maximum value in a given interval """
    return self.__find(self, a, b, -self.coeff)

  def __find(self, a, b, coeff):
    if coeff[2] == 0:
      if coeff[1] > 0:
        return a
      else:
        return b
    elif coeff[2] > 0:
      tmp = 0.5 * (self.x[0] + self.x[1]) - 0.5 * coeff[1] / coeff[2];
      if a <= tmp <= b:
        return tmp
      elif self(a) < self(b):
        return a
      else:
        return b
    elif  self(a) < self(b):
      return a
    else:
      return b

class MCS(optimizer.Optimizer):
  """ A Multilevel Coordinate Search """
  def __init__(self, **kwargs):
    """
    Needs to have :
      - an object function to optimize (function), alternatively a function ('fun'), gradient ('gradient'), ...
      - a criterion to stop the optimization (criterion)
      - the two corners of bound box (2 points of dimension n, u, v)
      - optionally the starting point (1 point of dimension n, x0, default is u+v/2)
      - optionally 3 starting points (3 point of dimension n, x, default is x0,u,v)
      - optionally the maximum search level (1 float, default is smax=50*n)
    """
    optimizer.Optimizer.__init__(self, **kwargs)
    
    self.bound1 = kwargs['u']
    self.bound2 = kwargs['v']
    if 'x0' not in kwargs:
      if 'x' in kwargs:
        self.optimal_parameters = np.array(kwargs['x'])
      else:
        self.optimal_parameters = np.array([(self.bound1 + self.bound2) / 2, self.bound1, self.bound2])
    else:
      self.optimal_parameters = np.array([kwargs['x0'], self.bound1, self.bound2])
    self.smax = kwargs.get('smax', 50*len(self.bound1))

    self.optimal_values = [self.function(x) for x in self.optimal_parameters]
    self.initialize_box()
    print self.boxes
    self.state["best_parameters"] = self.optimal_parameters[0]
    self.state["best_value"] = self.optimal_values[0]
    
    self.record_history(**self.state)

  def initialize_box(self):
    """ This methods first computes all the interesting points passed as parameters and then creates the first boxes for the algorithm """
    x0, f0 = self.initialize_x()
    self.initialize_splitting(x0)
    
  def initialize_x(self):
    """ After starting with a given x0, the method adds also to the mix new other points based on the initial distribution. """
    x0 = np.array(self.optimal_parameters[0])
    f0 = self.optimal_values[0]

    self.best = np.zeros(len(x0), dtype=np.int)

    for i in range(len(x0)):
      for j in range(1, len(self.optimal_parameters)):
        x0[i] = self.optimal_parameters[j][i]
        f1 = self.function(x0)
        if f1 < f0:
          self.best[i] = j
          f0 = f1
      x0[i] = self.optimal_parameters[self.best[i]][i]
    return x0, f0
    
  def initialize_splitting(self, x0):
    """ Create sthe first computation boxes """
    self.boxes = [(-1, 0, 0, None, (self.bound1, self.bound2))]
    parent = 0

    tempx = np.array(x0)

    for i in range(len(x0)):
      child = 1
      self.boxes.append((parent, self.boxes[parent][1] + 1, -child, self.function(x0), ()))

      x0[i] = tempx[i]

  def optimize(self):
    return self.state["best_parameters"]
   
class Rosenbrock(object):
  """
  The Rosenbrock function
  """
  def __init__(self):
    self.count = 0

  def __call__(self, x):
    """
    Get the value of the Rosenbrock function at a specific point
    """
    self.count = self.count+1
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1. - x[:-1])**2.0)

if __name__ == "__main__":
  from numpy.testing import *
  startPoint = np.array((-1.01, 1.01), np.float)
  u = np.array((-2.0, -2.0), np.float)
  v = np.array((2.0, 2.0), np.float)

  optimi = MCS(function=Rosenbrock(), criterion=criterion.OrComposition(criterion.MonotonyCriterion(0.00001), criterion.IterationCriterion(10000)), x0=startPoint, u=u, v=v)
  assert_almost_equal(optimi.optimize(), np.ones(2, np.float), decimal=1)
