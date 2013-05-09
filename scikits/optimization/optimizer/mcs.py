#!/usr/bin/env python

import numpy as np

from scikits.optimization import *
from scikits.optimization.optimizer import optimizer

class Quadratic(object):
  """ A MCS specific quadratif function """
  def __init__(self, x, f):
    """ Initialize the class with coeffs based on values and associated cost """
    f23 = (f[2] - f[1])/(x[2] - x[1])
    d = (f[1] - f[0])/(x[1] - x[0])
    self.coeff = f[0], d, (f23 - d)/(x[2] - x[0]);
    self.x = x
  
  def __call__(self, x):
    """ Computes the quadratic function at a given point """
    return self.coeff[0] + self.coeff[1] * (x - self.x[0]) + self.coeff[2] * (x - self.x[0]) * (x - self.x[1]);

  def find_min(self, a, b):
    """ Finds the minimum value in a given interval """
    return self.__find(a, b, self.coeff)

  def find_max(self, a, b):
    """ Finds the maximum value in a given interval """
    return self.__find(a, b, -self.coeff)

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

    self.__initialize_box()
    self.state["boxes"] = self.boxes
    
    self.record_history(**self.state)

  def __initialize_box(self):
    """ This methods first computes all the interesting points passed as parameters and then creates the first boxes for the algorithm """
    x0, f0 = self.__initialize_x()
    self.state["new_parameters"] = x0
    self.state["new_value"] = f0
    self.__initialize_splitting(x0)
    
  def __initialize_x(self):
    """ After starting with a given x0, the method adds also to the mix new other points based on the initial distribution. """
    x0 = np.array(self.optimal_parameters[0])
    f0 = self.function(x0)

    self.best = np.zeros(len(x0), dtype=np.int)
    self.best_values = np.zeros((len(x0), len(self.optimal_parameters)))
    self.best_values[0,0] = f0

    for i in range(len(x0)):
      if i != 0:
        self.best_values[i,0] = self.best_values[self.best[i-1],0]
      for j in range(1, len(self.optimal_parameters)):
        x0[i] = self.optimal_parameters[j][i]
        f1 = self.function(x0)
        self.best_values[i,j] = f1
        if f1 < f0:
          self.best[i] = j
          f0 = f1
      x0[i] = self.optimal_parameters[self.best[i]][i]
    return x0, f0
    
  def __initialize_splitting(self, x0):
    """ Create sthe first computation boxes """
    import math
    # a box is parent, level, nogain, cost, split, boundaries
    self.boxes = [[-1, 0, False, None, None, (self.bound1, self.bound2)]]
    parent = 0

    tempx = np.array(x0)

    for i in range(len(x0)):
      bound1 = np.array(self.boxes[parent][-1][0])
      bound2 = np.array(self.boxes[parent][-1][1])

      coordinates = self.optimal_parameters[:,i]
      sortorder = np.argsort(coordinates)

      # Try to find the minimum box so that it can be split in next dimension
      d = Quadratic(coordinates[sortorder[:3]], self.best_values[i, sortorder[:3]])
      xl = d.find_min(coordinates[sortorder[0]], coordinates[sortorder[2]])

      if self.best[i] == sortorder[0]:
        if xl < coordinates[sortorder[0]]:
          newparent = len(self.boxes)
        else:
          newparent = len(self.boxes) + 1

      x0[i] = coordinates[sortorder[0]]
      # if the lowest coordinate is not on the boundary, we create a box from the boundary to the coordinate
      if coordinates[sortorder[0]] != self.bound1[i]:
        bound2[i] = coordinates[sortorder[0]]
        self.boxes.append([parent, self.boxes[parent][1] + 1, False, self.best_values[i, sortorder[0]], None, (np.array(bound1), np.array(bound2), np.array(x0))])

      # Between two coordinates, create 2 new boxes with differnet level but same parent
      for j in range(len(coordinates) - 1):
        oldx0 = np.array(x0)
        bound1[i] = coordinates[sortorder[j]]
        x0[i] = coordinates[sortorder[j+1]]
        # Split so that the biggest share is given to the box with the lowest f
        if self.best_values[i, sortorder[j]] < self.best_values[i, sortorder[j+1]]:
          bound2[i] = coordinates[sortorder[j]] + 0.5 * (-1 + math.sqrt(5)) * (coordinates[sortorder[j+1]] - coordinates[sortorder[j]]);
          s = 1
        else:
          bound2[i] = coordinates[sortorder[j]] + 0.5 * (3 - math.sqrt(5)) * (coordinates[sortorder[j+1]] - coordinates[sortorder[j]]);
          s = 2

        self.boxes.append([parent, self.boxes[parent][1] + s, False, self.best_values[i, sortorder[j]], None, (np.array(bound1), np.array(bound2), oldx0)])

        oldx0 = np.array(x0)
        # Try to find the minimum box so that it can be split in next dimension, follow up if the best was not 0
        if j and self.best[i] == sortorder[j]:
          if xl < coordinates[sortorder[j]]:
            newparent = len(self.boxes) - 1
          else:
            newparent = len(self.boxes)

        if 1 < j < len(coordinates) - 2:
          d = Quadratic(coordinates[sortorder[j:j+3]], self.best_values[i, sortorder[j:j+3]])
          xl = d.find_min(coordinates[sortorder[j]], coordinates[sortorder[j+2]])

        bound1[i] = bound2[i]
        bound2[i] = coordinates[sortorder[j+1]]
        self.boxes.append([parent, self.boxes[parent][1] + 3 - s, False, self.best_values[i, sortorder[j+1]], None, (np.array(bound1), np.array(bound2), oldx0)])

      # if the highest coordinate is not on the boundary, we create a box from the coordinate to the boundary
      if coordinates[sortorder[-1]] != self.bound2[i]:
        bound1[i] = coordinates[sortorder[-1]]
        bound2[i] = self.bound2[i]
        self.boxes.append([parent, self.boxes[parent][1] + 1, False, self.best_values[i, sortorder[-1]], None, (np.array(bound1), np.array(bound2), oldx0)])

      x0[i] = tempx[i]
      self.boxes[parent][1] = -1
      self.boxes[parent][4] = -i
      parent = newparent

  def iterate(self):
    #start a new sweep = new iteration
    self.state["old_value"] = self.state["new_value"]
    self.state["old_parameters"] = self.state["new_parameters"]

    records, minlevel = self.__find_ranks()

    print "Starting sweep"
    while minlevel < self.smax:
      splits, x, y = self.__get_box_info(self.boxes[records[minlevel]])

      print "level", minlevel
      print self.boxes[records[minlevel]]
      print splits, x, y

      split = 0
      if minlevel > 2 * len(splits) * (np.min(splits) + 1):
        split = 1 # split by rank

      # Determine how to split it
      #  split by rank?
      #  split by gain?

      if split == 0:
        self.boxes[records[minlevel]][1] += 1
        if records[minlevel+1] == -1:
          records[minlevel+1] = records[minlevel]
        elif self.boxes[records[minlevel]][3] < self.boxes[records[minlevel+1]][3]:
          records[minlevel+1] = records[minlevel]
      elif split == 1: # split by rank
        pass
      # split it properly

      minlevel += 1
      while minlevel < self.smax:
        if records[minlevel] == -1:
          minlevel += 1
        else:
          break

    self.state["new_value"] = self.state["old_value"]
    self.state["new_parameters"] = self.state["old_parameters"]

  def __find_ranks(self):
    records = np.zeros(self.smax, dtype=np.int) - 1
    level = self.smax
    for i in range(len(self.boxes)):
      print self.boxes[i]
      if self.boxes[i][1] >= 0:
        current_level = self.boxes[i][1]
        level = min(current_level, level)
        if records[current_level] == -1 or self.boxes[records[current_level]][3] > self.boxes[i][3]:
          records[current_level] = i

    return records, level

  def __get_box_info(self, box):
    """ Determines mandatory box information as split and vertex positions x and the furthest y """
    x = np.array(box[-1][-1])
    y = np.array(box[-1][-1])
    splits = np.zeros(len(x), dtype=np.int)

    boxtraversal = box

    while boxtraversal[0] != -1:
      boxtraversal = self.boxes[boxtraversal[0]]
      splits[boxtraversal[4]] += 1

    for i in range(len(x)):
      if x[i] - box[-1][0][i] > box[-1][1][i] - x[i]:
        y[i] = box[-1][0][i]
      else:
        y[i] = box[-1][1][i]

    return splits, x, y

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

# six-hump camel : f(x,y)=x^2*(4-2.1*x^2+x^4/3)+x*y+y^2*(-4+4*y^2)

if __name__ == "__main__":
  from numpy.testing import *
  startPoint = np.array((-1.01, 1.01), np.float)
  u = np.array((-2.0, -2.0), np.float)
  v = np.array((2.0, 2.0), np.float)

  optimi = MCS(function=Rosenbrock(), criterion=criterion.criterion(iterations_max = 1000, ftol = 0.00001), x0=startPoint, u=u, v=v)
  assert_almost_equal(optimi.optimize(), np.ones(2, np.float), decimal=1)
