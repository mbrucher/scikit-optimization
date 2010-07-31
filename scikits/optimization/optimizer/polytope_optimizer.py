
"""
A polytope/Nelder Mead optimizer
"""

import numpy

import optimizer

class PolytopeOptimizer(optimizer.Optimizer):
  """
  A polytope/simplex/Nelder-Mead optimizer
  """
  def __init__(self, **kwargs):
    """
    Needs to have :
      - an object function to optimize (function), alternatively a function ('fun'), gradient ('gradient'), ...
      - a criterion to stop the optimization (criterion)
      - an array of starting points (n+1 points of dimension n, x0)
    """
    optimizer.Optimizer.__init__(self, **kwargs)
    self.optimal_point = kwargs['x0']

    self.sort_save()
    self.record_history(**self.state)

  def sort_save(self):
    """
    Sorts the current points/values and save them
    """
    values = numpy.array([self.function(point) for point in self.optimal_point])
    sorted_indices = numpy.argsort(values)
    self.state['new_parameters'] = numpy.array(self.optimal_point[sorted_indices])
    self.state['new_value'] = values[sorted_indices]

  def get_value(self, mean, discarded_point, t):
    """
    Compute the new point and its associated value
    """
    point = mean + t * (discarded_point - mean)
    return point, self.function(point)
    
  def iterate(self):
    """
    Implementation of the optimization. Does one iteration
    """
    self.state['old_parameters'] = self.optimal_point = self.state['new_parameters']
    self.state['old_value'] = self.state['new_value']

    mean = numpy.mean(self.optimal_point[:-1], axis=0)
    discarded_point = self.optimal_point[-1]
    
    point, value = self.get_value(mean, discarded_point, -1)
    if value < self.state['old_value'][-2]:
      if value > self.state['old_value'][0]:
        self.optimal_point = numpy.vstack((self.optimal_point[:-1], point))
      else:
        point_expansion, value_expansion = self.get_value(mean, discarded_point, -2)
        if value_expansion < value:
          self.optimal_point = numpy.vstack((self.optimal_point[:-1], point_expansion))
        else:
          self.optimal_point = numpy.vstack((self.optimal_point[:-1], point))
    else:
      if value < self.state['old_value'][-1]:
        point_contraction, value_contraction = self.get_value(mean, discarded_point, -.5)
        if value_contraction < value:
          self.optimal_point = numpy.vstack((self.optimal_point[:-1], point_contraction))
        else:
          self.optimal_point = (self.optimal_point + self.optimal_point[0])/2
      else:
        point_contraction, value_contraction = self.get_value(mean, discarded_point, .5)
        if value_contraction < self.state['old_value'][-1]:
          self.optimal_point = numpy.vstack((self.optimal_point[:-1], point_contraction))
        else:
          self.optimal_point = (self.optimal_point + self.optimal_point[0])/2

    self.sort_save()

    self.record_history(**self.state)

