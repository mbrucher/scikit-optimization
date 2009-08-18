
# Matthieu Brucher
# Last Change : 2007-08-10 23:13

"""
A standard optimizer
"""

import optimizer

class StandardOptimizer(optimizer.Optimizer):
  """
  A standard optimizer, takes a step and finds the best candidate
  Must give in self.optimalPoint the optimal point after optimization
  """
  def __init__(self, **kwargs):
    """
    Needs to have :
      - an object function to optimize (function), alternatively a function ('fun'), gradient ('gradient'), ...
      - a way to get a new point, that is a step (step)
      - a criterion to stop the optimization (criterion)
      - a starting point (x0)
      - a way to find the best point on a line (lineSearch)
    Can have :
      - a step modifier, a factor to modulate the step (stepSize = 1.)
    """
    optimizer.Optimizer.__init__(self, **kwargs)
    self.stepKind = kwargs['step']
    self.optimalPoint = kwargs['x0']
    self.lineSearch = kwargs['line_search']

    self.state['new_parameters'] = self.optimalPoint
    self.state['new_value'] = self.function(self.optimalPoint)

    self.recordHistory(**self.state)

  def iterate(self):
    """
    Implementation of the optimization. Does one iteration
    """
    self.state['old_parameters'] = self.optimalPoint
    self.state['old_value'] = self.state['new_value']

    step = self.stepKind(self.function, self.optimalPoint, state = self.state)

    self.optimalPoint = self.lineSearch(origin = self.optimalPoint, function = self.function, state = self.state)
    self.state['new_parameters'] = self.optimalPoint

    self.state['new_value'] = self.function(self.optimalPoint)

    self.recordHistory(**self.state)

