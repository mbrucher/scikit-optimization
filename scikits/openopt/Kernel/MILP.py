
from ooMisc import assignScript
from BaseProblem import MatrixProblem
from numpy import asarray, ones, inf, dot, nan, zeros

from LP import lp_init



class MILP(MatrixProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for MILP constructor, must be 0..2 + (optionaly) some kwargs')

        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: kwargs2['intVars'] = args[1]
        self.probType = 'MILP'
        MatrixProblem.__init__(self)
        lp_init(self, kwargs2)

    def objFunc(self, x):
        return dot(self.f, x)



