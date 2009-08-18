from ooMisc import assignScript
from BaseProblem import NonLinProblem
from numpy import asarray, ones, inf

from NLP import nlp_init



class NSP(NonLinProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for NSP constructor, must be 0..2 + (optionaly) some kwargs')

        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: kwargs2['x0'] = args[1]
        NonLinProblem.__init__(self)



        self.allowedGoals = ['minimum', 'min', 'maximum', 'max']
        self.showGoal = True
        #TODO: set here default tolx, tolcon, diffInt etc for NS Problem

        nlp_init(self, kwargs2)
        self.probType = 'NSP'






