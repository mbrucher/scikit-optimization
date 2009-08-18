from ooMisc import assignScript
from BaseProblem import NonLinProblem
from numpy import asarray, ones, inf, array, iterable
from NLP import nlp_init

class GLP(NonLinProblem):
    __optionalData__ = ['lb', 'ub']
    def __init__(self, *args, **kwargs):
        if len(args) > 1: self.err('incorrect args number for GLP constructor, must be 0..1 + (optionaly) some kwargs')

        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        NonLinProblem.__init__(self)

        glp_init(self, kwargs2)


def glp_init(p, kwargs):

    p.probType = 'GLP'
    p.goal = 'minimum'
    p.allowedGoals = ['minimum', 'min', 'maximum', 'max']
    p.showGoal = True
    p.plotOnlyCurrentMinimum= True

    f = kwargs['f']

    if kwargs.has_key('lb'):
        p.n = len(kwargs['lb'])
        if not kwargs.has_key('x0'): kwargs['x0'] = kwargs['lb']
    elif kwargs.has_key('ub'):
        p.n = len(kwargs['ub'])
        if not kwargs.has_key('x0'): kwargs['x0'] = kwargs['ub']

    p.lb = -inf * ones([p.n,1])
    p.ub =  inf * ones([p.n,1])


    if isinstance(f, basestring):
        p.err("Isn't implemented yet")
        # TODO: implement me!
        # p. f, p.fName = ..., f
    elif callable(f):
        p.f, p.fName = f, f.__name__
    elif iterable(f):
        p.f, p.fName = f, "undefined"
    else: p.err('incorrect objFun')


    return assignScript(p, kwargs)
