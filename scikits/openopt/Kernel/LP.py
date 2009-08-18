from ooMisc import assignScript
from BaseProblem import MatrixProblem
from numpy import asarray, ones, inf, dot, nan, zeros
import NLP

class LP(MatrixProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    def __init__(self, *args, **kwargs):
        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: self.err('incorrect args number for LP constructor, must be 0..1 + (optionaly) some kwargs')
        self.probType = 'LP'
        MatrixProblem.__init__(self)
        lp_init(self, kwargs2)


    def objFunc(self, x):
        return dot(self.f, x)

    def lp2nlp(self, solver, **solver_params):
        ff = lambda x: dot(x, self.f)
        dff = lambda x: self.f
        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff)
        else: p = NLP.NLP(ff, zeros(self.n), df=dff)
        self.inspire(p)
        self.iprint = -1

        # for LP plot is via NLP
        p.show = self.show
        p.plot, self.plot = self.plot, 0

        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf
        return r


def lp_init(prob, kwargs):

    prob.goal = 'minimum'
    prob.allowedGoals = ['minimum', 'min']#TODO: add handling of maximization problems
    prob.showGoal = True

    f = asarray(kwargs['f'], float)
    kwargs['f'] = f

    prob.n = len(f)
    if prob.x0 is nan: prob.x0 = zeros(prob.n)
    prob.lb = -inf * ones(prob.n)
    prob.ub =  inf * ones(prob.n)

    return assignScript(prob, kwargs)



