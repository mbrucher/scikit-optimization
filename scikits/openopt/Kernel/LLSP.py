from ooMisc import assignScript
from BaseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, nan, zeros, any, all, isfinite, eye
from numpy.linalg import norm
import NLP

class LLSP(MatrixProblem):
    __optionalData__ = ['damp', 'X', 'c']
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for LLSP constructor, must be 0..2 + (optionaly) some kwargs')
        if len(args) > 0: kwargs['C'] = args[0]
        if len(args) > 1: kwargs['d'] = args[1]

        MatrixProblem.__init__(self)
        llsp_init(self, kwargs)

    def objFunc(self, x):
        r = norm(dot(self.C, x) - self.d) ** 2  /  2.0
        if not self.damp is None:
            r += self.damp * norm(x-self.X)**2 / 2.0
        if any(isfinite(self.f)): r += dot(self.f, x)
        return r

    def llsp2nlp(self, solver, **solver_params):
        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff, d2f=d2ff)
        else: p = NLP.NLP(ff, zeros(self.n), df=dff, d2f=d2ff)
        p.args.f = self # DO NOT USE p.args = self IN PROB ASSIGNMENT!
        self.inspire(p)
        self.iprint = -1
        # for LLSP plot is via NLP
        p.show = self.show
        p.plot, self.plot = self.plot, 0
        #p.checkdf()
        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf
        return r

    def __prepare__(self):
        MatrixProblem.__prepare__(self)
        if not self.damp is None and not any(isfinite(self.X)):
            self.X = zeros(self.n)




def llsp_init(prob, kwargs):

    prob.probType = 'LLSP'
    prob.goal = 'minimum'
    prob.allowedGoals = ['minimum', 'min']
    prob.showGoal = False

    kwargs['C'] = asfarray(kwargs['C'])

    prob.n = kwargs['C'].shape[1]
    prob.lb = -inf * ones(prob.n)
    prob.ub =  inf * ones(prob.n)
    if not kwargs.has_key('damp'): kwargs['damp'] = None
    if not kwargs.has_key('X'): kwargs['X'] = nan*ones(prob.n)
    if not kwargs.has_key('f'): kwargs['f'] = nan*ones(prob.n)

    if prob.x0 is nan: prob.x0 = zeros(prob.n)

    return assignScript(prob, kwargs)

#def ff(x, LLSPprob):
#    r = dot(LLSPprob.C, x) - LLSPprob.d
#    return dot(r, r)
ff = lambda x, LLSPprob: LLSPprob.objFunc(x)
def dff(x, LLSPprob):
    r = dot(LLSPprob.C.T, dot(LLSPprob.C,x)  - LLSPprob.d)
    if not LLSPprob.damp is None: r += LLSPprob.damp*(x - LLSPprob.X)
    if all(isfinite(LLSPprob.f)) : r += LLSPprob.f
    return r

def d2ff(x, LLSPprob):
    r = dot(LLSPprob.C.T, LLSPprob.C)
    if not LLSPprob.damp is None: r += LLSPprob.damp*eye(x.size)
    return r
