from BaseProblem import NonLinProblem
from NLP import nlp_init
from numpy import sum, dot, asfarray
import NLP

class LSP(NonLinProblem):
    __optionalData__ = []
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for LSP constructor, must be 0..2 + (optionaly) some kwargs')

        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: kwargs2['x0'] = args[1]

        NonLinProblem.__init__(self)
        self.probType = 'LSP'
        self.allowedGoals = ['minimum', 'min']
        self.showGoal = False
        self.isObjFunValueASingleNumber = False

        return nlp_init(self, kwargs2)


    def objFuncMultiple2Single(self, fv):
        return (fv ** 2).sum()

    def lsp2nlp(self, solver, **solver_params):

        ff = lambda x: sum(asfarray(self.f(x))**2)
        if hasattr(self, 'df'):
            dff = lambda x: dot(2*asfarray(self.f(x)), asfarray(self.df(x)))
            p = NLP.NLP(ff, self.x0, df=dff)
        else:
            p = NLP.NLP(ff, self.x0)
        #p = NLP.NLP(FF, self.x0)
        self.inspire(p, sameConstraints=True)


        def lsp_iterfcn(*args,  **kwargs):
            p.primalIterFcn(*args,  **kwargs)
            p.xk = self.xk
            p.fk = p.f(p.xk)
            p.rk = self.rk
            # TODO: add nNaNs

            p.istop = self.istop

        p.primalIterFcn,  p.iterfcn = self.iterfcn, lsp_iterfcn
        p.goal = 'min'

        self.iprint = -1
        p.show = False

        r = p.solve(solver, **solver_params)
        #r.ff = ff(r.xf)

        return r
