from lp_solve import lp_solve as lps
from scikits.openopt.Kernel.BaseAlg import BaseAlg
from numpy import asarray, inf, ones, nan

from scikits.openopt.Kernel.ooMisc import LinConst2WholeRepr

def List(x):
    if x == None or x.size == 0: return None
    else: return x.tolist()

class lpSolve(BaseAlg):
    __name__ = 'lpSolve'
    __license__ = "LGPL"
    __authors__ = "Michel Berkelaar, michel@es.ele.tue.nl"
    __homepage__ = 'http://sourceforge.net/projects/lpsolve, http://www.cs.sunysb.edu/~algorith/implement/lpsolve/implement.shtml, http://www.nabble.com/lp_solve-f14350i70.html'
    __alg__ = "lpsolve"
    __info__ = 'use p.scale = 1 or True to turn scale mode on'
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars']
    def __init__(self): pass
    def __solver__(self, p):

        LinConst2WholeRepr(p)
        #FIXME: if problem is search for MAXIMUM, not MINIMUM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        f = - asarray(p.f) # sign '-' because lp_solve by default searches for maximum, not minimum
        scalemode = False
        if p.scale in [1, True]:
            scalemode = True
        elif not (p.scale in [None, 0, False]):
            p.warn(self.__name__ + ' requires p.scale from [None, 0, False, 1, True], other value obtained, so scale = True will be used')
            scalemode = True
        [obj, x_opt, duals] = lps(List(f.flatten()), List(p.Awhole), List(p.bwhole.flatten()), List(p.dwhole.flatten()), \
        List(p.lb.flatten()), List(p.ub.flatten()), (1+asarray(p.intVars)).tolist(), scalemode)
        if obj != []:
            p.ff = - obj # sign '-' because lp_solve by default searches for maximum, not minimum
            p.xf = asarray(x_opt).reshape((-1,1))
            p.duals = duals
            p.istop = 1
        else:
            p.ff = nan
            p.xf = nan*ones([p.n,1])
            p.duals = []
            p.istop = -1

