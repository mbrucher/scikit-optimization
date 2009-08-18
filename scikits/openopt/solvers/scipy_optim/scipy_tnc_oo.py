from scipy.optimize.tnc import *
import scikits.openopt
from scikits.openopt.Kernel.setDefaultIterFuncs import *
from scikits.openopt.Kernel.ooMisc import WholeRepr2LinConst
from scikits.openopt.Kernel.BaseAlg import BaseAlg

class scipy_tnc(BaseAlg):
    __name__ = 'scipy_tnc'
    __license__ = "BSD"
    __authors__ = "Stephen G. Nash"
    __alg__ = "undefined"
    __info__ = 'box-bounded NLP solver, can handle lb<=x<=ub constraints, some lb-ub coords can be +/- inf'
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = lambda self, p: True

    def __init__(self): pass

    def __solver__(self, p):
        WholeRepr2LinConst(p)#TODO: remove me
        bounds = []
        for i in xrange(p.n): bounds.append((p.lb[i], p.ub[i]))
        messages = 0#TODO: edit me

        maxfun=p.maxFunEvals
        if maxfun > 1e8:
            p.warn('tnc cannot handle maxFunEvals > 1e8, the value will be used')
            maxfun = int(1e8)

        xf, nfeval, rc = fmin_tnc(p.f, x0 = p.x0, fprime=p.df, args=(), approx_grad=0, bounds=bounds, messages=messages, maxfun=maxfun, ftol=p.ftol, xtol=p.xtol, pgtol=p.gtol)

        if rc in (INFEASIBLE, NOPROGRESS): istop = FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
        elif rc == FCONVERGED: istop = SMALL_DELTA_F
        elif rc == XCONVERGED: istop = SMALL_DELTA_X
        elif rc == MAXFUN: istop = IS_MAX_FUN_EVALS_REACHED
        elif rc == LSFAIL: istop = IS_LINE_SEARCH_FAILED
        elif rc == CONSTANT: istop = IS_ALL_VARS_FIXED
        elif rc == LOCALMINIMUM: istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
        else:
            #TODO: IMPLEMENT USERABORT
            p.err('unknown stop reason')
        msg = RCSTRINGS[rc]
        p.istop, p.msg = istop, msg
        p.xf = xf

##        INFEASIBLE   = -1 # Infeasible (low > up)
##        LOCALMINIMUM =  0 # Local minima reach (|pg| ~= 0)
##        FCONVERGED   =  1 # Converged (|f_n-f_(n-1)| ~= 0)
##        XCONVERGED   =  2 # Converged (|x_n-x_(n-1)| ~= 0)
##        MAXFUN       =  3 # Max. number of function evaluations reach
##        LSFAIL       =  4 # Linear search failed
##        CONSTANT     =  5 # All lower bounds are equal to the upper bounds
##        NOPROGRESS   =  6 # Unable to progress
##        USERABORT    =  7 # User requested end of minimization

##RCSTRINGS = {
##        INFEASIBLE   : "Infeasible (low > up)",
##        LOCALMINIMUM : "Local minima reach (|pg| ~= 0)",
##        FCONVERGED   : "Converged (|f_n-f_(n-1)| ~= 0)",
##        XCONVERGED   : "Converged (|x_n-x_(n-1)| ~= 0)",
##        MAXFUN       : "Max. number of function evaluations reach",
##        LSFAIL       : "Linear search failed",
##        CONSTANT     : "All lower bounds are equal to the upper bounds",
##        NOPROGRESS   : "Unable to progress",
##        USERABORT    : "User requested end of minimization"
##}

if __name__ == '__main__':
    import sys, os.path as P
    sys.path.insert(0,P.split(P.split(P.split(P.split(P.realpath(P.dirname(__file__)))[0])[0])[0])[0])
    from scikits.openopt import NLP
    from numpy import *

    N = 750

    ff = lambda x: (((x/32 - 1)**4).sum() + 64*((x/4 -1 )**2).sum())
##    ff = lambda x: (arctan((x-arange(x.size))/arange(x.size))**2).sum()
##    for solver in ('scipy_tnc',  'scipy_lbfgsb', 'ALGENCAN'):
    for solver in ( 'lincher', 'scipy_lbfgsb', 'ALGENCAN'):
##        lb = 0.1*N + arange(N)
##        ub = 0.8*N + arange(N)
        #x0 = cos(arange(N))
        lb = 50*sin(arange(N))+15
        ub = lb + 15
        x0 = (lb + ub)/2
        p = NLP(ff, x0, lb = lb, ub = ub, plot = 0, maxFunEvals = 1e8, maxIter = 1e4, gtol = 1e-3)
        #p.c = lambda x: x[0]**2 - (lb[0] + ub[0])**2 / 4.0
        #p = NLP(ff, x0, maxFunEvals = 1e8)
        r = p.solve(solver)
        #assert r.istop > 0





