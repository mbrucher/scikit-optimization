from scipy.optimize import fmin_slsqp
import scikits.openopt
from scikits.openopt.Kernel.setDefaultIterFuncs import *
from scikits.openopt.Kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from scikits.openopt.Kernel.BaseAlg import BaseAlg
from numpy import *

class EmptyClass: pass

class scipy_slsqp(BaseAlg):
    __name__ = 'scipy_slsqp'
    __license__ = "BSD"
    __authors__ = """Dieter Kraft, connected to scipy by Rob Falck, connected to OO by Dmitrey"""
    __alg__ = "Sequential Least SQuares Programming"
    __info__ = 'constrained NLP solver'
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']

    def __init__(self): pass
    def __solver__(self, p):
        bounds = []
        if any(isfinite(p.lb)) or any(isfinite(p.ub)):
            for i in xrange(p.n):
                bounds.append((p.lb[i], p.ub[i]))
        if p.userProvided.c:
            C = lambda x: -hstack((p.c(x), p.matmult(p.A, x) - p.b))
            fprime_ieqcons = lambda x: -vstack((p.dc(x), p.A))
        else: C,  fprime_ieqcons = None,  None
        if p.userProvided.h:
            H = lambda x: hstack((p.h(x), p.matmult(p.Aeq, x) - p.beq))
            fprime_eqcons = lambda x: vstack((p.dh(x), p.Aeq))
        else: H,  fprime_eqcons = None,  None
       # fprime_cons = lambda x: vstack((p.dh(x), p.Aeq, p.dc(x), p.A))

        x, fx, its, imode, smode = fmin_slsqp(p.f, p.x0, bounds=bounds,  f_eqcons = H, f_ieqcons = C, full_output=1, iprint=-1, fprime = p.df, fprime_eqcons = fprime_eqcons, fprime_ieqcons = fprime_ieqcons, acc = p.contol, iter = p.maxIter)
        p.msg = smode
        if imode == 0: p.istop = 1000
        #elif imode == 9: p.istop = ? CHECKME that OO Kernel is capable of handling the case
        else: p.istop = -1000
	p.xf, p.ff = array(x), fx

