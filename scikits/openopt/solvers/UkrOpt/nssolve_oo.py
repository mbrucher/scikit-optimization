from numpy import asfarray, argmax, sign, inf, log10
from numpy.linalg import norm
from scikits.openopt.Kernel.BaseAlg import BaseAlg
from scikits.openopt import NSP
from string import rjust
from scikits.openopt.Kernel.setDefaultIterFuncs import IS_MAX_FUN_EVALS_REACHED, FVAL_IS_ENOUGH

class nssolve(BaseAlg):
    __name__ = 'nssolve'
    __license__ = "BSD"
    __authors__ = 'Dmitrey Kroshko'
    __alg__ = "based on Naum Z. Shor r-alg"
    __iterfcnConnected__ = True
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    __isIterPointAlwaysFeasible__ = lambda self, p: p.isUC
    __info__ = """
    Solves system of non-smooth or noisy equations
    via (by default) minimizing max residual using NSP solver (default UkrOpt.ralg).

    Can handle user-supplied gradient/subradient (p.df field)
    If the one is not available -
    splitting equations to separate functions is recommended
    (to speedup calculations):
    f = [func1, func2, ...] or f = ([func1, func2, ...)

    ns- can be interpreted as
    NonSmooth
    or NoiSy
    or Naum Shor (Ukrainian academician, my teacher, r-algorithm inventor)
    """

    def __init__(self):pass
    def __solver__(self, p):

        f = lambda x: max(abs(p.f(x)))
        def df(x):
            F = p.f(x)
            ind = argmax(abs(F))
            return p.df(x, ind) * sign(F[ind])

        def iterfcn(*args,  **kwargs):
            p2.primalIterFcn(*args,  **kwargs)
            #p2.solver.__decodeIterFcnArgs__(p2,  *args,  **kwargs)
            p.xk = p2.xk.copy()
            Fk = norm(p.f(p.xk), inf)
            p.rk = p.getMaxResidual(p.xk)

            #TODO: ADD p.rk

            if p.nEvals['f'] > p.maxFunEvals:
                p.istop = p2.istop = IS_MAX_FUN_EVALS_REACHED
            elif p2.istop!=0:
                if Fk < p.ftol and p.rk < p.contol:
                    p.istop = 15
                    if p.isUC: msg_contol = ''
                    else: msg_contol = 'and contol '
                    p.msg = 'solution with required ftol ' + msg_contol+ 'has been reached'
                else: p.istop = p2.istop

            p.iterfcn()


        p2 = NSP(f, p.x0, df=df, xtol = p.xtol/1e16, gtol = p.gtol/1e16,\
        A=p.A,  b=p.b,  Aeq=p.Aeq,  beq=p.beq,  lb=p.lb,  ub=p.ub, \
        maxFunEvals = p.maxFunEvals, fEnough = p.ftol, maxIter=p.maxIter, iprint = -1, \
        maxtime = p.maxTime, maxCPUTime = p.maxCPUTime,  noise = p.noise)

        if p.isUC: p2.ftol = p.ftol / 1e16
        else: p2.ftol = 0

        if p.userProvided.c:
            p2.c,  p2.dc = p.c,  p.dc
        if p.userProvided.h:
            p2.h,  p2.dh = p.h,  p.dh

        p2.primalIterFcn,  p2.iterfcn = p2.iterfcn, iterfcn

        if p.debug: p2.iprint = 1
        r2 = p2.solve('ralg')
        #xf = fsolve(p.f, p.x0, fprime=p.df, xtol = p.xtol, maxfev = p.maxFunEvals)
        xf = r2.xf
        p.xk = p.xf = xf
        p.fk = p.ff = asfarray(norm(p.f(xf), inf)).flatten()
        #p.istop is defined in iterfcn
