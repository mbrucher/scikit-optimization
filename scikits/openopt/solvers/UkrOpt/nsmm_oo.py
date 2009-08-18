from numpy import asfarray, argmax, inf, log10
from numpy.linalg import norm
from scikits.openopt.Kernel.BaseAlg import BaseAlg
from scikits.openopt import NSP
from string import rjust

class nsmm(BaseAlg):
    __name__ = 'nsmm'
    __license__ = "BSD"
    __authors__ = 'Dmitrey Kroshko'
    __alg__ = "based on Naum Z. Shor r-alg"
    __iterfcnConnected__ = True
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    __info__ = """
    Solves mini-max problem
    via minimizing max (F[i]) using NSP solver (default UkrOpt.ralg).

    Can handle user-supplied gradient/subradient (p.df field)
    If the one is not available -
    splitting equations to separate functions is recommended
    (to speedup calculations):
    f = [func1, func2, ...] or f = ([func1, func2, ...)
    """

    def __init__(self):pass
    def __solver__(self, p):
        self.lastIterTextOutputWasInvolved = False
        self.isConstrained = not p.__isUnconstrained__()
        f = lambda x: max(p.f(x))
        def df(x):
            F = p.f(x)
            ind = argmax(F)
            return p.df(x, ind)

        def iterfcn(*args,  **kwargs):
            p2.primalIterFcn(*args,  **kwargs)

            p.xk = p2.xk.copy()
            p.fk = p2.fk#p.objFuncMultiple2Single(p.f(p.xk))
            p.rk = p2.rk
            #p.isFinished = p2.isFinished

            p.istop = p2.istop


            cond1 = p.iprint>0 and p.iter>0 and p.iter % p.iprint == 0
            #or len(p2.iterValues.r)>1 and p2.iterValues.r[-2]>p2.contol)
            cond2 = (p.iter == 0 or (p.istop and (p2.rk <= p2.contol ))) \
            and p.iprint>=0 and not self.lastIterTextOutputWasInvolved
            #print p2.rk,  p.getMaxResidual(p.xk),   p2.rk-  p.getMaxResidual(p.xk)
            if p.istop and p2.rk <= p2.contol:
                self.lastIterTextOutputWasInvolved = True
                p.msg = p2.msg
            #print 'iter',  p.iter,  'cond2',  cond2,  'istop',  p.istop,  'self.lastIterTextOutputWasInvolved',  self.lastIterTextOutputWasInvolved

            p.iterfcn()

        p2 = NSP(f, p.x0, df=df, xtol = p.xtol, ftol = p.ftol, gtol = p.gtol,\
        A=p.A,  b=p.b,  Aeq=p.Aeq,  beq=p.beq,  lb=p.lb,  ub=p.ub, \
        maxFunEvals = p.maxFunEvals, fEnough = p.fEnough, maxIter=p.maxIter, iprint = -1, \
        maxtime = p.maxTime, maxCPUTime = p.maxCPUTime,  noise = p.noise)

        if p.userProvided.c:
            p2.c,  p2.dc = p.c,  p.dc
        if p.userProvided.h:
            p2.h,  p2.dh = p.h,  p.dh


        p2.primalIterFcn,  p2.iterfcn = p2.iterfcn, iterfcn

        r2 = p2.solve('ralg')
        #xf = fsolve(p.f, p.x0, fprime=p.df, xtol = p.xtol, maxfev = p.maxFunEvals)
        xf = r2.xf
        p.xk = p.xf = xf
        p.fk = p.ff = asfarray(norm(p.f(xf), inf)).flatten()
        #p.istop is defined in iterfcn
