from scipy.optimize import fmin_bfgs
from scikits.openopt.Kernel.ooMisc import isSolved
from scikits.openopt.Kernel.BaseAlg import BaseAlg

class scipy_bfgs(BaseAlg):
    __name__ = 'scipy_bfgs'
    __license__ = "BSD"
    #__authors__ =
    __alg__ = "BFGS"
    __info__ = 'unconstrained NLP solver'
    __iterfcnConnected__ = True

    def __init__(self):pass
    def __solver__(self, p):

        def iterfcn(x):
            p.xk, p.fk = x, p.f(x)
            p.iterfcn()
            if p.istop: raise isSolved

#        try:
        #p.iterfcn(p.x0)
        p.xk = p.xf = fmin_bfgs(p.f, p.x0, fprime=p.df, disp = 0, gtol=p.gtol, maxiter=p.maxIter, callback=iterfcn)
        p.istop = 1000
#        except isSolved:
#            xf = p.xk
#
#        ff = p.f(xf)
#        p.xk = p.xf = xf
#        p.fk = p.ff = ff
#        p.istop = 1000
#        p.iterfcn()

