from scipy.optimize import fmin_ncg
from scikits.openopt.Kernel.ooMisc import isSolved
from scikits.openopt.Kernel.BaseAlg import BaseAlg

class scipy_ncg(BaseAlg):
    __name__ = 'scipy_ncg'
    __license__ = "BSD"
    #__authors__ =
    __alg__ = "Newton-CG"
    __info__ = 'unconstrained NLP solver, can handle 2nd derivatives'
    __iterfcnConnected__ = True

    def __init__(self):pass
    def __solver__(self, p):

        def iterfcn(x):
            p.xk, p.fk = x, p.f(x)
            p.iterfcn()
            if p.istop: raise isSolved

        if p.userProvided.d2f: fhess = p.d2f
        else: fhess = None

        xf = fmin_ncg(p.f, p.x0, p.df, fhess = fhess, maxiter = p.maxIter+15, disp = 0, callback=iterfcn)

        ff = p.f(xf)
        p.istop = 1000
        p.xk = p.xf = xf
        p.fk = p.ff = ff

