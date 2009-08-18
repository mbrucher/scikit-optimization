from scipy.optimize import fsolve
from numpy import asfarray
from scikits.openopt.Kernel.BaseAlg import BaseAlg


class scipy_fsolve(BaseAlg):
    __name__ = 'scipy_fsolve'
    __license__ = "BSD"
    #__authors__ =
    #__alg__ = ""
    __info__ = """
    solves system of n non-linear equations with n variables.
    """

    def __init__(self):pass
    def __solver__(self, p):
        xf = fsolve(p.f, p.x0, fprime=p.df, xtol = p.xtol, maxfev = p.maxFunEvals, warning = (p.iprint>=0))
        p.istop = 1000
        p.iterfcn(xf)


