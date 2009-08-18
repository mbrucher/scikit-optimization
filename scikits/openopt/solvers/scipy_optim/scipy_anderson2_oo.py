from scipy.optimize import anderson2
from numpy import asfarray
from scikits.openopt.Kernel.BaseAlg import BaseAlg

class scipy_anderson2(BaseAlg):
    __name__ = 'scipy_anderson2'
    __license__ = "BSD"
    #__authors__ = 
    #__alg__ = ""  
    __info__ = """
    solves system of n non-linear equations with n variables. 
    """

    def __init__(self):pass
    def __solver__(self, p):
        
        p.xk = p.x0.copy()
        p.fk = asfarray(max(abs(p.f(p.x0)))).flatten()
        
        p.iterfcn()
        if p.istop:
            p.xf, p.ff = p.xk, p.fk
            return 
        
        try: xf = anderson2(p.f, p.x0, iter = p.maxIter)
        except: 
            p.istop = -1000
            return

        p.xk = p.xf = asfarray(xf)
        p.fk = p.ff = asfarray(max(abs(p.f(xf)))).flatten()
        p.istop = 1000
        p.iterfcn()
