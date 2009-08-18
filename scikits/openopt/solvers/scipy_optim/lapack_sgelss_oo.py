from scipy.linalg.flapack import sgelss
from numpy.linalg import norm
from numpy import dot, asfarray, atleast_1d
from scikits.openopt.Kernel.BaseAlg import BaseAlg

class lapack_sgelss(BaseAlg):
    __name__ = 'lapack_sgelss'
    __license__ = "BSD"
    __authors__ = 'Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., Courant Institute, Argonne National Lab, and Rice University'
    #__alg__ = "SVD"
    __info__ = 'wrapper to LAPACK sgelss routine (single precision), requires scipy & LAPACK 3.0 or newer installed'
    def __init__(self):pass
    def __solver__(self, p):
        v,x,s,rank,info = sgelss(p.C, p.d)
        xf = x[:p.C.shape[1]]
        ff = atleast_1d(asfarray(p.F(xf)))
        p.xf = p.xk = xf
        p.ff = p.fk = ff
        if info == 0: p.istop = 1000
        else: p.istop = -1000
