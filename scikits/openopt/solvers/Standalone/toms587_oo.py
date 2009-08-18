from numpy.linalg import norm
from numpy import dot, asfarray, atleast_1d,  zeros,  ones,  int,  float128,  float64, where, inf, hstack, vstack, array
from scikits.openopt.Kernel.BaseAlg import BaseAlg
from toms_587 import lsei
from scikits.openopt.Kernel.ooMisc import xBounds2Matrix

f = lambda x: norm(dot(p.C, x) - p.d)

class toms587(BaseAlg):
    __name__ = 'toms587'
    __license__ = "BSD"
    __authors__ = 'R. J. HANSON AND K. H. HASKELL'
    #__alg__ = ''
    __info__ = 'requires manual compilation of toms_587.f by f2py, see OO online doc for details'
    #__optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']

    T = float64
    def __init__(self): pass

    def __solver__(self, p):
        xBounds2Matrix(p)
        T = self.T
        n = p.n
        #xf = zeros(n, T)
        xf = zeros(1, T)
        A, B = T(p.C),  T(p.d).reshape(-1, 1)
        G, H = T(p.A),  T(p.b).reshape(-1, 1)
        E, F =  T(p.Aeq),  T(p.beq).reshape(-1, 1)
        me, ma, mg = F.size, B.size, H.size
        #mdw = me + ma + mg
        prgopt = 1.0
        w = vstack((hstack((E, F)), hstack((A, B)), hstack((G, H))))
        mode = -15
        rnorme, rnorml = -15.0, -15.0
        ip = array((-15, -15, -15))
        ws = array((-15.0))

        #bl, bu = p.lb.copy(), p.ub.copy()
#        bl[where(bl==-inf)[0]] = -self.BVLS_inf
#        bu[where(bu==inf)[0]] = self.BVLS_inf
#        if hasattr(bvls,  'boundedleastsquares'):#f90 version
#            p.debugmsg('using BVLS.f90')
#            xf,loopa = bvls.boundedleastsquares.bvls(key,a,b,bl,bu,istate)
#        else:
#            p.debugmsg('using BVLS.f')
#            bvls.bvls(key, a, b, bl, bu, xf,  w, act, zz,  istate, loopa)
        #lsei(w,me,ma,mg,n,prgopt,xf,rnorme,rnorml,mode,ws,ip,[mdw])
        lsei(w.flatten(),me,ma,mg,n,prgopt,xf,rnorme,rnorml,mode,ws,ip)

        #p.iter = loopa
        ff = atleast_1d(asfarray(f(xf)))
        p.xf = p.xk = xf
        p.ff = p.fk = ff
        p.istop = 1000



