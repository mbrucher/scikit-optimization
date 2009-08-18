# created by Dmitrey
from numpy import copy, isnan, array, argmax, abs, zeros, any, isfinite, all, where, asscalar, sign, dot, sqrt, array_equal
__docformat__ = "restructuredtext en"
empty_arr = array(())

class Point:
    """
    the class is used to prevent calling non-linear constraints more than once
    f, c, h are funcs for obtaining objFunc, non-lin ineq and eq constraints.
    df, dc, dh are funcs for obtaining 1st derivatives.
    """
    __expectedArgs__ = ['x', 'f', 'mr']
    def __init__(self, p, x, *args, **kwargs):
        self.p = p
        self.x = copy(x)
        for i, arg in enumerate(args):
            setattr(self, '_' + self.__expectedArgs__[i], args[i])
        for name, val in kwargs.iteritems():
            setattr(self, '_' + name, val)
        #assert self.x is not None

    def f(self):
        if not hasattr(self, '_f'): self._f = self.p.f(self.x)
        return copy(self._f)

    def df(self):
        if not hasattr(self, '_df'): self._df = self.p.df(self.x)
        return copy(self._df)

    def c(self, ind=None):
        if not self.p.userProvided.c: return empty_arr.copy()
        if ind is None:
            if not hasattr(self, '_c'): self._c = self.p.c(self.x)
            return copy(self._c)
        else:
            if hasattr(self, '_c'): return copy(self._c[ind])
            else: return copy(self.p.c(self.x, ind))


    def dc(self, ind=None):
        if not self.p.userProvided.c: return empty_arr.copy().reshape(0, self.p.n)
        if ind is None:
            if not hasattr(self, '_dc'): self._dc = self.p.dc(self.x)
            return copy(self._dc)
        else:
            if hasattr(self, '_dc'): return copy(self._dc[ind])
            else: return copy(self.p.dc(self.x, ind))


    def h(self, ind=None):
        if not self.p.userProvided.h: return empty_arr.copy()
        if ind is None:
            if not hasattr(self, '_h'): self._h = self.p.h(self.x)
            return copy(self._h)
        else:
            if hasattr(self, '_h'): return copy(self._h[ind])
            else: return copy(self.p.h(self.x, ind))

    def dh(self, ind=None):
        if not self.p.userProvided.h: return empty_arr.copy().reshape(0, self.p.n)
        if ind is None:
            if not hasattr(self, '_dh'): self._dh = self.p.dh(self.x)
            return copy(self._dh)
        else:
            if hasattr(self, '_dh'): return copy(self._dh[ind])
            else: return copy(self.p.dh(self.x, ind))

    def d2f(self):
        if not hasattr(self, '_d2f'): self._d2f = self.p.d2f(self.x)
        return copy(self._d2f)

    def lin_ineq(self):
        if not hasattr(self, '_lin_ineq'): self._lin_ineq = self.p.__get_AX_Less_B_Residuals__(self.x)
        return copy(self._lin_ineq)

    def lin_eq(self):
        if not hasattr(self, '_lin_eq'): self._lin_eq = self.p.__get_AeqX_eq_Beq_Residuals__(self.x)
        return copy(self._lin_eq)

    def __all_lin_ineq(self):
        if not hasattr(self, '_all_lin_ineq'):
            lb, ub, lin_ineq = self.lb(), self.ub(), self.lin_ineq()
            r = 0
            # TODO: CHECK IT - when 0 (if some nans), when contol
            threshold = 0
#            if all(isfinite(self.f())): threshold = self.p.contol
#            else: threshold = 0

            lb, ub = self.lb(), self.ub()
            lin_ineq = self.lin_ineq()
            lin_eq = self.lin_eq()
            ind_lb, ind_ub = where(lb>threshold)[0], where(ub>threshold)[0]
            ind_lin_ineq = where(lin_ineq>threshold)[0]
            ind_lin_eq = where(abs(lin_eq)>threshold)[0]

            if ind_lb.size != 0:
                r += sum(lb[ind_lb] ** 2)
            if ind_ub.size != 0:
                r += sum(ub[ind_ub] ** 2)
            if ind_lin_ineq.size != 0:
                r += sum(lin_ineq[ind_lin_ineq] ** 2)
            if ind_lin_eq.size != 0:
                r += sum(lin_eq[ind_lin_eq] ** 2)
            self._all_lin_ineq = sqrt(r)
        return copy(self._all_lin_ineq)

    def __all_lin_ineq_gradient(self):
        if not hasattr(self, '_all_lin_ineq_gradient'):
            p = self.p
            d = zeros(p.n)
            contol = p.contol

            # TODO: CHECK IT - when 0 (if some nans), when contol
#            if all(isfinite(self.f())): threshold = contol
#            else: threshold = 0
            threshold = 0

            lb, ub = self.lb(), self.ub()
            lin_ineq = self.lin_ineq()
            lin_eq = self.lin_eq()
            ind_lb, ind_ub = where(lb>threshold)[0], where(ub>threshold)[0]
            ind_lin_ineq = where(lin_ineq>threshold)[0]
            ind_lin_eq = where(abs(lin_eq)>threshold)[0]

            if ind_lb.size != 0:
                d[ind_lb] -= lb[ind_lb]# d/dx((x-lb)^2) for violated constraints
            if ind_ub.size != 0:
                d[ind_ub] += ub[ind_ub]# d/dx((x-ub)^2) for violated constraints
            if ind_lin_ineq.size != 0:
                a = p.A[ind_lin_ineq]
                b = p.b[ind_lin_ineq]
                d += dot(a.T, dot(a, self.x)  - b) # d/dx((Ax-b)^2)
            if ind_lin_eq.size != 0:
                aeq = p.Aeq[ind_lin_eq]
                beq = p.beq[ind_lin_eq]
                d += dot(aeq.T, dot(aeq, self.x)  - beq) # 0.5*d/dx((Aeq x - beq)^2)
            devider = self.__all_lin_ineq()
            if devider != 0:
                self._all_lin_ineq_gradient = d / devider
            else:
                self._all_lin_ineq_gradient = d
        return copy(self._all_lin_ineq_gradient)

    def lb(self):
        if not hasattr(self, '_lb'): self._lb = self.p.lb - self.x
        return copy(self._lb)

    def ub(self):
        if not hasattr(self, '_ub'): self._ub = self.x - self.p.ub
        return copy(self._ub)

    def mr(self, retAll = False):
        # returns max residual
        return self.__mr(retAll)

    def __mr(self, retAll = False):
        if not hasattr(self, '_mr'):
            r, fname, ind = 0, None, None
            for field in ('c',  'lin_ineq', 'lb', 'ub'):
                fv = array(getattr(self, field)()).flatten()
                if fv.size > 0:
                    ind_max = argmax(fv)
                    val_max = fv[ind_max]
                    if r < val_max:
                        r, ind, fname = val_max, ind_max, field
            for field in ('h', 'lin_eq'):
                fv = array(getattr(self, field)()).flatten()
                if fv.size > 0:
                    fv = abs(fv)
                    ind_max = argmax(fv)
                    val_max = fv[ind_max]
                    if r < val_max:
                        r, ind, fname = val_max, ind_max, field
            self._mr, self._mrName,  self._mrInd= r, fname, ind
        if retAll:
            return asscalar(copy(self._mr)), self._mrName, asscalar(copy(self._mrInd))
        else: return asscalar(copy(self._mr))

    def mr_alt(self, retAll = False):
        if not hasattr(self, '_mr_alt'):
            mr, fname, ind = self.__mr(retAll = True)
            self._mr_alt, self._mrName_alt,  self._mrInd_alt= mr, fname, ind
            c, h= self.c(), self.h()
            all_lin_ineq = self.__all_lin_ineq()
            r = 0
            Type = 'all_lin_ineq'
            if c.size != 0:
                ind_max = argmax(c)
                val_max = c[ind_max]
                if val_max > r:
                    r = val_max
                    Type = 'c'
                    ind = ind_max
            if h.size != 0:
                h = abs(h)
                ind_max = argmax(h)
                val_max = h[ind_max]
                #hm = abs(h).max()
                if val_max > r:
                    r = val_max
                    Type = 'h'
                    ind = ind_max
#            if lin_eq.size != 0:
#                l_eq = abs(lin_eq)
#                ind_max = argmax(l_eq)
#                val_max = l_eq[ind_max]
#                if val_max > r:
#                    r = val_max
#                    Type = 'lin_eq'
#                    ind = ind_max

            if  r <= all_lin_ineq:
                self._mr_alt, self._mrName_alt,  self._mrInd_alt = all_lin_ineq, 'all_lin_ineq', 0
            else:
                self._mr_alt, self._mrName_alt,  self._mrInd_alt = r, Type, ind
        if retAll:
            return asscalar(copy(self._mr_alt)), self._mrName_alt, asscalar(copy(self._mrInd_alt))
        else: return asscalar(copy(self._mr_alt))


    def dmr(self, retAll = False):
        # returns direction for max residual decrease
        #( gradient for equality < 0 residuals ! )
        return self.__dmr(retAll)

    def __dmr(self, retAll = False):
        if not hasattr(self, '_dmr') or (retAll and not hasattr(self, '_dmrInd')):
            g = zeros(self.p.n)
            maxResidual, resType, ind = self.mr(retAll=True)
            if resType == 'lb':
                g[ind] -= 1 # N * (-1), -1 = dConstr/dx = d(lb-x)/dx
            elif resType == 'ub':
                g[ind] += 1 # N * (+1), +1 = dConstr/dx = d(x-ub)/dx
            elif resType == 'lin_ineq':
                g += self.p.A[ind]
            elif resType == 'lin_eq':
                rr = self.p.matmult(self.p.Aeq[ind], self.x)-self.p.beq[ind]
                if rr < 0:  g -= self.p.Aeq[ind]
                else:  g += self.p.Aeq[ind]
            elif resType == 'c':
                dc = self.dc(ind=ind).flatten()
                g += dc
            elif resType == 'h':
                dh = self.dh(ind=ind).flatten()
                if self.p.h(self.x, ind) < 0:  g -= dh#CHECKME!!
                else: g += dh#CHECKME!!
            else:
                # TODO: error or debug warning
                pass
                #self.p.err('incorrect resType')

            self._dmr, self._dmrName,  self._dmrInd = g, resType, ind
        if retAll:
            return copy(self._dmr),  self._dmrName,  copy(self._dmrInd)
        else:
            return copy(self._dmr)

    def betterThan(self, *args, **kwargs):
        """
        usage: result = involvedPoint.better(pointToCompare)

        returns True if the involvedPoint is better than pointToCompare
        and False otherwise
        (if NOT better, mb same fval and same residuals or residuals less than desired contol)
        """
        return self.__betterThan__(*args, **kwargs)

    def __betterThan__(self, point2compare, altLinInEq = False):
        if self.p.isUC:
            return self.f() < point2compare.f()

        if altLinInEq:
            mr_field = 'mr_alt'
        else:
            mr_field = 'mr'
        point2compareResidual = getattr(point2compare, mr_field)()

        criticalResidualValue = max((self.p.contol, point2compareResidual))

        if hasattr(self, '_'+mr_field):
            if getattr(self, '_'+mr_field) > criticalResidualValue: return False
        else:
            #TODO: simplify it!
            #for fn in Residuals: (...)
            if altLinInEq:
                if self.__all_lin_ineq() > criticalResidualValue: return False
            else:
                if any(self.lb() > criticalResidualValue): return False
                if any(self.ub() > criticalResidualValue): return False
                if any(self.lin_ineq() > criticalResidualValue): return False
                if any(abs(self.lin_eq()) > criticalResidualValue): return False
            if any(abs(self.h()) > criticalResidualValue): return False
            if any(self.c() > criticalResidualValue): return False

        mr = getattr(self, mr_field)()

        if not self.p.isNaNInConstraintsAllowed:
            if point2compare.__nNaNs__()  > self.__nNaNs__(): return True
            elif point2compare.__nNaNs__()  < self.__nNaNs__(): return False
            # TODO: check me
            if mr <= self.p.contol and point2compareResidual <= self.p.contol and self.__nNaNs__() != 0: return mr < point2compareResidual

        if mr < point2compareResidual and self.p.contol < point2compareResidual: return True

        point2compareF_is_NaN = isnan(point2compare.f())
        selfF_is_NaN = isnan(self.f())

        if not point2compareF_is_NaN: # f(point2compare) is not NaN
            if not selfF_is_NaN: # f(newPoint) is not NaN
                return self.f() < point2compare.f()
            else: # f(newPoint) is NaN
                return False
        else: # f(point2compare) is NaN
            if selfF_is_NaN: # f(newPoint) is NaN
                return mr < point2compareResidual
            else: # f(newPoint) is not NaN
                return True

    def isFeas(self, **kwargs):
        return self.__isFeas__(**kwargs)

    def __isFeas__(self, altLinInEq = False):
        if not all(isfinite(self.f())): return False
        contol = self.p.contol
        if altLinInEq:
            if hasattr(self, '_mr_alt'):
                if self._mr_alt > contol or (not self.p.isNaNInConstraintsAllowed and self.__nNaNs__() != 0): return False
            else:
                #TODO: simplify it!
                #for fn in Residuals: (...)
                if self.all_lin_ineq() > contol: return False
        else:
            if hasattr(self, '_mr'):
                if self._mr > contol or (not self.p.isNaNInConstraintsAllowed and self.__nNaNs__() != 0): return False
            else:
                #TODO: simplify it!
                #for fn in Residuals: (...)
                if any(self.lb() > contol): return False
                if any(self.ub() > contol): return False
                if any(abs(self.lin_eq()) > contol): return False
                if any(self.lin_ineq() > contol): return False
        if any(abs(self.h()) > contol): return False
        if any(self.c() > contol): return False
        return True

    def __nNaNs__(self):
        # returns number of nans in constraints
        r = 0
        c, h = self.c(), self.h()
        r += len(where(isnan(c))[0])
        r += len(where(isnan(h))[0])
        return r

    def directionType(self, *args, **kwargs):
        return self.__directionType__(*args, **kwargs)

    def __directionType__(self, *args, **kwargs):
        if not hasattr(self, 'dType'):
            self.__getDirection__(*args, **kwargs)
        return self.dType

    #def __getDirection__(self, useCurrentBestFeasiblePoint = False):
    def __getDirection__(self, altLinInEq = False):
        if hasattr(self, 'direction'):
            return self.direction
        p = self.p
        contol = p.contol
        maxRes, fname, ind = self.mr_alt(1)
        if self.isFeas(altLinInEq=altLinInEq):
        #or (useCurrentBestFeasiblePoint and hasattr(p, 'currentBestFeasiblePoint') and self.f() - p.currentBestFeasiblePoint.f() > self.mr()):
        #if (maxRes <= p.contol and all(isfinite(self.df())) and (p.isNaNInConstraintsAllowed or self.__nNaNs__() == 0)) :
            self.direction, self.dType = self.df(),'f'
            return self.direction
        else:
            #d = zeros(p.n)
            #if any(self.lb()>contol) or any(self.ub()>contol) or any(self.lin_eq()>contol) or any(self.lin_ineq()>contol):
#            lin_eq = self.lin_eq()
#            c = self.c()
#            h = self.h()

#            LB = lb[lb>contol/2]
#            UB = ub[ub>contol/2]
#            LIN_INEQ = lin_ineq[lin_ineq>contol/2]
#            nActiveLinInEq = LB.size + UB.size + LIN_INEQ.size
#            LinConstraints = sum(LB** 2) + sum(UB ** 2)
#            if  LIN_INEQ.size > 0: LinConstraints+= sum(LIN_INEQ ** 2)
#            #if  lin_eq.size > 0: LinConstraints+= sum(lin_eq[abs(lin_eq)>contol] ** 2)
##
#            maxNonLinConstraint = 0.0
#            if c.size > 0: maxNonLinConstraint = max((maxNonLinConstraint, max(c)))
#            if h.size > 0: maxNonLinConstraint = max((maxNonLinConstraint, max(abs(h))))

            if fname == 'all_lin_ineq':
                d = self.__all_lin_ineq_gradient()
                self.direction, self.dType = d, 'all_lin_ineq'
            elif fname == 'lin_eq':
                raise "OpenOpt kernal error"
                #d = self.dmr()
                #self.dType = 'lin_eq'
            elif fname == 'c':
                d = self.dmr()
                if p.debug: assert array_equal(self.dc(ind).flatten(), self.dmr())
                self.dType = 'c'
            elif fname == 'h':
                d = self.dmr()#sign(self.h(ind))*self.dh(ind)
                if p.debug: assert array_equal(self.dh(ind).flatten(), self.dmr())
                self.dType = 'h'
            else:
                p.err('error in getRalgDirection (unknown residual type ' + fname + ' ), you should report the bug')
            self.direction = d.flatten()
            return self.direction

#    def __getDirection__(self, useCurrentBestFeasiblePoint = False):
#        if hasattr(self, 'direction'):
#            return self.direction
#        p = self.p
#        contol = p.contol
#        maxRes, fname, ind = self.mr(retAll=True)
#        if (maxRes <= p.contol and all(isfinite(self.df())) and (p.isNaNInConstraintsAllowed or self.__nNaNs__() == 0)) \
#        or (useCurrentBestFeasiblePoint and hasattr(p, 'currentBestFeasiblePoint') and self.f() - p.currentBestFeasiblePoint.f() > self.mr()):
#        #if (maxRes <= p.contol and all(isfinite(self.df())) and (p.isNaNInConstraintsAllowed or self.__nNaNs__() == 0)) :
#            self.direction, self.dType = self.df(),'f'
#            return self.direction
#        else:
#            d = zeros(p.n)
#            #if any(self.lb()>contol) or any(self.ub()>contol) or any(self.lin_eq()>contol) or any(self.lin_ineq()>contol):
#            lb = self.lb()
#            ub = self.ub()
#            lin_ineq = self.lin_ineq()
#            lin_eq = self.lin_eq()
#            c = self.c()
#            h = self.h()
#
#            LB = lb[lb>contol/2]
#            UB = ub[ub>contol/2]
#            LIN_INEQ = lin_ineq[lin_ineq>contol/2]
#            nActiveLinInEq = LB.size + UB.size + LIN_INEQ.size
#            LinConstraints = sum(LB** 2) + sum(UB ** 2)
#            if  LIN_INEQ.size > 0: LinConstraints+= sum(LIN_INEQ ** 2)
#            #if  lin_eq.size > 0: LinConstraints+= sum(lin_eq[abs(lin_eq)>contol] ** 2)
##
#            maxNonLinConstraint = 0.0
#            if c.size > 0: maxNonLinConstraint = max((maxNonLinConstraint, max(c)))
#            if h.size > 0: maxNonLinConstraint = max((maxNonLinConstraint, max(abs(h))))
#
#            if nActiveLinInEq != 0 and LinConstraints/nActiveLinInEq >= p.contol * maxNonLinConstraint and (lin_eq.size == 0 or LinConstraints/nActiveLinInEq >= p.contol * abs(lin_eq).max()):#fname in ['lb',  'ub',  'lin_eq',  'lin_ineq']:# or tmp > maxNonLinConstraint:
#                threshold = contol
#                ind_lb = where(lb>contol)[0]
#                ind_ub = where(ub>contol)[0]
#                ind_lin_ineq = where(lin_ineq>threshold)[0]
#                #ind_lin_eq = where(abs(lin_eq)>threshold)[0]
#
#                if ind_lb.size != 0:
#                    d[ind_lb] -= lb[ind_lb]# 0.5*d/dx((x-lb)^2) for violated constraints
#                if ind_ub.size != 0:
#                    d[ind_ub] += ub[ind_ub]# 0.5*d/dx((x-ub)^2) for violated constraints
#                if ind_lin_ineq.size != 0:
#                    a = p.A[ind_lin_ineq]
#                    b = p.b[ind_lin_ineq]
#                    d += dot(a.T, dot(a, self.x)  - b) # 0.5*d/dx((Ax-b)^2)
##                if ind_lin_eq.size != 0:
##                    aeq = p.Aeq[ind_lin_eq]
##                    beq = p.beq[ind_lin_eq]
##                    d += dot(aeq.T, dot(aeq, self.x)  - beq) # 0.5*d/dx((Ax-b)^2)
#                self.direction, self.dType = d/p.contol/nActiveLinInEq, 'linear'
#            elif fname == 'lin_eq':
#                d = self.dmr()
#                self.direction = d.flatten()
#                self.dType = 'lin_eq'
#            elif fname == 'c':
#                d = self.dmr()#self.dc(ind)
#                self.direction = d.flatten()
#                self.dType = 'c'
#            elif fname == 'h':
#                d = self.dmr()#sign(self.h(ind))*self.dh(ind)
#                self.direction = d.flatten()
#                self.dType = 'h'
#            else:
#                p.err('error in getRalgDirection (unknown residual type ' + fname + ' ), you should report the bug')
#            return self.direction
