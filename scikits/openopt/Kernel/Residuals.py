__docformat__ = "restructuredtext en"
from numpy import concatenate, asfarray, array, where, argmax, zeros, isfinite, copy, all, isnan
from copy import deepcopy
empty_arr = asfarray([])


class Residuals:
    def __init__(self):
        pass
    def __get_nonLinInEq_Residuals__(self, x):
        if hasattr(self.userProvided, 'c') and self.userProvided.c: return self.c(x)
        else: return empty_arr

    def __get_nonLinEq_Residuals__(self, x):
        if hasattr(self.userProvided, 'h') and self.userProvided.h: return self.h(x)
        else: return empty_arr

    def __get_AX_Less_B_Residuals__(self, x):
        #TODO: CHECK FUTURE VERSIONS OF NUMPY IS flatten() required
        if self.A != None and self.A.size > 0  : return self.matmult(self.A, x) - self.b
        else: return empty_arr

    def __get_AeqX_eq_Beq_Residuals__(self, x):
        #TODO: CHECK FUTURE VERSIONS OF NUMPY IS flatten() required
        if self.Aeq != None and self.Aeq.size>0 : return self.matmult(self.Aeq, x).flatten() - self.beq
        else: return empty_arr

    def __getLbResiduals__(self, x):
        return self.lb - x

    def __getUbResiduals__(self, x):
        return x - self.ub

    def __getResiduals__(self, x):
#        if self.prevVal['r'].has_key('x') and all(x == self.prevVal['r']['x']):
#            return self.prevVal['r']['Val']
        # TODO: add quadratic constraints
        r = EmptyClass()
        # TODO: simplify it!
        if self.__baseClassName__ == 'NonLin':
            r.c = self.__get_nonLinInEq_Residuals__(x)
            r.h = self.__get_nonLinEq_Residuals__(x)
        else:
            r.c = r.h = 0
        r.lin_ineq = self.__get_AX_Less_B_Residuals__(x)
        r.lin_eq= self.__get_AeqX_eq_Beq_Residuals__(x)
        r.lb = self.__getLbResiduals__(x)
        r.ub = self.__getUbResiduals__(x)
#        self.prevVal['r']['Val'] = deepcopy(r)
#        self.prevVal['r']['x'] = copy(x)
        return r

    def getMaxResidual(self, x, retAll = False):
        """
        if retAll:  returns
        1) maxresidual
        2) name of residual type (like 'lb', 'c', 'h', 'Aeq')
        3) index of the constraint of given type
        (for example 15, 'lb', 4 means maxresidual is equal to 15, provided by lb[4])
        don't forget about Python indexing from zero!
        if retAll == False:
        returns only r
        """

        residuals = self.__getResiduals__(x)
        r, fname, ind = 0, None, None
        for field in ('c',  'lin_ineq', 'lb', 'ub'):
            fv = array(getattr(residuals, field)).flatten()
            if fv not in ([], ()) and fv.size>0:
                ind_max = argmax(fv)
                val_max = fv[ind_max]
                if r < val_max:
                    r, ind, fname = val_max, ind_max, field
        for field in ('h', 'lin_eq'):
            fv = array(getattr(residuals, field)).flatten()
            if fv not in ([], ()) and fv.size>0:
                fv = abs(fv)
                ind_max = argmax(fv)
                val_max = fv[ind_max]
                if r < val_max:
                    r, ind, fname = val_max, ind_max, field
#        if self.probType == 'NLSP':
#            fv = abs(self.f(x))
#            ind_max = argmax(fv)
#            val_max = fv[ind_max]
#            if r < val_max:
#                r, ind, fname = val_max, ind_max, 'f'
        if retAll:
            return r, fname, ind
        else:
            return r

    def __getMaxConstrGradient2__(self, x):
        g = zeros(self.n)
        mr0 = self.getMaxResidual(x)
        for j in xrange(self.n):
            x[j] += self.diffInt
            g[j] = self.getMaxResidual(x)-mr0
            x[j] -= self.diffInt
        g /= self.diffInt
        return g

    def getMaxConstrGradient(self, x,  retAll = False):
        g = zeros(self.n)
        maxResidual, resType, ind = self.getMaxResidual(x, retAll=True)
        if resType == 'lb':
            g[ind] -= 1 # N * (-1), -1 = dConstr/dx = d(lb-x)/dx
        elif resType == 'ub':
            g[ind] += 1 # N * (+1), +1 = dConstr/dx = d(x-ub)/dx
        elif resType == 'A':
            g += self.A[ind]
        elif resType == 'Aeq':
            rr = self.matmult(self.Aeq[ind], x)-self.beq[ind]
            if rr < 0:  g -= self.Aeq[ind]
            else:  g += self.Aeq[ind]
        elif resType == 'c':
            dc = self.dc(x, ind).flatten()
            g += dc
        elif resType == 'h':
            dh = self.dh(x, ind).flatten()
            if self.h(x, ind) < 0:  g -= dh#CHECKME!!
            else: g += dh#CHECKME!!
        if retAll:
            return g,  resType,  ind
        else:
            return g

    def __getLagrangeResiduals__(self, x, lm):
        #lm is Lagrange multipliers
        residuals = self.getResiduals(x)
        r = 0

        for field in ['c', 'h', 'A', 'Aeq', 'lb', 'ub']:
            fv = getattr(residuals, field)
            if fv not in ([], ()) and fv.size>0: r += p.dotwise(fv, getattr(lm, field))
        return r
        #return r.nonLinInEq * lm.nonLinInEq + r.nonLinEq * lm.nonLinEq + \
                   #r.aX_Less_b * lm.aX_Less_b + r.aeqX_ineq_beq * lm.aeqX_ineq_beq + \
                   #r.res_lb * lm.res_lb + r.res_ub * lm.res_ub

    def isFeas(self, x):
        if hasattr(self, 'isNaNInConstraintsAllowed') and not self.isNaNInConstraintsAllowed and \
        (any(isnan(self.__get_nonLinEq_Residuals__(x))) or any(isnan(self.__get_nonLinInEq_Residuals__(x)))):
            return False
        is_X_finite = all(isfinite(x))
        is_ConTol_OK = self.getMaxResidual(x) <= self.contol
        cond1 = is_ConTol_OK and is_X_finite and all(isfinite(self.objFunc(x)))
        if self.probType == 'NLSP': return cond1 and self.F(x) < self.ftol
        else: return cond1


class EmptyClass:
    pass


