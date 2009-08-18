__docformat__ = "restructuredtext en"
from numpy import *
from string import lower
from scikits.openopt.Kernel.BaseAlg import BaseAlg
#from scikits.openopt.Kernel.setDefaultIterFuncs import SMALL_DF
import pyipopt, re
from scikits.openopt.Kernel.ooMisc import isSolved

class ipopt(BaseAlg):
    __name__ = 'ipopt'
    __license__ = "CPL"
    __authors__ = 'Carl Laird (Carnegie Mellon University) and Andreas Wachter'
    __alg__ = "A. Wachter and L. T. Biegler, On the Implementation of a Primal-Dual Interior Point Filter Line Search Algorithm for Large-Scale Nonlinear Programming, Mathematical Programming 106(1), pp. 25-57, 2006 "
    __homepage__ = 'http://www.coin-or.org/'
    __info__ = "requires pyipopt made by Eric Xu You"
    __cannotHandleExceptions__ = True
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']


    # CHECK ME!
    __isIterPointAlwaysFeasible__ = lambda self, p: False
    #__isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()

    optFile = 'auto'
    options = ''

    def __init__(self): pass
    def __solver__(self, p):
        nvar = p.n
        x_L = p.lb
        x_U = p.ub

        ncon = p.nc + p.nh + p.b.size + p.beq.size

        g_L, g_U = zeros(ncon), zeros(ncon)
        g_L[:p.nc] = -inf
        g_L[p.nc+p.nh:p.nc+p.nh+p.b.size] = -inf


        # IPOPT non-linear constraints, both eq and ineq
        nnzj = ncon * p.n #TODO: is reduction possible?

        def eval_g(x):
            r = array(())
            if p.userProvided.c: r = p.c(x)
            if p.userProvided.h: r = hstack((r, p.h(x)))
            r = hstack((r, p.__get_AX_Less_B_Residuals__(x), p.__get_AeqX_eq_Beq_Residuals__ (x)))
            return r

        def eval_jac_g(x, flag, userdata = None):
            r = array(()).reshape(0, p.n)
            if p.userProvided.c: r = p.dc(x)
            if p.userProvided.h: r = vstack((r, p.dh(x)))
            r = vstack((r, p.A, p.Aeq))
            if flag:
                return where(ones(r.shape))
            else:
                return r.flatten()

        """ This function might be buggy, """ # // comment by Eric
        nnzh = 0
        def eval_h(lagrange, obj_factor, flag):
            return None



#        def apply_new(x):
#            return True

        nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, p.f, p.df, eval_g, eval_jac_g)

        if self.optFile == 'auto':
            lines = ['# generated automatically by OpenOpt\n','print_level -2\n']
            lines.append('tol ' + str(p.ftol)+ '\n')
            lines.append('constr_viol_tol ' + str(p.contol)+ '\n')
            lines.append('max_iter ' + str(min(15000, p.maxIter))+ '\n')
            if self.options != '' :
                for s in re.split(',|;', self.options):
                    lines.append(s.strip().replace('=', ' ',  1) + '\n')
            if lower(p.castFrom) in ('lp', 'qp', 'llsp'):
                lines.append('jac_c_constant yes\n')
                lines.append('jac_d_constant yes\n')
                lines.append('hessian_constant yes\n')


            ipopt_opt_file = open('ipopt.opt', 'w')
            ipopt_opt_file.writelines(lines)
            ipopt_opt_file.close()

        x, zl, zu, obj = nlp.solve(p.x0)


        if p.istop == 0: p.istop = 1000
        p.xk, p.fk = x, obj
        nlp.close()


