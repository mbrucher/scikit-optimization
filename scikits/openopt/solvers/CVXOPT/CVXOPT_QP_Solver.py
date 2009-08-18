from numpy import asfarray,  ones, all, isfinite, copy, nan, concatenate, dot
from scikits.openopt.Kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from cvxopt_misc import *
import cvxopt.solvers as cvxopt_solvers

def CVXOPT_QP_Solver(p, solverName):
        if solverName == 'native_CVXOPT_QP_Solver': solverName = None
        if p.iprint <= 0: 
            cvxopt_solvers.options['show_progress'] = False
            cvxopt_solvers.options['MSK_IPAR_LOG'] = 0
        xBounds2Matrix(p)
        #FIXME: if problem is search for MAXIMUM, not MINIMUM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        f = copy(p.f).reshape(-1,1)
        
        
        # CVXOPT have some problems with x0 so currently I decided to avoid using the one
        #if  p.x0.size>0 and p.x0.flatten()[0] != None and all(isfinite(p.x0)):
        #    sol= cvxopt_solvers.solvers.lp(Matrix(p.f), Matrix(p.A), Matrix(p.b), Matrix(p.Aeq), Matrix(p.beq), solverName)
        #else:
        
        sol = cvxopt_solvers.qp(Matrix(p.H), Matrix(p.f), Matrix(p.A), Matrix(p.b), Matrix(p.Aeq), Matrix(p.beq), solverName)
        
##        try:
##            sol = cvxopt_solvers.qp(Matrix(p.H), Matrix(p.f), Matrix(p.A), Matrix(p.b), Matrix(p.Aeq), Matrix(p.beq), solverName)
##        except:
##            msg_err = 'Error: YOU SHOULD INSTALL ' 
##            if solverName is not None: msg_err += solverName + ' and '
##            msg_err += 'CVXOPT'    
##            print  msg_err
##            p.istop, p.msg = -1, 'unknown'
##            return
        
        p.msg = sol['status']
        if p.msg == 'optimal' :  p.istop = 1000
        else: p.istop = -100
        
        
        if sol['x'] is not None:
            p.xf = xf = asfarray(sol['x']).flatten()
            p.ff = asfarray(0.5*dot(xf, dot(p.H, xf)) + p.dotmult(p.f, xf).sum()).flatten()
            p.duals = concatenate((asfarray(sol['y']).flatten(), asfarray(sol['z']).flatten()))
        else:
            p.ff = nan
            p.xf = nan*ones([p.n,1])
