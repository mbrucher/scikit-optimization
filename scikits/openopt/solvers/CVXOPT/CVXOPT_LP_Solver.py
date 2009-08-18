from numpy import asarray,  ones, all, isfinite, copy, nan, concatenate, array
from scikits.openopt.Kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from cvxopt_misc import *
import cvxopt.solvers as cvxopt_solvers
from cvxopt.base import matrix
from scikits.openopt.Kernel.setDefaultIterFuncs import SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON,  IS_MAX_ITER_REACHED, IS_MAX_TIME_REACHED, FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON, UNDEFINED

def CVXOPT_LP_Solver(p, solverName):
    if solverName == 'native_CVXOPT_LP_Solver': solverName = None
    if p.iprint <= 0: 
        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['LPX_K_MSGLEV'] = 0
        cvxopt_solvers.options['MSK_IPAR_LOG'] = 0
    xBounds2Matrix(p)
    WholeRepr2LinConst(p)
    #FIXME: if problem is search for MAXIMUM, not MINIMUM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   
    f = copy(p.f).reshape(-1,1)
    
    # CVXOPT have some problems with x0 so currently I decided to avoid using the one
    #if  p.x0.size>0 and p.x0.flatten()[0] != None and all(isfinite(p.x0)):
    #    sol= cvxopt_solvers.solvers.lp(Matrix(p.f), Matrix(p.A), Matrix(p.b), Matrix(p.Aeq), Matrix(p.beq), solverName)
    #else:
    
    if len(p.intVars)>0 and solverName == 'glpk':
        from cvxopt.glpk import ilp
        c = Matrix(p.f)
        A, b = Matrix(p.Aeq),  Matrix(p.beq)
        G, h = Matrix(p.A),  Matrix(p.b)
        if A is None: 
            A = matrix(0.0,  (0, p.n))
            b = matrix(0.0,(0,1))
        if G is None: 
            G = matrix(0.0,  (0, p.n))
            h = matrix(0.0,(0,1))
        
        (status, x) = ilp(c, G, h, A, b, set(p.intVars), B=set(p.binVars))
        if status == 'optimal': p.istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
        elif status == 'maxiters exceeded': p.istop = IS_MAX_ITER_REACHED
        elif status == 'time limit exceeded': p.istop = IS_MAX_TIME_REACHED
        elif status == 'unknown': p.istop = UNDEFINED
        else: p.istop = FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
        if x is None: 
            p.xf = nan*ones(p.n)
        else:
            p.xf = array(x).flatten()#w/o flatten it yields incorrect result in ff!
        p.ff = sum(p.dotmult(p.f, p.xf))
        p.msg = status
    else:
        sol = cvxopt_solvers.lp(Matrix(p.f), Matrix(p.A), Matrix(p.b), Matrix(p.Aeq), Matrix(p.beq), solverName)
        p.msg = sol['status']
        if p.msg == 'optimal' :  p.istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
        else: p.istop = -100
        if sol['x'] is not None:
            p.xf = asarray(sol['x']).flatten()
            p.ff = sum(p.dotmult(p.f, p.xf))
            p.duals = concatenate((asarray(sol['y']).flatten(), asarray(sol['z']).flatten()))
        else:
            p.ff = nan
            p.xf = nan*ones([p.n,1])
