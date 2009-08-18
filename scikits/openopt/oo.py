__docformat__ = "restructuredtext en"




from Kernel.BaseProblem import *
from numpy import asarray
from Kernel.LP import LP as CLP
from Kernel.QP import QP as CQP
from Kernel.MILP import MILP as CMILP
from Kernel.NSP import NSP as CNSP
from Kernel.NLP import NLP as CNLP
from Kernel.NLSP import NLSP as CNLSP
from Kernel.LSP import LSP as CLSP
from Kernel.GLP import GLP as CGLP
from Kernel.LLSP import LLSP as CLLSP
from Kernel.MMP import MMP as CMMP
from Kernel.LLAVP import LLAVP as CLLAVP

def MILP(*args, **kwargs):
    """
    MILP: constructor for Mixed Integer Linear Problem assignment
    f' x -> min
    subjected to
    lb <= x <= ub
    A x <= b
    Aeq x = beq
    for all i from intVars: i-th coordinate of x is required to be integer
    for all j from binVars: j-th coordinate of x is required to be from {0, 1}

    Examples of valid calls:
    p = MILP(f, <params as kwargs>)
    p = MILP(f=objFunVector, <params as kwargs>)
    p = MILP(f, A=A, intVars = myIntVars, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub, binVars = binVars)
    See also: /examples/milp_*.py

    :Parameters:
    - intVars : Python list of those coordinates that are required to be integers.
    - binVars : Python list of those coordinates that are required to be binary.
    all other input parameters are same to LP class constructor ones

    :Returns:
    OpenOpt MILP class instance

    Notes
    -----
    Solving of MILPs is performed via
    r = p.solve(string_name_of_solver)
    r.xf - desired solution (NaNs if a problem occured)
    r.ff - objFun value (<f,x_opt>) (NaN if a problem occured)
    (see also other r fields)
    Solvers available for now:
    lpSolve (LGPL) - requires lpsolve + Python bindings installations (all mentioned is available in http://sourceforge.net/projects/lpsolve)
    glpk (GPL 2) - requires glpk + CVXOPT v >= 1.0 installations (read OO MILP webpage for more details)
    """
    return CMILP(*args, **kwargs)

def LP(*args, **kwargs):
    """
    LP: constructor for Linear Problem assignment
    f' x -> min
    subjected to
    lb <= x <= ub
    A x <= b
    Aeq x = beq

    valid calls are:
    p = LP(f, <params as kwargs>)
    p = LP(f=objFunVector, <params as kwargs>)
    p = LP(f, A=A, Aeq=Aeq, Awhole=Awhole, b=b, beq=beq, bwhole=bwhole, dwhole=dwhole, lb=lb, ub=ub)
    See also: /examples/lp_*.py

    :Parameters:
    f: size n x 1 vector
    A: size m1 x n matrix, subjected to A * x <= b
    Aeq: size m2 x n matrix, subjected to Aeq * x = beq
    b, beq: corresponding vectors with lengthes m1, m2
    lb, ub: size n x 1 vectors, some coords may be +/- inf

    :Returns:
    OpenOpt LP class instance

    Notes
    -----
    Solving of LPs is performed via
    r = p.solve(string_name_of_solver)
    r.xf - desired solution (NaNs if a problem occured)
    r.ff - objFun value (<f,x_opt>) (NaN if a problem occured)
    (see also other r fields)
    Solvers available for now:
    lpSolve (LGPL) - requires lpsolve + Python bindings installations (all mentioned is available in http://sourceforge.net/projects/lpsolve)
    cvxopt_lp (GPL) - requires CVXOPT (http://abel.ee.ucla.edu/cvxopt)
    cvxopt_glpk(GPL2) - requires CVXOPT(http://abel.ee.ucla.edu/cvxopt) & glpk (www.gnu.org/software/glpk)
    converter to NLP. Example: r = p.solve('nlp:ipopt')
    """
    return CLP(*args, **kwargs)

def QP(*args, **kwargs):
    """
    QP: constructor for Quadratic Problem assignment
    1/2 x' H x  + f' x -> min
    subjected to
    A x <= b
    Aeq x = beq
    lb <= x <= ub

    Examples of valid calls:
    p = QP(H, f, <params as kwargs>)
    p = QP(numpy.ones((3,3)), f=numpy.array([1,2,4]), <params as kwargs>)
    p = QP(f=range(8)+15, H = numpy.diag(numpy.ones(8)), <params as kwargs>)
    p = QP(H, f, A=A, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub, <other params as kwargs>)
    See also: /examples/qp_*.py

    INPUT:
    H: size n x n matrix, symmetric, positive-definite
    f: size n x 1 vector
    lb, ub: size n x 1 vectors, some coords may be +/- inf
    A: size m1 x n matrix, subjected to A * x <= b
    Aeq: size m2 x n matrix, subjected to Aeq * x = beq
    Alternatively to A/Aeq you can use Awhole matrix as it's described in LP documentation (or both A, Aeq, Awhole)
    OUTPUT: OpenOpt QP class instance

    Solving of QPs is performed via
    r = p.solve(string_name_of_solver)
    r.xf - desired solution (NaNs if a problem occured)
    r.ff - objFun value (NaN if a problem occured)
    (see also other r fields)
    Solvers available for now:
    cvxopt_qp (GPL) - requires CVXOPT (http://abel.ee.ucla.edu/cvxopt)
    converter to NLP. Example: r = p.solve('nlp:ipopt')
    """
    return CQP(*args, **kwargs)

def NLP(*args, **kwargs):
    """
    NLP: constructor for general Non-Linear Problem assignment

    f(x) -> min (or -> max)
    subjected to
    c(x) <= 0
    h(x) = 0
    A x <= b
    Aeq x = beq
    lb <= x <= ub

    Examples of valid usage:
    p = NLP(f, x0, <params as kwargs>)
    p = NLP(f=objFun, x0 = myX0, <params as kwargs>)
    p = NLP(f, x0, A=A, df = objFunGradient, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub)
    See also: /examples/nlp_*.py

    INPUTS:
    f: objFun
    x0: start point, vector of length n

    Optional:
    name: problem name (string), is used in text & graphics output
    df: user-supplied gradient of objective function
    c, h - functions defining nonlinear equality/inequality constraints
    dc, dh - functions defining 1st derivatives of non-linear constraints

    A: size m1 x n matrix, subjected to A * x <= b
    Aeq: size m2 x n matrix, subjected to Aeq * x = beq
    b, beq: corresponding vectors with lengthes m1, m2
    lb, ub: vectors of length n subjected to lb <= x <= ub constraints, may include +/- inf values

    isNaNInConstraintsAllowed = {False} | True : is nan (not a number) allowed in optim point for non-linear constraints.

    iprint = {10}: print text output every <iprint> iteration
    goal = {'minimum'} | 'min' | 'maximum' | 'max' - minimize or maximize objective function
    diffInt = {1e-7} : finite-difference gradient aproximation step, scalar or vector of length nVars
    scale = {None} : scale factor, see /examples/badlyScaled.py for more details
    check.df, check.dc, check.dh: if set to True, OpenOpt will check user-supplied gradients.
    args (or args.f, args.c, args.h) - additional arguments to objFunc and non-linear constraints,
        see /examples/userArgs.py for more details.

    contol: max allowed residual in optim point
    (for any constraint from problem constraints:
    constraint(x_optim) < contol is required from solver)

    stop criteria:
    maxIter {400}
    maxFunEvals {1e5}
    maxCPUTime {inf}
    maxTime {inf}
    maxLineSearch {500}
    fEnough {-inf for min problems, +inf for max problems}:
        stop if objFunc vulue better than fEnough and all constraints less than contol
    ftol {1e-6}: used in stop criterium || f[iter_k] - f[iter_k+1] || < ftol
    xtol {1e-6}: used in stop criterium || x[iter_k] - x[iter_k+1] || < xtol
    gtol {1e-6}: used in stop criteria || gradient(x[iter_k]) || < gtol

    callback - user-defined callback function(s), see /examples/userCallback.py

    Notes:
    1) for more safety default values checking/reassigning (via print p.maxIter / prob.maxIter = 400) is recommended
    (they may change in future OpenOpt versions and/or not updated in time in the documentation)
    2) some solvers may ignore some of the stop criteria above and/or use their own ones
    3) for NSP constructor ftol, xtol, gtol defaults may have other values

    graphic options:
    plot = {False} | True : plot figure (now implemented for UC problems only), requires matplotlib installed
    color = {'blue'} | black | ... (any valid matplotlib color)
    specifier = {'-'} | '--' | ':' | '-.' - plot specifier
    show = {True} | False : call pylab.show() after solver finish or not
    xlim {(nan, nan)}, ylim {(nan, nan)} - initial estimation for graphical output borders
    (you can use for example p.xlim = (nan, 10) or p.ylim = [-8, 15] or p.xlim=[inf, 15], only real finite values will be taken into account)
    for constrained problems ylim affects only 1st subplot
    p.graphics.xlabel or p.xlabel = {'time'} | 'cputime' | 'iter' # desired graphic output units in x-axe, case-unsensetive


    Note: some Python IDEs have problems with matplotlib!

    Also, after assignment NLP instance you may modify prob fields inplace:
    p.maxIter = 1000
    p.df = lambda x: cos(x)

    OUTPUT: OpenOpt NLP class instance

    Solving of NLPs is performed via
    r = p.solve(string_name_of_solver)
    r.xf - desired solution (NaNs if a problem occured)
    r.ff - objFun value (NaN if a problem occured)
    (see also other fields, such as CPUTimeElapsed, TimeElapsed, isFeasible, iter etc, via dir(r))

    Solvers available for now:
    single-variable:
        goldenSection, scipy_fminbound (latter is not recommended)
        (both these solvers require finite lb-ub and ignore user-supplied gradient)
    unconstrained:
        scipy_bfgs, scipy_cg, scipy_ncg, scipy_powell (latter cannot handle user-provided gradient)
    box-bounded:
        scipy_lbfgsb, scipy_tnc
    all constraints:
        ralg
        ipopt (requires ipopt + pyipopt installed)
        scipy_slsqp (requires scipy from svn 25-Dec-2007 or later)
        scipy_cobyla (this one cannot handle user-supplied gradients)
        lincher (requires CVXOPT QP solver),
        algencan (ver. 2.0.3 or more recent, very powerful constrained solver, GPL,
        requires ALGENCAN + Python interface installed,
        see http://www.ime.usp.br/~egbirgin/tango/)

    """
    return CNLP(*args, **kwargs)

def NSP(*args, **kwargs):
    """
    Non-Smooth Problem constructor
    Same usage as NLP (see help(NLP) and /examples/nsp_*.py), but default values of contol, xtol, ftol, diffInt may differ
    Also, some solvers (like UkrOpt ralg) will take NS into account and behave differently.
    Solvers available for now:
        ralg - all constraints, medium-scale (nVars = 1...1000), can handle user-provided gradient/subgradient
        ShorEllipsoid (unconstrained for now) - small-scale, nVars=1...10, requires r0: ||x0-x*||<=r0
    """
    return CNSP(*args, **kwargs)

def NLSP(*args, **kwargs):
    """
    Solving systems of n non-linear equations with n variables
    Parameters and usage: same as NLP
    (see help(NLP) and /examples/nlsp_*.py)
    Solvers available for now:
        scipy_fsolve (can handle df);
        converter to NLP. Example: r = p.solve('nlp:ipopt');
        nssolve (primarily for non-smooth and noisy funcs; can handle all types of constraints and 1st derivatives df,dc,dh; splitting equations to Python list or tuple is recommended to speedup calculations)
    (these ones below are very unstable and can't use user-supplied gradient - at least, for scipy 0.6.0)
        scipy_anderson
        scipy_anderson2
        scipy_broyden1
        scipy_broyden2
        scipy_broyden3
        scipy_broyden_generalized
    """
    return CNLSP(*args, **kwargs)

def LSP(*args, **kwargs):
    """
    Given set of non-linear equations
        f1(x)=0, f2(x)=0, ... fm(x)=0
    search for x: f1(x, <optional params>)^2 + ,,, + fm(x, <optional params>)^2 -> min

    Parameters and usage: same as NLP
    (see help(scikits.openopt.NLP) and /examples/lsp_*.py)
    Solvers available for now:
        scipy_leastsq (requires scipy installed)
        converter to NLP. Example: r = p.solve('nlp:ipopt')
    """
    return CLSP(*args, **kwargs)


def GLP(*args, **kwargs):
    """
    GLP: constructor for general GLobal Problem
    p = GLP(f, <params as kwargs>)
    search for global minimum of general non-linear (maybe discontinious) function
    Parameters and usage: same as NLP  (see help(NLP) and /examples/glp_*.py)
    See also: /examples/glp_*.py

    Solvers available:
        galileo - a GA-based solver by Donald Goodman, requires finite lb <= x <= ub
        pswarm (requires PSwarm installed), license: BSD
    """
    return CGLP(*args, **kwargs)


def LLSP(*args, **kwargs):
    """
    LLSP: constructor for Linear Least Squares Problem assignment
    0.5*||C*x-d||^2 + 0.5*damp*||x-X||^2 + <f,x> -> min

    subjected to:
    lb <= x <= ub

    Examples of valid calls:
    p = LLSP(C, d, <params as kwargs>)
    p = LLSP(C=my_C, d=my_d, <params as kwargs>)

    p = LLSP(C, d, lb=lb, ub=ub)

    See also: /examples/llsp_*.py

    :Parameters:
    C - float m x n numpy.ndarray, numpy.matrix or Python list of lists
    d - float array of length m (numpy.ndarray, numpy.matrix, Python list or tuple)
    damp - non-negative float number
    X - float array of length n (by default all-zeros)
    f - float array of length n (by default all-zeros)
    lb, ub - float arrays of length n (numpy.ndarray, numpy.matrix, Python list or tuple)

    :Returns:
    OpenOpt LLSP class instance

    Notes
    -----
    Solving of LLSPs is performed via
    r = p.solve(string_name_of_solver)
    r.xf - desired solution (NaNs if a problem occured)
    r.ff - objFun value (NaN if a problem occured)
    (see also other r fields)
    Solvers available for now:
    lapack_dgelss - slow but stable, requires scipy; unconstrained
    lapack_sgelss - single precesion, requires scipy; unconstrained
    bvls - requires installation from OO LLSP webpage, can handle lb, ub
    converter to nlp. Example: r = p.solve('nlp:ralg', plot=1, iprint =15, <...>)
    """
    return CLLSP(*args, **kwargs)

def MMP(*args, **kwargs):
    """
    MMP: constructor for Mini-Max Problem
    search for minimum of max(func0(x), func1(x), ... funcN(x))
    See also: /examples/mmp_*.py

    Parameters and usage: same as NLP  (see help(NLP) and /examples/mmp_*.py)
    Solvers available:
        nsmm (currently unconstrained, NonSmooth-based MiniMax, uses NSP ralg solver)
    """
    return CMMP(*args, **kwargs)

def LLAVP(*args, **kwargs):
    """
   LLAVP : constructor for Linear Least Absolute Value Problem assignment
    ||C * x - d||_1  + damp*||x-X||_1-> min

    subjected to:
    lb <= x <= ub

    Examples of valid calls:
    p = LLAVP(C, d, <params as kwargs>)
    p = LLAVP(C=my_C, d=my_d, <params as kwargs>)

    p = LLAVP(C, d, lb=lb, ub=ub)

    See also: /examples/llavp_*.py

    :Parameters:
    C - float m x n numpy.ndarray, numpy.matrix or Python list of lists
    d - float array of length m (numpy.ndarray, numpy.matrix, Python list or tuple)
    damp - non-negative float number
    X - float array of length n (by default all-zeros)
    lb, ub - float arrays of length n (numpy.ndarray, numpy.matrix, Python list or tuple)

    :Returns:
    OpenOpt LLAVP class instance

    Notes
    -----
    Solving of LLAVPs is performed via
    r = p.solve(string_name_of_solver)
    r.xf - desired solution (NaNs if a problem occured)
    r.ff - objFun value (NaN if a problem occured)
    (see also other r fields)
    Solvers available for now:
    nsp:<NSP_solver_name> - converter llavp2nsp. Example: r = p.solve('nsp:ralg', plot=1, iprint =15, <...>)
    """
    return CLLAVP(*args, **kwargs)
