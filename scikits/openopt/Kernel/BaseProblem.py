__docformat__ = "restructuredtext en"
from numpy import *
from oologfcn import *
from ooGraphics import Graphics
from setDefaultIterFuncs import setDefaultIterFuncs, IS_MAX_FUN_EVALS_REACHED, denyingStopFuncs
from objFunRelated import objFunRelated
from Residuals import Residuals
from ooIter import ooIter
from Point import Point
from ooCheckGradient import ooCheckGradient
from ooIterPrint import ooTextOutput
from ooMisc import setNonLinFuncsNumber
from ooVar import oovar
from Function import oofun
from copy import copy as Copy

ProbDefaults = {'diffInt': 1e-7,  'xtol': 1e-6,  'noise': 0}
from runProbSolver import runProbSolver
from GUI import manage



class user:
    def __init__(self):
        pass

class oomatrix:
    def __init__(self):
        pass
    def matmult(self, x, y):
        return dot(x, y)
        #return asarray(x) ** asarray(y)
    def dotmult(self, x, y):
        return x * y
        #return asarray(x) * asarray(y)

class autocreate:
    def __init__(self): pass

class BaseProblem(oomatrix, Residuals, ooTextOutput):
    isObjFunValueASingleNumber = True

    name = 'unnamed'
    state = 'init'# other: paused, running etc
    castFrom = '' # used by converters qp2nlp etc
    nonStopMsg = ''
    xlabel = 'time'
    plot = False # draw picture or not
    show = True # use command pylab.show() after solver finish or not

    iter = 0
    cpuTimeElapsed = 0.
    TimeElapsed = 0.
    isFinished = False
    invertObjFunc = False # True for goal = 'max' or 'maximum'
    nEvals = {}

    lastPrintedIter = -1
    data4TextOutput = ['objFunVal', 'log10(maxResidual)']
    debug = 0
    # graphics.lowerBoundForPlotEstim = 0#for future implement

    iprint = 10
    #if iprint<0 -- no output
    #if iprint==0 -- final output only

    maxIter = 400
    maxFunEvals = 10000 # TODO: move it to NinLinProblem class?
    maxCPUTime = inf
    maxTime = inf
    maxLineSearch = 500 # TODO: move it to NinLinProblem class?
    xtol = ProbDefaults['xtol'] # TODO: move it to NinLinProblem class?
    gtol = 1e-6 # TODO: move it to NinLinProblem class?
    ftol = 1e-6
    contol = 1e-6

    minIter = 0
    minFunEvals = 0
    minCPUTime = 0.0
    minTime = 0.0

    userStop = False # becomes True is stopped by user

    x0 = nan

    noise = ProbDefaults['noise'] # TODO: move it to NinLinProblem class?

    showFeas = False

    # A * x <= b inequalities
    A = None
    b = None

    # Aeq * x = b equalities
    Aeq = None
    beq = None

    scale = None

    goal = None# should be redefined by child class
    # possible values: 'maximum', 'min', 'max', 'minimum', 'minimax' etc
    showGoal = False# can be redefined by child class, used for text & graphic output

    color = 'b' # blue, color for plotting
    specifier = '-'# simple line for plotting
    plotOnlyCurrentMinimum = False # some classes like GLP change the default to True
    xlim = (nan,  nan)
    ylim = (nan,  nan)
    legend = ''

    fixedVars = None # numbers of fixed variables, for future implementation

    istop = 0

    fEnough = -inf # if value less than fEnough will be obtained
    # and all constraints no greater than contol
    # then solver will be stopped.
    # this param is handled in iterfcn of OpenOpt Kernel
    # so it may be ignored with some solvers not closely connected to OO Kernel

    callback = [] # user-defined callback function(s)

    def __init__(self):
        self.norm = linalg.norm
        self.denyingStopFuncs = denyingStopFuncs()
        self.iterfcn = lambda *args, **kwargs: ooIter(self, *args, **kwargs)# this parameter is only for OpenOpt developers, not common users
        self.graphics = Graphics()
        self.user = user()
        self.F = lambda x: self.objFuncMultiple2Single(self.objFunc(x)) # TODO: should be changes for LP, MILP, QP classes!

        self.point = lambda *args,  **kwargs: Point(self, *args,  **kwargs)

        self.timeElapsedForPlotting = [0.]
        self.cpuTimeElapsedForPlotting = [0.]
        #user can redirect these ones, as well as debugmsg
        self.debugmsg = lambda msg: oodebugmsg(self,  msg)
        self.err = ooerr
        self.warn = oowarn
        self.oassert = ooassert # user can already have other assert func
        #persistent warning, is called no more than 1 times per session
        self.pWarn = ooPWarn
        self.info = ooinfo
        self.hint = oohint

        self.solverParams = autocreate()

        self.userProvided = autocreate()

        self.special = autocreate()

        self.intVars = [] # for problems like MILP
        self.binVars = [] # for problems like MILP
        self.optionalData = []#string names of optional data like 'c', 'h', 'Aeq' etc

    def __finalize__(self):
        pass

    def objFunc(self, x):
        return self.f(x) # is overdetermined in LP, QP, LLSP etc classes

    def __isFiniteBoxBounded__(self): # TODO: make this function 'lazy'
        return all(isfinite(self.ub)) and all(isfinite(self.lb))

    def __isNoMoreThanBoxBounded__(self): # TODO: make this function 'lazy'
        s = ((), [], array([]), None)
        return self.b.size ==0 and self.beq.size==0 and not self.userProvided.c and not self.userProvided.h

#    def __1stBetterThan2nd__(self,  f1, f2,  r1=None,  r2=None):
#        if self.isUC:
#            #TODO: check for goal = max/maximum
#            return f1 < f2
#        else:#then r1, r2 should be defined
#            return (r1 < r2 and  self.contol < r2) or (((r1 <= self.contol and r2 <=  self.contol) or r1==r2) and f1 < f2)
#
#    def __1stCertainlyBetterThan2ndTakingIntoAcoountNoise__(self,   f1, f2,  r1=None,  r2=None):
#        if self.isUC:
#            #TODO: check for goalType = max
#            return f1 + self.noise < f2 - self.noise
#        else:
#            #return (r1 + self.noise < r2 - self.noise and  self.contol < r2) or \
#            return (r1 < r2  and  self.contol < r2) or \
#            (((r1 <= self.contol and r2 <=  self.contol) or r1==r2) and f1 + self.noise < f2 - self.noise)


    def solve(self, *args, **kwargs):
        return runProbSolver(self, *args, **kwargs)

    def objFuncMultiple2Single(self, f):
        #this function can be overdetermined by child class
        if asfarray(f).size != 1: self.err('unexpected f size. The function should be redefined in OO child class, inform OO developers')
        return f

        self.err('OpenOpt error: this function should be overdetermined by child class')

    def inspire(self, newProb, sameConstraints=True):
        # fills some fields of new prob with old prob values
        newProb.castFrom = self.probType

        #TODO: hold it in single place

        fieldsToAssert = ['userProvided', 'contol', 'xtol', 'ftol', 'gtol', 'iprint', 'maxIter', 'maxTime', 'maxCPUTime','fEnough', 'goal', 'color', 'debug', 'maxFunEvals', 'xlabel']
        if sameConstraints: fieldsToAssert+= ['lb', 'ub', 'A', 'Aeq', 'b', 'beq']

        for key in ['userProvided', 'lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'contol', 'xtol', 'ftol', 'gtol', 'iprint', 'plot', 'maxIter', 'maxTime', 'maxCPUTime','fEnough', 'goal', 'color', 'debug', 'maxFunEvals', 'xlabel'] :
            if hasattr(self, key): setattr(newProb, key, getattr(self, key))

        # note: because of 'userProvided' from prev line
        #self self.userProvided is same to newProb.userProvided
        if sameConstraints:
            for key in ['c','dc','h','dh','d2c','d2h']:
                if hasattr(self.userProvided, key):
                    if getattr(self.userProvided, key):
                        setattr(newProb, key, getattr(self, key))
                    else:
                        setattr(newProb, key, None)



class MatrixProblem(BaseProblem):
    __baseClassName__ = 'Matrix'
    #obsolete, should be removed
    # still it is used by lpSolve
    # Awhole * x {<= | = | >= } b
    Awhole = None # matrix m x n, n = len(x)
    bwhole = None # vector, size = m x 1
    dwhole = None #vector of descriptors, size = m x 1
    # descriptors dwhole[j] should be :
    # 1 : <Awhole, x> [j] greater (or equal) than bwhole[j]
    # -1 : <Awhole, x> [j] less (or equal) than bwhole[j]
    # 0 : <Awhole, x> [j] = bwhole[j]
    def __init__(self):
        BaseProblem.__init__(self)
        self.kernelIterFuncs = setDefaultIterFuncs('Matrix')

    def __prepare__(self):
        pass

    # TODO: move the function to child classes
    def __isUnconstrained__(self):
        s = ((), [], array([]), None)
        return self.b.size ==0 and self.beq.size==0 and (self.lb in s or all(isinf(self.lb))) and (self.ub in s or all(isinf(self.ub)))


class Parallel:
    def __init__(self):
        self.f = False# 0 - don't use parallel calclations, 1 - use
        self.c = False
        self.h = False
        #TODO: add paralell func!
        #self.parallel.fun = dfeval

class args:
    def __init__(self): pass
    f, c, h = (), (), ()

class NonLinProblem(BaseProblem, objFunRelated, args):
    __baseClassName__ = 'NonLin'
    isNaNInConstraintsAllowed = False
    consMode = 'all' # TODO: remove it?
    diffInt = ProbDefaults['diffInt']        #finite-difference gradient aproximation step
    #non-linear constraints
    c = None # c(x)<=0
    h = None # h(x)=0
    #lines with |info_user-info_numerical| / (|info_user|+|info_numerical+1e-15) greater than maxViolation will be shown
    maxViolation = 1e-2
    def __init__(self):
        BaseProblem.__init__(self)
        #self.check = Check()
        self.args = args()
        self.prevVal = {}
        for fn in ['f', 'c', 'h', 'df', 'dc', 'dh', 'd2f', 'd2c', 'd2h']:
            self.prevVal[fn] = {'key':None, 'val':None}

        self.functype = {}

        #self.isVectoriezed = False

#        self.fPattern = None
#        self.cPattern = None
#        self.hPattern = None
        self.kernelIterFuncs = setDefaultIterFuncs('NonLin')

    def checkdf(self, *args,  **kwargs):
        return ooCheckGradient(self, 'df', *args,  **kwargs)

    def checkdc(self, *args,  **kwargs):
        return ooCheckGradient(self, 'dc', *args,  **kwargs)

    def checkdh(self, *args,  **kwargs):
        return ooCheckGradient(self, 'dh', *args,  **kwargs)

    def __makeCorrectArgs__(self):
        argslist = dir(self.args)
        if not ('f' in argslist and 'c' in argslist and 'h' in argslist):
            tmp, self.args = self.args, autocreate()
            self.args.f = self.args.c = self.args.h = tmp
        for j in ('f', 'c', 'h'):
            v = getattr(self.args, j)
            if type(v) != type(()): setattr(self.args, j, (v,))

    def __finalize__(self):
        if (self.userProvided.c and any(isnan(self.c(self.xf)))) or (self.userProvided.h and any(isnan(self.h(self.xf)))):
            self.warn('some non-linear constraints are equal to NaN')
        if hasattr(self, 'oovars') and len(self.oovars)>0:
            xf, k = {}, 0
            for var in self.oovars:
                xf[var.name] = self.xf[k:k+var.size]
                k += var.size
            self.xf = xf

    def __construct_x_from_ooVars__(self):
        self.oovars = set([])
        for FuncType in ['f', 'c', 'h']:
            Funcs = getattr(self, FuncType)
            if Funcs is None: continue
            if isinstance(Funcs, oofun):
                Funcs.__connect_ooVars__(self)
            else:
                if type(Funcs) not in [tuple, list]:
                    self.err('when x0 is absent, oofuns (with oovars) are expected')
                for fun in Funcs:
                    if type(Funcs) not in [tuple, list]:
                        self.err('when x0 is absent, oofuns (with oovars) are expected')
                    fun.__connect_ooVars__(self)
        assert len(self.oovars) > 0
        n = 0
        for fn in ['x0', 'lb', 'ub']:
            if not hasattr(self, fn): continue
            val = getattr(self, fn)
            if val is not None and any(isfinite(val)):
                self.err('while using oovars providing x0, lb, ub for whole prob is forbidden, use for each oovar instead')

        x0, lb, ub = [], [], []

        for var in self.oovars:
            var.dep = range(n, n+var.size)
            n += var.size
            x0 += list(atleast_1d(asarray(var.v0)))
            lb += list(atleast_1d(asarray(var.lb)))
            ub += list(atleast_1d(asarray(var.ub)))
        self.n = n
        self.x0 = x0
        self.lb = lb
        self.ub = ub

    def __prepare__(self):

        # TODO: simplify it
        if hasattr(self, 'solver'):
            if not self.solver.__iterfcnConnected__:
                #self.connectIterFcn('df')
                if self.solver.__funcForIterFcnConnection__ == 'f':
                    if not hasattr(self, 'f_iter'):
                        self.f_iter = max((self.n, 4))
                else:
                    if not hasattr(self, 'df_iter'):
                        self.df_iter = True


        if hasattr(self, 'prepared') and self.prepared == True:
            return

        # TODO: remove GLP, make other workaround
        if (not hasattr(self, 'x0') or self.x0 is nan) and self.probType != 'GLP':
            # hence oovar(s) are used
            self.__construct_x_from_ooVars__()

        self.x0 = ravel(self.x0)
        self.__makeCorrectArgs__()
        for s in ('f', 'df', 'd2f', 'c', 'dc', 'd2c', 'h', 'dh', 'd2h'):
            derivativeOrder = len(s)-1
            self.nEvals[Copy(s)] = 0
            if hasattr(self, s) and getattr(self, s) is not None:
                setattr(self.userProvided, s, True)

                A = getattr(self,s)

                if not type(A) in [list, tuple]: #TODO: add or ndarray(A)
                    A = (A,)#make tuple
                setattr(self.user, s, A)
            else:
                setattr(self.userProvided, s, False)
            if derivativeOrder == 0:
                setattr(self, s, lambda x, IND=None, userFunctionType= s, ignorePrev=False, getDerivative=False: \
                        self.wrapped_func(x, IND, userFunctionType, ignorePrev, getDerivative))
            elif derivativeOrder == 1:
                setattr(self, s, lambda x, ind=None, funcType=s[-1], ignorePrev = False:
                        self.wrapped_1st_derivatives(x, ind, funcType, ignorePrev))
            elif derivativeOrder == 2:
                setattr(self, s, getattr(self, 'user_'+s))
            else:
                self.err('incorrect non-linear function case')

        self.diffInt = ravel(self.diffInt)

        #initialization, getting nf, nc, nh etc:
        for s in ['c', 'h', 'f']:
            if not getattr(self.userProvided, s):
                setattr(self, 'n'+s, 0)
            else:
                setNonLinFuncsNumber(self,  s)

        self.prepared = True
        #self.currentBestFeasiblePoint = None

    # TODO: move the function to child classes
    def __isUnconstrained__(self):
        s = ((), [], array([]), None)
        return self.b.size ==0 and self.beq.size==0 and not self.userProvided.c and not self.userProvided.h \
            and (self.lb in s or all(isinf(self.lb))) and (self.ub in s or all(isinf(self.ub)))

    manage = manage # GUI func



