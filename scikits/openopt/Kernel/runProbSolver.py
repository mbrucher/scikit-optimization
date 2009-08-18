__docformat__ = "restructuredtext en"
from time import time, clock
from numpy import asfarray, copy, inf, nan, isfinite, ones, ndim, all, atleast_1d, any, isnan, array_equiv, asscalar, asarray
from setDefaultIterFuncs import stopcase,  SMALL_DELTA_X,  SMALL_DELTA_F
from ooCheck import ooCheck
import copy
import os, string
from ooMisc import isSolved, killThread
from string import lower
from BaseProblem import ProbDefaults
from BaseAlg import BaseAlg
from nonOptMisc import getSolverFromStringName
#from scikits.openopt.Kernel.ooMisc import __solverPaths__
ConTolMultiplier = 0.8

#if __solverPaths__ is None:
#    __solverPaths__ = {}
#    file = string.join(__file__.split(os.sep)[:-1], os.sep)
#    for root, dirs, files in os.walk(os.path.dirname(file)+os.sep+'solvers'):
#        rd = root.split(os.sep)
#        if '.svn' in rd: continue
#        rd = rd[rd.index('solvers')+1:]
#        for file in files:
#            print file
#            if len(file)>6 and file[-6:] == '_oo.py':
#                __solverPaths__[file[:-6]] = 'scikits.openopt.solvers.' + string.join(rd,'.') + '.'+file[:-3]

#import pickle
#f = open('solverPaths.py', 'w')
#solverPaths = pickle.load(f)
from solverPaths import solverPaths


def runProbSolver(p_, solver_str_or_instance=None, *args, **kwargs):
    #p = copy.deepcopy(p_, memo=None, _nil=[])
    p = p_
    if args is not (): p.err('unexpected args for p.solve()')
    if hasattr(p, 'was_involved'): p.err("please use re-assigned prob struct, Python deepcopy can't handle it properly for now, it's intended to be fixed in Python 2.6")
    else: p.was_involved = True

    if solver_str_or_instance is None:
        if hasattr(p, 'solver'): solver_str_or_instance = p.solver
        elif kwargs.has_key('solver'): solver_str_or_instance = kwargs['solver']

    if type(solver_str_or_instance) is str and ':' in solver_str_or_instance:
        isConverter = True
        probTypeToConvert,  solverName = solver_str_or_instance.split(':', 1)
        converterName = lower(p.probType)+'2'+probTypeToConvert
        converter = getattr(p, converterName)
        p.solver = getSolverFromStringName(p, solverName)
        solver_params = {}
        #return converter(solverName, *args, **kwargs)
    else:
        isConverter = False
        if type(solver_str_or_instance) is str:
            p.solver = getSolverFromStringName(p, solver_str_or_instance)
        else:
            p.solver = solver_str_or_instance
            for key, value  in solver_str_or_instance.fieldsForProbInstance.iteritems():
                setattr(p, key, value)


    if kwargs.has_key('debug'):
       p.debug =  kwargs['debug']


    #p.solver = solverClass()
#    p.solverName = p.solver.__name__
#    setattr(p, p.solverName, EmptyClass())
    solver = p.solver.__solver__

    for key, value in kwargs.iteritems():
        if hasattr(p.solver, key):
            if isConverter:
                solver_params[key] = value
            else:
                setattr(p.solver, key, value)
        elif hasattr(p, key):
            setattr(p, key, value)
        else: p.warn('incorrect parameter for prob.solve(): "' + str(key) + '" - will be ignored (this one has been not found in neither prob nor ' + p.solver.__name__ + ' solver parameters)')

    p.iterValues = EmptyClass()

    p.iterCPUTime = []
    p.iterTime = []
    p.iterValues.x = [] # iter points
    p.iterValues.f = [] # iter ObjFunc Values
    p.iterValues.r = [] # iter MaxResidual
    p.iterValues.rt = [] # iter MaxResidual Type: 'c', 'h', 'lb' etc
    p.iterValues.ri = [] # iter MaxResidual Index



    if p.goal in ['max','maximum']: p.invertObjFunc = True

    #TODO: remove it!
    p.advanced = EmptyClass()

    p.istop = 0
    p.iter = 0
    p.graphics.nPointsPlotted = 0
    #for fn in p.nEvals.keys(): p.nEvals[fn] = 0 # NB! f num is used in LP/QP/MILP/etc stop criteria check

    p.msg = ''
    if not type(p.callback) in (list,  tuple): p.callback = [p.callback]
    if hasattr(p, 'xlabel'): p.graphics.xlabel = p.xlabel
    if p.graphics.xlabel == 'nf': p.iterValues.nf = [] # iter ObjFunc evaluation number

    p.__prepare__()
    for fn in ['FunEvals', 'Iter', 'Time', 'CPUTime']:
        if hasattr(p,'min'+fn) and hasattr(p,'max'+fn) and getattr(p,'max'+fn) < getattr(p,'min'+fn):
            p.warn('min' + fn + ' (' + str(getattr(p,'min'+fn)) +') exceeds ' + 'max' + fn + '(' + str(getattr(p,'max'+fn)) +'), setting latter to former')
            setattr(p,'max'+fn, getattr(p,'min'+fn))

    if p.probType in ('LP', 'MILP', 'QP') and p.plot:
        p.warn("plotting for LP/MILP/QP isn't implemented/tested yet")

    for fn in ['maxFunEvals', 'maxIter']: setattr(p, fn, int(getattr(p, fn)))# to prevent warnings from numbers like 1e7

    if hasattr(p, 'x0'): p.x0 = atleast_1d(asfarray(p.x0).copy())
    for fn in ['lb', 'ub', 'b', 'beq', 'scale', 'diffInt']:
        if hasattr(p, fn):
            fv = getattr(p, fn)
            if fv != None:# and fv != []:
                setattr(p, fn, asfarray(fv, dtype='float').flatten())
            elif fn != 'scale':
                setattr(p, fn, asfarray([]))


    if p.scale is not None and hasattr(p, 'diffInt'):
        if p.diffInt.size>1 or p.diffInt[0] != ProbDefaults['diffInt']:
            p.info('using both non-default scale & diffInt is not recommended. diffInt = diffInt/scale will be used')
        p.diffInt = p.diffInt / p.scale

    if p.lb.size == 0:
        p.lb = -inf * ones(p.n)
    if p.ub.size == 0:
        p.ub = inf * ones(p.n)
    for fn in ('A', 'Aeq'):
        fv = getattr(p, fn)
        if fv != None:# and fv != []:
            afv = asfarray(fv)
            if ndim(afv) > 1:
                if afv.shape[1] != p.n:
                    p.err('incorrect ' + fn + ' size')
            else:
                if afv.shape != () and afv.shape[0] == p.n: afv = afv.reshape(1,-1)
            setattr(p, fn, afv)
        else:
            setattr(p, fn, asfarray([]).reshape(0, p.n))


    p.stopdict = {}

    for s in ['b','beq']:
        if hasattr(p, s): setattr(p, 'n'+s, len(getattr(p, s)))

    #if p.probType not in ['LP', 'QP', 'MILP', 'LLSP']: p.objFunc(p.x0)

    p.isUC = p.__isUnconstrained__()
    if p.solver.__isIterPointAlwaysFeasible__ is True or \
    (not p.solver.__isIterPointAlwaysFeasible__ is False and p.solver.__isIterPointAlwaysFeasible__(p)):
        assert p.data4TextOutput[-1] == 'log10(maxResidual)'
        p.data4TextOutput = p.data4TextOutput[:-1]

    if p.showFeas and p.data4TextOutput[-1] != 'isFeasible': p.data4TextOutput.append('isFeasible')

    if not p.solver.__iterfcnConnected__:
        p.kernelIterFuncs.pop(SMALL_DELTA_X)
        p.kernelIterFuncs.pop(SMALL_DELTA_F)

#    p.xf = nan * ones([p.n, 1])
#    p.ff = nan
    #todo : add scaling, etc
    p.primalConTol = p.contol
    p.contol *= ConTolMultiplier

    p.timeStart = time()
    p.cpuTimeStart = clock()


    ############################
    # Start solving problem:

    if p.iprint >= 0:
        print '-----------------------------------------------------'
        s = 'solver: ' +  p.solver.__name__ +  '   problem: ' + p.name
        if p.showGoal: s += '   goal: ' + p.goal
        print s



    try:
        if isConverter:
            # TODO: will R be somewhere used?
            R = converter(solverName, **solver_params)
        else:
            nErr = ooCheck(p)
            if nErr: p.err("prob check results: " +str(nErr) + "ERRORS!")#however, I guess this line will be never reached.

            p.iterfcn(p.x0)
            solver(p)
#    except killThread:
#        if p.plot:
#            print 'exiting pylab'
#            import pylab
#            if hasattr(p, 'figure'):
#                print 'closing figure'
#                #p.figure.canvas.draw_drawable = lambda: None
#                pylab.ioff()
#                pylab.close()
#                #pylab.draw()
#            #pylab.close()
#            print 'pylab exited'
#        return None
    except isSolved:
#        p.fk = p.f(p.xk)
#        p.xf = p.xk
#        p.ff = p.objFuncMultiple2Single(p.fk)

        if p.istop == 0: p.istop = 1000
    ############################
    p.contol = p.primalConTol

    # Solving finished
    p.isFinished = True
    if hasattr(p, 'xf') and (not hasattr(p, 'xk') or array_equiv(p.xk, p.x0)): p.xk = p.xf
    if not hasattr(p,  'xf') or all(p.xf==nan): p.xf = p.xk

    p.fk = p.objFunc(p.xk)
    if not hasattr(p,  'ff') or any(p.ff==nan): p.ff = p.objFunc(p.xf)

    if not hasattr(p, 'fk'): p.fk = p.ff
    if p.invertObjFunc:  p.fk, p.ff = -p.fk, -p.ff

    if asfarray(p.ff).size > 1: p.ff = p.objFuncMultiple2Single(p.fk)

    #p.ff = p.objFuncMultiple2Single(p.ff)
    #if not hasattr(p, 'xf'): p.xf = p.xk

    p.xf = p.xf.flatten()
    p.rf = p.getMaxResidual(p.xf)
    if p.isFeas(p.xf):
        p.isFeasible = True
    else: p.isFeasible = False
    if not p.isFeasible and p.istop > 0: p.istop = -100-p.istop/1000.0
    p.stopcase = stopcase(p)

    if p.invertObjFunc: p.iterfcn(p.xf, -p.ff)
    else: p.iterfcn(p.xf, p.ff)

    p.__finalize__()

    r = OpenOptResult()
    r.elapsed = dict()
    r.elapsed['solver_time'] = round(100.0*(time() - p.timeStart))/100.0
    r.elapsed['solver_cputime'] = clock() - p.cpuTimeStart

    for fn in ('ff', 'istop', 'duals', 'isFeasible', 'msg', 'stopcase', 'iterValues',  'special'):
        if hasattr(p, fn):  setattr(r, fn, getattr(p, fn))

    r.xf = copy.deepcopy(p.xf)
    r.rf = asscalar(asarray(p.rf))
    r.ff = asscalar(asarray(r.ff))

    r.solverInfo = dict()
    for fn in ('homepage',  'alg',  'authors',  'license',  'info', 'name'):
        r.solverInfo[fn] =  getattr(p.solver,  '__' + fn + '__')


    #TODO: add scaling handling!!!!!!!
#    for fn in ('df', 'dc', 'dh', 'd2f', 'd2c', 'd2h'):
#        if hasattr(p, '_' + fn): setattr(r, fn, getattr(p, '_'+fn))

    if p.plot:
        #for df in p.graphics.drawFuncs: df(p)    #TODO: include time spent here to (/cpu)timeElapsedForPlotting
        r.elapsed['plot_time'] = round(100*p.timeElapsedForPlotting[-1])/100 # seconds
        r.elapsed['plot_cputime'] = p.cpuTimeElapsedForPlotting[-1]
    else:
        r.elapsed['plot_time'] = 0
        r.elapsed['plot_cputime'] = 0

    r.elapsed['solver_time'] -= r.elapsed['plot_time']
    r.elapsed['solver_cputime'] -= r.elapsed['plot_cputime']

    r.evals = p.nEvals
    for fn in r.evals.keys():
        if type(r.evals[fn]) != int:
            r.evals[fn] = round(r.evals[fn] *10) /10.0
    r.evals['iter'] = p.iter

    p.invertObjFunc = False

    finalTextOutput(p, r)
    finalShow(p)
    return r

##################################################################
def finalTextOutput(p, r):
    if p.iprint >= 0:
        if p.msg is not '':  print "istop: ", r.istop , '(' + p.msg +')'
        else: print "istop: ", r.istop

        print 'Solver:   Time Elapsed = ' + str(r.elapsed['solver_time']) + ' \tCPU Time Elapsed = ' + str(r.elapsed['solver_cputime'])
        if p.plot:
            print 'Plotting: Time Elapsed = '+ str(r.elapsed['plot_time'])+ ' \tCPU Time Elapsed = ' + str(r.elapsed['plot_cputime'])
        if not p.isFeasible:
            print 'NO FEASIBLE SOLUTION is obtained (max residual = %0.2g, objFunc = %0.8g)' % (r.rf, r.ff)
        else:
            msg = "objFunValue: %0.8g" % r.ff
            if not p.isUC: msg += ' (feasible, max constraint =  %g)' % r.rf
            print msg

##################################################################
def finalShow(p):
    if not p.plot: return
    pylab = __import__('pylab')
    pylab.ioff()
    if p.show:
        pylab.show()

class EmptyClass: pass
class OpenOptResult: pass
