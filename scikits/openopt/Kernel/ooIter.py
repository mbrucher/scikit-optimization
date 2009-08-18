__docformat__ = "restructuredtext en"

from time import time, clock
from numpy import isreal,  array_equal
from ooMisc import isSolved
from setDefaultIterFuncs import USER_DEMAND_STOP, BUTTON_ENOUGH_HAS_BEEN_PRESSED, IS_NAN_IN_X, SMALL_DELTA_X, IS_MAX_ITER_REACHED, IS_MAX_CPU_TIME_REACHED, IS_MAX_TIME_REACHED, IS_MAX_FUN_EVALS_REACHED

has_Tkinter = True
try:
    import Tkinter
except:
    has_Tkinter = False

NoneType = type(None)

def ooIter(p, *args,  **kwargs):
    """
    this func is called from iter to iter
    it is default iter function of OpenOpt Kernel
    lots of solvers use this one
    it provides basic graphics output (provided plot option is turned on),
    maybe in future some text output will also be generated here.
    also, some stop criteria are handled via the func.
    """

    if has_Tkinter:
        if p.state == 'paused':
            p.GUI_root.wait_variable(p.statusTextVariable)

    if not hasattr(p, 'timeStart'): return#called from check 1st derivatives

    p.currtime = time()
    if not p.iter:
        p.lastDrawTime = p.currtime
        p.lastDrawIter = 0

    if not p.isFinished or len(p.iterValues.f) == 0:
        p.solver.__decodeIterFcnArgs__(p,  *args,  **kwargs)
    if p.graphics.xlabel == 'nf': p.iterValues.nf.append(p.nEvals['f'])

    if (p.iter == 1 and array_equal(p.xk,  p.iterValues.x[0]) and not p.probType == 'GLP') \
    or (p.istop != 0 and len(p.iterValues.x) >= 2 and array_equal(p.iterValues.x[-2],  p.iterValues.x[-1])):
        for fn in dir(p.iterValues):
            attr = getattr(p.iterValues,  fn)
            if type(attr) == list:
                attr.pop(-1)
            elif type(attr) not in [str, NoneType]:
                p.warn('Found incorrect type ' + str(type(attr)) +' in p.iterValues (Python list expected), it can lead to error(s)!')
        #TODO: handle case x0 = x1 = x2 = ...
        return

    p.iterPrint()

    p.iterCPUTime.append(clock() - p.cpuTimeStart)
    p.iterTime.append(p.currtime - p.timeStart)
    #todo: same with norm(p.constraints,1) and norm(p.constraints,inf)

    #TODO: turn off xtol and ftol for artifically iterfcn funcs

    if not p.isFinished and not p.userStop:
        for key, fun in p.kernelIterFuncs.iteritems():
            r =  fun(p)
            if r is not False:
                p.stopdict[key] = True
                if p.istop == 0 or not (key in [IS_MAX_ITER_REACHED, IS_MAX_CPU_TIME_REACHED, IS_MAX_TIME_REACHED, IS_MAX_FUN_EVALS_REACHED]):
                    p.istop = key
                    if type(r) == tuple:
                        p.msg = r[1]
                    else:
                        p.msg = 'unkown, if you see the message inform openopt developers'
        if p.stopdict.has_key(IS_NAN_IN_X):pass
        elif p.stopdict.has_key(SMALL_DELTA_X) and array_equal(p.iterValues.x[-1], p.iterValues.x[-2]): pass
        else:
            p.nonStopMsg = ''
            for fun in p.denyingStopFuncs.keys():
                if not fun(p):
                    p.istop = 0
                    p.stopdict = {}
                    p.msg = ''
                    p.nonStopMsg = p.denyingStopFuncs[fun]
                    break
            for fun in p.callback:
                r =  fun(p)
                if r not in [0,  False]:
                    if r in [True,  1]:  p.istop = USER_DEMAND_STOP
                    elif isreal(r):
                        p.istop = r
                        p.msg = 'user-defined'
                    else:
                        p.istop = r[0]
                        p.msg = r[1]
                    p.stopdict[p.istop] = True
                    p.userStop = True

    if p.istop and not p.solver.__iterfcnConnected__ and not p.isFinished and not p.solver.__cannotHandleExceptions__:
        raise isSolved

    T, cpuT = 0., 0.

    if p.plot and (p.iter == 0 or p.iter <2 or p.isFinished or \
    p.currtime - p.lastDrawTime > p.graphics.rate * (p.currtime - p.iterTime[p.lastDrawIter] - p.timeStart)):
    #(p.timeElapsedForPlotting[-1]-p.timeElapsedForPlotting[1]) /  (p.currtime - p.timeStart - p.timeElapsedForPlotting[-1]) < p.graphics.rate):
        for df in p.graphics.drawFuncs: df(p)
        T = time() - p.timeStart - p.iterTime[-1]
        cpuT = clock() - p.cpuTimeStart - p.iterCPUTime[-1]
        p.lastDrawTime = time()
        p.lastDrawIter = p.iter
    if p.plot:
        p.timeElapsedForPlotting.append(T+p.timeElapsedForPlotting[-1])
        p.cpuTimeElapsedForPlotting.append(cpuT+p.cpuTimeElapsedForPlotting[-1])

    p.iter += 1



