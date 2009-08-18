__docformat__ = "restructuredtext en"
from ooCheckGradient import ooCheckGradient
from numpy import array, isfinite, any, asarray
def ooCheck(p):
    """
    this func is called from runProbSolver(), you don't need to call the one
    """
    nErrors = 0

    if not (p.goal in p.allowedGoals):
        p.err('goal '+ p.goal+' is not available for the '+ p.probType + ' class (at least not implemented yet)')

#    for fn in p.__optionalData__:
#        if not fn in p.solver.__optionalDataThatCanBeHandled__:
#            p.err('the solver ' + p.solverName + ' cannot handle ' + "'" + fn + "' data")

    for fn in p.__optionalData__:
        if hasattr(p, fn):
            attr = getattr(p, fn)
            if not fn in p.solver.__optionalDataThatCanBeHandled__ \
            and \
            ((callable(attr) and getattr(p.userProvided, fn)) or (not callable(attr) and attr not in ([], (), None) and asarray(attr).size>0 and any(isfinite(attr)))):
                p.err('the solver ' + p.solver.__name__ + ' cannot handle ' + "'" + fn + "' data")


    return nErrors

