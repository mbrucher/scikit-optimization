from solverPaths import solverPaths
from BaseAlg import BaseAlg
from oologfcn import OpenOptException

##################################################################
def getSolverFromStringName(p, solver_str):
    if p.debug:
        solverClass =  getattr(my_import(solverPaths[solver_str]), solver_str)
    else:
        try:
            solverClass = getattr(my_import(solverPaths[solver_str]), solver_str)
        except:
            p.err('incorrect solver is called, maybe the solver "' + solver_str +'" is not installed. Maybe setting p.debug=1 could specify the matter more precisely')
    return solverClass()

##################################################################
def my_import(name):
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def oosolver(solverName, *args,  **kwargs):
    if args != ():
        raise OpenOptException("Error: oosolver() doesn't consume any *args, use **kwargs only")
    try:
        solverClass = getattr(my_import(solverPaths[solverName]), solverName)
        solverClassInstance = solverClass()
        solverClassInstance.fieldsForProbInstance = {}
        for key, value in kwargs.iteritems():
            if hasattr(solverClassInstance, key):
                setattr(solverClassInstance, key, value)
            else:
                solverClassInstance.fieldsForProbInstance[key] = value
        solverClassInstance.isInstalled = True
    except:
        solverClassInstance = BaseAlg()
        solverClassInstance.__name__ = solverName
        solverClassInstance.isInstalled = False
    return solverClassInstance
