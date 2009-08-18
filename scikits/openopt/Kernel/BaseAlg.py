__docformat__ = "restructuredtext en"
from numpy import atleast_1d,  all, asarray, ndarray, copy, ravel
from Point import Point

class BaseAlg:
    def __init__(self):pass
    __name__ = 'Undefined. If you are a user and got the message, inform developers please.'
    __license__ = "Undefined. If you are a user and got the message, inform developers please."
    __authors__ = "Undefined. If you are a user and got the message, inform developers please."
    __alg__ = "Undefined"
    __solver__ = "Undefined. If you are a user and got the message, inform developers please."
    __homepage__ = 'Undefined. Use web search'
    __info__ = 'None'

    """__cannotHandleExceptions__  is
    True for ipopt and mb some others,
    somehow exceptions raised in OO and
    passed through ipopt
    cannot be catched by OO
    """
    __cannotHandleExceptions__ = False

    __optionalDataThatCanBeHandled__ = []
    __isIterPointAlwaysFeasible__ = lambda self, p: p.isUC#TODO: provide possibility of simple True, False
    __iterfcnConnected__ = False
    __funcForIterFcnConnection__ = 'df' # the field is used for non-linear solvers with not-connected iter function


    # these ones below are used in iterfcn (ooIter.py)
    # to decode input args
    # and can be overdetermined by child class (LP, QP, network etc)
    __expectedArgs__ = ['xk',  'fk',  'rk'] #point, objFunVal, max residual
    def __decodeIterFcnArgs__(self,  p,  *args,  **kwargs):
        """
        decode and assign x, f, maxConstr
        (and/or other fields) to p.iterValues
        """
        fArg,  rArg = True,  True

        if len(args)>0 and isinstance(args[0], Point):
            if len(args) != 1: p.err('incorrect iterfcn args, if you see this contact OO developers')
            point = args[0]
            p.xk, p.fk = point.x, point.f()
            p.rk, p.rtk, p.rik = point.mr(True)
        else:
            if len(args)>0: p.xk = args[0]
            elif kwargs.has_key('xk'): p.xk = kwargs['xk']
            elif not hasattr(p, 'xk'): p.err('iterfcn must get x value, if you see it inform oo developers')

            if len(args)>1: p.fk = args[1]
            elif kwargs.has_key('fk'): p.fk = kwargs['fk']
            else: fArg = False

            if len(args)>2: p.rk = args[2]
            elif kwargs.has_key('rk'): p.rk = kwargs['rk']
            else: rArg = False

            p.rk, p.rtk, p.rik = p.getMaxResidual(p.xk, True)

        #TODO: handle kwargs correctly! (decodeIterFcnArgs)

#        for key in kwargs.keys():
#            if p.debug: print 'decodeIterFcnArgs>>',  key,  kwargs[key]
#            setattr(p, key, kwargs[key])

        p.iterValues.x.append(copy(p.xk))

        p.iterValues.rt.append(p.rtk)
        p.iterValues.ri.append(p.rik)

        if not fArg:
            p.Fk = p.F(p.xk)
            p.fk = copy(p.Fk)
        else:
            if asarray(p.fk).size >1:
                if p.debug and p.iter <= 1: p.warn('please fix solver iter output func, objFuncVal should be single number (use p.F)')
                p.Fk = p.objFuncMultiple2Single(asarray(p.fk))
            else:
                p.Fk = p.fk

        #if p.isObjFunValueASingleNumber: p.Fk = p.fk
        #else: p.Fk = p.objFuncMultiple2Single(fv)

        v = ravel(p.Fk)[0]
        if p.invertObjFunc: v = -v

        p.iterValues.f.append(v)

        if not rArg:
            p.rk = ravel(p.getMaxResidual(p.xk))[0]
        p.iterValues.r.append(p.rk)

