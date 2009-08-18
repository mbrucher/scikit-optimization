from numpy import zeros, ones, copy, isfinite, where, asarray, inf, array, asfarray
__docformat__ = "restructuredtext en"

__solverPaths__ = None

def Len(arg):
    if arg == None or arg == [] or (arg.size==1 and arg == array(None, dtype=object)):
        return 0
    elif type(arg) in [int, float]:
        return 1
    else:
        return len(arg)

def xBounds2Matrix(p):
    """
    transforms lb - ub bounds into (A, x) <= b, (Aeq, x) = beq conditions
    this func is developed for those solvers that can handle lb, ub only via c(x)<=0, h(x)=0
    """

    #TODO: is reshape/flatten required in newest numpy versions?
    indLB, indUB, indEQ = \
    asarray(where(isfinite(p.lb).flatten() & ~(p.lb.reshape((-1,1)) == p.ub.reshape((-1,1))).flatten())).flatten(), \
    asarray(where(isfinite(p.ub).flatten() & ~asarray(p.lb.reshape((-1,1)) == p.ub.reshape((-1,1))).flatten())).flatten(), \
    asarray(where(p.lb.reshape((-1,1)) == p.ub.reshape((-1,1)))).flatten()

    initLenB = Len(p.b)
    initLenBeq = Len(p.beq)
    nLB, nUB, nEQ = Len(indLB), Len(indUB), Len(indEQ)

    if  nLB>0 or nUB>0:
        A, b = copy(p.A), copy(p.b)
        p.A = zeros([Len(p.b) + nLB+nUB, p.n])
        p.b = zeros([Len(p.b) + nLB+nUB])
        p.b[:Len(b)] = b.flatten() # sometimes flatten is needed when called before runProbSolver(), from tests
        p.A[:Len(b)] = A
        for i in  xrange(len(indLB)):
            p.A[initLenB+i, indLB[i]] = -1
            p.b[initLenB+i] = -p.lb[indLB[i]]
        for i in  xrange(len(indUB)):
            p.A[initLenB+len(indLB)+i, indUB[i]] = 1
            p.b[initLenB+len(indLB)+i] = p.ub[indUB[i]]

    if nEQ>0:
        Aeq, beq = copy(p.Aeq), copy(p.beq)
        p.Aeq = zeros([Len(p.beq) + nEQ, p.n])
        p.beq = zeros([Len(p.beq) + nEQ])
        p.beq[:Len(beq)] = beq
        p.Aeq[:Len(beq)] = Aeq
        for i in xrange(len(indEQ)):
            p.Aeq[initLenBeq+i, indEQ[i]] = 1
            p.beq[initLenBeq+i] = p.lb[indEQ[i]] # = p.ub[indEQ[i]], because they are the same

    p.lb = -inf*ones([p.n])
    p.ub = inf*ones([p.n])


def LinConst2WholeRepr(p):
    """
    transforms  (A, x) <= b, (Aeq, x) = beq into Awhole, bwhole, dwhole constraints (see help(LP))
    this func is developed for those solvers that can handle linear (in)equality constraints only via Awhole
    """
    if p.A == None and p.Aeq == None:
        return

    Awhole = copy(p.Awhole) # maybe it's already present and not equal to None
    p.Awhole = zeros([Len(p.b) + Len(p.beq) + Len(p.bwhole), p.n])
    if Awhole.size>0: p.Awhole[:Len(p.bwhole)] = Awhole

    p.Awhole[Len(p.bwhole):Len(p.bwhole)+Len(p.b)] = p.A
    p.A = None
    if p.Aeq.size: p.Awhole[Len(p.bwhole)+Len(p.b):] = p.Aeq
    p.Aeq = None


    bwhole = copy(p.bwhole)
    p.bwhole = zeros([Len(p.b) + Len(p.beq) + Len(p.bwhole)])
    p.bwhole[:Len(bwhole)] = bwhole

    p.bwhole[Len(bwhole):Len(bwhole)+Len(p.b)] = p.b

    p.bwhole[Len(bwhole)+Len(p.b):] = p.beq


    dwhole = copy(p.dwhole)
    p.dwhole = zeros([Len(p.bwhole)])
    if dwhole.size: p.dwhole[:Len(bwhole)] = dwhole
    p.dwhole[Len(bwhole):Len(bwhole)+Len(p.b)] = -1
    p.dwhole[Len(bwhole)+Len(p.b):] = 0

    p.b = None
    p.beq = None

def WholeRepr2LinConst(p):
    """
    transforms  Awhole, bwhole, dwhole  into (A, x) <= b, (Aeq, x) = beq constraints (see help(LP))
    this func is developed for those solvers that can handle linear (in)equality constraints only via Awhole
    """
    if p.dwhole == None:
        return
    #TODO: is flatten required in newest numpy versions?
    ind_less = asarray(where(p.dwhole == -1)).flatten()
    ind_greater = asarray(where(p.dwhole == 1)).flatten()
    ind_equal = asarray(where(p.dwhole == 0)).flatten()

    if len(ind_equal) != 0:
        Aeq, beq = copy(p.Aeq) , copy(p.beq)
        p.Aeq = zeros([Len(p.beq)+len(ind_equal), p.n])
        if Aeq.size: p.Aeq[:Len(p.beq)] = Aeq
        if len(ind_equal): p.Aeq[Len(p.beq):] = p.Awhole[ind_equal]
        p.beq = zeros([Len(p.beq)+len(ind_equal)])
        if beq.size: p.beq[:Len(beq)] = beq
        if len(ind_equal): p.beq[Len(beq):] = p.bwhole[ind_equal]

    if len(ind_less) + len(ind_greater)>0:
        A, b = copy(p.A) , copy(p.b)
        p.A = zeros([Len(p.b)+len(ind_less)+len(ind_greater), p.n])
        if A.size: p.A[:Len(p.b)] = A
        p.A[Len(p.b):Len(p.b)+len(ind_less)] = p.Awhole[ind_less]
        p.A[Len(p.b)+len(ind_less):] = -p.Awhole[ind_greater]
        p.b = zeros(Len(p.b)+len(ind_less)+len(ind_greater))
        if b.size: p.b[:Len(b)] = b
        if len(ind_less): p.b[Len(b):Len(b)+len(ind_less)] = p.bwhole[ind_less]
        if len(ind_greater): p.b[Len(b)+len(ind_less):] = -p.bwhole[ind_greater]

    p.Awhole = None
    p.bwhole = None
    p.dwhole = None

def assignScript(p, *args, **kwargs):
    for key in args[0].keys():
        setattr(p, key, args[0][key])
    if len(args)>1 or len(kwargs)>0:
        p.err("Error in assignment or this feature isn't implemented yet")

def setNonLinFuncsNumber(p,  userFunctionType):
    # userFunctionType  should be 'f', 'c' or 'h'
    args = getattr(p.args, userFunctionType)
    fv = getattr(p.user, userFunctionType)

    if len(fv) == 1: p.functype[userFunctionType] = 'single func'
    if fv is None or (type(fv) in [list, tuple] and fv[0] is None):
        setattr(p, 'n'+userFunctionType, 0)
    elif type(fv) in [list, tuple] and len(fv)>1:
        # TODO: handle problems w/o x0, like GLP
        number = 0
        arr = []
        for func in fv:
#                    if func.__module__ == 'scikits.openopt.Kernel.Function':
#                        func.x = p.x0
            number += asarray(func(*(p.x0,) + args)).size
            arr.append(number)
        if len(arr) < number: p.functype[userFunctionType] = 'block'
        elif len(arr) > 1: p.functype[userFunctionType] = 'some funcs'
        else: assert p.functype[userFunctionType] == 'single func'
        setattr(p, 'n' + userFunctionType, number)
        if p.functype[userFunctionType] == 'block':
            setattr(p, 'arr_of_indexes_' + userFunctionType, array(arr)-1)
    elif p.x0 is not None:
        FV = fv
        if not (type(FV) in [list, tuple]):
            FV = [FV]
        setattr(p, 'n'+userFunctionType, asfarray(FV[0](*(p.x0, ) + args)).size)
    else:
        FV = fv
        if not (type(FV) in [list, tuple]):
            FV = [FV]
        setattr(p, 'n'+userFunctionType, asfarray(FV[0](*(0.5*(p.lb+p.ub),  ) + args)).size)



class isSolved:
    def __init__(self): pass
class killThread:
    def __init__(self): pass
