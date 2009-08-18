__docformat__ = "restructuredtext en"
from numpy import *
from setDefaultIterFuncs import USER_DEMAND_EXIT
from ooMisc import killThread, setNonLinFuncsNumber
from Function import oofun

none_type = type(None)

class objFunRelated:
    def __init__(self): pass

    def wrapped_func(p, x, IND, userFunctionType, ignorePrev, getDerivative):
        if not getattr(p.userProvided, userFunctionType): return array([])
        if p.istop == USER_DEMAND_EXIT:
            if p.solver.__cannotHandleExceptions__:
                return nan
            else:
                raise killThread
        #userFunctionType should be 'f', 'c', 'h'
        funcs = getattr(p.user, userFunctionType)
        funcs_num = getattr(p, 'n'+userFunctionType)
        if IND is not None:
            ind = p.getCorrectInd(IND)
        else: ind = None

        # this line had been added because some solvers pass tuple instead of
        # x being vector p.n x 1 or matrix X=[x1 x2 x3...xk], size(X)=[p.n, k]
        if type(x) != ndarray: x = asfarray(x)

        prevKey = p.prevVal[userFunctionType]['key']
        # TODO: move it into runprobsolver or baseproblem
        if p.prevVal[userFunctionType]['val'] is None:
            p.prevVal[userFunctionType]['val'] = zeros(getattr(p, 'n'+userFunctionType))
        R = None
        if prevKey is not None and p.iter > 0 and array_equal(x,  prevKey) and ind is None and not ignorePrev:
            #TODO: add counter of the situations
            if  getDerivative:
                R = copy(p.prevVal[userFunctionType]['val'])
            else:

                r = copy(p.prevVal[userFunctionType]['val'])
                #if p.debug: assert array_equal(r,  p.wrapped_func(x, IND, userFunctionType, True, getDerivative))
                if ind is not None: r = r[ind]

                if userFunctionType == 'f':
                    if p.isObjFunValueASingleNumber: r = r.sum(0)
                    if p.invertObjFunc: r = -r
                    if  p.solver.__funcForIterFcnConnection__=='f' and any(isnan(x)):
                        p.nEvals['f'] += 1
                        if p.nEvals['f']%p.f_iter == 0:
                            p.iterfcn(x, fk = r)
                return r

        args = getattr(p.args, userFunctionType)

        #if p.iter == 0:
        if not hasattr(p, 'n'+userFunctionType): setNonLinFuncsNumber(p,  userFunctionType)


        if ind is None:
            nFuncsToObtain = getattr(p, 'n'+ userFunctionType)
        else:
            nFuncsToObtain = len(ind)

        if x.shape[0] != p.n: p.err('incorrect x passed to obj fun')

        #TODO: code cleanup (below)
        if getDerivative or x.ndim <= 1 or x.shape[1] == 1:
            nXvectors = 1
            x_0 = copy(x)
        else:
            nXvectors = x.shape[1]
            x_0 = x[:, 0]

        if getDerivative:
            r = zeros((nFuncsToObtain, p.n))
        else:
            r = zeros((nFuncsToObtain, nXvectors))

        extractInd = None

        if ind is not None and p.functype[userFunctionType] == 'block':
            if len(ind) > 1:
                # TODO! Don't forget to remove ind[0] and use ind instead
                p.err("multiple index for block problems isn't implemented yet")

            #getting number of block and shift
            arr_of_indexes = getattr(p, 'arr_of_indexes_' + userFunctionType)
            left_arr_ind = searchsorted(arr_of_indexes, ind[0]) # CHECKME! is it index of block?

            if left_arr_ind != 0:
                num_of_funcs_before_arr_left_border = arr_of_indexes[left_arr_ind-1]
                inner_ind = ind[0] - num_of_funcs_before_arr_left_border - 1
            else:
                inner_ind = ind[0]
            Funcs = (funcs[left_arr_ind], )
            extractInd = inner_ind

        elif ind is not None and len(funcs) > 1:
            assert p.functype[userFunctionType] == 'some funcs'
            Funcs = [funcs[i] for i in ind]
        else:
            Funcs = funcs

        doInplaceCut = ind is not None  and len(funcs) == 1

        agregate_counter = 0
        #result_need_div_diffInt = getDerivative

        for fun in Funcs:
            if isinstance(fun, oofun):
                objectFlag = True
            else:
                objectFlag = False

            if getDerivative and objectFlag:
                    v = fun.D(x)
                    if extractInd is not None:  v = atleast_2d(v)[extractInd]
                    if doInplaceCut: v = atleast_2d(v)[ind]
                    r[agregate_counter:agregate_counter+v.shape[0]] = v

            elif nXvectors == 1:
                if objectFlag:
                    Args = ()
                else:
                    Args = args

                if R is None:
                    v = ravel(fun(*((x,) + Args)))
                    if extractInd is not None:  v = v[extractInd]
                    if doInplaceCut: v = v[ind]
                    r[agregate_counter:agregate_counter+v.size,0] = v
                    if  (ind is None or funcs_num == 1) and not ignorePrev:
                        #TODO: ADD COUNTER OF THE CASE
                        p.prevVal[userFunctionType]['val'][agregate_counter:agregate_counter+v.size] = v


                """                                                 geting derivatives                                                 """
                if getDerivative:
                    diffInt = copy(p.diffInt)

                    if R is None:
                        r0 = copy(r[agregate_counter:agregate_counter+v.size, 0])

                    if hasattr(fun, 'dep') and fun.dep is not None:
                        derivativeInd = fun.dep
                    else:
                        derivativeInd = xrange(p.n)

                    for i in derivativeInd:
                        if p.diffInt.size == 1:
                            finiteDiffNumber = p.diffInt[0]
                        else:
                            finiteDiffNumber = p.diffInt[i]

                        x[i] += finiteDiffNumber

                        v = ravel(fun(*((x,) + getattr(p.args, userFunctionType))))

                        x[i] -= finiteDiffNumber

                        assert not (extractInd is not None and doInplaceCut)

                        if extractInd is not None: v = v[extractInd]
                        if doInplaceCut: v = v[ind]

                        if not all(isfinite(v)):
                            x[i] -= finiteDiffNumber
                            v = ravel(fun(*((x,) + getattr(p.args, userFunctionType))))
                            x[i] += finiteDiffNumber
                            if extractInd is not None: v = v[extractInd]
                            if doInplaceCut: v = v[ind]

                        if i == derivativeInd[0]:
                            r[agregate_counter:agregate_counter+v.size, :] = 0
                            if R is not None: r0 = R[agregate_counter:agregate_counter+v.size]

                        r[agregate_counter:agregate_counter+v.size, i] = v - r0

                    if p.diffInt.size > 1:
                        if extractInd is not None: diffInt = diffInt[extractInd]
                        if doInplaceCut: diffInt = diffInt[ind]
                    r[agregate_counter:agregate_counter+v.size] /= diffInt
            else:
                for i in xrange(nXvectors): # TODO: add vectoriezed case
                    v = ravel(fun(*((x[:,i],) + getattr(p.args, userFunctionType))))
                    if i==0 and (ind is None or funcs_num==1) and not ignorePrev:
                        p.prevVal[userFunctionType]['val'][agregate_counter:agregate_counter+v.size] = v
                    if extractInd is not None:  v = v[extractInd]
                    if doInplaceCut: v = v[ind]
                    r[agregate_counter:agregate_counter+v.size,i] = v

            agregate_counter += atleast_1d(asarray(v)).shape[0]

#        if getDerivative and result_need_div_diffInt:
#            if asarray(p.diffInt).size == 1:
#                r /= p.diffInt
#            else:
#                for j in xrange(nFuncsToObtain):
#                    r[j, :] /= p.diffInt

        if userFunctionType == 'f' and p.isObjFunValueASingleNumber: r = r.sum(0)

        if nXvectors == 1  and  not getDerivative: r = r.flatten()

        if p.invertObjFunc and userFunctionType=='f':
            r = -r

        if (ind is None or funcs_num==1) and not ignorePrev and x.ndim <= 1: p.prevVal[userFunctionType]['key'] = copy(x_0)
        if ind is None:
            p.nEvals[userFunctionType] += nXvectors
        else:
            p.nEvals[userFunctionType] = p.nEvals[userFunctionType] + float(nXvectors * len(ind)) / getattr(p, 'n'+ userFunctionType)

        if getDerivative:
            assert x.size == p.n#TODO: add python list possibility here
            x = x_0 # for to suppress numerical instability effects while x +/- delta_x

        if userFunctionType == 'f' and hasattr(p, 'solver') and p.solver.__funcForIterFcnConnection__=='f':
            if p.nEvals['f']%p.f_iter == 0:
                p.iterfcn(x, fk = r)

        return r




    def wrapped_1st_derivatives(p, x, ind_, funcType, ignorePrev):
        if ind_ is not None:
            ind = p.getCorrectInd(ind_)
        else: ind = None

        if p.istop == USER_DEMAND_EXIT:
            if p.solver.__cannotHandleExceptions__:
                return nan
            else:
                raise killThread
        derivativesType = 'd'+ funcType
        prevKey = p.prevVal[derivativesType]['key']
        if prevKey is not None and p.iter > 0 and array_equal(x, prevKey) and ind is None and not ignorePrev:
            #TODO: add counter of the situations
            assert p.prevVal[derivativesType]['val'] is not None
            return copy(p.prevVal[derivativesType]['val'])

        if ind is None and not ignorePrev: p.prevVal[derivativesType]['ind'] = copy(x)

        #TODO: patterns!
        nFuncs = getattr(p, 'n'+funcType)
        if not getattr(p.userProvided, derivativesType):
            #x, IND, userFunctionType, ignorePrev, getDerivative
            derivatives = p.wrapped_func(x, ind, funcType, True, True)
            if ind is None:
                p.nEvals[derivativesType] -= 1
            else:
                p.nEvals[derivativesType] = p.nEvals[derivativesType] - float(len(ind)) / nFuncs
        else:

            if ind is not None:
                ind = p.getCorrectInd(ind)
                #nFuncs = len(ind)

            derivatives = empty((nFuncs, p.n))
            agregate_counter = 0
            for fun in getattr(p.user, derivativesType):
                tmp = atleast_1d(fun(*(x,)+getattr(p.args, funcType)))
                if mod(tmp.size, p.n) != 0:
                    if funcType=='f':
                        p.err('incorrect user-supplied (sub)gradient size of objective function')
                    elif funcType=='c':
                        p.err('incorrect user-supplied (sub)gradient size of non-lin inequality constraints')
                    elif funcType=='h':
                        p.err('incorrect user-supplied (sub)gradient size of non-lin equality constraints')
                if tmp.ndim == 1: m= 1
                else: m = tmp.shape[0]
                derivatives[agregate_counter : agregate_counter + m] =  tmp.reshape(tmp.size/p.n,p.n)
                agregate_counter += m
            #TODO: inline ind modification!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if ind is None:
                p.nEvals[derivativesType] += 1
            else:
                derivatives = derivatives[ind]
                p.nEvals[derivativesType] = p.nEvals[derivativesType] + float(len(ind)) / nFuncs

            if funcType=='f':
                if p.invertObjFunc: derivatives = -derivatives
                if p.isObjFunValueASingleNumber: derivatives = derivatives.flatten()

        if ind is None and not ignorePrev: p.prevVal[derivativesType]['val'] = derivatives

        if funcType=='f':
            if hasattr(p, 'solver') and not p.solver.__iterfcnConnected__  and p.solver.__funcForIterFcnConnection__=='df':
                if p.df_iter is True: p.iterfcn(x)
                elif p.nEvals[derivativesType]%p.df_iter == 0: p.iterfcn(x) # call iterfcn each {p.df_iter}-th df call

        return derivatives


    # the funcs below are not implemented properly yet
    def user_d2f(p, x):
        assert x.ndim == 1
        p.nEvals['d2f'] += 1
        assert(len(p.user.d2f)==1)
        r = p.user.d2f[0](*(x, )+p.args.f)
        if p.invertObjFunc and userFunctionType=='f': r = -r
        return r

    def user_d2c(p, x):
        return ()

    def user_d2h(p, x):
        return ()

    def user_l(p, x):
        return ()

    def user_dl(p, x):
        return ()

    def user_d2l(p, x):
        return ()

    def getCorrectInd(p, ind):
        if type(ind) in [none_type, list, tuple]:
            result = ind
        else:
            try:
                result = atleast_1d(ind).tolist()
            except:
                raise ValueError('%s is an unknown func index type!'%type(ind))
        return result

