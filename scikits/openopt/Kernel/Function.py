__docformat__ = "restructuredtext en"
# created by Dmitrey
#from numpy import copy, isnan, array, argmax, abs, zeros
from numpy import inf, asfarray, copy, all, any, empty, atleast_2d, zeros, dot, asarray, atleast_1d, empty, ones, ndarray, where, isfinite, array, nan, ix_
from numpy.linalg import norm
from oologfcn import OpenOptException
from copy import deepcopy
from ooVar import oovar

class oofun:
    #initialized = False
    #__allowedFields__ = ['d', 'dep']
    name = 'unnamed'
    input = None # if None then x will be used
    args = ()
    ooVarsAreAlreadyConnected = False

    # finite-difference aproximation step
    diffInt = 1.5e-8

    def __init__(self, fun, *args, **kwargs):
        assert len(args) == 0
        self.fun = fun

        #TODO: modify for cases where output can be partial
        self.evals = 0
        self.same = 0

        for key, item in kwargs.iteritems():
            #assert key in self.__allowedFields__ # TODO: make set comparison
            setattr(self, key, item)

    """                                             getInput                                              """
    def __getInput__(self):
        if self.input is None:
            self.inputTotalLength = len(self.x)
            return (self.x, )
            #return None
        elif not type(self.input) in (list, tuple):
            self.input = [self.input]
        r = []
        self.inputTotalLength = 0
        for item in self.input:
            if isinstance(item, oofun):
                item.x = self.x
                r.append(item())
                self.inputTotalLength += item().size
            elif isinstance(item, oovar):
                if item.fixed:
                    r.append(item.v0)
                else:
                    r.append(self.x[item.dep])
                self.inputTotalLength += item.size
            elif not callable(item): r.append(item)
            else:  r.append(item())

        return tuple(r)

    """                                                getDep                                             """
    def __getDep__(self):
        if hasattr(self, 'dep'):
            return self.dep
        elif self.input is None:
            self.dep = None
        else:
            r = set([])
            #r.fill(False)
            if not type(self.input) in (list, tuple):
                self.input = [self.input]
            for oofunInstance in self.input:
                if oofunInstance.fixed: continue
#                if not hasattr(oofunInstance, 'x'):
#                    oofunInstance.x = self.x
                tmp = oofunInstance.__getDep__()
                if tmp is None:
                    r = self.dep = None # depends on all x coords
                    break
                else:
                    if type(tmp) in (ndarray, list, tuple):
                        tmp = set(tmp)
                    elif type(tmp) == int:
                        tmp = set([tmp])
                    elif type(tmp) != set:
                        raise OpenOptException('unknown type of oofun or oovar dependence')
                    r.update(tmp)
            if r is not None:
                self.dep = array(list(r))
        return self.dep


    """                                                getFunc                                             """
    def __getFunc__(self, x=None):
        if self.fixed and hasattr(self, 'f_key_prev'):
            return deepcopy(self.f_val_prev)
            
        if x is None: x = self.x
        else: self.x = x

        dep = self.__getDep__()

        # TODO: remove it
        if dep is None: key_to_compare = x
        else: key_to_compare = x[dep]

        if not hasattr(self, 'f_key_prev') or any(self.f_key_prev != key_to_compare):
            self.evals += 1
            if type(self.args) != tuple:
                self.args = (self.args, )
            Input = self.__getInput__()
            if self.args != ():
                Input += self.args
            self.f_val_prev = asfarray(self.fun(*Input))
            self.outputTotalLength = self.f_val_prev.size # TODO: omit reassigning
            self.f_key_prev = copy(key_to_compare)
        else:
            self.same += 1

        return deepcopy(self.f_val_prev)


    """                                                getFunc                                             """
    __call__ = lambda self, *args: self.__getFunc__(*args)


    """                                              derivatives                                           """
    D = lambda self, *args: self.__d(*args)


    """
    def __recursiveDerivatives__(self, derivativeSelf, val_0, agregate_counter, Input, item, ind_arr):
        if type(item) in (tuple, list):
            for ind, item2 in enumerate(item):
                self.__recursiveDerivatives__(derivativeSelf, val_0, agregate_counter, Input, item2, ind_arr +(ind, ))
#            derivativeSelf[:, agregate_counter] = 1
#            agregate_counter += 1
        elif type(item) == ndarray and item.size > 1:
            if self.input is None and self.dep is not None:
                indexes = asarray(self.dep).tolist()
            else:
                indexes = xrange(len(item))

            for j in indexes:
                item[j] += self.diffInt
                v = atleast_1d(self.fun(*Input))
                item[j] -= self.diffInt
                if not all(isfinite(v)):
                    item[j] -= self.diffInt
                    v = atleast_1d(self.fun(*Input))
                    item[j] += self.diffInt
                assert v.ndim == 1
                derivativeSelf[:, agregate_counter] += (v.reshape(val_0.shape)-val_0) / self.diffInt
                if v.ndim == 1:
                    agregate_counter += 1
                else:
                    agregate_counter += v.shape[1]
        else:
            item += self.diffInt
            v = atleast_1d(self.fun(*Input))
            item -= self.diffInt
            if not all(isfinite(v)):
                item -= self.diffInt
                v = atleast_1d(self.fun(*Input))
                item += self.diffInt
            derivativeSelf[:, agregate_counter] = (v.reshape(val_0.shape)-val_0) / self.diffInt
            agregate_counter += 1
    """

    def __checkDerivatives__(self, x=None, eps = 1e-5):
        #  TODO: CHECK ME!
        #wasFixed, self.fixed = self.fixed, False

        if not hasattr(self, 'd') or self.d is None:
            raise OpenOptException('you should provide derivatives for the oofun to be tested')

        if x is None: x = self.x
        else: self.x = x

        Input_ = self.__getInput__()
        if self.args == (): Input = list(Input_)
        else: Input = list(Input_ + self.args)

        d_tmp, self.d = self.d, None
        derivativeNumerical = self.__getDerivativeSelf__(Input, Input_)
        self.d = d_tmp
        derivativeSelf = atleast_2d(self.d(*Input))
        if derivativeSelf.shape != derivativeNumerical.shape:
            raise OpenOptException('incorrect shape for oofun '+ self.name + ' derivative: ' + str(derivativeNumerical.shape) + ' expected,'+\
                                   str(derivativeSelf.shape) + ' obtained')
        #r = norm(derivativeNumerical.flatten() - derivativeSelf.flatten(), inf)
        d = abs(derivativeNumerical.flatten() - derivativeSelf.flatten())
        D = abs(derivativeNumerical - derivativeSelf)
        (I, J) = where(D > eps)
        for i in xrange(len(I)):
            ind = (I[i], J[i])
            print(I[i], J[i], derivativeNumerical[ind], derivativeSelf[ind], derivativeNumerical[ind] - derivativeSelf[ind])
        ind = d.argmax()
        maxDiff = d[ind]
        ii, jj = divmod(ind, derivativeNumerical.shape[1])
        print('max difference: '+str(maxDiff))
        return derivativeNumerical, derivativeSelf, maxDiff, ii, jj


    def __d(self, x=None):
        if self.fixed: 
            # TODO: 
            # 1) handle sparsity if possible
            # 2) try to handle the situation in the level above
            return zeros((self.outputTotalLength, self.inputTotalLength))

        if x is None: x = self.x
        else: self.x = x

        if not hasattr(self, 'outputTotalLength'):
            self.__getFunc__(x)

        dep = self.__getDep__()
        if dep is None: key_to_compare = x
        else: key_to_compare = x[dep]

        Input_ = self.__getInput__()

        if self.args == (): Input = list(Input_)
        else: Input = list(Input_ + self.args)

        if hasattr(self, 'd') and self.d is not None:
            derivativeSelf = atleast_2d(self.d(*Input))
        else:
            derivativeSelf = self.__getDerivativeSelf__(Input, Input_)

            # TODO: mb copy self.input to prevent numerical noise -> other values -> recalculate self.__getFunc__
#            if self.input is not None:


            # !!! enumerate() returns copy and hence is unsuitable here
            # not xrange(len(Input))! because Input is already with args here


        ##########################
        if not hasattr(self, 'd_key_prev') or any(self.d_key_prev != key_to_compare):
            if self.input is not None:
                ########################################
                agregate_counter = 0
                rr = zeros((self.inputTotalLength, len(x)))
                #rr = zeros((self.inputTotalLength, self.outputTotalLength))
                has_oovar = False
                for i, inp in enumerate(self.input):
                    # get derivatives of i-th input
                    if isinstance(inp, oovar):
                        if inp.fixed: continue
                        #has_oovar = True
                        #rr[agregate_counter:agregate_counter + inp.size] = nan
                        for ii in inp.dep:
                            rr[agregate_counter, ii] = 1
                            agregate_counter += 1
                        #agregate_counter += inp.size
                    else:
                        tmp = atleast_2d(inp.D(x))
                        rr[agregate_counter:agregate_counter+tmp.shape[0]] = tmp
                        agregate_counter += tmp.shape[0]

                if derivativeSelf.size == 1:
                    r = derivativeSelf * rr
                #elif derivativeSelf.ndim > 1:

                elif derivativeSelf.shape[0] == 1 and rr.shape[0] == 1:
                    r = dot(derivativeSelf.T, rr)
                    #r = dot(derivativeSelf.flatten(), rr.flatten())
#
#                elif derivativeSelf.ndim>1:
#                    pass
                    #derivativeSelf = derivativeSelf.T
                else:
                    r = dot(derivativeSelf, rr)

                if has_oovar:
                    agregate_counter = 0
                    derivativeSelf = atleast_2d(derivativeSelf)
                    for i, inp in enumerate(self.input):
                        if isinstance(inp, oovar):
                            r[agregate_counter, inp.dep] = derivativeSelf[agregate_counter]#, inp.dep]
                            agregate_counter += inp.size
                        else:
                            agregate_counter += inp.__getInput__().size()

            else:
                r = derivativeSelf

            self.d_val_prev = r
            self.d_key_prev = copy(key_to_compare)
        return copy(self.d_val_prev)



    D2 = lambda self, x: self.__d2(self, x)

    def __d2(self, x):
        raise OpenOptException('2nd derivatives for obj-funcs are not implemented yet')
#        if not self.initialized:
#            self.nFuncs = r.size
#            self.initialized = True
        return r

    def __connect_ooVars__(self, p):
        if self.ooVarsAreAlreadyConnected: return
        self.fixed = True
        if self.input is None: p.err('got oofun w/o connection to oovar (empty input instead). Use x0 or connect oovars.')
        if not type(self.input) in (list, tuple):
            self.input = [self.input]
        # p.oovars is set

        for inp in self.input:
            if isinstance(inp, oovar):
                if not inp.initialized: inp.__initialize__(p)
                if not inp.fixed: p.oovars.add(inp)
            elif isinstance(inp, oofun):
                inp.__connect_ooVars__(p) # recursive
            else: p.err('incorrect input for oofun instance')
            if inp.fixed == False: self.fixed = False

        self.ooVarsAreAlreadyConnected = True

    def __getDerivativeSelf__(self, Input, Input_):

        if not hasattr(self, 'inputTotalLength'): self.__getInput__()
        derivativeSelf = zeros((self.outputTotalLength, self.inputTotalLength))
        agregate_counter = 0
        val_0 = self.__getFunc__(self.x)

        for i in xrange(len(Input_)):
            inp = Input_[i]

            assert asarray(inp).ndim <= 1
            if isinstance(inp, oovar):
                #if not inp.fixed: 
                derivativeSelf[:, agregate_counter] = 1
                agregate_counter += 1
            if type(inp) in (ndarray, tuple, list):
                # TODO: handle Python dict, mb Python class here
                if self.input is None and self.dep is not None:
                    indexes = asarray(self.dep).tolist()
                else:
                    indexes = xrange(len(inp))
                for j in indexes:
                    Input[i][j] += self.diffInt
                    v = atleast_1d(self.fun(*Input))
                    Input[i][j] -= self.diffInt
                    if not all(isfinite(v)):
                        Input[i][j] -= self.diffInt
                        v = atleast_1d(self.fun(*Input))
                        Input[i][j] += self.diffInt
                    assert v.ndim == 1
                    derivativeSelf[:, agregate_counter] += (v.reshape(val_0.shape)-val_0) / self.diffInt
                    if v.ndim == 1:
                        agregate_counter += 1
                    else:
                        agregate_counter += v.shape[1]
            else:
                # TODO: ASSERT isscalar(Input[i])
                Input[i] += self.diffInt
                v = atleast_1d(self.fun(*Input))
                Input[i] -= self.diffInt
                if not all(isfinite(v)):
                    Input[i] -= self.diffInt
                    v = atleast_1d(self.fun(*Input))
                    Input[i] += self.diffInt
                #assert v.ndim == 1
                derivativeSelf[:, agregate_counter] =  (v.reshape(val_0.shape)-val_0) / self.diffInt#(v-val_0) / self.diffInt
                if v.ndim == 1:
                    agregate_counter += 1
                else:
                    agregate_counter += v.shape[1]

        # TODO: mb copy self.input to prevent numerical noise -> other values -> recalculate self.__getFunc__
        return derivativeSelf





class oolin(oofun):
    def __init__(self, C, d=0, *args, **kwargs):
        # returns Cx + d
        # TODO: handle FIXED variables here
        mtx = atleast_2d(array(C))
        d = array(d, float)
        
        # TODO: use p.err instead assert
        assert d.ndim <= 1, 'passing d with ndim>1 into oolin Cx+d is forbidden'
        if d.size != mtx.shape[0]:
            if d.size == 1: OpenOptException('Currently for Cx+d using d with size 1 is forbidden for C.shape[0]>1 for the sake of more safety and for openopt users code to be clearer')
        
        ind_zero = where(all(mtx==0, 0))[0]
        def oolin_objFun(*x):
            if len(x) == 1:
                x = x[0]
            X = asfarray(x).copy()
            X[ind_zero] = 0
            r = dot(mtx, X) + d # case c = 0 or all-zeros yields insufficient additional calculations, so "if c~=0" can be omitted
            return r
        oofun.__init__(self, oolin_objFun, *args, **kwargs)
        self.d = lambda *x: mtx.copy()

