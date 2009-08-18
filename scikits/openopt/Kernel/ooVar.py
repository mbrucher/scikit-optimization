__docformat__ = "restructuredtext en"
# created by Dmitrey

from numpy import nan, asarray, isfinite, empty, zeros, inf, any, array
#from numpy import inf, asfarray, copy, all, any, empty, atleast_2d, zeros, dot, asarray, atleast_1d, empty, ones, ndarray, where, isfinite
from oologfcn import OpenOptException
#from copy import deepcopy

class oovar:
    size = nan # number of variables
    shape = nan
    fixed = False
    initialized = False

    def __init__(self, name, *args, **kwargs):

        self.name = name

        if len(args) > 1: raise OpenOptException('incorrect args number for oovar constructor')

        if len(args) > 0:
            self.v0 = array(args[0], float)

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def __getDep__(self):
        return self.dep

    def D(self, x):
        #TODO: remove it
        r = zeros((self.size, x.size))
        for i in range(self.size):
            r[i, self.dep[i]] = 1
        return r

    def __initialize__(self, p):

        """                                               Handling size and shape                                               """
        sizes = set([])
        shapes = set([])
        for fn in ['v0', 'lb', 'ub']:
            if hasattr(self, fn):
                setattr(self, fn, asarray(getattr(self, fn)))
                shapes.add(getattr(self, fn).shape)
                sizes.add(getattr(self, fn).size)
        if self.shape is not nan: sizes.add(self.size)
        if self.size is not nan: sizes.add(self.size)
        if len(shapes) > 1: p.err('for oovar fields (if present) lb, ub, v0 should have same shape')
        elif len(shapes) == 1: self.shape = shapes.pop()
        if len(sizes) > 1: p.err('for oovar fields (if present) lb, ub, v0 should have same size')
        else: self.size = sizes.pop()

        if self.shape is nan:
            assert isfinite(self.size)
            self.shape = (self.size, )
        if self.size is nan: self.size = asarray(self.shape).prod()

        """                                                     Handling init value                                                   """
        if not hasattr(self, 'lb'):
            self.lb = empty(self.shape)
            self.lb.fill(-inf)
        if not hasattr(self, 'ub'):
            self.ub = empty(self.shape)
            self.ub.fill(inf)
        if any(self.lb > self.ub):
            p.err('lower bound exceeds upper bound, solving impossible')
        if not hasattr(self, 'v0'):
            #p.warn('got oovar w/o init value')
            v0 = zeros(self.shape)

            ind = isfinite(self.lb) & isfinite(self.ub)
            v0[ind] = 0.5*(self.lb[ind] + self.ub[ind])

            ind = isfinite(self.lb) & ~isfinite(self.ub)
            v0[ind] = self.lb[ind]

            ind = ~isfinite(self.lb) & isfinite(self.ub)
            v0[ind] = self.ub[ind]

            self.v0 = v0
            
        self.initialized = True



