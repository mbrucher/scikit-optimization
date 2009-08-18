"""
the example illustrates 
how you can declare and use
fixed oovars

here it is used for objFunc only 
but of course it can be used 
for non-linear equality and inequality constraints as well
"""

n = 5

from scikits.openopt import NLP,  oofun,  oovar
from numpy import inf

v0 = oovar('v0', [-4.0, -15.0]) # 2nd arg (if provided) is start value
v1 = oovar('v1', range(n), fixed=True)  # range(n) is [0, 1, ..., n-1], see also: numpy.arange

#set objFunc
f = oofun(lambda z0, z1: (z0[0]-15)**4 + (z0[1]-80)**4 + (z1**2).sum(), input = [v0, v1])

# assign prob
p = NLP(f)

# solve
r = p.solve('ralg')

print 'solution:', r.xf
print 'optim value:', r.ff

"""
using oofun-style for fixed vars is more effective than same problem in classic style fix via lb=ub

some mature solvers (but not ralg) have efficient handling of fixed vars, 
but they can't take advantage from deeply nested parts of code fixed due to all used variables in that parts are fixed.
so using oovars + oofuns can yield serious benefites even for those ones.
"""
lb = [-inf, -inf] + range(n)
ub = [inf, inf] + range(n)
p2 = NLP(lambda x: (x[0]-15) **4 + (x[1]-80)**4 + (x[2:] ** 2).sum(), [-4.0, -15.0] + range(n), lb=lb, ub=ub)
r2 = p2.solve('ralg')




