"""
This example illustrates a great speedup that can be achieved
via using oofun and oovar vs "classic" style,
even for unconstrained functions,
provided the solver deals with at least 1st derivatives,
so scipy_cobyla, goldenSection, scipy_fminbound, scipy_powell
or GLP solvers are inappropriate

The speedup is due to changes in derivatives numerical approximation approach:
instead of handling whole dF/dx = d(g(f))/dx
we find dg/df, df/dx and
dF/dx = dg/df * df/dx

Here's example of unconstrained problem, but constrained ones can be used as well
for NLP, NSP, NLSP, LSP classes

Concider the NL problem
g(f(x)) -> min
g is costly and g derivative are not available
f(x) = (x[0]-0)^2 + (x[1]-1)^2 + ... + (x[N-1]-(N-1))^2
see below for definition of g
here I have chosed g: R -> R for the sake of simplicity,
but R^m -> R^k can be handled as well
"""
def CostlyFunction(z):
    counter['g'] += 1
    r = z
    for k in xrange(1, K+2):
        r += z ** (1 / k**1.5)
    return r

def f(z):
    counter['f'] += 1
    return ((z-aN)**2).sum()

solver = 'scipy_ncg'# try also scipy_cg, scipy_ncg, ralg, algencan etc
N, K = 150, 500
ftol, xtol, gtol = 1e-6, 1e-6, 1e-6
iprint = 5

from scikits.openopt import NLP,  oofun,  oovar
from numpy import arange, zeros
aN = arange(N)

"""                      1: using oovar & oofun                      """
counter = {'f':0, 'g':0}
v = oovar('v', size = N) # start value will be zeros(N)
ff = oofun(f, input = v)
g = oofun(CostlyFunction, input = ff)
p = NLP(g, maxIter=1e4, iprint=iprint, ftol=ftol, xtol=xtol, gtol=gtol)
print 'using oofun:'
r = p.solve(solver)
print 'evals f:', counter['f'], '  evals of costly func g:', counter['g']
"""                               2: classic                                  """
counter = {'f':0, 'g':0}
g = CostlyFunction
p = NLP(lambda x: g(f(x)), x0=zeros(N), maxIter=1e4, ftol=ftol, xtol=xtol, gtol=gtol, iprint=iprint)
print '\nwithout oofun:'
r = p.solve(solver)
print 'evals f:', counter['f'], '  evals of costly func g:', counter['g']
"""
using oofun:
-----------------------------------------------------
solver: scipy_ncg   problem: unnamed   goal: minimum
 iter    objFunVal
    0  2.228e+06
    5  4.876e+04
   10  4.953e+02
   15  4.924e+02
   20  4.901e+02
   25  4.881e+02
   30  4.862e+02
   31  4.862e+02
istop:  1000
Solver:   Time Elapsed = 1.51   CPU Time Elapsed = 1.49
objFunValue: 486.20891
evals f: 17887   evals of costly func g: 305

without oofun:
-----------------------------------------------------
solver: scipy_ncg   problem: unnamed   goal: minimum
 iter    objFunVal
    0  2.228e+06
    5  4.957e+02
   10  4.926e+02
   15  4.903e+02
   20  4.882e+02
   25  4.864e+02
   27  4.861e+02
istop:  1000
Solver:   Time Elapsed = 15.09  CPU Time Elapsed = 14.29
objFunValue: 486.07635
evals f: 13660   evals of costly func g: 13660
"""
