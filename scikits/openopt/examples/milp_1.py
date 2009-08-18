__docformat__ = "restructuredtext en"

from numpy import *
from scikits.openopt import MILP

f = [1, 2, 3, 4, 5, 4, 2, 1]

# indexing starts from ZERO!
# while in native lpsolve-python wrapper from 1
# so if you used [5,8] for native lp_solve python binding
# you should use [4,7] instead
intVars = [4, 7]

lb = -1.5 * ones([8,1])
ub = 15 * ones([8,1])
A = zeros((5, 8))
b = zeros(5)
for i in xrange(5):
    for j in xrange(8):
        A[i,j] = -8+sin(8*i) + cos(15*j)
    b[i] = -150 + 80*sin(80*i)

p = MILP(f=f, lb=lb, ub=ub, A=A, b=b, intVars=intVars)
#r = p.solve('lpSolve')
r = p.solve('glpk')
print 'f_opt:', r.ff # 25.801450769161505
print 'x_opt:', r.xf # [ 15. 10.15072538 -1.5 -1.5 -1.  -1.5 -1.5 15.]

"""
if you have installed glpk+cvxopt 1.0 or later 
(with BUILD_GLPK=1 in setup.py file) 
you can handle MILP problems with binary constraints 
(coords x from p.binVars should be in {0, 1}):

p = MILP(f=f, lb=lb, ub=ub, A=A, b=b, intVars=intVars, binVars=[1])
#intVars, binVars indexing from ZERO!
r = p.solve('glpk') 

print 'f_opt:', r.ff # 26.566058805272387
print 'x_opt:', r.xf # [15.  1.  -1.5 -1.5 -1. -1.5 8.0330294 15.]
"""
