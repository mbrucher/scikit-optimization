"""
Example:
Let's concider the problem
15x1 + 8x2 + 80x3 -> min        (1)
subjected to
x1 + 2x2 + 3x3 <= 15              (2)
8x1 +  15x2 +  80x3 <= 80      (3)
8x1  + 80x2 + 15x3 <=150      (4)
100x1 +  10x2 + x3 >= 800     (5)
80x1 + 8x2 + 15x3 = 750         (6)
x1 + 10x2 + 100x3 = 80           (7)
x1 >= 4                                     (8)
-8 >= x2 >= -80                        (9)
"""

from numpy import *
from scikits.openopt import LP
f = array([15,8,80])
A = mat('1 2 3; 8 15 80; 8 80 15; -100 -10 -1') # numpy.ndarray is also allowed
b = [15, 80, 150, -800] # numpy.ndarray, matrix etc are also allowed
Aeq = mat('80 8 15; 1 10 100') # numpy.ndarray is also allowed
beq = (750, 80)

lb = [4, -80, -inf]
ub = [inf, -8, inf]
p = LP(f, A=A, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub)
#or p = LP(f=f, A=A, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub)
r = p.solve('cvxopt_lp') # CVXOPT must be installed
"""
Other:
r = p.solve('glpk') # CVXOPT & glpk must be installed
r = p.solve('lpSolve')#lp_solve must be installed
or using converter lp2nlp:
r = p.solve('nlp:ralg', xtol=1e-8, ftol=1e-7) # for ralg reducing xtol, ftol is usually required
r = p.solve('nlp:algencan')
r = p.solve('nlp:ipopt')
r = p.solve('nlp:scipy_slsqp')
"""
print 'objFunValue:', r.ff # should print 204.48841578
print 'x_opt:', r.xf # should print [ 9.89355041 -8.          1.5010645 ]
