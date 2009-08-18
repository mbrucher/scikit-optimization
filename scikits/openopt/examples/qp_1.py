"""
Example:
Concider the problem
0.5 * (x1^2 + 2x2^2 + 3x3^2) + 15x1 + 8x2 + 80x3 -> min        (1)
subjected to
x1 + 2x2 + 3x3 <= 150            (2)
8x1 +  15x2 +  80x3 <= 800    (3)
x2 - x3 = 25                              (4)
x1 <= 15                                  (5)
"""

from numpy import diag, matrix, inf
from scikits.openopt import QP
p = QP(diag([1,2,3]), [15,8,80], A = matrix('1 2 3; 8 15 80'), b = [150, 800], Aeq = [0, 1, -1], beq = 25, ub = [15,inf,inf])
# or p = QP(H=diag([1,2,3]), f=[15,8,80], A = matrix('1 2 3; 8 15 80'), b = [150, 800], Aeq = [0, 1, -1], beq = 25, ub = [15,inf,inf])
r = p.solve('cvxopt_qp', iprint = 0)
# r = p.solve('nlp:ipopt'), r = p.solve('nlp:algencan')
f_opt, x_opt = r.ff, r.xf
# x_opt = array([-14.99999995,  -2.59999996, -27.59999991])
# f_opt = -1191.90000013
