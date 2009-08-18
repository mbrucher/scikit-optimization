"""
Example for oolin - linear oofun

For the given input vector z it returns dot(f, z) + c 
(by default c=0)

Note! For oolin NaN * 0 had been implemented as 0.
For example oolin([0, 0, 1]) with x = [nan, 10, 15] will yield 15
"""
from scikits.openopt import *
from numpy import mat
v0 = oovar('v0', [-4, -15]) # 2nd arg (if provided) is start value
v1 = oovar('v1', [1, 2], ub=[0.1, 0.2]) # ub is upper bound (optional)
v2 = oovar('v2', (3, 4), lb=[-2, 0], ub=[-1.5, 1]) # lb is lower bound (optional)

f0 = oofun(lambda z: z[0]**2 + z[0]  + 2*z[1]**2 , input = v0)
f1 = oofun(lambda z: (z-1).sum() ** 2, input = v1)
f2 = oofun(lambda z: z.sum() ** 2, input = v2)


"""
f3 returns z0 + 2 * z1 + 3 * z2
this is same to 
f3 = oofun(lambda z0, z1, z2: z0 + 2*z1 + 3*z2] , input = [f0, f1, f2], d = lambda x: [1, 2, 3])
"""
f3 = oolin([1, 2, 3], input=[f0, f1, f2])

"""
f4 returns 15 * f0 + 8 * f3 + 80 * f2 + 15
this is same to
f4 = oofun(lambda z0, z1, z2: 15*f0 + 8*f3 + 80*f2 + 15] , input = [f0, f4, f2], d = lambda x: [15, 8, 80])
"""
f4 = oolin([15, 8, 80], 15, input=[f0, f3, f2]) 

"""
f5 returns matrix_C * vector_input + vector_d
We will use it as 3 non-linear inequalities C * input + d <= 0 
(input is non-linear, hence the constraints are non-linear):
z1 + 2*z2 + 4*z3 - 10 <= 0
4*z1 + 5*z2 + 8*z3 - 20 <= 0
7*z1 + 8*z2 + 10*z3 - 30 <= 0
Of course, we could use f5 somewhere as a part of (recursive) calculations for obtaining objFunc
"""
C = mat('1 2 4; 4 5 8; 7 8 10') # numpy.array or array-like object is OK as well
d = [-10, -20, -30] # numpy.array or array-like object is OK as well
f5 = oolin(C, d, input=[f2, f1, f0])

# objective function:
F = oolin([1, 1, 1], 15, input = [f2, f3, f4]) # returns f2 + f3 + f4 + 15

# assign prob:
p = NLP(F, c=f5) # Note - it already has some lb <= x <= ub constraints rased from v1, v2
 
# solve:
r = p.solve('ralg')
#r = p.solve('scipy_slsqp')

print 'solution:', r.xf
print 'optim value:', r.ff
"""
solution: {'v0': array([ -5.00012873e-01,   8.20344763e-05]), 'v1': array([ 0.10000018,  0.19999917]), 'v2': array([-1.49999962,  1.00000076])}
optim value: 103.019917653
"""
