"""
Solving system of equations:
x[0]**3+x[1]**3-9 = 0
x[0]-0.5*x[1] = 0
cos(x[2])+x[0]-1.5 = 0
with some constraints:
150 <= x[2] <= 158
and possible non-linear constraint:
(x[2] - 150.8)**2 <= 1.5

Note:
1. Using Ax <= b constraints is also allowed
2. You can try using equality constraints (h(x)=0, Aeq x = beq) as well.
3. Required function tolerance is p.ftol, constraints tolerance is p.contol,
and hence using h(x)=0 constraints is not 100% same
to some additional f coords
"""

from scikits.openopt import NLSP
from numpy import asfarray, zeros, cos, sin

# you can define f in several ways:
f = lambda x: (x[0]**3+x[1]**3-9, x[0]-0.5*x[1], cos(x[2])+x[0]-1.5)
#f = (lambda x: x[0]**3+x[1]**3-9, lambda x: x[0]-0.5*x[1], lambda x: cos(x[2])+x[0]-1.5)
# Python list, numpy.array are allowed as well:
#f = lambda x: [x[0]**3+x[1]**3-9, x[0]-0.5*x[1], cos(x[2])+x[0]-1.5]
#or f = lambda x: asfarray((x[0]**3+x[1]**3-9, x[0]-0.5*x[1], cos(x[2])+x[0]-1.5))

#optional: gradient
def df(x):
    df = zeros((3,3))
    df[0,0] = 3*x[0]**2
    df[0,1] = 3*x[1]**2
    df[1,0] = 1
    df[1,1] = -0.5
    df[2,0] = 1
    df[2,2] = -sin(x[2])
    return df

x0 = [8,15, 80]

#w/o gradient:
#p = NLSP(f, x0)

p = NLSP(f, x0, df = df, maxFunEvals = 1e5, iprint = 10, plot=1, ftol = 1e-8, contol=1e-15)

#optional: user-supplied gradient check:
#p.checkdf()

#optional: graphical output, requires matplotlib installed
#plot doesn't work correctly for constrained NLSP yet
#p.plot = 1


p.lb[2] = 150
p.ub[2] = 158

# you could try also comment/uncomment nonlinear constraints:
p.c = lambda x: (x[2] - 150.8)**2-1.5
# optional: gradient
p.dc = lambda x: asfarray((0, 0, 2*(x[2]-150.8)))
# also you could set it via p=NLSP(f, x0, ..., c = c, dc = dc)

#optional: user-supplied dc check:
#p.checkdc()

#r = p.solve('nssolve', debug=0, maxIter=1e9)
# using nlsp2nlp converter, try to minimize sum(f_i(x)^2):
r = p.solve('nlp:ralg')

print 'solution:', r.xf
print 'max residual:', r.ff

