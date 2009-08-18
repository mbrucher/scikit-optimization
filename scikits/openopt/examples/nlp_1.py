"""
Example:
(x0-5)^2 + (x2-5)^2 + ... +(x149-5)^2 -> min

subjected to

# lb<= x <= ub:
x4 <= 4
8 <= x5 <= 15

# Ax <= b
x0+...+x149 >= 825
x9 + x19 <= 3
x10+x11 <= 9

# Aeq x = beq
x100+x101 = 11

# c(x) <= 0
2*x0^4-32 <= 0
x1^2+x2^2-8 <= 0

# h(x) = 0
(x[149]-1)**6 = 0
(x[148]-1.5)**6 = 0
"""


from scikits.openopt import NLP

from numpy import cos, arange, ones, asarray, zeros, mat, array
N = 150

# 1st arg - objective function
# 2nd arg - x0
p = NLP(lambda x: ((x-5)**2).sum(), 8*cos(arange(N)), iprint = 50, maxIter = 1e3)

# f(x) gradient (optional):
p.df = lambda x: 2*(x-5)

# c(x) <= 0 constraints
p.c = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]

# dc(x)/dx: non-lin ineq constraints gradients (optional):
def DC(x):
    r = zeros((len(p.c(p.x0)), p.n))
    r[0,0] = 2 * 4 * x[0]**3
    r[1,1] = 2 * x[1]
    r[1,2] = 2 * x[2]
    return r
p.dc = DC

# h(x) = 0 constraints
h1 = lambda x: (x[149]-1)**6
h2 = lambda x: (x[148]-1.5)**6
p.h = [h1, h2]

### dh(x)/dx: non-lin eq constraints gradients (optional):
def DH(x):
    r = zeros((2, p.n))
    r[0, -1] = 6*(x[149]-1)**5
    r[1, -2] = 6*(x[148]-1.5)**5
    return r
p.dh = DH

p.lb = -6*ones(p.n)
p.ub = 6*ones(p.n)
p.ub[4] = 4
p.lb[5], p.ub[5] = 8, 15

p.A = zeros((3, p.n))
p.A[0, 9] = 1
p.A[0, 19] = 1
p.A[1, 10:12] = 1
p.A[2] = -ones(p.n)
p.b = [7, 9, -825]


p.Aeq = zeros(p.n)
p.Aeq[100:102] = 1
p.beq = 11

##p.ftol = 1e-4# one of stop criteria, default 1e-6
##p.xtol = 1e-5# one of stop criteria, default 1e-6

p.contol = 1e-3 # required constraints tolerance, default for NLP is 1e-6

# ALGENCAN solver ignores xtol and ftol; using maxTime, maxCPUTime, maxIter, maxFunEvals, fEnough is recommended.

# Note that in algencan gradtol means norm of projected gradient of  the Augmented Lagrangian
# so it should be something like 1e-3...1e-5
p.gtol = 1e-7 # (default gtol = 1e-6)

##p.debug = 1

#optional: user-supplied 1st derivatives check
p.checkdf()
p.checkdc()
p.checkdh()


p.maxIter = 10000

#optional: graphic output, requires pylab (matplotlib)
p.plot = 1
p.maxFunEvals = 1e7

r = p.solve('ralg')

# r.xf and r.ff are optim point and optim objFun value
# r.ff should be something like 128.08949
