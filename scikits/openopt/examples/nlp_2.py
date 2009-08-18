from scikits.openopt import NLP
from numpy import cos, arange, ones, asarray, abs, zeros
N = 30
M = 5
ff = lambda x: ((x-M)**2).sum()
p = NLP(ff, cos(arange(N)))
p.df =  lambda x: 2*(x-M)
p.c = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]

def DC(x):
    r = zeros((2, p.n))
    r[0,0] = 2 * 4 * x[0]**3
    r[1,1] = 2 * x[1]
    r[1,2] = 2 * x[2]
    return r    
p.dc = DC

h1 = lambda x: 1e1*(x[-1]-1)**4
h2 = lambda x: (x[-2]-1.5)**4
p.h = (h1, h2)

def DH(x):
    r = zeros((2, p.n))
    r[0,-1] = 1e1*4*(x[-1]-1)**3
    r[1,-2] = 4*(x[-2]-1.5)**3
    return r
p.dh = DH

p.lb = -6*ones(p.n)
p.ub = 6*ones(p.n)
p.lb[3] = 5.5
p.ub[4] = 4.5

r = p.solve('lincher')
#r = p.solve('algencan')
#!! fmin_cobyla can't use user-supplied gradient 
#r = p.solve('scipy_cobyla')

print 'objfunc val:', r.ff

