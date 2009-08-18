from numpy import cos, arange, ones, asarray, zeros, mat, array
from scikits.openopt import NLP, oofun


def test(complexity=0, **kwargs):
    n = 15 * (complexity+1)

    x0 = 15*cos(arange(n)) + 8

    f = lambda x: ((x-15)**2).sum()
    df = lambda x: 2*(x-15)

    c1 = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]


    def dc2(x):
        r = zeros((2, n))
        r[0,3] = 2 * 4 * x[3]**3
        r[1,4] = 2 * x[4]
        r[1,5] = 2 * x[5]
        return r

    #c2 = oofun(lambda x: [2* x[3] **4-32, x[4]**2+x[5]**2 - 8], d = dc2)
    c2 = oofun(lambda x: [2* x[3] **4-32, x[4]**2+x[5]**2 - 8], dep = [3, 4, 5])
    #c2 = oofun(lambda x: [2* x[3] **4-32, x[4]**2+x[5]**2 - 8])

    c = [c1, c2]

    hp = 2
    h1 = lambda x: 1e2*(x[-1]-13)**hp
    h2 = lambda x: (x[-2]-17)**hp
    h = [h1, h2]

    # dh(x)/dx: non-lin eq constraints gradients (optional):
    def dh(x):
        r = zeros((2, n))
        r[0, -1] = 1e2*hp*(x[-1]-13)**(hp-1)
        r[1, -2] = hp*(x[-2]-17)**(hp-1)
        return r

    lb = -8*ones(n)
    ub = 15*ones(n)+8*cos(arange(n))

    ind = 3

    A = zeros((2, n))
    A[0, ind:ind+2] = 1
    A[1, ind+2:ind+4] = 1
    b = [15,  8]

    Aeq = zeros(n)
    Aeq[ind+4:ind+8] = 1
    beq = 45
    ########################################################
    colors = ['b', 'k', 'y', 'g', 'r']
    #solvers = ['ipopt', 'ralg','scipy_cobyla']
    solvers = ['ralg','ipopt','scipy_slsqp']
    solvers = [ 'ralg','scipy_slsqp']
    solvers = [ 'ralg']
#    solvers = ['ipopt']
    ########################################################
    for i, solver in enumerate(solvers):
        p = NLP(f, x0, df=df, c=c, h=h, dh=dh, A=A, b=b, Aeq=Aeq, beq=beq, maxIter = 1e3, \
                show = solver==solvers[-1], color=colors[i],  **kwargs )

        if not kwargs.has_key('iprint'): p.iprint = -1
#        p.checkdf()
#        p.checkdc()
#        p.checkdh()
        r = p.solve(solver)
    if r.istop>0: return True, r, p
    else: return False, r, p

if __name__ == '__main__':
    res, r, p = test(iprint= 100, plot=0, ftol = 1e-7)
