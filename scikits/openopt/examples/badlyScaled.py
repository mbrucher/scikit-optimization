from numpy import *
from scikits.openopt import *

coeff = 1e-8

f = lambda x: (x[0]-20)**2+(coeff * x[1] - 80)**2 # objFun
c = lambda x: (x[0]-14)**2-1 # non-lin ineq constraint(s) c(x) <= 0
# for the problem involved: f_opt =25, x_opt = [15.0, 8.0e9]

x0 = [-4,4]
# even modification of stop criteria can't help to achieve the desired solution:
someModifiedStopCriteria = {'gradtol': 1e-15,  'ftol': 1e-15,  'xtol': 1e-15}

# using default diffInt = 1e-7 is inappropriate:
p = NLP(f, x0, c=c, **someModifiedStopCriteria)
r = p.solve('ralg')
print r.ff,  r.xf #  will print something like "6424.9999886000014 [ 15.0000005   4.       ]"
"""
 for to improve the solution we will use
 changing either p.diffInt from default 1e-7 to [1e-7,  1]
 or p.scale from default None to [1,  1e-7]

 latter (using p.scale) is more recommended
 because it affects xtol for those solvers
 who use OO stop criteria
 (ralg, lincher, nsmm, nssolve and mb some others)
  xtol will be compared to scaled x shift:
 is || (x[k] - x[k-1]) * scale || < xtol

 You can define scale and diffInt as
 numpy arrays, matrices, Python lists, tuples
 """
p = NLP(f, x0, c=c, scale = [1,  coeff],  **someModifiedStopCriteria)
r = p.solve('ralg')
print r.ff,  r.xf # "24.999996490694787 [  1.50000004e+01   8.00004473e+09]" - much better
"""
Full Output:
starting solver ralg (license: BSD)  with problem  unnamed
itn 0 : Fk= 6975.9999935999995 MaxResidual= 323.0
itn 10  Fk: 6424.9985147662055 MaxResidual: 2.96e-04 ls: 5
itn 20  Fk: 6424.9999835226936 MaxResidual: 2.02e-06 ls: 4
itn 30  Fk: 6424.9999885998468 MaxResidual: 1.00e-06 ls: 5
itn 40  Fk: 6424.999988599995 MaxResidual: 1.00e-06 ls: 5
itn 50  Fk: 6424.9999886000005 MaxResidual: 1.00e-06 ls: 78
itn 51  Fk: 6424.9999886000014 MaxResidual: 1.00e-06 ls: 0
ralg has finished solving the problem unnamed
istop:  2 (|| gradient F(X[k]) || < gradtol)
Solver:   Time Elapsed = 0.54 	CPU Time Elapsed = 0.39
objFunValue: 6424.9999886000014 (feasible, max constraint =  1e-06)
6424.9999886000014 [ 15.0000005   4.       ]
starting solver ralg (license: BSD)  with problem  unnamed
itn 0 : Fk= 6975.9999935999995 MaxResidual= 323.0
itn 10  Fk: 6424.9985147649186 MaxResidual: 2.96e-04 ls: 5
itn 20  Fk: 6424.9999824449724 MaxResidual: 1.80e-06 ls: 4
itn 30  Fk: 6424.9959805950612 MaxResidual: 1.00e-06 ls: 99
itn 40  Fk: 25.121367939538644 MaxResidual: 0.00e+00 ls: 1
itn 50  Fk: 25.000287679235381 MaxResidual: 0.00e+00 ls: -1
itn 60  Fk: 24.999999424995089 MaxResidual: 1.47e-07 ls: 1
itn 62  Fk: 24.999996226675954 MaxResidual: 7.95e-07 ls: -1
ralg has finished solving the problem unnamed
istop:  2 (|| gradient F(X[k]) || < gradtol)
Solver:   Time Elapsed = 1.33 	CPU Time Elapsed = 1.07
objFunValue: 24.999996489689014 (feasible, max constraint =  7.42082e-07)
24.999996489689014 [  1.50000004e+01   8.00004473e+09]
"""
