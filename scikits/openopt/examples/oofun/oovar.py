from scikits.openopt import NLP,  oofun,  oovar

v0 = oovar('v0', [-4.0, -15.0]) # 2nd arg (if provided) is start value
"""
1st arg is name
r.ff will be dictionary with keys = these names
and values = optim point coords

alternatively you can set
v0 = oovar('my_var_0', size = 2)
or
v0 = oovar('my_var_0', lb=[-15, -80])
and size will be inhereted from lb (or ub)

setting i-th coord of the oovar start value (if that one is absent)
will be performed via rule:
coords i of lb and ub are
finite   finite -> (lb[i] + ub[i]) / 2
-inf      finite -> ub[i]
finite   +inf  -> lb[i]
-inf      +inf -> 0
"""
v1 = oovar('v1', [1, 2], ub=[0.1, 0.2]) # ub is upper bound (optional)
v2 = oovar('v2', (3, 4), lb=[-2, 0.0], ub=[-1.5, 1]) # lb is lower bound (optional)
# also, Python tuple or numpy ndarray or matrix can be used instead of Python list

f0 = oofun(lambda z: z[0]**2 + z[0]  + 2*z[1]**2 , input = v0)
"""However, indexing (like above) is not recommended.
Try to create separate oovars instead:
v0_1 = oovar(-4, lb=...), v0_2 = oovar(-15, ub=...)
f1 = oofun(lambda y, z: y**2 + y  + 2*z , input = v0_1, v0_2)
"""

f1 = oofun(lambda z: (z-1).sum() ** 2, input = v1)
f2 = oofun(lambda z: z.sum() ** 2, input = v2)
f3 = oofun(lambda y, z, t: y.sum() ** 2 + 2*z.sum() + 4 * t, input = [f1, f2, f0])

p = NLP(f3, gtol=1e-7)
r = p.solve('ralg')

print 'solution:', r.xf
print 'optim value:', r.ff
"""
solution: {'v0': array([ -5.00002645e-01,   3.52068631e-06]), 'v1': array([ 0.10000075,  0.20000047]), 'v2': array([-1.50000237,  0.9999997 ])}
optim value: 7.85208138965
"""






