"""
oofun (OpenOpt Function) is alternative approach to classic numerical optimization problems syntax (like nlp_1.py).
Requires openopt v > 0.18 (latest tarball or svn is ok).

Classic syntax was formed in accordance with Fortran or C langueges,
while Python with object-oriented features (and lots of other benefits)
allows more powerful and convenient style.

oofun provides much powerful capabilities for writing optimization programs:
 - Good separate handling of deeply merged recursive funcs: F(G(H(...(Z(x)))))
        The situation is rather common for engineering problems: for example,
        Z(x) is mass of spoke,
        H(Z, other data) is mass of wheel,
        G(H, other data) is mass of bicycle
        you can provide derivative for some of the funcs from F, G, H, ...
        and other will be calculated via finite-difference approximation
        and automatically form derivatives for their superposition
 - Preventing of same code blocks recalculating
        For example, H() can be used in some non-linear constraints, objFun or their derivatives
 - Reducing func calls via informing what are block inputs
        for example, if you have H(z) = a*z**2 + b*z you have to obtain only 2 calls
        H(z) and H(z+dz), no matter how great is number of all optimization variables for the problem involved
        Normally you have to call H(x+dx_i) nVars times, because H=H(Z)=H(Z(x)), so even providing sparse pattern
        isn't helpful here.
 - etc

I call this style "all-included", because all available info is stored in single place.
Fields to be added to oofun in future: convex(true/false), unimodal(true/false), d2 (2nd derivatives) etc.
Changing something in function (or turn it on/off in prob instance) doesn't require
immidiate fix for lots of other files that calculate derivatives, supply dependency patterns etc,
you can just temporary turn it off and openopt will use finite-difference derivatives obtaining
until you'll provide updated info.
And, there are no such ugly constructing of derivatives like
for <...>
    r[last_ind1:last_ind1+4,last_ind2:last_ind2+5] = <...>
all they are gathered automatically.

Note also oofun has mechanism of preventing recalculating assigned funcs
for twice (i.e. for same x as called before)

I intend to continue develop of oofun class, there are still many other ideas to be implemented
(named outputs, fixed variables (that are absent in OO classic style yet as well),
inner dependency patterns of input-output (currently it's a 1-dim python list only),
oovar (OpenOpt Variable), etc)
Some of the ideas are researched by some of our optimization department workers
using Visual C++ and Rational Rose.

Also, some ideas similar to my intentions for openopt oofun you can view at
http://control.ee.ethz.ch/~joloef/yalmip.php
yalmip is free optimization toolbox that translates yalmip scripts to MATLAB.

"""
