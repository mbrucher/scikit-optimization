
from scikits.openopt.Kernel.BaseAlg import BaseAlg
__docformat__ = "restructuredtext en"
from scikits.openopt.solvers.optimizers import optimizer, criterion, line_search, step

class Function: 
    def __init__(self): pass # python 2.5 can use just class Function: pass
    
class nlpSolver1(BaseAlg):
    def __init__(self):
        self.__name__ = 'nlpSolver1'
        self.__license__ = "BSD"
        self.__authors__ = "Matthieu Brucher <matthieu.brucher@gmail.com>"
        self.__alg__ = "unknown"  
        
        self.__constraints__ = []
        #Matthieu, I will implement the field some time later, this one shows which constraints can handle the solver: 'A', 'Aeq', 'c', 'h', 'lb', 'ub'
        #openopt will automatically check can solver handle problem
    
    def __solver__(self, p):
        
        p.xk = p.x0
        p.fk = p.f(p.x0)
        
        p.iterfcn()
        if p.istop:
            p.xf, p.ff = p.xk, p,fk
            return 
        
        
        F = Function()
        F.__call__ = p.f
        F.gradient = p.df
        
        optimi = optimizer.StandardOptimizer(function = F, step = step.RestartNotOrthogonalConjugateGradientStep(step.FRConjugateGradientStep(), 0.1), criterion = criterion.criterion(iterations_max = p.maxIter, ftol = p.ftol), x0 = p.x0, line_search = line_search.StrongWolfePowellRule())
        xf =  optimi.optimize()
        
        p.istop = 1000
        p.xk, p.fk = xf, p.f(xf)
        p.iterfcn()
        p.xf, p.ff = p.xk, p.fk
        # you should set something negative if error(s) occure or solver failed to solve the problem
        # 0 - if situation unknown
        # or use those enum numbers from Kernel/setdefaultiterfuncs.py
        # but 1) those ones will be in openopt (i.e. from openopt import SMALL_DF etc or import *)
        # 2) it's better to connect iterfcn and use native openopt stop criteria
        
        
        
        
