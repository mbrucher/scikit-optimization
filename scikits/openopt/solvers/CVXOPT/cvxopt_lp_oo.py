from scikits.openopt.Kernel.BaseAlg import BaseAlg
from CVXOPT_LP_Solver import CVXOPT_LP_Solver

class cvxopt_lp(BaseAlg):
    __name__ = 'cvxopt_lp'
    __license__ = "LGPL"
    __authors__ = "http://abel.ee.ucla.edu/cvxopt"
    __alg__ = "see http://abel.ee.ucla.edu/cvxopt"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    def __init__(self): pass
    def __solver__(self, p):
        return CVXOPT_LP_Solver(p, 'native_CVXOPT_LP_Solver')
##if __name__ == '__main__':
##    a =cvxopt_lp ()

