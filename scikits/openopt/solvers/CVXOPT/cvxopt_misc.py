##try:
    #cvxopt_solvers =  __import__('cvxopt.solvers') 
    #cvxopt_base =  __import__('cvxopt.base')
import cvxopt.base
matrix = cvxopt.base.matrix
sparse = cvxopt.base.sparse
##except:
##    raise  'error: cvxopt is absent'

from numpy import asarray

def Matrix(x):
    if x == None or x.size == 0:
        return None
    else:
        x = asarray(x)
        #float - to avoid integer devision
        if x[x == 0].size > 0.7*x.size and x.ndim > 1: #todo: replace 0.7 by prob param
            return sparse(x.tolist()).T # without tolist currently it doesn't work
        else:  return matrix(x)
