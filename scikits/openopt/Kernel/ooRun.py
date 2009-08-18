def ooRun(prob, solvers, opts = None):
    r = []
    for solver in solvers:
        r.append(runProbSolver(prob.copy(), solver, opts))
    return r