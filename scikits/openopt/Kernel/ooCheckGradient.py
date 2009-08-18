from numpy import hstack, ceil, floor, log10, inf, tile, argmax, abs, asfarray, ravel, any
from string import rjust, ljust
class autocreate:
    def __init__(self):pass


#def ooCheckGradient(p, fun_, xCheck=None, separator='========================'):
def ooCheckGradient(p, fun_, *args, **kwargs):
    """
    fun_ is string name like 'df', 'dc', 'dh'
    for to avoid problems with scaling, the func must be called before p.solve() has been executed.
    lately I intend to fix the problem. // D.
    """
    if not hasattr(p,  fun_):
        p.warn('you must provide gradient for check ' + fun_+'. Turning the option off')
        return
#
#    x0 = asfarray(p.x0)
#    p.x0 = asfarray(p.x0)
#
    if len(args)>0:
        if len(args)>1 or kwargs.has_key('x'):
            p.err('checkd<func> funcs can have single argument x only (then x should be absent in kwargs )')
        xCheck = asfarray(args[0])
    elif kwargs.has_key('x'):
        xCheck = asfarray(kwargs['x'])
    else:
        xCheck = asfarray(p.x0)

    if kwargs.has_key('maxViolation'):
        setattr(p, 'maxViolation', kwargs['maxViolation'])

    separator='========================'

    if p.isObjFunValueASingleNumber: singleColumn, doubleColumn = ['df'] , ['dc', 'dh']
    else: singleColumn, doubleColumn = [], ['df', 'dc', 'dh']


    genericUserFunc1 = getattr(p, fun_) # df, dc, dh
    genericUserFunc2 = getattr(p, fun_[1:]) # f, c, h

#    p.nEvals = {}
#    p.nEvals[fun_[1:]] = 0
#    p.nEvals[fun_] = 0

    #for genericUserFunc in [genericUserFunc1, genericUserFunc2]:
#    for fn in [fun_[1:], fun_]:
#        genericUserFunc = getattr(p, fn)
#        if type(genericUserFunc) in [list, tuple]:
#            setattr(p.user, fn, genericUserFunc)
#        else:
#            setattr(p.user, fn, [genericUserFunc])
#        setattr(p, fn, getattr(p, 'user_' + fn))

    p.__prepare__()

    #p.diffInt = ravel(p.diffInt)

    #p.__makeCorrectArgs__()

    setattr(p.userProvided, fun_, False)
    info_numerical = getattr(p, fun_)(xCheck, ind=None, ignorePrev=True)
    setattr(p.userProvided, fun_, True)

    info_user = getattr(p, fun_)(xCheck, ignorePrev=True)

    if info_numerical.shape != info_user.shape:
        p.err('user-supplied gradient ' + fun_ + ' has other size than the one, obtained numerically: '+ \
        str(info_numerical.shape) + ' expected, ' + str(info_user.shape) + ' obtained')
        return inf

    #S = abs(info_user) +abs(info_numerical) + 1e-15
    #Diff = abs(info_user-info_numerical) / S
    Diff = 1 - (info_user+1e-8)/(info_numerical + 1e-8) # 1e-8 to suppress zeros
    log10_RD = log10(abs(Diff)/p.maxViolation+1e-150)

    d = hstack((info_user.reshape(-1,1), info_numerical.reshape(-1,1), Diff.reshape(-1,1)))

    #nskiplines = sum(abs(Diff.flatten()) < p.check.maxViolation)

    print('OpenOpt checks user-supplied gradient ' + fun_ + ' (shape: ' + str(info_user.shape) + ' )')
    print('according to:')
    print('    prob.diffInt = ' + str(p.diffInt))#TODO: ADD other parameters: allowed epsilon, maxDiffLines etc
    print('    |1 - info_user/info_numerical| <= prob.maxViolation = '+ str(p.maxViolation))

    if any(abs(Diff) >= p.maxViolation):
        ss = '    '
        if fun_ in doubleColumn:
            ss = ' i,j:' + fun_ + '[i]/dx[j]'

        s = fun_ + ' num  ' + ss + '   user-supplied     numerical               RD'
        print(s)

    ns = ceil(log10(d.shape[0]))
    counter = 0
    fl_info_user = info_user.flatten()
    fl_info_numerical = info_numerical.flatten()
    if len(Diff.shape) == 1:
        Diff = Diff.reshape(-1,1)
        log10_RD = log10_RD.reshape(-1,1)
    for i in xrange(Diff.shape[0]):
        for j in xrange(Diff.shape[1]):
            if abs(Diff[i,j]) < p.maxViolation: continue
            counter += 1
            k = Diff.shape[1]*i+j
            nSpaces = ns - floor(log10(k+1))+2
            if fun_ in doubleColumn:  ss = str(i) + ' / ' + str(j)
            else: ss = ''

            if len(Diff.shape) == 1 or Diff.shape[1] == 1: n2 = 0
            else: n2 = 15
            s = '    ' + ljust('%d' % k,5) + rjust(ss, n2) + rjust('%+0.3e' % fl_info_user[k],19) + rjust('%+0.3e' % fl_info_numerical[k], 15) + rjust('%d' % int(ceil(log10_RD[i,j])), 15)
            print(s)

    diff_d = abs(d[:,0]-d[:,1])
    ind_max = argmax(diff_d)
    val_max = diff_d[ind_max]
    if any(abs(Diff) >= p.maxViolation):
        print('max(abs('  + fun_ + '_user - ' + fun_ + '_numerical)) = ' + str(val_max))
        print('(is registered in '+ fun_+ ' number ' + str(ind_max) + ')')
    else:
        print('derivatives are equal')
    #print('sum(abs('  + fun_ + '_user - ' + fun_ + '_numerical)) = ' + str(p.norm(d[:,2],1)))

    print(separator)

    #delattr(p.user, fun_)
    #delattr(p.user, fun_[1:])
    #setattr(p, fun_, genericUserFunc1) # df, dc, dh
    #setattr(p, fun_[1:], genericUserFunc2) # f, c, h
    #p.x0 = asfarray(x0)
    p.nEvals[fun_[1:]] = 0
    p.nEvals[fun_] = 0

    #return counter




