from numpy import diag, array, sqrt,  ones, inf, any, copy, zeros, dot, where, all, tile, sum, nan, isfinite, float64, isnan, log10, max, sign, array_equal
from numpy.linalg import norm
try:
    from numpy.linalg import cond
except:
    print 'warning: no cond in numpy.linalg, matrix B rejuvenation check will be omitted'
    cond = lambda Matrix: 1

from scikits.openopt.Kernel.BaseAlg import *
from scikits.openopt.Kernel.Point import Point
from scikits.openopt.Kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F,  SMALL_DF,  IS_LINE_SEARCH_FAILED
from UkrOptMisc import getBestPointAfterTurn

class ralg(BaseAlg):
    __name__ = 'ralg'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "Naum Z. Shor R-algorithm with adaptive space dilation & some modifications"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    __iterfcnConnected__ = True

    #ralg default parameters
    alp, h0, nh, q1, q2  = 2.0, 1.0, 3, 'default:0.9 for NLP, 1.0 for NSP', 1.1
    hmult = 0.5
    S = 0
    T = float64
    dilationType = 'auto'

    showLS = False
    show_hs = False
    showRej = False
    showRes = False
    show_nnan = False
    doBackwardSearch = 1
    ls_0 = 3.3
    j_multiplier = [1.0, 2.0, 3.0]

    def needRej(self, p, b, g, g_dilated):
        return 1e15 * p.norm(g_dilated) < p.norm(g)
    #checkTurnByGradient = True

    def __init__(self): pass
    def __solver__(self, p):

        alp, h0, nh, q1, q2 = self.alp, self.h0, self.nh, self.q1, self.q2

        if type(q1) == str:
            if p.probType== 'NLP' and p.isUC: q1 = 0.9
            else: q1 = 1.0
        T = self.T
        # alternatively instead of alp=self.alp etc you can use directly self.alp etc



        n = p.n
        b = diag(ones(n,  T))
#        B_f = diag(ones(n))
#        B_constr = diag(ones(n))
        hs = T(h0)
        ls_arr = []
        w = T(1.0/alp-1.0)
        #best_feas_objfunc_value = inf

        """                            Shor r-alg engine                           """
        prevIterPoint = p.point(atleast_1d(T(copy(p.x0))))
        bestPoint = prevIterPoint

        g = prevIterPoint.__getDirection__()
        moveDirection = g
        if not any(g) and all(isfinite(g)):
            # TODO: create ENUMs
            if prevIterPoint.isFeas():
                p.istop = 14
            else:
                p.istop = -14

            p.msg = 'move direction has all-zero coords'
            return

        p.hs = [hs]
#        #pass-by-ref! not copy!
#        if p.isFeas(p.x0): b = B_f
#        else: b = B_constr

        """                           Ralg main cycle                                    """

        for itn in xrange(1500000):
            doDilation = True

            #ls1 = 0
            # TODO: is (g^T b)^T better?
            g_tmp = self.__economyMult__(b.T, moveDirection)
            if any(g_tmp): g_tmp /= p.norm(g_tmp)
            g1 = p.matmult(b, g_tmp)


            """                           Forward line search                          """

            x = prevIterPoint.x.copy()
            prevPrevPoint = prevIterPoint
            hs_cumsum = 0
            for ls in xrange(p.maxLineSearch):
                if ls > 20:
                    hs *= 2.0
                elif ls > 10:
                    hs *= 1.5
                elif ls > 2:
                    hs *= 1.05

                x -= hs * g1#dotwise
                hs_cumsum += hs

                newPoint = p.point(x)
                if self.show_nnan: p.info('ls: %d nnan: %d' % (ls, newPoint.__nnan__()))

                if ls == 0:
                    oldPoint = prevIterPoint
                elif ls >= 2:
                    newPoint._lin_ineq = prevIterPoint.lin_ineq() + hs_cumsum / (hs_cumsum - hs) * (oldPoint.lin_ineq() - prevIterPoint.lin_ineq())
                    newPoint._lin_eq = prevIterPoint.lin_eq() + hs_cumsum / (hs_cumsum - hs) * (oldPoint.lin_eq() - prevIterPoint.lin_eq())


                #if not self.checkTurnByGradient:

                if newPoint.betterThan(oldPoint, altLinInEq=True):
                    if newPoint.betterThan(bestPoint): bestPoint = newPoint
                    if ls !=0: prevPrevPoint = oldPoint
                    oldPoint, newPoint = newPoint,  None
                else:
                    break

            if ls == p.maxLineSearch-1:
                p.istop,  p.msg = IS_LINE_SEARCH_FAILED,  'maxLineSearch (' + str(p.maxLineSearch) + ') has been exceeded'
                return

            #changes!
            if ls > 10:
                pass
                #newPoint, ls2 = getBestPointAfterTurn(prevPrevPoint, newPoint, altLinInEq=True, maxLS = 4)
            #end of changes!

            g2 = newPoint.__getDirection__() # used for dilation direction obtaining

            iterPoint  = newPoint

            """                          Backward line search                          """


            if ls == 0 and self.doBackwardSearch:
                iterPoint, ls_backward = getBestPointAfterTurn(prevIterPoint, iterPoint, altLinInEq = True)

                # TODO: extract last point from backward search, that one is better than iterPoint
                if iterPoint.betterThan(bestPoint): bestPoint = iterPoint

                hs *= 2 ** ls_backward

            """                      iterPoint has been obtained                     """

            moveDirection = iterPoint.__getDirection__()

            ls_arr.append(ls)
            if ls >= 2 and len(ls_arr) >= len(self.j_multiplier):
                j_arr = array(ls_arr[-3:]) * self.j_multiplier#array((ls_arr[-3], 2.0*ls_arr[-2],  3.0*ls_arr[-1]))
                j_arr[j_arr<0] = 0 # for more safety
                j_mean = j_arr.sum() / sum(self.j_multiplier)
                if j_mean > self.ls_0:
                    hs *= sqrt(j_mean - self.ls_0 + 1.0)
                else:
                    hs *= sqrt(j_mean / self.ls_0)

            if itn == 0:
                p.debugmsg('hs: ' + str(hs))
                p.debugmsg('ls: ' + str(ls))
            if self.showLS: p.info('ls: ' + str(ls))
            if self.show_hs: p.info('hs: ' + str(hs))
            if self.show_nnan: p.info('nnan: ' + str(iterPoint.__nnan__()))
            if self.showRes:
                r, fname, ind = iterPoint.mr(True)
                p.info(fname+str(ind))

            """                         Set dilation direction                            """

            #if sum(p.dotmult(g, g2))>0:
                #p.debugmsg('ralg warning: slope angle less than pi/2. Mb dilation for the iter will be omitted.')
                #doDilation = False

            prevIterPointIsFeasible = prevIterPoint.isFeas(altLinInEq=True)
            currIterPointIsFeasible = iterPoint.isFeas(altLinInEq=True)
            r_p, ind_p, fname_p = prevIterPoint.mr(1)
            r_, ind_, fname_ = iterPoint.mr(1)

            if self.dilationType == 'normalized' and (not fname_p in ('lb', 'ub', 'lin_eq', 'lin_ineq') or not fname_ in ('lb', 'ub', 'lin_eq', 'lin_ineq')) and (fname_p != fname_  or ind_p != ind_):
                G2,  G = g2/norm(g2), g/norm(g)
            else:
                G2,  G = g2, g

            if prevIterPointIsFeasible == currIterPointIsFeasible == True:
                g1 = G2 - G
            elif prevIterPointIsFeasible == currIterPointIsFeasible == False:
                g1 = G2 - G
            elif prevIterPointIsFeasible:
                g1 = G2.copy()
            else:
                g1 = G.copy()
                #g1 = -G.copy() # signum doesn't matter here

#            #pass-by-ref! not copy!
#            if currIterPointIsFeasible: b = B_f
#            else: b = B_constr


            """                             Perform dilation                               """

            g = self.__economyMult__(b.T, g1)
            ng = p.norm(g)
            p._df = g2.copy()

            if self.needRej(p, b, g1, g):
                if self.showRej or p.debug:
                    p.info('debug msg: matrix B restoration in ralg solver')
                b = diag(ones(n))
                hs = 0.5*p.norm(prevIterPoint.x - iterPoint.x)
            if all(isfinite(g)) and ng > 1e-50 and doDilation:
                g = (g / ng).reshape(-1,1)
                vec1 = self.__economyMult__(b, g).reshape(-1,1)
                vec2 = w * g.T
                b += p.matmult(vec1, vec2)

            """                               Call OO iterfcn                                """

            if iterPoint.isFeas(altLinInEq=True):
                if hasattr(iterPoint, '_df'):
                    p._df = iterPoint._df
            p.iterfcn(iterPoint)


            """                             Check stop criteria                           """

            cond_same_point = array_equal(iterPoint.x, prevIterPoint.x)
            if cond_same_point and not p.istop:
                p.istop = 14
                p.msg = 'X[k-1] and X[k] are same'
                p.stopdict[SMALL_DELTA_X] = True
                return

            s2 = 0
            if not p.istop and not p.userStop:
                if p.stopdict.has_key(SMALL_DF):
                    if currIterPointIsFeasible: s2 = p.istop
                    p.stopdict.pop(SMALL_DF)
                if p.stopdict.has_key(SMALL_DELTA_F):
                    if currIterPointIsFeasible: s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_F)
                if p.stopdict.has_key(SMALL_DELTA_X):
                    if currIterPointIsFeasible or not prevIterPointIsFeasible or cond_same_point: s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_X)
                if s2 and (any(isnan(iterPoint.c())) or any(isnan(iterPoint.h()))) \
                and not p.isNaNInConstraintsAllowed\
                and not cond_same_point:
                    s2 = 0
                if not s2 and any(p.stopdict.values()):
                    for key,  val in p.stopdict.iteritems():
                        if val == True:
                            s2 = key
                            break
                p.istop = s2


            """                                If stop required                                """

            if p.istop:
                if self.needRej(p, b, g1, g):
                    b = diag(ones(n))
                    hs = 0.5*p.norm(prevIterPoint.x - iterPoint.x)
                    p.istop = 0
                else:
#                    if newPoint.betterThan(bestPoint):
#                        optimIterPoint = newPoint
#                    else:
#                        optimIterPoint = bestPoint
                    #istop, msg = p.istop, p.msg
                    #if any(bestPoint.x != iterPoint.x):
                    p.iterfcn(bestPoint)
                    #p.istop, p.msg = istop, msg
                    return


            """                Some final things for ralg main cycle                """
            p.hs.append(hs)
            #g = moveDirection.copy()
            g = g2.copy()

            prevIterPoint, iterPoint = iterPoint, None

    def __economyMult__(self, M, V):
        #return dot(M, V)
        if all(V): # all v coords are non-zeros
            return dot(M, V)
        else:
            ind = where(V != 0)[0]
            r = dot(M[:,ind], V[ind])
            return r




