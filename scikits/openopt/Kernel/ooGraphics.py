#from gtk import Window
from string import lower
from scikits.openopt import __version__ as ooversion
problemsWithPylab = False
try:
    import pylab
except:
    problemsWithPylab = True


from setDefaultIterFuncs import stopcase

from numpy import asfarray, log10, array,  isfinite
class Graphics:
    def __init__(self):
        self.drawFuncs = [self.oodraw]
        self.specifierStart = 'd'
        self.specifierFailed = 'x'
        self.specifierOK = 'p'
        self.specifierUndefined = 'o'
        self.specifierContinueFeasible = 'v'
        self.specifierContinueInfeasible = '>'
        self.specifierError = 's'
        self.REDUCE = 1e8
        self.axLineStyle= '-'
        self.axLineWidth= 2

        self.axMarker = ''
        self.axMarkerSize = 1
        self.markerEdgeWidth = 1
        self.axMarkerEdgeColor = 'b'
        self.axFaceColor = 'y'

        # figure updating rate, (time elapsed for graphics) / (time passed)
        self.rate = 0.5

        self.drawingInOneWindow = True#some solvers for the same problem
        #what do you want on label x?
        #self.xlabel = 'time'#case-unsensitive
        #other values: CPUTime, iter
        #ignores time, spent on figure updatings
        #iter not recomended because iterations of different solvers take
        #different time
        #cputime is unrecomended on computers with several CPU
        #because some solvers can handle different number of CPU units
        #so time is best provided no other programs consume much cputime

        self.markerSize = 12
        self.iterMarkerSize = 1
        self.plotIterMarkers = True
        #self.plotOnlyCurrentMinimum = 0


    def oodraw(self, p): #note that self is same as p.graphics
        if problemsWithPylab:
            p.warn('to use OpenOpt graphics you need pylab (Python module) installed. Turning graphics off...')
            p.plot = 0
            return

        #isNewFigure = (not isempty(lastTask) and not isequal(lastTask, {p.name p.primal.fName})) ...
            #or (~self.drawingInOneWindow and globstat.iter ==1)...
            #or ~isfield(self, 'figureHandler')...
            #or (isequal(p.env, 'matlab') and ~ishandle(self.figureHandler))

        #todo: fix me later!
        needNewFigure = not p.iter

        #lastTask={p.name p.primal.fName} % x0 can be absent
        colors = ['b', 'k', 'c', 'r', 'g']
        specifiers = ['-', ':', '-.', '--']
        pylab.ion()
        if needNewFigure:

            #self.figureHandler = pylab.figure
            self.colorCount = -1
            self.specifierCount = 0
            self.nTrajectories = 0
            self.ghandlers = []
            #self.solverNames = []

            #TODO: the condition should be handled somewhere other place, not here.
            if p.probType == 'NLSP':
                Y_LABELS = ['log10(maxResidual)']
                #pylab.plot([0], [log10(p.contol)-1.5])
            elif p.probType == 'LSP':
                Y_LABELS = ['sum(residuals^2)']
            else:
                Y_LABELS = ['objFunc(x)']

            if not (p.__isUnconstrained__() or p.solver.__isIterPointAlwaysFeasible__(p)):
                self.isMaxConstraintSubplotRequired = True
                Y_LABELS.append('log10(maxConstraint)')
            else: self.isMaxConstraintSubplotRequired = False

            if self.isMaxConstraintSubplotRequired: self.nSubPlots = 2
            else: self.nSubPlots = 1


        #creating new trajectory, if needed
        isNewTrajectory = not p.iter # FIXME
        if isNewTrajectory:
            self.colorCount += 1
            if self.drawingInOneWindow:
                if self.colorCount > len(colors) - 1 :
                    self.colorCount = 0
                    self.specifierCount += 1
                    if self.specifierCount > len(specifiers) - 1 :
                        p.warn('line types number exeeded')
                        self.specifierCount = 0

        #setting color & specifier
        #color = colors[self.colorCount]
        #specifier = specifiers[self.specifierCount]
        color = p.color
        specifier = p.specifier


        #setting xlabel, ylabel, title etc
        tx = lower(p.xlabel)
        if isNewTrajectory:

            self.nTrajectories += 1

            #win = gtk.Window()
            #win.set_name("OpenOpt " + str(oover) + ", license: BSD 2.0")
            pTitle = 'problem: ' + p.name
            if p.showGoal: pTitle += '       goal: ' + p.goal
            if self.nSubPlots>1: pylab.subplot(self.nSubPlots, 1, 1)
            p.figure = pylab.gcf()
            pylab.title(pTitle)
            p.figure.canvas.set_window_title('OpenOpt ' + ooversion)

            if tx == 'cputime':
                xlabel = 'CPU Time elapsed (without graphic output), sec'
                d_x = 0.01
            elif tx == 'time':
                xlabel = 'Time elapsed (without graphic output), sec'
                d_x = 0.01
            elif tx in ['niter',  'iter']:
                xlabel = 'iter'
                d_x = 4
            elif tx == 'nf':
                xlabel = 'Number of objective function evaluations'
                d_x = 4
            else:  p.err('unknown graphic output xlabel: "' + tx + '", should be in "time", "cputime", "iter", "nf"')
            self.nPointsPlotted = 0

            for ind in range(self.nSubPlots):
                if self.nSubPlots > 1:
                    pylab.subplot(self.nSubPlots, 1, ind+1)
                pylab.hold(1)
                pylab.grid(1)
                pylab.ylabel(Y_LABELS[ind])
            pylab.xlabel(xlabel)


        ################ getting data to plot ###############
        if p.iter>0:
            IND_start, IND_end = self.nPointsPlotted-1, p.iter+1#note: indexing from zero assumed
            #if p.isFinished: IND_end = p.iter
        else: IND_start, IND_end = 0, 1

        if p.plotOnlyCurrentMinimum:
            yy = array(p.iterValues.f[IND_start:IND_end])
            if isNewTrajectory: self.currMin = yy[0]
            k = 0
            for j in xrange(IND_start,IND_end-int(p.isFinished)): #todo: min is slow in 1x1 comparison vs if-then-else
                yy[k] = min(self.currMin, p.iterValues.f[j])
                self.currMin = yy[k]
                k += 1
        else: yy = array(p.iterValues.f[IND_start:IND_end])

        if tx == 'iter': xx = range(IND_start, IND_end)
        elif tx == 'cputime':
            if len(p.iterTime) != len(p.cpuTimeElapsedForPlotting): p.iterTime.append(p.iterTime[-1])
            xx = asfarray(p.iterCPUTime[IND_start:IND_end]) - asfarray(p.cpuTimeElapsedForPlotting[IND_start:IND_end])
        elif tx == 'time':
            if len(p.iterTime) != len(p.timeElapsedForPlotting): p.iterTime.append(p.iterTime[-1])
            xx = asfarray(p.iterTime[IND_start:IND_end]) - asfarray(p.timeElapsedForPlotting[IND_start:IND_end])
        elif tx == 'nf':
            xx = asfarray(p.iterValues.nf[IND_start:IND_end])
        else: p.err('unknown labelX case')

        if len(xx)>len(yy):
            if p.isFinished: xx = xx[:-1]
            else:  p.err('OpenOpt graphics ERROR - FIXME!')

        if p.probType == 'NLSP':
            yy = log10(yy+p.ftol/self.REDUCE)
        YY = [yy]

        if self.isMaxConstraintSubplotRequired:
            YY.append(log10(p.contol/self.REDUCE+asfarray(p.iterValues.r[IND_start:IND_end])))

        if needNewFigure:
            if self.nSubPlots > 1:
                pylab.subplot(2, 1, 2)
                pylab.plot([xx[0]],[log10(p.contol)-1.5])
                pylab.plot([xx[0]+d_x],[log10(p.contol)-1.5])
                pylab.plot([xx[0]], [YY[1][0]+1])
                pylab.plot([xx[0]+d_x], [YY[1][0]+1])
                pylab.subplot(2, 1, 1)
#            if p.probType == 'NLSP':
#                [xmin, xmax] = pylab.xlim()
#                pylab.scatter([xx[0],  xmax], [log10(yy[0]), log10(yy[0])], s=1, c='w', faceted=False,  marker='o')
#                pylab.axhline(y=log10(p.primalConTol), linewidth = self.axLineWidth, ls = self.axLineStyle, color='g',\
#                marker = self.axMarker, ms = self.axMarkerSize, mew = self.markerEdgeWidth, mec = self.axMarkerEdgeColor, mfc = self.axFaceColor)
#                pylab.xlim([xmin, xmax])



            pylab.plot([xx[0]],[YY[0][0]])
            pylab.plot([xx[0]+d_x],[YY[0][0]])

#            if self.nSubPlots > 1:
#                for i in [1, 2]:
#                    pylab.subplot(2, 1, i)



        ##########################################
        if self.plotIterMarkers: usualMarker = 'o'
        else: usualMarker = ''

        for ind in range(self.nSubPlots):
            if self.nSubPlots > 1: pylab.subplot(self.nSubPlots,1,ind+1)
            yy2 = YY[ind]
#            if len(yy2)<len(xx):
#                if p.debug: p.warn('FIXME! - different len of xx and yy in ooGraphics')
#                yy2 = yy2.tolist()+[yy2[-1]]
            if isNewTrajectory:
                if isfinite(p.xlim[0]): pylab.plot([p.xlim[0]],  [yy2[0]],  color='w')
                if isfinite(p.xlim[1]): pylab.plot([p.xlim[1]],  [yy2[0]],  color='w')
                if ind==0:
                    if isfinite(p.ylim[0]): pylab.plot([xx[0]],  [p.ylim[0]],  color='w')
                    if isfinite(p.ylim[1]): pylab.plot([xx[0]],  [p.ylim[1]],  color='w')
                    if p.probType == 'NLSP': pylab.plot([xx[0]], [log10(p.ftol / self.REDUCE)],  color='w')
            if ind == 1:
                pylab.plot([xx[0], xx[-1]], [log10(p.primalConTol), log10(p.primalConTol)], ls = self.axLineStyle, linewidth = self.axLineWidth, color='g',\
                marker = self.axMarker, ms = self.axMarkerSize, mew = self.markerEdgeWidth, mec = self.axMarkerEdgeColor, mfc = self.axFaceColor)
            elif p.probType == 'NLSP':
                pylab.plot([xx[0], xx[-1]], [log10(p.ftol), log10(p.ftol)], ls = self.axLineStyle, linewidth = self.axLineWidth, color='g',\
                marker = self.axMarker, ms = self.axMarkerSize, mew = self.markerEdgeWidth, mec = self.axMarkerEdgeColor, mfc = self.axFaceColor)
            if isNewTrajectory:
                p2 = pylab.plot([xx[0]], [yy2[0]], color = color, marker = self.specifierStart,  markersize = self.markerSize)
                p3 = pylab.plot([xx[0], xx[0]+1e-50], [yy2[0], yy2[0]], color = color, markersize = self.markerSize)
                if p.legend == '': pylab.legend([p3[0]], [p.solver.__name__], shadow = True)
                elif type(p.legend) in (tuple, list): pylab.legend([p3[0]], p.legend, shadow = True)
                else: pylab.legend([p3[0]], [p.legend], shadow = True)
                pylab.plot(xx[1:], yy2[1:], color,  marker = usualMarker, markersize = self.markerSize/3)
            else:
                pylab.plot(xx, yy2, color + specifier, marker = usualMarker, markersize = self.iterMarkerSize)

            if p.isFinished:
                #xMin, xMax = [], []
                if p.istop<0:
                    if stopcase(p.istop) == 0: # maxTime, maxIter, maxCPUTime, maxFunEvals etc exeeded
                        if p.isFeas(p.xf):  s = self.specifierContinueFeasible
                        else: s = self.specifierContinueInfeasible
                    else: s = self.specifierFailed
                else:
                    if p.isFeasible:
                        if p.istop > 0:
                            s = self.specifierOK
                        else:# p.istop = 0
                            s = self.specifierUndefined
                    else: s = self.specifierError
                #pylab.scatter([xx[-1]], [yy2[-1]], color, marker = s, markersize = self.markerSize, style=1)
                if s == self.specifierOK: marker = (5, 1, 0)
                else: marker = s
                pylab.scatter([xx[-1]], [yy2[-1]], c=color, marker = marker, s=[150])

                #pylab.axis('auto')
                [xmin, xmax, ymin, ymax] = pylab.axis()
                if ymax - ymin > 25 * (yy2[-1] -ymin):
                    delta = 0.04 * (ymax - ymin)
                    pylab.scatter([(xmin+xmax)/2,  (xmin+xmax)/2], [ymin-delta, ymax+delta], s=1, c='w', faceted=False,  marker='o')
                    pylab.draw()
                if  ind == 0 and p.probType == 'NLSP':
                    pylab.plot([xmin, xmax], [log10(p.ftol), log10(p.ftol)],\
                        linewidth = self.axLineWidth, ls = self.axLineStyle, color='g',\
                        marker = self.axMarker, ms = self.axMarkerSize, \
                        mew = self.markerEdgeWidth, mec = self.axMarkerEdgeColor, mfc = self.axFaceColor)
                if  ind == 1:
                    pylab.plot([xmin, xmax], [log10(p.primalConTol), log10(p.primalConTol)],\
                        linewidth = self.axLineWidth, ls = self.axLineStyle, color='g',\
                        marker = self.axMarker, ms = self.axMarkerSize, \
                        mew = self.markerEdgeWidth, mec = self.axMarkerEdgeColor, mfc = self.axFaceColor)
                    pylab.subplot(self.nSubPlots, 1, 1)
                    pylab.plot([xmax], [YY[0][-1]],  color='w')



#                if p.probType == 'NLSP':
#                    pylab.axhline(y=log10(p.primalConTol),  xmin=xmin, xmax=xmax,\
#                        linewidth = self.axLineWidth, ls = self.axLineStyle, color='g',\
#                        marker = self.axMarker, ms = self.axMarkerSize, \
#                        mew = self.markerEdgeWidth, mec = self.axMarkerEdgeColor, mfc = self.axFaceColor)


#        if p.isFinished:
#            for ind in range(self.nSubPlots):
#                if self.nSubPlots>1: pylab.subplot(self.nSubPlots,1,ind+1)
                #pylab.xlim(min(xMin), max(xMax))
        self.nPointsPlotted = p.iter+1
        pylab.draw()


