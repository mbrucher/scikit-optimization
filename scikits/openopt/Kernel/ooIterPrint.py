from string import rjust
from numpy import atleast_1d, asfarray, log10
def signOfFeasible(p):
    r = '-'
    if p.isFeas(p.xk): r = '+'
    return r

textOutputDict = {\
'objFunVal': lambda p: '%0.3e' % p.iterValues.f[-1], \
'log10(maxResidual)': lambda p: '%0.2f' % log10(p.iterValues.r[-1]+1e-100), \
'isFeasible': signOfFeasible
}
delimiter = '   '

class ooTextOutput:
    def __init__(self):
        pass

    def iterPrint(self):

        if self.lastPrintedIter == self.iter: return

        if self.iter == 0 and self.iprint >= 0: # 0th iter (start)
            print ' iter' + delimiter,
            for fn in self.data4TextOutput:
                print fn + delimiter,
            print
        elif self.iprint<0 or \
        (((self.iprint>0 and self.iter % self.iprint != 0) or self.iprint==0)  and not(self.isFinished or self.iter == 0)):
            return

        s = rjust(str(self.iter), 5) + '  '
        for columnName in self.data4TextOutput:
            val = textOutputDict[columnName](self)
            #nWhole = length(columnName)
            s += rjust(val, len(columnName)) + ' '
        print s
        self.lastPrintedIter = self.iter
