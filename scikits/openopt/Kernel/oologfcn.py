class OpenOptException:
    def __init__(self,  msg):
        self.msg = msg
    def __str__(self):
        return self.msg
        #pass

def ooassert(cond, msg):
    assert cond, msg

def oowarn(msg):
    print 'OO Warning! ', msg

def ooerr(msg):
    print 'OO Error:' + msg
    raise OpenOptException(msg)

def ooPWarn(msg):
    print 'OO Warning! ', msg

def ooinfo(msg):
    print 'OO info: ', msg

def oohint(msg):
    print 'OO hint: ', msg

def oodebugmsg(p,  msg):
    if p.debug: print 'OpenOpt debug msg: ', msg
