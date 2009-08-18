# SAGE has some problems with Tkinter
try:
    from Tkinter import Tk, Toplevel, Button, Label, Frame, StringVar, DISABLED, ACTIVE
except:
    pass

from threading import Thread
from scikits.openopt import __version__ as ooversion
from setDefaultIterFuncs import BUTTON_ENOUGH_HAS_BEEN_PRESSED, USER_DEMAND_EXIT
from ooMisc import killThread

try:
    import pylab
except:
    pass

def manage(p, *args, **kwargs):
    # expected args are (solver, start) or (start, solver) or one of them
    p._args = args
    p._kwargs = kwargs
    start = True

    for arg in args:
        if type(arg) == str: p.solver = arg
        elif arg in (0, 1, True, False): start = arg
        else: p.err('Incorrect argument for manage()')

    if kwargs.has_key('start'): start = kwargs['start']

    if kwargs.has_key('solver'): p.solver = kwargs['solver']

    # root

    root = Tk()
    p.GUI_root = root


    # Title
    #root.wm_title('OpenOpt  ' + ooversion)


    p.GUI_buttons = {}

    """                                              Buttons                                               """

    # OpenOpt label
    Frame(root).pack(ipady=4)
    Label(root, text=' OpenOpt ' + ooversion + ' ').pack()
    Label(root, text=' Problem: ' + p.name + ' ').pack()
    Label(root, text=' Solver: ' + p.solver + ' ').pack()

    # Run
    t = StringVar()
    t.set("      Run      ")
    RunPause = Button(root, textvariable = t, command = lambda: invokeRunPause(p))
    Frame(root).pack(ipady=8)
    RunPause.pack(ipady=15)
    p.GUI_buttons['RunPause'] = RunPause
    p.statusTextVariable = t

    # Enough
    def invokeEnough():
        p.userStop = True
        p.istop = BUTTON_ENOUGH_HAS_BEEN_PRESSED
        if hasattr(p, 'stopdict'):  p.stopdict[BUTTON_ENOUGH_HAS_BEEN_PRESSED] = True
        p.msg = 'button Enough has been pressed'

        if p.state == 'paused':
            invokeRunPause(p, isEnough=True)
        else:
            RunPause.config(state=DISABLED)
            Enough.config(state=DISABLED)
    Frame(root).pack(ipady=8)
    Enough = Button(root, text = '   Enough!   ', command = invokeEnough)
    Enough.config(state=DISABLED)
    Enough.pack()
    p.GUI_buttons['Enough'] = Enough

    # Exit
    def invokeExit():
        p.userStop = True
        p.istop = USER_DEMAND_EXIT
        if hasattr(p, 'stopdict'): p.stopdict[USER_DEMAND_EXIT] = True

        # however, the message is currently unused
        # since openopt return r = None
        p.msg = 'user pressed Exit button'

        root.destroy()
    Frame(root).pack(ipady=8)
    Button(root, text="      Exit      ", command = invokeExit).pack(ipady=15)


    """                                            Start main loop                                      """
    state = 'paused'

    if start:
        Thread(target=invokeRunPause, args=(p, )).start()
    root.mainloop()


    """                                              Handle result                                       """

    if hasattr(p, 'tmp_result'):
        r = p.tmp_result
        delattr(p, 'tmp_result')
    else:
        r = None


    """                                                    Return                                           """
    return r

def invokeRunPause(p, isEnough=False):
    if isEnough:
        p.GUI_buttons['RunPause'].config(state=DISABLED)

    if p.state == 'init':
        p.probThread = Thread(target=doCalculations, args=(p, ))
        p.state = 'running'
        p.statusTextVariable.set('    Pause    ')
        p.GUI_buttons['Enough'].config(state=ACTIVE)
        p.GUI_root.update_idletasks()
        p.probThread.start()

    elif p.state == 'running':
        p.state = 'paused'
        if p.plot: pylab.ioff()
        p.statusTextVariable.set('      Run      ')
        p.GUI_root.update_idletasks()

    elif p.state == 'paused':
        p.state = 'running'
        if p.plot:
            pylab.ion()
        p.statusTextVariable.set('    Pause    ')
        p.GUI_root.update_idletasks()

def doCalculations(p):
    try:
        p.tmp_result = p.solve(*p._args, **p._kwargs)
    except killThread:
        if p.plot:
            if hasattr(p, 'figure'):
                #p.figure.canvas.draw_drawable = lambda: None
                pylab.ioff()
                pylab.close('all')


