#! /usr/bin/env python

from info import __version__
from oo import LP, NLP, NSP, MILP, QP, NLSP, LSP, GLP, LLSP,  MMP, LLAVP
from Kernel.Function import oofun, oolin
from Kernel.ooVar import oovar
from Kernel.GUI import manage
from Kernel.oologfcn import OpenOptException
from Kernel.nonOptMisc import oosolver

#__all__ = filter(lambda s:not s.startswith('_'),dir())

#from numpy.testing import NumpyTest
#test = NumpyTest().test


