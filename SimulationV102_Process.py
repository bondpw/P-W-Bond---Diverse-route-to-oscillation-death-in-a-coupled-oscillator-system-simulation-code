import numpy as np
import copy as cp
import math as ma
from SimulationV102_DefUpdate import point as po
import pandas as pd

class Process:

    def __init__(self,step,iters,Situation):
        self.step = step
        self.iters = iters
        self.Situation = Situation
        
    def movement (self):

        xcElapse = []
        xrElapse = []
        TimeElapse = []

        for i in range(self.iters):

            self.Situation.update(self.step)

            dataxc = cp.deepcopy(self.Situation.xc)
            dataxr = cp.deepcopy(self.Situation.xr)
            dataTime = i*self.step

            xcElapse.append(dataxc)               
            xrElapse.append(dataxr)
            TimeElapse.append(dataTime)

        np.save("xrData",xrElapse)
        np.save("xcData",xcElapse)
        np.save("timeData",TimeElapse)
