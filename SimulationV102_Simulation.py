from SimulationV102_Process import Process as pro 
from SimulationV102_DefUpdate import point as up
import math as ma

situation = up(1,1,1,1,1,1.3,1,2,1.5,1)

forward = pro(0.01,10000,situation)
forward.movement()