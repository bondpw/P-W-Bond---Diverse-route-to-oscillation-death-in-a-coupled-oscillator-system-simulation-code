import numpy as np
import math as ma
import copy as cp 

class point: 
    def __init__(self,xc,yc,xr,yr,alpha,epsilon,omegac,omegar,ac,ar):
        
        '''x component of the cardiac function'''
        self.xc = xc
        '''y component of the cardiac function'''
        self.yc = yc
        '''x component of the respiratory function'''
        self.xr = xr
        '''y component of the respiratory function'''
        self.yr = yr
        '''This is the amplitude of the noncoupled part of the oscillator'''
        self.alpha = alpha
        '''This is the coupling strength of the oscillator'''
        self.epsilon = epsilon
        '''This is the cardiac frequency'''
        self.omegac = omegac
        '''This is the respiratory frequency'''
        self.omegar = omegar
        '''ac and ar are the variable that shows the linear part of the non coupled oscillator'''
        self.ac = ac
        self.ar = ar
     
        
    '''This is deifning the parameters as either a constant or a value that can be edited'''
    def __repr__(self):
        return 'xc{0},yc{1},xr{2},yr{3},alpha{4:12.3e},epsilon{5:12.3e},omegac{6:12.3e},omegar{7:12.3e},ac{8:12.3e},ar{9:12.3e}'.format(self.xc,self.yr,self.xr,self.yr,self.alpha,self.epsilon,self.omegac,self.omegar,self.ac,self.ar)

    def update(self,step):
            '''First step of Runge-Kutta'''
            xck1 = step*(-self.xc*self.alpha*(ma.sqrt((self.xc)**2 + (self.yc)**2) - self.ac) - (self.yc*self.omegac) + self.epsilon*(self.xc + self.xr))
            yck1 = step*(-self.yc*self.alpha*(ma.sqrt((self.xc)**2 + (self.yc)**2) - self.ac) + (self.xc*self.omegac))
            xrk1 = step*(-self.xr*self.alpha*(ma.sqrt((self.xr)**2 + (self.yr)**2) - self.ar) - (self.yr*self.omegar) + self.epsilon*(self.xc + self.xr))
            yrk1 = step*(-self.yr*self.alpha*(ma.sqrt((self.xr)**2 + (self.yr)**2) - self.ar) + (self.xr*self.omegac))
            '''Second step of Runge-Kutta'''
            xck2 = step*(-(self.xc + (xck1/2))*self.alpha*(ma.sqrt((self.xc + (xck1/2))**2 + (self.yc + (yck1/2))**2) - self.ac) - ((self.yc + (yck1/2))*self.omegac) + self.epsilon*((self.xc + (xck1/2)) + (self.xr + (xrk1/2))))
            yck2 = step*(-(self.yc + (yck1/2))*self.alpha*(ma.sqrt((self.xc + (xck1/2))**2 + (self.yc + (yck1/2))**2) - self.ac) + ((self.xc + (xck1/2))*self.omegac))
            xrk2 = step*(-(self.xr + (xrk1/2))*self.alpha*(ma.sqrt((self.xr + (xrk1/2))**2 + (self.yr + (yrk1/2))**2) - self.ar) - ((self.yr + (yrk1/2))*self.omegar) + self.epsilon*((self.xc + (xck1/2)) + (self.xr + (xrk1/2))))
            yrk2 = step*(-(self.yr + (yrk1/2))*self.alpha*(ma.sqrt((self.xr + (xrk1/2))**2 + (self.yr + (yrk1/2))**2) - self.ar) + ((self.xr + (xrk1/2))*self.omegac))
            '''Third step of Runge-Kutta'''
            xck3 = step*(-(self.xc + (xck2/2))*self.alpha*(ma.sqrt((self.xc + (xck2/2))**2 + (self.yc + (yck2/2))**2) - self.ac) - ((self.yc + (yck2/2))*self.omegac) + self.epsilon*((self.xc + (xck2/2)) + (self.xr + (xrk2/2))))
            yck3 = step*(-(self.yc + (yck2/2))*self.alpha*(ma.sqrt((self.xc + (xck2/2))**2 + (self.yc + (yck2/2))**2) - self.ac) + ((self.xc + (xck2/2))*self.omegac))
            xrk3 = step*(-(self.xr + (xrk2/2))*self.alpha*(ma.sqrt((self.xr + (xrk2/2))**2 + (self.yr + (yrk2/2))**2) - self.ar) - ((self.yr + (yrk2/2))*self.omegar) + self.epsilon*((self.xc + (xck2/2)) + (self.xr + (xrk2/2))))
            yrk3 = step*(-(self.yr + (yrk2/2))*self.alpha*(ma.sqrt((self.xr + (xrk2/2))**2 + (self.yr + (yrk2/2))**2) - self.ar) + ((self.xr + (xrk2/2))*self.omegac))
            '''Fourth step of Runge-Kutta'''
            xck4 = step*(-(self.xc + (xck3))*self.alpha*(ma.sqrt((self.xc + (xck3))**2 + (self.yc + (yck3))**2) - self.ac) - ((self.yc + (yck3))*self.omegac) + self.epsilon*((self.xc + (xck3)) + (self.xr + (xrk3))))
            yck4 = step*(-(self.yc + (yck3))*self.alpha*(ma.sqrt((self.xc + (xck3))**2 + (self.yc + (yck3))**2) - self.ac) + ((self.xc + (xck3))*self.omegac))
            xrk4 = step*(-(self.xr + (xrk3))*self.alpha*(ma.sqrt((self.xr + (xrk3))**2 + (self.yr + (yrk3))**2) - self.ar) - ((self.yr + (yrk3))*self.omegar) + self.epsilon*((self.xc + (xck3)) + (self.xr + (xrk3))))
            yrk4 = step*(-(self.yr + (yrk3))*self.alpha*(ma.sqrt((self.xr + (xrk3))**2 + (self.yr + (yrk2))**2) - self.ar) + ((self.xr + (xrk3))*self.omegac))
            '''Updating the respective variable based on the fourth order Runge-Kutta method'''
            self.xc = self.xc + ((1/6)*(xck1 + (2*xck2) + (2*xck3) + xck4))
            self.yc = self.yc + ((1/6)*(yck1 + (2*yck2) + (2*yck3) + yck4))
            self.xr = self.xr + ((1/6)*(xrk1 + (2*xrk2) + (2*xrk3) + xrk4))
            self.yr = self.yr + ((1/6)*(yrk1 + (2*yrk2) + (2*yrk3) + yrk4))

   