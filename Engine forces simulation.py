"""

Title: Piston engine code

Purpose: Simulate the forces present in a piston engine
    
Created: Fri Aug 20 

Last Updated: Fri Aug 20

Author: Robert Salati, Adam Wickenheiser

"""

# Modules
import scipy as sci
from scipy import optimize 
from scipy import interpolate
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import timeit
import cv2 as cv
start = timeit.default_timer()

# Code Start

# constants
r = 3                                                       # length of crank [in]
l = 12                                                      # length of connecting rod [in]
w0 = 200                                                    # angular velocity of crank [rad/s]

# equations 
t = np.linspace(0,2*np.pi/w0,101)                           # time array for one rotation [s]
x = l - r**2/(4*l)+r*(np.cos(w0*t)+r/(4*l)*np.cos(2*w0*t))  # array for position of the piston [in]
v = -r*w0*(np.sin(w0*t)+r/(2*l)*np.sin(2*w0*t))             # array for velocity of the piston [in/s]
a = -r*w0**2*(np.cos(w0*t)+r/l*np.cos(2*w0*t))              # array for acceleration of the piston [in/s^2]

# plotting
_, ax = plt.subplots(3,sharex=True)
ax[0].plot(t,x)
ax[0].set_ylabel('X (in)')
ax[0].grid()
ax[1].plot(t,v)
ax[1].set_ylabel('V (in/s)')
ax[1].grid()
ax[2].plot(t,a)
ax[2].set_ylabel('A (in/s^2)')
ax[2].set_xlabel('time (s)')
ax[2].grid()


#==============================================================================
#                                  Problem 3
#==============================================================================

# constants 
mp = 0.1                                                    # mass of piston [slug]
mr = 0.075                                                  # mass of connecting rod [slug]

# derived constants
m = mp + mr/3                                               # effective mass of piston [slug]

# primary inertial forces
F1 = m*(-r*w0**2*(np.cos(w0*t)))                            # array of primary forces [lbf]

# secondary inertial forces
F2 = m*(-r*w0**2*(r/l*np.cos(2*w0*t)))                      # array of secndary forces [lbf]

# plotting
_, cx = plt.subplots()
cx.plot(t,F1,label='Primary force')
cx.plot(t,F2,label='Secondary force')
cx.set_xlabel('time (s)')
cx.set_ylabel('Force (lbf)')
cx.grid()
cx.legend()



#==============================================================================
#                                  Problem 4
#==============================================================================

# using same masses as above

phase = np.pi                                       # phase lag of pistons 2 & 3

FP14 = m*(-r*w0**2*(np.cos(w0*t)))                  # array of primary forces for pistons 1 and 4 [lbf]
FS14 = m*(-r*w0**2*(r/l*np.cos(2*w0*t)))           # array of secondary forces for pistons 1 and 4 [lbf]
FP23 = m*(-r*w0**2*(np.cos(w0*t-phase)))            # array of primary forces for pistons 2 and 3 [lbf]
FS23 = m*(-r*w0**2*(r/l*np.cos(2*(w0*t-phase))))    # array of secondary forces for pistons 2 and 3 [lbf]

FP = 2*FP14+2*FP23                                  # sum of primary forces [lbf]
FS = 2*FS14+2*FS23                                  # sum of secondary forces [lbf]

# plotting
_, dx = plt.subplots()
dx.plot(t,FP,label='Primary force')
dx.plot(t,FS,label='Secondary force')
dx.set_xlabel('time (s)')
dx.set_ylabel('Force (lbf)')
dx.grid()
dx.legend()

_, ex = plt.subplots()
ex.plot(t,FP14,label='Primary force')
ex.plot(t,FS14,label='Secondary force')
ex.set_xlabel('time (s)')
ex.set_ylabel('Force (lbf)')
ex.grid()
ex.legend()

_, fx = plt.subplots()
fx.plot(t,FP23,label='Primary force')
fx.plot(t,FS23,label='Secondary force')
fx.set_xlabel('time (s)')
fx.set_ylabel('Force (lbf)')
fx.grid()
fx.legend()

# Code End

stop = timeit.default_timer()
print('Runtime:', round(stop-start,3), 'seconds')
