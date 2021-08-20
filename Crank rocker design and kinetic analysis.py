"""
Title: Exam 2
    
Purpose: Design 4 bar linkage and measure necessary torque to run it.

Created on Thu Nov 19 09:19:56 2020

author: Robert Salati and Adam Wickenheiser
"""

# Modules
import math as m
import sympy as sy
import scipy as sci
from scipy.optimize import root_scalar
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# Code Start

# -----------------------------------------------------------------------------
#                           Designing Crank-rocker                           
# -----------------------------------------------------------------------------

def check_Grashof(a,b,c,d):
    # Prints "Grashof","non-Grashof", or "Special case Grashof"
    # based on link lengths a,b,c,d (in any order)
    
    S = min(a,b,c,d)
    L = max(a,b,c,d)
    SL = S + L
    PQ = a + b + c + d - SL
    if SL < PQ:
        print('Grashof')
    elif SL == PQ:
        print('Special case Grashof')
    else:
        print('non-Grashof')
        
# 2-position crank rocker synthesis

# givens
z_O4 = 0 + 0j                                       # Ground link for rocker
tht4 = -22.5*np.pi/180                              # Starting angle for rocker
phi = 45*np.pi/180                                  # Angle of sweep for rocker

# R4 from R2
R2 = 0.25/2                                         # Length of crank
R4 = R2/np.sin(phi/2)                               # Length of rocker
M = R4*(np.exp(1j*(phi+tht4))-np.exp(1j*tht4))      # Vector between both positions of rocker

# compute position of crank-ground joint
R3 = 0.5                                            # Length of coupler
K = (R2+R3)/np.abs(M)                               # Ratio
z_B1 = z_O4 + R4*np.exp(1j*tht4)                    # Vector from origin to end of position 1
z_O2 = z_B1 + K*M                                   # ground link of crank
z1 = z_O4-z_O2


# Print statements
print('Parameters for Crank-Rocker:')
print('Length of rocker connection point:',round(R4,3))
print('Total length of rocker:',round(R4*2,3))
print('Crank ground link:',round(np.real(z_O2),3),'+',round(np.imag(z_O2),3),'j')
check_Grashof(np.abs(z_O2-z_O4), R2, R3, R4)


# -----------------------------------------------------------------------------
#                              Kinematic Analysis                              
# -----------------------------------------------------------------------------

# Given Constants:
b = 0.025                                           # Height of bars [m]
h = 0.025                                           # Width of bars (into screen) [m]
rho = 7800                                          # Density of steel [kg/m^3]
theta20 = 148.89*np.pi/180                          # initial angle of crank [rad]
w = 0.25*2*np.pi                                    # Angular velocity of crank [rad/s]
alpha2 = 0                                          # Angular acceleration of crank [rad/s^2]
link_config = 'open'
z1=-np.abs(z1)

# Derived Constants: 
m2 = b*h*R2*rho                                     # Mass of crank [kg]
I2g = 825.20*10**-6                                 # Moment of inertia of crank about CoM [kg*m^2]
I2o4 = I2g+m2*(R2/2)**2                             # Moment of inertia of crank about rotation point [kg*m^2]

m3 = b*h*R3*rho                                     # Mass of coupler [kg]
I3g = 50908.20*10**-6                               # Moment of inertia of coupler about CoM [kg*m^2]

m4 = b*h*R4*2*rho                                   # Mass of rocker [kg]
I4g = 113804.85*10**-6                              # Moment of inertia of rocker about CoM [kg*m^2]
I4o2 = I4g+m4*R4**2                                 # Moment of inertia of rocker about rotation point [kg*m^2] 

# set up arrays for simulation
t = np.linspace(0,2*np.pi/w,1001)                   # time array [s]
tht2 = w*t + theta20                                # crank angle array [rad]
tht3 = np.zeros_like(t)                             # coupler angle array [rad]
tht4 = np.zeros_like(t)                             # rocker angle array [rad]

# choose which solution of tht4 to look for
if link_config == 'open':
    bracket = [0,np.pi]
    x0 = np.pi/2
else:
    bracket = [-np.pi,0]
    x0 = -np.pi/2

# function whose root we want to find within the bracket
def calc_tht4(tht4_guess):
    z4 = R4*np.exp(1j*tht4_guess)
    z5 = z1 + z4
    z3 = z5 - z2
    r = np.abs(z3)
    return r - R3

for i in range(t.size):
    z2 = R2*np.exp(1j*tht2[i])
    sol = root_scalar(calc_tht4,x0=x0,bracket=bracket)
    tht4[i] = sol.root
    z4 = R4*np.exp(1j*tht4[i])
    z5 = z1 + z4
    z3 = z5 - z2
    tht3[i] = np.angle(z3)
    
# Converting back to standard coordinates
tht2 = tht2 - 58.89*np.pi/180                       # new theta 2 array
tht3 = tht3 - 58.89*np.pi/180                       # new theta 3 array
tht4 = tht4 - 58.89*np.pi/180                       # new theta 4 array

# positions of A and B
z_B = R4*np.exp(1j*tht4)                            # position of coupler-rocker attachment
z_A = z_B+R3*np.exp(1j*tht3)                        # position of end of crank-coupler attachment

# positions of centers of gravity
z_g2 = z_O2 + R2/2*np.exp(1j*tht2)                  # CoM for crank
z_g3 = z_B + R3/2*np.exp(1j*tht3)                   # CoM for coupler
z_g4 = z_B                                          # CoM for rocker

# other positions
z_p = z_O4+R4*2*np.exp(1j*tht4)                     # point p on rocker

# Splines:

# Crank
x_g2_spline = interpolate.splrep(t,np.real(z_g2))   # spline of point g2 x-position array
y_g2_spline = interpolate.splrep(t,np.imag(z_g2))   # spline of point g2 y-position array 
tht2_spline = interpolate.splrep(t,tht2)            # spline of point theta 2 array
Vx_g2 = interpolate.splev(t,x_g2_spline,der=1)      # spline of G2 x velocity
Vy_g2 = interpolate.splev(t,y_g2_spline,der=1)      # spline of G2 y velocity
Ax_g2 = interpolate.splev(t,x_g2_spline,der=2)      # spline of G2 x acceleration
Ay_g2 = interpolate.splev(t,y_g2_spline,der=2)      # spline of G2 y acceleration
w_2 = interpolate.splev(t,tht2_spline,der=1)        # spline of crank angular velocity
al_2 = interpolate.splev(t,tht2_spline,der=2)       # spline of crank angular acceleration

# coupler
x_g3_spline = interpolate.splrep(t,np.real(z_g3))   # spline of point g3 x-position array
y_g3_spline = interpolate.splrep(t,np.imag(z_g3))   # spline of point g3 y-position array
tht3_spline = interpolate.splrep(t,tht3)            # spline of point theta 3 array
Vx_g3 = interpolate.splev(t,x_g3_spline,der=1)      # spline of G3 x velocity
Vy_g3 = interpolate.splev(t,y_g3_spline,der=1)      # spline of G3 y velocity
Ax_g3 = interpolate.splev(t,x_g3_spline,der=2)      # spline of G3 x acceleration
Ay_g3 = interpolate.splev(t,y_g3_spline,der=2)      # spline of G3 y acceleration
w_3 = interpolate.splev(t,tht3_spline,der=1)        # spline of coupler angular velocity
al_3 = interpolate.splev(t,tht3_spline,der=2)       # spline of coupler angular acceleration

# Rocker
x_g4_spline = interpolate.splrep(t,np.real(z_g4))   # spline of point g4 x-position array
y_g4_spline = interpolate.splrep(t,np.imag(z_g4))   # spline of point g4 y-position array
x_p_spline = interpolate.splrep(t,np.real(z_p))     # spline of point P4 x-position array
y_p_spline = interpolate.splrep(t,np.imag(z_p))     # spline of point P4 y-position array
tht4_spline = interpolate.splrep(t,tht4)            # spline of point theta 4 array
Vx_g4 = interpolate.splev(t,x_g4_spline,der=1)      # spline of G4 x velocity
Vy_g4 = interpolate.splev(t,y_g4_spline,der=1)      # spline of G4 y velocity
Ax_g4 = interpolate.splev(t,x_g4_spline,der=2)      # spline of G4 x acceleration
Ay_g4 = interpolate.splev(t,y_g4_spline,der=2)      # spline of G4 y acceleration
Vx_p = interpolate.splev(t,x_p_spline,der=1)        # spline of p x velocity
Vy_p = interpolate.splev(t,y_p_spline,der=1)        # spline of p y velocity
Ax_p = interpolate.splev(t,x_p_spline,der=2)        # spline of p x acceleration
Ay_p = interpolate.splev(t,y_p_spline,der=2)        # spline of p y acceleration
w_4 = interpolate.splev(t,tht4_spline,der=1)        # spline of rocker angular velocity
al_4 = interpolate.splev(t,tht4_spline,der=2)       # spline of rocker angular acceleration

# -----------------------------------------------------------------------------
#                               Kinetic Analysis                               
# -----------------------------------------------------------------------------

T12d = []                                           # empty array for T12 (downstroke)
T12u = []                                           # empty array for T12 (upstroke)
td = []                                             # empty array for t (downstroke)
tu = []                                             # empty array for t (upstroke)
F = [100,-50]                                       # f values for downstroke and upstroke respectively
for i in range(len(t)):
    if Vy_p[i] < 0:                                 # downstroke
        t12 = (I3g*al_3[i]*w_3[i]+I4o2*al_4[i]*w_4[i]-F[0]*Vy_p[i])/w_2[i]
        T12d.append(t12)
        td.append(t[i])
    else:                                           # upstroke
        t12 = (I3g*al_3[i]*w_3[i]+I4o2*al_4[i]*w_4[i]-F[1]*Vy_p[i])/w_2[i]
        T12u.append(t12)
        tu.append(t[i])
T12avg = np.average(T12d+T12u)
T12avg = np.ones_like(t)*T12avg
print('Max torque =', round(np.max(T12d),3),'N*m')
print('Average torque =', round(T12avg[0],3),'N*m')

# -----------------------------------------------------------------------------
#                                   Plotting                                   
# -----------------------------------------------------------------------------

# positions
_, ax = plt.subplots()
ax.plot(np.real(z_g2),np.imag(z_g2),label='CG2')
ax.plot(np.real(z_g3),np.imag(z_g3),label='CG3')
ax.plot(np.real(z_g4),np.imag(z_g4),label='CG4')
ax.plot(np.real(z_A),np.imag(z_A),label='A')
ax.plot(np.real(z_p),np.imag(z_p),label='P')
ax.set_xlabel('x position (m)')
ax.set_ylabel('y position (m)')
ax.legend()

# x velocities
_, bx = plt.subplots(4,sharex=True)
bx[0].plot(t,Vx_g2)
bx[0].set_ylabel('Vcg2x')
bx[1].plot(t,Vx_g3)
bx[1].set_ylabel('Vcg3x')
bx[2].plot(t,Vx_g4)
bx[2].set_ylabel('Vcg4x')
bx[3].plot(t,Vx_p)
bx[3].set_ylabel('Vpx')
bx[3].set_xlabel('time (s)')

# y velocities
_, cx = plt.subplots(4,sharex=True)
cx[0].plot(t,Vy_g2)
cx[0].set_ylabel('Vcg2y')
cx[1].plot(t,Vy_g3)
cx[1].set_ylabel('Vcg3y')
cx[2].plot(t,Vy_g4)
cx[2].set_ylabel('Vcg4y')
cx[3].plot(t,Vy_p)
cx[3].set_ylabel('Vpy')
cx[3].set_xlabel('time (s)')

# x accelerations
_, dx = plt.subplots(4,sharex=True)
dx[0].plot(t,Ax_g2)
dx[0].set_ylabel('Acg2x')
dx[1].plot(t,Ax_g3)
dx[1].set_ylabel('Acg3x')
dx[2].plot(t,Ax_g4)
dx[2].set_ylabel('Acg4x')
dx[3].plot(t,Ax_p)
dx[3].set_ylabel('Apx')
dx[3].set_xlabel('time (s)')

# y accelerations
_, ex = plt.subplots(4,sharex=True)
ex[0].plot(t,Ay_g2)
ex[0].set_ylabel('Acg2y')
ex[1].plot(t,Ay_g3)
ex[1].set_ylabel('Acg3y')
ex[2].plot(t,Ay_g4)
ex[2].set_ylabel('Acg4y')
ex[3].plot(t,Ay_p)
ex[3].set_ylabel('Apy')
ex[3].set_xlabel('time (s)')

# angles
_, fx = plt.subplots(3,sharex=True)
fx[0].plot(t,tht2*180/np.pi)
fx[0].set_ylabel('theta 2')
fx[1].plot(t,tht3*180/np.pi)
fx[1].set_ylabel('theta 3')
fx[2].plot(t,tht4*180/np.pi)
fx[2].set_ylabel('theta 4')
fx[2].set_xlabel('Time (s)')

# angular velocities
_, gx = plt.subplots(3,sharex=True)
gx[0].plot(t,w_2*180/np.pi)
gx[0].set_ylim(-100,100)
gx[0].set_ylabel('omega 2')
gx[1].plot(t,w_3*180/np.pi)
gx[1].set_ylabel('omega 3')
gx[2].plot(t,w_4*180/np.pi)
gx[2].set_ylabel('omega 4')
gx[2].set_xlabel('Time (s)')

# angular accelerations
_, hx = plt.subplots(3,sharex=True)
hx[0].plot(t,al_2*180/np.pi)
hx[0].set_ylabel('alpha 2')
hx[0].set_ylim(-50,50)
hx[1].plot(t,al_3*180/np.pi)
hx[1].set_ylabel('alpha 3')
hx[2].plot(t,al_4*180/np.pi)
hx[2].set_ylabel('alpha 4')
hx[2].set_xlabel('Time (s)')

# torque required:
_, ix = plt.subplots()
td.remove(td[-1])               # Removing the last data point. This is strictly for plotting purposes.                    
T12d.remove(T12d[-1])
ix.plot(td,T12d,label='Downstroke')
ix.plot(tu,T12u,label='Upstroke')
ix.plot(t,T12avg,label='Average')
ix.set_xlabel('time (s)')
ix.set_ylabel('T12 (N*m)')
ix.legend()
