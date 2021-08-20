"""
Title: Machine Design Homework 7
    
Purpose: Analysis of pistons and 4 bar linkages

Created on Tue Oct 27 23:59:06 2020

author: Robert Salati
"""

#Modules

from scipy.optimize import root_scalar
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt


#Code Start

#==============================================================================
#                                  Problem 1
#==============================================================================

# geometric constants
O2 = 0 + 0j
O4 = 2.22 + 0j
z1 = O4 - O2                         # ground link vector
R1 = np.abs(z1)                      # ground link length
R2 = 0.86                            # crank length [m]
R3 = 1.85                            # coupler length [m]
R4 = 1.35                            # rocker length [m]
Rap = 1.85                           # distance from A to P [m]
link_config = 'open'                 # 'open' or 'crossed'
b = 0.025                            # width of bars [m]
h = 0.055                            # height of bars [m]

# Initial conditions
w20 = 0                               # initial angular velocity of crank [rad/s]
alpha2 = 0                            # angular acceleration of crank [rad/s^2]
theta20 = 0                           # intial angle of crank [rad]

# physical constants
g = 9.81                             # gravity acceleration [m/s^2]            
f = 50                               # force applied to link 3 [N]
rho = 7800                           # density of steel [kg/m^3]

# derived constants
# crank
m2 = R2*b*h*rho                      # m2 value for crank [kg]
Ig2 = 0.57                           # mass moment of inertia of [kg*m^2]
CG2 = R2/2                           # distance from O2 of center of mass [in]

# coupler
m3 = R3*b*h*rho                      # m3 value for coupler [kg]
Ig3 = 5.66                           # mass moment of inertia of coupler [kg*m^2]
CG3 = R3/2                           # distance from A of center of mass [m]

# rocker
m4  = R4*b*h*rho                     # m4 value for [lbf]
Ig4 = 2.20                           # mass moment of inertia of [lb-in-sec^2]
CG4 = R4/2                           # distance from O4 of center of mass [m]

# derived constants
Ia2 = Ig2 + CG2*m2                   # moment of inertia of A around O2

# set up arrays for simulation
t = np.linspace(0,5,5001)            # time array [s]
w = w20 + 8*(1-np.exp(-t))            # angular velocity array [rad/s]
tht2 = theta20 + w*t + alpha2*t**2 # crank angle array [rad]
tht3 = np.zeros_like(t)              # coupler angle array [rad]
tht4 = np.zeros_like(t)              # rocker angle array [rad]

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
    
# positions of A and B
z_A = R2*np.exp(1j*tht2)+O2                      # position of end of coupler
z_B = z_A + R3*np.exp(1j*tht3)                   # position of end of crank

# positions of centers of gravity
z_g2 = O2 + CG2*np.exp(1j*tht2)                  # CoM for crank
z_g3 = z_A + CG3*np.exp(1j*tht3)                 # CoM for coupler
z_g4 = O4 + CG4*np.exp(1j*(tht4))   # CoM for rocker

# other positions
z_p4 = z_A + Rap*np.exp(1j*(tht3))

# spline curve fits
x_g2_spline = interpolate.splrep(t,np.real(z_g2))   # spline of point g2 x-position array
y_g2_spline = interpolate.splrep(t,np.imag(z_g2))   # spline of point g2 y-position array

x_g3_spline = interpolate.splrep(t,np.real(z_g3))   # spline of point g3 x-position array
y_g3_spline = interpolate.splrep(t,np.imag(z_g3))   # spline of point g3 y-position array

x_g4_spline = interpolate.splrep(t,np.real(z_g4))   # spline of point g4 x-position array
y_g4_spline = interpolate.splrep(t,np.imag(z_g4))   # spline of point g4 y-position array

x_p4_spline = interpolate.splrep(t,np.real(z_p4))   # spline of point P4 x-position array
y_p4_spline = interpolate.splrep(t,np.imag(z_p4))   # spline of point P4 y-position array

tht2_spline = interpolate.splrep(t,tht2)            # spline of point theta 2 array
tht3_spline = interpolate.splrep(t,tht3)            # spline of point theta 3 array
tht4_spline = interpolate.splrep(t,tht4)            # spline of point theta 4 array


# computing velocities and accelerations

# Crank 
V_g2 = interpolate.splev(t,x_g2_spline,der=1) + interpolate.splev(t,y_g2_spline,der=1)*1j
A_g2 = interpolate.splev(t,x_g2_spline,der=2) + interpolate.splev(t,y_g2_spline,der=2)*1j
w_2 = interpolate.splev(t,tht2_spline,der=1)
al_2 = interpolate.splev(t,tht2_spline,der=2)

# coupler
V_g3 = interpolate.splev(t,x_g3_spline,der=1) + interpolate.splev(t,y_g3_spline,der=1)*1j
A_g3 = interpolate.splev(t,x_g3_spline,der=2) + interpolate.splev(t,y_g3_spline,der=2)*1j
w_3 = interpolate.splev(t,tht3_spline,der=1)
al_3 = interpolate.splev(t,tht3_spline,der=2)

# Rocker
V_g4 = interpolate.splev(t,x_g4_spline,der=1) + interpolate.splev(t,y_g4_spline,der=1)*1j
A_g4 = interpolate.splev(t,x_g4_spline,der=2) + interpolate.splev(t,y_g4_spline,der=2)*1j
V_p4 = interpolate.splev(t,x_p4_spline,der=1) + interpolate.splev(t,y_p4_spline,der=1)*1j
A_p4 = interpolate.splev(t,x_p4_spline,der=2) + interpolate.splev(t,y_p4_spline,der=2)*1j
w_4 = interpolate.splev(t,tht4_spline,der=1)
al_4 = interpolate.splev(t,tht4_spline,der=2)

# plotting:

# theta
_, ax = plt.subplots(3,sharex=True)
ax[0].plot(t,tht2*180/np.pi)
ax[0].set_ylabel('theta 2 (deg)')
ax[1].plot(t,tht3*180/np.pi)
ax[1].set_ylabel('theta 3 (deg)')
ax[2].plot(t,tht4*180/np.pi)
ax[2].set_ylabel('theta 4 (deg)')
ax[2].set_xlabel('Time (s)')

# alpha
_, bx = plt.subplots(3,sharex=True)
bx[0].plot(t,al_2*180/np.pi)
bx[0].set_ylabel('alpha 2 (deg/s^2)')
bx[1].plot(t,al_3*180/np.pi)
bx[1].set_ylabel('alpha 3 (deg/s^2)')
bx[2].plot(t,al_4*180/np.pi)
bx[2].set_ylabel('alpha 4 (deg/s^2)')
bx[2].set_xlabel('Time (s)')


# x and y components
Ag2x = np.real(A_g2)
Ag2y = np.imag(A_g2)
Ag3x = np.real(A_g3)
Ag3y = np.imag(A_g3)
Ag4x = np.real(A_g4)
Ag4y = np.imag(A_g4)

# relative positions
Rcg2o2 = z_g2 - O2                              # vector from CG2 to O2
Rao2 = z_A - O2                                 # vector from point A to O2
Racg3 = z_g3 - z_A                              # vector from point A to CG3
Rbcg3 = z_B - z_g3                              # vector from CG3 to point B
Rbcg4 = z_g4 - z_B                              # vector from point B to CG4
Rcg4o4 = z_g4 - O4                              # vector from CG4 to O4
Rpcg4 = z_g4-z_p4                               # vector from CG4 to P

Rcg2o2x = abs(np.real(Rcg2o2))
Rcg2o2y = abs(np.imag(Rcg2o2))
Rao2x = abs(np.real(Rao2))
Rao2y = abs(np.imag(Rao2))
Racg3x = abs(np.real(Racg3))
Racg3y = abs(np.imag(Racg3))
Rbcg3x = abs(np.real(Rbcg3))
Rbcg3y = abs(np.imag(Rbcg3))
Rbcg4x = abs(np.real(Rbcg4))
Rbcg4y = abs(np.imag(Rbcg4))
Rcg4o4x = abs(np.real(Rcg4o4))
Rcg4o4y = abs(np.imag(Rcg4o4))
Rpcg4x = abs(np.real(Rpcg4))
Rpcg4y = abs(np.imag(Rpcg4))


#------------------------------kinetic analysis--------------------------------

Fo2 = []
Fa = []
Fb = []
Fo4 = []
t12 = []


for i in range(len(t)):
    A = np.array([
            [1,0,1,0,0,0,0,0,0],
            [0,-1,0,-1,0,0,0,0,0],
            [0,0,Rao2y[i],-Rao2x[i],0,0,0,0,1],
            [0,0,1,0,1,0,0,0,0],
            [0,0,0,-1,0,-1,0,0,0],
            [0,0,Racg3y[i],Racg3x[i],-Rbcg3y[i],-Rbcg3x[i],0,0,0],
            [0,0,0,0,1,0,-1,0,0],
            [0,0,0,0,0,-1,0,1,0],
            [0,0,0,0,-Rbcg4y[i],Rbcg4x[i],-Rcg4o4y[i],Rcg4o4x[i],0]
            ])
    
    C = np.array([
            [m2*Ag2x[i]],
            [m2*Ag2y[i]+m2*g],
            [Ia2*al_2[i]+m2*g*Rcg2o2x[i]],
            [m3*Ag3x[i]],
            [m3*Ag3y[i]+m3*g],
            [Ig3*al_3[i]+f*Rpcg4y[i]],
            [m4*Ag4x[i]],
            [m4*Ag4y[i]+m4*g],
            [Ig4*al_4[i]],
            ])

    F = np.linalg.solve(A,C)

    FO2, F12Angle = F[0]+1j*F[1], np.angle(F[0]+1j*F[1])*180/np.pi
    FA, F23Angle = F[2]+1j*F[3], np.angle(F[2]+1j*F[3])*180/np.pi
    FB, F34Angle = F[4]+1j*F[5], np.angle(F[4]+1j*F[5])*180/np.pi
    FO4, F14Angle = F[6]+1j*F[7], np.angle(F[6]+1j*F[7])*180/np.pi
    T12 = np.abs(F[8])
    Fo2.append(FO2)
    Fa.append(FA)
    Fb.append(FB)
    Fo4.append(FO4)
    t12.append(T12)


# plotting:
    
# x components
_, cx = plt.subplots(2,sharex=True)
cx[0].plot(t,np.real(Fa),label='FA')
cx[0].plot(t,np.real(Fb),label='FB')
cx[0].set_ylabel("Internal Pin x Forces (N)")
cx[1].plot(t,np.real(Fo2),label='FO2')
cx[1].plot(t,np.real(Fo4),label='FO4')
cx[1].set_ylabel("Ground Link x Forces (N)")
cx[0].legend()
cx[1].legend()
cx[1].set_xlabel('Time (s)')

# y components
_, dx = plt.subplots(2,sharex=True)
dx[0].plot(t,np.imag(Fa),label='FA')
dx[0].plot(t,np.imag(Fb),label='FB')
dx[0].set_ylabel("Internal Pin y Forces (N)")
dx[1].plot(t,np.imag(Fo2),label='FO2')
dx[1].plot(t,np.imag(Fo4),label='FO4')
dx[1].set_ylabel("Ground Link y Forces (N)")
dx[0].legend()
dx[1].legend()
dx[1].set_xlabel('Time (s)')

# torque
_, ex = plt.subplots()
ex.plot(t,t12,label='T12')
ex.legend()
ex.set_xlabel('Time (s)')
ex.set_ylabel('Torque (N*m)')