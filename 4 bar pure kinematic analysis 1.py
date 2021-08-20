"""
Title: Machine Design Homework 6
    
Purpose: Run kinematic and kinetic analysis on a 4 bar linkage

Created on Sun Oct 18 15:52:58 2020

author: Robert Salati and Adam Wickenheiser
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

#-----------------------------------Part A-------------------------------------

# crank position
w = 100*2*np.pi/60                   # rotation rate [rad/s]
t = np.linspace(0,2*np.pi/w,10000)   # time array [s]
R2 = 30                              # radius [cm]
z_A = R2*np.exp(1j*w*t)              # complex position [cm]

# slider position
R3 = 100                                    # coupler length [cm]
D = 40                                      # slider vertical offset [cm]
tht3 = np.arcsin((D-R2*np.sin(w*t))/R3)     # coupler angle [rad]
z_B = z_A + R3*np.exp(1j*tht3)              # complex position [cm]

# plot trajectory
plt.figure()
plt.plot(np.real(z_A),np.imag(z_A),label='crank')
plt.plot(np.real(z_B),np.imag(z_B),label='slider')
plt.axis('equal')
plt.legend()

# spline curve fits
x_A_spline = interpolate.splrep(t,np.real(z_A))   # spline of crank x-position array
y_A_spline = interpolate.splrep(t,np.imag(z_A))   # spline of crank y-position array
x_B_spline = interpolate.splrep(t,np.real(z_B))   # spline of slider x-position array
y_B_spline = interpolate.splrep(t,np.imag(z_B))   # spline of slider y-position array

# numerically differentiate spline fits
_, ax = plt.subplots(2,1)

ax[0].plot(t,interpolate.splev(t,x_A_spline,der=2),label='A x-acceleration')
ax[0].plot(t,interpolate.splev(t,x_B_spline,der=2),label='B x-acceleration')
ax[0].set_xlabel('Time')
ax[0].legend()

ax[1].plot(t,interpolate.splev(t,y_A_spline,der=2),label='A y-acceleration')
ax[1].plot(t,interpolate.splev(t,y_B_spline,der=2),label='B y-acceleration')
ax[1].set_xlabel('Time')
ax[1].legend()

#-----------------------------------Part B--------------0----------------------

# tangential and normal components of acceleration

V_A = interpolate.splev(t,x_A_spline,der=1) + interpolate.splev(t,y_A_spline,der=1)*1j
A_A = interpolate.splev(t,x_A_spline,der=2) + interpolate.splev(t,y_A_spline,der=2)*1j
a_At = (np.real(V_A)*np.real(A_A)+np.imag(V_A)*np.imag(A_A))/np.abs(V_A)
A_At = a_At*np.exp(1j*np.angle(V_A))
A_An = A_A - A_At
a_An = np.abs(A_An)

plt.figure()
plt.plot(t,a_At,label='Point A tangential acceleration')
plt.plot(t,a_An,label='Point A normal acceleration')
plt.ylabel('Acceleration [cm/s^2]')
plt.xlabel('Time [s]')
plt.legend()

#-----------------------------------Part C-------------------------------------

# point in the middle of link 3

z_C = z_A + R3/2*np.exp(1j*tht3)
x_C_spline = interpolate.splrep(t,np.real(z_C))   # spline of crank x-position array
y_C_spline = interpolate.splrep(t,np.imag(z_C))   # spline of crank y-position array

_, bx = plt.subplots()

bx.plot(t,interpolate.splev(t,x_C_spline,der=2),label='C x-acceleration')
bx.plot(t,interpolate.splev(t,y_C_spline,der=2),label='C y-acceleration')
bx.set_xlabel('Time')
bx.legend()

#-----------------------------------Part D-------------------------------------

# unit vector along theta 3:

A_C = abs(interpolate.splev(t,x_C_spline,der=2) + interpolate.splev(t,y_C_spline,der=2)*1j)

A_Ct = A_C*np.cos(tht3)
A_Cn = A_C*np.cos(tht3-90)
_, cx = plt.subplots()
cx.plot(t, A_Ct, label='C tangential acceleration')
cx.plot(t, A_Cn, label='C normal acceleration')
cx.set_xlabel('Time')
cx.legend()


#==============================================================================
#                                   Problem 2
#==============================================================================

#-----------------------------kinematic analysis-------------------------------

# givens
R1 = 2.22                            # ground link length [m]
R2 = 1                               # crank length [m]
R3 = 2.06                            # coupler length [m]
R4 = 2.33                            # rocker length [m]
Rap = 3.06                           # distance from A to P [m]
w0 = 10                              # initial angular velocity of crank [rad/s]
al = 5                               # angular acceleration of crank [rad/s^2]
link_config = 'open'                 # 'open' or 'crossed'
z1 = R1                              # ground link

# set up arrays for simulation
t = np.linspace(0,1,1001)            # time array [s]
w = w0 + al * t
tht2 = np.zeros_like(t)+60*np.pi/180 # crank angle array [rad]
tht3 = np.zeros_like(t)              # coupler angle array [rad]
tht4 = np.zeros_like(t)              # rocker angle array [rad]

plt.figure()
plt.plot(t,w)
plt.ylabel('Crank rotation rate [rad/s]')
plt.xlabel('Time [s]')

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
    if i == 0:
        tht2[i] = 60*np.pi/180
    else:
        tht2[i] = w[i]*(t[i]-t[i-1]) + tht2[i-1]
    z2 = R2*np.exp(1j*tht2[i])
    sol = root_scalar(calc_tht4,x0=x0,bracket=bracket)
    tht4[i] = sol.root
    z4 = R4*np.exp(1j*tht4[i])
    z5 = z1 + z4
    z3 = z5 - z2
    tht3[i] = np.angle(z3)
    

# specific theta values:

print('Kinematics:')

print('Theta 2 =',round(tht2[0]*180/np.pi,3),'\nTheta 3 =',round(tht3[0]*180/np.pi,3),'\nTheta 4 =',round(tht4[0]*180/np.pi,3))

# positions of A B and P
z_A = R2*np.exp(1j*tht2)
z_B = z_A + R3*np.exp(1j*tht3)
z_P = z_A+Rap*np.exp(1j*(tht3-31*np.pi/180))
# positions of centers of gravity for links 2, 3, 4

z4 = R4*np.exp(1j*tht4)

cm = 1.6476*np.exp(1j*(tht3+(-31+12.2648)*np.pi/180))

z_g2 = R2/2*np.exp(1j*tht2)
z_g3 = z_A + cm
z_g4 = R1 + z4/2

# spline curve fits
x_g2_spline = interpolate.splrep(t,np.real(z_g2))   # spline of point g2 x-position array
y_g2_spline = interpolate.splrep(t,np.imag(z_g2))   # spline of point g2 y-position array
x_g3_spline = interpolate.splrep(t,np.real(z_g3))   # spline of point g3 x-position array
y_g3_spline = interpolate.splrep(t,np.imag(z_g3))   # spline of point g3 y-position array
x_g4_spline = interpolate.splrep(t,np.real(z_g4))   # spline of point g4 x-position array
y_g4_spline = interpolate.splrep(t,np.imag(z_g4))   # spline of point g4 y-position array


# compute tangential and normal components of acceleration of point A

V_g2 = interpolate.splev(t,x_g2_spline,der=1) + interpolate.splev(t,y_g2_spline,der=1)*1j
A_g2 = interpolate.splev(t,x_g2_spline,der=2) + interpolate.splev(t,y_g2_spline,der=2)*1j

V_g3 = interpolate.splev(t,x_g3_spline,der=1) + interpolate.splev(t,y_g3_spline,der=1)*1j
A_g3 = interpolate.splev(t,x_g3_spline,der=2) + interpolate.splev(t,y_g3_spline,der=2)*1j

V_g4 = interpolate.splev(t,x_g4_spline,der=1) + interpolate.splev(t,y_g4_spline,der=1)*1j
A_g4 = interpolate.splev(t,x_g4_spline,der=2) + interpolate.splev(t,y_g4_spline,der=2)*1j

# accelerations:

Ag2x = np.real(A_g2[0])
Ag2y = np.imag(A_g2[0])
Ag3x = np.real(A_g3[0])
Ag3y = np.imag(A_g3[0])
Ag4x = np.real(A_g4[0])
Ag4y = np.imag(A_g4[0])
print('Ag2x =', round(Ag2x,3),'\nAg2y =', round(Ag2y,3),'\nAg3x =', round(Ag3x,3),'\nAg3y =', round(Ag3y,3),'\nAg4x =', round(Ag4x,3),'\nAg4y =', round(Ag4y,3))

# angular velocities and accelerations:

tht3_spline = interpolate.splrep(t,tht3)
tht4_spline = interpolate.splrep(t,tht4)

w_3 = interpolate.splev(t,tht3_spline,der=1)
w_4 = interpolate.splev(t,tht4_spline,der=1)

al_3 = interpolate.splev(t,tht3_spline,der=2)
al_4 = interpolate.splev(t,tht4_spline,der=2)

w3 = w_3[0]
w4 = w_4[0]
al3 = al_3[0]
al4 = al_4[0]

print('omega3 =', round(w3,3),'\nomega4 =', round(w4,3),'\nalpha3 =', round(al3,3),'\nalpha4 =', round(al4,3))

#------------------------------kinetic analysis--------------------------------

# givens:

F = 100                                 # force applied [N]
al2 = al
rho1 = 2710                             # density of aluminum [kg/m^3]
rho2 = 7850                             # density of steel [kg/m^3]
v2 = R2*0.05*0.025                      # volume of crank [m^3]
v3 = Rap*R3*np.sin(31*np.pi/180)*0.025  # volume of triangle [m^3]
v4 = R4*0.05*0.025                      # volume of rocker [m^3]
m2 = rho2*v2                            # mass of crank [kg]
m3 = rho1*v3                            # mass of triangle [kg]
m4 = rho2*v4                            # mass of rocker [kg]

I2 = 0.82                               # mass moment of inertia of body 2 [kg*m^2]
I3 = 10.35                              # mass moment of inertia of body 3 [kg*m^2]
I4 = 50.13                              # mass moment of inertia of body 4 [kg*m^2]

i=0

R12x = abs(np.real(z_g2[i]))
R12y = abs(np.imag(z_g2[i]))
R32x = abs(np.real(z_g2[i]))
R32y = abs(np.imag(z_g2[i]))
R14x = abs(np.real(z4[i]/2))
R14y = abs(np.imag(z4[i]/2))
R34x = abs(np.real(z4[i]/2))
R34y = abs(np.imag(z4[i]/2))
R23x = abs(np.real(cm[i]))
R23y = abs(np.imag(cm[i]))
R43x = abs(np.real(z_B[i]-z_g3[i]))
R43y = abs(np.imag(z_B[i]-z_g3[i]))
Rpx = abs(np.real(z_P[i]-z_g3[i]))
Rpy = abs(np.imag(z_P[i]-z_g3[i]))


A = np.array([
        [1,0,-1,0,0,0,0,0,0],
        [0,1,0,-1,0,0,0,0,0],
        [R12y,-R12x,R32y,-R32x,0,0,0,0,1],
        [0,0,0,0,-1,0,-1,0,0],
        [0,0,0,0,0,1,0,1,0],
        [0,0,0,0,R34y,-R34x,-R14y,R14x,0],
        [0,0,-1,0,-1,0,0,0,0],
        [0,0,0,-1,0,1,0,0,0],
        [0,0,-R23y,R23x,R43y,R43x,0,0,0]
        ])

C = np.array([
        [m2*Ag2x],
        [m2*Ag2y],
        [I2*al2],
        [m4*Ag4x],
        [m4*Ag4y],
        [I4*al4],
        [m3*Ag3x],
        [m3*Ag3y],
        [I3*al3+F*Rpx]
        ])

F = np.linalg.solve(A,C)

F12, F12Angle = np.abs(F[0]+1j*F[1]), np.angle(F[0]+1j*F[1])*180/np.pi
F32, F32Angle = np.abs(F[2]+1j*F[3]), np.angle(F[2]+1j*F[3])*180/np.pi
F43, F43Angle = np.abs(F[4]+1j*F[5]), np.angle(F[4]+1j*F[5])*180/np.pi
F14, F14Angle = np.abs(F[6]+1j*F[7]), np.angle(F[6]+1j*F[7])*180/np.pi
T12 = np.abs(F[8])
Fshake, FshakeAngle = np.abs(F12+F14), np.angle(-F12-F14)

print('Kinematics:')
print('F12 =', round(F12[0],2), 'at', round(F12Angle[0],2), 'Degrees')
print('F32 =', round(F32[0],2), 'at', round(F32Angle[0],2), 'Degrees')
print('F43 =', round(F43[0],2), 'at', round(F43Angle[0],2), 'Degrees')
print('F14 =', round(F14[0],2), 'at', round(F14Angle[0],2)+360, 'Degrees')
print('T12 =', round(T12[0],2), 'CCW wrt positive x-axis')
print('Fshake =', round(Fshake[0],2), 'at', round(FshakeAngle[0],2), 'Degrees')

