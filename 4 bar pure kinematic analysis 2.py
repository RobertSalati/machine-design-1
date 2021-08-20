"""
Title: Machine Design Exam 1
    
Purpose: Calculate positions and velocities for crank slider links 

Created on Mon Oct 12 08:54:54 2020

author: Robert Salati & Professor Wickenheiser
"""

#Modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy import interpolate

#Code Start

# =============================================================================
#                                Ground Links
# =============================================================================

def rigid_body_position_update(z_C1,z_D1,z_C2,z_D2,z_E1,z_F1):
    # Given absolute complex positions of C1,D1,C2,D2,E1,F1, compute E2,F2
    
    # compute translation (not needed) and rotation from position 1 to 2
    # z21 = z_C2 - z_C1
    z_D1C1 = z_D1 - z_C1
    z_D2C2 = z_D2 - z_C2
    tht21 = np.angle(z_D2C2) - np.angle(z_D1C1)
    
    # relative posotions of E1 and F1 with respect to C1
    z_E1C1 = z_E1 - z_C1
    z_F1C1 = z_F1 - z_C1
    
    # rotate these relative positions through angle tht2
    z_E2C2 = z_E1C1*np.exp(1j*tht21)
    z_F2C2 = z_F1C1*np.exp(1j*tht21)
    
    # compute absolute positions of E2 and F2
    z_E2 = z_C2 + z_E2C2
    z_F2 = z_C2 + z_F2C2
    
    return z_E2, z_F2
        
def intersect_perp_bisectors(z_A1,z_B1,z_A2,z_B2):
    # Find the intersection of the perpendicular bisectors between A1 and A2 and between 
    # B1 and B2
    
    # find midpoints
    z_Am = 0.5*(z_A1+z_A2)
    z_Bm = 0.5*(z_B1+z_B2)
    
    # compute angles of bisectors
    thtA = np.angle(z_A2-z_A1) + np.pi/2
    thtB = np.angle(z_B2-z_B1) + np.pi/2
    
    # solve a*x = b
    a = [[np.cos(thtA), -np.cos(thtB)],[np.sin(thtA), -np.sin(thtB)]]
    b = [np.real(z_Bm-z_Am), np.imag(z_Bm-z_Am)]
    x = np.linalg.solve(a, b)
    
    z_O = z_Am + x[0]*np.exp(1j*thtA)
    
    return z_O
    
# 3-position coupler output synthesis

# Constants

z_C1 = 35 + 0j
z_D1 = 35 + 16j
z_C2 = 17.742 + 12.743j
z_D2 = 7.457 + 25j
z_C3 = 15.757 + 32.778j
z_D3 = 0 + 30j

z_E1 = 40 + 0j
z_F1 = 40 + 16j

# Calculations

z_E2, z_F2 = rigid_body_position_update(z_C1,z_D1,z_C2,z_D2,z_E1,z_F1)
print('z_E2 = ',z_E2)
print('z_F2 = ',z_F2)
z_E3, z_F3 = rigid_body_position_update(z_C1,z_D1,z_C3,z_D3,z_E1,z_F1)
print('z_E3 = ',z_E3)
print('z_F3 = ',z_F3)

z_O2 = intersect_perp_bisectors(z_E1,z_E2,z_E2,z_E3)
print('z_O2 = ',z_O2)
z_O4 = intersect_perp_bisectors(z_F1,z_F2,z_F2,z_F3)
print('z_O4 = ',z_O4)

# =============================================================================
#                               Crank-Coupler 
# =============================================================================

# Constants

tht4 = -114.31*np.pi/180            # [rad] - angle of the link at position 1
phi = (-178.36+114.31)*np.pi/180    # [rad] - sweep angle of the link
k = 1.5                             # ratio for length of coupler link
# E1 and E3 are the same values as calculated above.

# vector calculations

M = z_E3-z_E1                               # Vector connecting position 1 and 3
R2 = np.abs(M)/2                            # [cm] - length of crank
z_O6 = z_E1+k*M+R2*np.exp(1j*np.angle(M))   # position of O6 (ground link for the crank)
R3 = k*abs(M)
print('Position of crank ground link (z_O6) =',z_O6)
print('Length of crank (R2) =', R2)
print('Length of coupler (R3) =', R3)


# =============================================================================
#                                   Positions 
# =============================================================================

# Constants
R1 = 76.995                         # [cm] - ground link length 
R2 = R2                             # [cm] - crank length (calculated above)
R3 = R3                             # [cm] - coupler length (calculated above)
R4 = np.abs(z_O2-z_E1)              # [cm] - rocker length           
w = 1                               # [rad/s] - rotation rate 
z1 = R1*np.exp(1j*-28.28*np.pi/180) # ground link

# Finding angles:
t = np.linspace(0,2*np.pi,100)     # time array [s]
tht2 = np.zeros_like(t)              # crank angle array [rad]
tht3 = np.zeros_like(t)              # coupler angle array [rad]
tht4 = np.zeros_like(t)              # rocker angle array [rad]
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
    tht2[i] = w*t[i]
    z2 = R2*np.exp(1j*tht2[i])
    sol = root_scalar(calc_tht4,x0=x0,bracket=bracket)
    tht4[i] = sol.root
    z4 = R4*np.exp(1j*tht4[i])
    z5 = z1 + z4
    z3 = z5 - z2
    tht3[i] = np.angle(z3)
    
    
# positions of A and B
z_A = R2*np.exp(1j*tht2)
z_B = z_A + R3*np.exp(1j*tht3)

# Plotting

plt.figure()
plt.plot(np.real(z_A)+np.real(z_O6),np.imag(z_A)+np.imag(z_O6),label='Point A')
plt.plot(np.real(z_B)+np.real(z_O6),np.imag(z_B)+np.imag(z_O6),label='Point B')
# The vector zO6 was added to these to account for the shifted origin
plt.ylabel('y-position')
plt.xlabel('x-position')
plt.axis('equal')
plt.legend()

# =============================================================================
#                                 Velocities 
# =============================================================================

# Splines
x_A_spline = interpolate.splrep(t,np.real(z_A))   # spline of Point A x-position array
y_A_spline = interpolate.splrep(t,np.imag(z_A))   # spline of Point A y-position array
x_B_spline = interpolate.splrep(t,np.real(z_B))   # spline of Point B x-position array
y_B_spline = interpolate.splrep(t,np.imag(z_B))   # spline of Point B y-position array

_, ax = plt.subplots()

# Magnitudes of velocity
vb = []
for i in range(len(interpolate.splev(t,x_B_spline,der=1))):
    mag = np.sqrt(interpolate.splev(t,x_B_spline,der=1)[i]**2+interpolate.splev(t,y_B_spline,der=1)[i]**2)
    vb.append(mag)

# Plotting
ax.plot(t,vb)
ax.set_xlabel('Time')
ax.set_ylabel('Velocity')
ax.set_title('Magnitude of velocity of point B')
