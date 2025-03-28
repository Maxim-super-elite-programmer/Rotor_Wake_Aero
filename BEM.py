import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from functions import solve_stream

# Extract data from file
airfoil = 'polar DU95W180.csv'
data = pd.read_csv(airfoil)

polar_alpha = data['Alfa'].values
polar_cl = data['Cl'].values
polar_cd = data['Cd'].values

# define the blade geometry
delta_r_R = 0.01
r_R = np.arange(0.2, 1+delta_r_R/2, delta_r_R)

# blade shape
pitch = -2 # degrees
chord_distribution = 3*(1-r_R)+1 # meters
twist_distribution = 14*(1-r_R)+pitch # degrees

# flow conditions
u_inf = 10 
TSR = 10 # tip speed ratio
Radius = 50
Omega = u_inf*TSR/Radius
N_blades = 3

tip_R =  1
root_R =  0.2

# solve BEM model
results =np.zeros([len(r_R)-1,6]) 
alpha = np.zeros([len(r_R-1)])
phi = np.zeros_like(alpha)

for i in range(len(r_R)-1):
    chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
    twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
    
    results[i,:], alpha[i], phi[i] = solve_stream(u_inf, r_R[i], r_R[i+1], root_R, tip_R , Omega, Radius, N_blades, chord, twist, polar_alpha, polar_cl, polar_cd )

phi = phi *180/np.pi

# plot results
areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2
dr = (r_R[1:]-r_R[:-1])*Radius
CT = np.sum(dr*results[:,3]*N_blades/(0.5*u_inf**2*np.pi*Radius**2))
CP = np.sum(dr*results[:,4]*results[:,2]*N_blades*Radius*Omega/(0.5*u_inf**3*np.pi*Radius**2))
Torque_blade_element = np.sum(dr * results[:, 4] * r_R[1:] * N_blades*Radius*Omega * 1.225)

print("CT is equal to:", CT)
print("CP is equal to:", CP)

T = CT * 0.5 * 1.225 * u_inf**2 * (2 * np.pi * Radius**2)
P = CP * 0.5 * 1.225 * u_inf**3 * (2 * np.pi * Radius**2)

print('Total Thrust:', T)
print('Total Power:', P)
print('torque BEM', Torque_blade_element)


plot = 1
if plot:
    fig_alpha = plt.figure(figsize=(12, 6))
    plt.title('Angle of attack and inflow angle over span')
    plt.plot(r_R, alpha, '-', label=r'$\alpha$')
    plt.plot(r_R, phi, '-', label=r'$\phi$')
    plt.grid()
    plt.xlabel('r/R')
    plt.ylabel('DEG')
    plt.legend()
    plt.show()

    fig_induction = plt.figure(figsize=(12, 6))
    plt.title('Induction factors over span')
    plt.plot(r_R[1:], results[:, 0], '-', label='Axial induction factor')
    plt.plot(r_R[1:], results[:, 1], '-', label='Tangential induction factor')
    plt.grid()
    plt.xlabel('r/R')
    plt.ylabel('-')
    plt.legend()
    plt.show()

    fig_loading = plt.figure(figsize=(12, 6))
    plt.title('Normal and azimuthal loading')
    plt.plot(r_R[1:], results[:, 3], '-', label='Normal loading')
    plt.plot(r_R[1:], results[:, 4], '-', label='Azimuthal loading')
    plt.grid()
    plt.xlabel('r/R')
    plt.ylabel('N')
    plt.legend()
    plt.show()
