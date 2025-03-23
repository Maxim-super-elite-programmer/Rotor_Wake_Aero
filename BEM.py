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
pitch = 2 # degrees
chord_distribution = 3*(1-r_R)+1 # meters
twist_distribution = -14*(1-r_R)+pitch # degrees

# flow conditions
u_inf = 10 
TSR = 6 # tip speed ratio
Radius = 50
Omega = u_inf*TSR/Radius
N_blades = 3

tip_R =  1
root_R =  0.2

# solve BEM model
results =np.zeros([len(r_R)-1,6]) 

for i in range(len(r_R)-1):
    chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
    twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
    
    results[i,:] = solve_stream(u_inf, r_R[i], r_R[i+1], root_R, tip_R , Omega, Radius, N_blades, chord, twist, polar_alpha, polar_cl, polar_cd )

# plot results
areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2
dr = (r_R[1:]-r_R[:-1])*Radius
CT = np.sum(dr*results[:,3]*N_blades/(0.5*u_inf**2*np.pi*Radius**2))
CP = np.sum(dr*results[:,4]*results[:,2]*N_blades*Radius*Omega/(0.5*u_inf**3*np.pi*Radius**2))

print("CT is equal to:", CT)
print("CP is equal to:", CP)

plot = 1
if plot:
    fig1 = plt.figure(figsize=(12, 6))
    plt.title('Axial and tangential induction')
    plt.plot(results[:,2], results[:,0], '-', label=r'$a$')
    plt.plot(results[:,2], results[:,1], '-', label=r'$a^,$')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    plt.show()

    fig1 = plt.figure(figsize=(12, 6))
    plt.title(r'Normal and tagential force, non-dimensioned by $\frac{1}{2} \rho U_\infty^2 R$')
    plt.plot(results[:,2], results[:,3]/(0.5*u_inf**2*Radius), '-', label=r'F_norm')
    plt.plot(results[:,2], results[:,4]/(0.5*u_inf**2*Radius), '-', label=r'F_tan')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    plt.show()

    fig1 = plt.figure(figsize=(12, 6))
    plt.title(r'Circulation distribution, non-dimensioned by $\frac{\pi U_\infty^2}{\Omega * NBlades } $')
    plt.plot(results[:,2], results[:,5]/(np.pi*u_inf**2/(N_blades*Omega)), '-', label=r'$\Gamma$')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    plt.show()
