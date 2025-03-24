import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from functions import solve_stream, solve_stream_yaw

# Extract data from file
airfoil = 'polar DU95W180.csv'
data = pd.read_csv(airfoil)

polar_alpha = data['Alfa'].values
polar_cl = data['Cl'].values
polar_cd = data['Cd'].values

# define the blade geometry
delta_r_R = [1, 0.5, 0.25, 0.1, 0.05, 0.01]

def solver(dr):
    r_R = np.arange(0.2, 1+dr/2, dr)

    # blade shape
    pitch = -2 # degrees
    chord_distribution = 3*(1-r_R)+1 # meters
    twist_distribution = 14*(1-r_R)+pitch # degrees

    # flow conditions
    u_inf = 10 
    TSR = 8 # tip speed ratio
    Radius = 50
    Omega = u_inf*TSR/Radius
    N_blades = 3

    tip_R =  1
    root_R =  0.2

    # solve BEM model
    results = np.zeros([len(r_R)-1,6]) 
    alpha = np.zeros([len(r_R)-1])
    phi = np.zeros_like(alpha)
    yaw = 0

    for i in range(len(r_R)-1):
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
        if yaw == 0:
            results[i,:], alpha[i], phi[i] = solve_stream(u_inf, r_R[i], r_R[i+1], root_R, tip_R , Omega, Radius, N_blades, chord, twist, polar_alpha, polar_cl, polar_cd)
        elif yaw != 0:
            results[i,:], alpha[i], phi[i] = solve_stream_yaw(u_inf, r_R[i], r_R[i+1], root_R, tip_R , Omega, Radius, N_blades, chord, twist, polar_alpha, polar_cl, polar_cd, yaw)

    phi = phi * 180 / np.pi

    return r_R, alpha, phi, results

# Initialize lists to store results for different dr values
all_r_R = []
all_alpha = []
all_phi = []
all_results = []

for dr in delta_r_R:
    r_R, alpha, phi, results = solver(dr)
    all_r_R.append(r_R)
    all_alpha.append(alpha)
    all_phi.append(phi)
    all_results.append(results)

# Plot results
fig_alpha = plt.figure(figsize=(12, 6))
plt.title('Angle of attack and inflow angle over span')
for i, dr in enumerate(delta_r_R):
    plt.plot(all_r_R[i][:-1], all_alpha[i], '-', label=fr'$\alpha$ (dr={dr})')
    plt.plot(all_r_R[i][:-1], all_phi[i], '--', label=fr'$\phi$ (dr={dr})')
plt.grid()
plt.xlabel('r/R')
plt.ylabel('DEG')
plt.legend()
plt.show()

fig_induction = plt.figure(figsize=(12, 6))
plt.title('Induction factors over span')
for i, dr in enumerate(delta_r_R):
    plt.plot(all_r_R[i][1:], all_results[i][:, 0], '-', label=fr'Axial induction factor (dr={dr})')
    plt.plot(all_r_R[i][1:], all_results[i][:, 1], '--', label=fr'Tangential induction factor (dr={dr})')
plt.grid()
plt.xlabel('r/R')
plt.ylabel('-')
plt.legend()
plt.show()

fig_loading = plt.figure(figsize=(12, 6))
plt.title('Normal and azimuthal loading')
for i, dr in enumerate(delta_r_R):
    plt.plot(all_r_R[i][1:], all_results[i][:, 3], '-', label=fr'Normal loading (dr={dr})')
    plt.plot(all_r_R[i][1:], all_results[i][:, 4], '--', label=fr'Azimuthal loading (dr={dr})')
plt.grid()
plt.xlabel('r/R')
plt.ylabel('N')
plt.legend()
plt.show()