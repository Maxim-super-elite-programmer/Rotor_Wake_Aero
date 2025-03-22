import numpy as np
import matplotlib.pyplot as plt

# propeller constants
r_prop = 0.7
N_blades_prop = 6
blade_start_prop = 0.25 #r/R
twist_prop = [-50, 35] #coefficient, degrees
collective_blade_pitch_prop = 46 #degrees at r/R = 0.7
chord_distribution_prop = [0.18, -0.06] #coefficient
airfoil_prop = 'ARA-D8%'

# propeller operational constants
u_prop = 60 #m/s
rpm_prop = 1200
altitude_prop = 2000 #m
incidience_angle_prop = 0 #degrees
yaw_angle_prop = [0, 15, 30] #degrees