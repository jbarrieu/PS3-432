#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:43:08 2024

@author: jbarrieu
"""

import numpy as np
import matplotlib.pyplot as plt

n = 100         # Size of the grid
steps = 100000   # Number of time steps 
dt = 2          # Time Step Size
dx = 1          # Grid Step Size
D = 0.1         # Diffusion constant
g = 0.01        # Gravitational acceleration
alpha = 10      # Angle of the slope in degrees

### Important scalars

C = D*dt/dx**2                # Set value obeying Courant
x = np.arange(0, n, dx)       # Setting Scale of Grid
angle = np.radians(alpha)     # Conversion to radians for practicality
a = g * np.sin(angle)         # Acceleration of Lava Formula 
y = -a/D * (0.5*x**2 - x[-1]*x) # Analytical result for steady-state
A = (1 + 2*B)*np.eye(n) - C*np.eye(n, k=1) - C*np.eye(n, k=-1) # Whole formula

# Boundary conditions
A[0] = np.zeros(n)  # We need the velocity at the ground to be 0.
A[0][0] = 1
A[-1][-1] = 1 + C   # This avoids strain on surface of lava

f = np.zeros(n)     # Velocity Field with initial value of 0

### Plot Setup 

plt.ion()
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, "--", color="orange")    # Plots Steady State 
plt_obj, = ax.plot(x, f, color="red") # Plots initial field

fig.canvas.draw() 

### Loop

for t in range(steps):
    
    f = np.linalg.solve(A, f)   # Solves the repeated system
    
    f[1:] += a * dt     # Current Acceleration
    
    if t % 200 == 0: 
        plt_obj.set_ydata(f)
        ax.set_title('t = ' + str(t * dt))
        fig.canvas.draw()
        plt.pause(0.001)
