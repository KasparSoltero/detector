import numpy as np
import matplotlib.pyplot as plt

# Define the distance range from 1 to 100 meters
distance = np.linspace(1, 50, 400)
# Calculate the amplitude using the inverse square law, assuming a constant initial amplitude
intensity = 1 / distance**2

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distance, intensity, linewidth=2, label='Intensity (1/x**2)')
# plot sqrt of intensity
plt.plot(distance, np.sqrt(intensity), linewidth=2, label='Sqrt Intensity (1/x)')
plt.xlabel('Distance (m)')
plt.ylabel('Relative to Initial')
plt.legend()
plt.grid(True)

# Set x-axis and y-axis limits
plt.xlim(0, 50)  # Set x-axis limits from 0 to 50
plt.ylim(0, 1)   # Set y-axis limits from 0 to 1
# horizontal lines at intersection points where

plt.show()
