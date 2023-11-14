
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Generate an array of angles from 0 to 2*pi
theta = np.linspace(0, 300*np.pi, 30_000)

# Compute the complex numbers using e^(ix)
# r = np.exp(1j * theta) + np.exp(1j * theta * np.pi)
r = np.exp(1j *  np.pi * theta)
# fig, ax = plt.subplots(subplo_kw={'projection': 'polar'})
fig, ax = plt.subplots()
line = ax.plot(theta[0], r[0], label='real part')[0]

def update(frame):
    frame *= 100
    # update the line plot:
    line.set_xdata(theta[:frame])
    line.set_ydata(r[:frame])
    return line


ani = animation.FuncAnimation(fig=fig, func=update, frames=300, interval=5)
plt.show()
