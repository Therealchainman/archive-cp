import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Original vector
vector = (3, 2)

# Create a figure and axis
fig, ax = plt.subplots()

# Rotate the vector by 45 degrees
angles = np.linspace(0, 2 * np.pi, 16)
rotated_vectors = complex(*vector) * np.exp(1j * angles)

# Plot the original vector
quiver = ax.quiver(0, 0, rotated_vectors.real[0], rotated_vectors.imag[0], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector')

# Set axis limits
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)


# Set aspect ratio to be equal
ax.set_aspect('equal', adjustable='box')

# Add a legend
ax.legend()

def update(frame):
    print(frame)
    # update the line plot:
    ax.setquiver(0, 0, rotated_vectors.real[frame], rotated_vectors.imag[frame], angles='xy', scale_units='xy', scale=1, color='blue', label='Rotated Vector')
    return quiver

ani = animation.FuncAnimation(fig=fig, func=update, frames=16, interval=3)
plt.show()