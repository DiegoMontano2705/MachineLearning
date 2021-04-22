import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers


def animation_frame(i):
    scat.set_offsets(np.c_[x[i], y[i]])
    return scat


# Cargamos los datos
x = []
y = []
with open('conv_0.dat', 'r') as f:
    for row in f.readlines():
        line = [float(i) for i in row.split(" ")[:-1]]
        x.append(line[::2])
        y.append(line[1::2])
x = np.array(x)
y = np.array(y)
# Creamos el plot
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Evolución temporal del PSO')
ax.set_xlabel('X')
ax.set_ylabel('Y')
scat = ax.scatter(x[0], y[0], alpha=0.25)
# Graficamos las curvas de nivel
delta = 0.025
x1 = np.arange(-5, 5, delta)
y1 = np.arange(-5, 5, delta)
X, Y = np.meshgrid(x1, y1)
Z = X**2 + Y**2
ax.contour(X, Y, Z)
# Hacemos la animacion
animation = FuncAnimation(fig, func=animation_frame, frames=len(x), interval=10)
# Guardamos la animacion
Writer = writers['ffmpeg']
writer = Writer(fps=15, metadata={'artist': 'Me'}, bitrate=1800)
animation.save('Particulas.mp4', writer)
