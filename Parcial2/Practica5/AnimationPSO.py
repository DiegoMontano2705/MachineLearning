import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def animation_frame(i):
    scat.set_offsets(np.c_[x[i], y[i]])
    return scat
x = []
y = []
#Separacion del archivo para crear dataset
with open('conv_0.dat', 'r') as f:
    for row in f.readlines():
        line = [float(i) for i in row.split(" ")[:-1]]
        x.append(line[::2])
        y.append(line[1::2])
x = np.array(x)
y = np.array(y)

# Creacion de grafica
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Desplazamiento particulas PSO")
ax.set_xlabel('X')
ax.set_ylabel('Y')
scat = ax.scatter(x[0], y[0], alpha=0.25)

# Graficacio de las curvas de nivel de la funcion
delta = 0.025
x1 = np.arange(-5, 5, delta)
y1 = np.arange(-5, 5, delta)
X, Y = np.meshgrid(x1, y1)
Z = X**2 + Y**2
ax.contour(X, Y, Z)
# Creacion animacion
anim = animation.FuncAnimation(fig, func=animation_frame, frames=len(x), interval=10)
# Creacion de la animacion 
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Diego'), bitrate=1000)
# Guardar animacion
anim.save('animationPSO.mp4', writer=writer)
#En caso de no poder cargar la animacion, se genera esta imagen
plt.savefig('PSOAnimacion.jpg')