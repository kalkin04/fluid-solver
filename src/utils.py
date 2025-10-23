import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import subprocess
from src.params import *
cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=-vmag_color_cap / 3, vmax=vmag_color_cap)
bg_color = '#fdf6e3'

def render(pos, v, i, title, path):
    v_mag = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])
    plt.figure(facecolor=bg_color)
    plt.scatter(pos[:,0], pos[:, 1], c=v_mag, s=16e3/(n*n), cmap=cmap, norm=norm)
    plt.gca().set_facecolor(bg_color)
    plt.ylim(0, n*dh)
    plt.xlim(0, n*dh)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(f'render/{path}/{i}')
    plt.close()

    print(f"Rendered frame {i}/{num_iter}")

def plot(grid, title, label):
    plt.imshow(grid)
    plt.colorbar(label=label)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.close()

def initialize():
    # initialize all velocities to zero
    v = np.zeros((num_particles, 2))
    v_grad_x = np.zeros((num_particles, 2))
    v_grad_y = np.zeros((num_particles, 2))

    # initialize particles to be vertically stacked
    i, j = np.meshgrid(
        dh * (0.5 + np.arange(0, init_num_cols, 1)),
        dh * (0.5 + np.arange(0, n, 1)),
        indexing='xy'
    )
    np.random.seed(0)
    pos = np.stack((i, j), axis=-1)

    pos = np.append(pos, pos)
    pos = np.append(pos, pos)

    pos = pos.reshape(-1, 2)
    pos += dh * (np.random.random((num_particles, 2)) - 0.5)
    return pos, v, v_grad_x, v_grad_y

def make_video(name):
    command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'quiet',
        '-framerate', f'{framerate}',
        '-i', f'render/{name}/%d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        f'videos/{name}.mp4'
    ]
    subprocess.run(command, check=True)
    