dt = .005
num_iter = 1000
g = -9.81
particle_mass = 1
n = 50
dh = .05
one_by_dh = 1 / dh
eps = 0.005
eps_tiny = 1e-6
init_num_cols = max(int(n / 2), 1)
num_particles = n * init_num_cols * 4
framerate = 120

x_cols = n + 1
x_rows = n
y_cols = x_rows
y_rows = x_cols
vmag_color_cap = 5
