import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from src.params import *

two_thirds = 2/3.
one_sixth = 1/6.

# cubic spline for basis function, avoid division & branches
def N(x):
    abs_x = np.abs(x)
    abs_x_square = abs_x**2
    result = np.zeros_like(abs_x)
    mask1 = (abs_x >= 0) & (abs_x < 1)
    mask2 = (abs_x >= 1) & (abs_x < 2)
    result[mask1] = 0.5 * abs_x_square[mask1] * abs_x[mask1] - abs_x_square[mask1] + two_thirds
    result[mask2] = one_sixth * (2 - abs_x[mask2])**3
    return result

# particle to grid transfer
def P2G(v, pos):
    u_x = np.zeros((x_rows, x_cols))
    u_y = np.zeros((y_rows, y_cols))
    u_y_gravity = np.zeros((y_rows, y_cols))

    Nx, Ny, Nx_half, Ny_half = compute_weights(pos)

    for i in range(0, y_rows):
        for j in range(0, y_cols):
            weight = particle_mass * Ny[i, :] * Nx_half[j, :]
            weight_sum = np.sum(weight)
            u_y[i, j] = np.sum(v[:, 1] * weight) / (weight_sum + eps_tiny)
            u_y_gravity[i, j] = u_y[i, j]
            u_y_gravity[i, j] += g*dt*weight_sum / (weight_sum + eps_tiny)

    for i in range(0, x_rows):
        for j in range(0, x_cols):
            weight = particle_mass * Ny_half[i, :] * Nx[j, :]
            weight_sum = np.sum(weight)
            u_x[i, j] = np.sum(v[:, 0] * weight) / (weight_sum + eps_tiny)

    return u_x, u_y, u_y_gravity

def grid_idx_to_pos_x(i, j):
    # i:row, j:col
    # (0, 0) -> (0, h/2)
    # (0, 1) -> (1, h / 2)
    # (1, 0) -> (0, 3h / 2)
    return np.array([j*dh, 0.5*(2*i + 1) * dh])

def grid_idx_to_pos_y(i, j):
    # i:row, j:col
    # (0, 0) -> (h/2, 0)
    # (0, 1) -> (3h / 2, 0)
    # (1, 0) -> (h/2, h)
    return np.array([0.5*(2*j + 1)*dh, i*dh])

def precompute_grid_positions():
    pos_x = np.zeros((x_rows, x_cols, 2))
    pos_y = np.zeros((y_rows, y_cols, 2))

    for i in range(x_rows):
        for j in range(x_cols):
            pos_x[i, j] = grid_idx_to_pos_x(i, j)
    
    for i in range(y_rows):
        for j in range(y_cols):
            pos_y[i, j] = grid_idx_to_pos_y(i, j)
    
    return pos_x, pos_y

def affine_P2G(v, v_grad_x, v_grad_y, pos, gravity=True):
    u_x = np.zeros((x_rows, x_cols))
    u_y = np.zeros((y_rows, y_cols))

    Nx, Ny, Nx_half, Ny_half = compute_weights(pos)

    for i in range(0, y_rows):
        for j in range(0, y_cols):
            weight = particle_mass * Ny[i, :] * Nx_half[j, :]
            weight_sum = np.sum(weight)

            grid_pos = grid_idx_to_pos_y(i, j)
            linear_part = np.sum((grid_pos - pos) * v_grad_y, axis=1)

            u_y[i, j] = np.sum((v[:, 1] + linear_part) * weight) / (weight_sum + eps_tiny)
            u_y[i, j] += g*dt*weight_sum/(weight_sum + eps_tiny)*bool(gravity)

    for i in range(0, x_rows):
        for j in range(0, x_cols):
            weight = particle_mass * Ny_half[i, :] * Nx[j, :]
            weight_sum = np.sum(weight)

            grid_pos = grid_idx_to_pos_x(i, j)
            linear_part =  np.sum((grid_pos - pos) * v_grad_x, axis=1)

            u_x[i, j] = np.sum((v[:, 0] + linear_part) * weight) / (weight_sum + eps_tiny)

    return u_x, u_y

# pressure projection
def resolve_forces(u_x, u_y, grid_state):
    # build divergence matrix
    # num rows = num grid boxes
    div = np.zeros((n * n, 2 * x_rows * x_cols))
    row = 0
    off = x_rows * x_cols
    for j in range(0, n):
        for i in range(0, n):
            if (not grid_state[j][i]):
                continue

            div[row][(i) + (j)*(x_cols)] = 1 * (i != 0)
            div[row][(i) + (j)*(y_cols) + off] = 1 * (j != 0)

            div[row][(i+1) + (j)*(x_cols)] = -1 * (i != n-1)
            div[row][(i) + (j+1)*(y_cols) + off] = -1 * (j != n-1)

            row += 1
    div *= one_by_dh
    div = csr_matrix(div)

    div_div_t = div @ div.T

    # dirichlet boundary conditions
    u_x[:, 0] = 0
    u_x[:, -1]  = 0

    u_y[0, :] = 0
    u_y[-1, :] = 0

    # update grid velocities by taking projection
    u = np.append(u_x.flatten(), u_y.flatten())
    p = cg(div_div_t, div @ u)[0]

    u = u - div.T @ p
    u_x = u[:x_rows*x_cols].reshape(x_rows, x_cols)
    u_y = u[x_rows*x_cols:].reshape(y_rows, y_cols)

    return u_x, u_y

# grid to particle transfer
def G2P(u_x, u_y, pos):
    Nx, Ny, Nx_half, Ny_half = compute_weights(pos)
    
    v = np.zeros((num_particles, 2))
    for i in range(num_particles):
        grid_weights_x = np.outer(Ny_half[:, i], Nx[:, i])
        v[i, 0] = np.sum(grid_weights_x * u_x)
        grid_weights_y = np.outer(Ny[:, i], Nx_half[:, i])
        v[i, 1] = np.sum(grid_weights_y * u_y)
    return v

def affine_G2P(u_x, u_y, pos, pos_x, pos_y):
    Nx, Ny, Nx_half, Ny_half = compute_weights(pos)
    
    v = np.zeros((num_particles, 2))
    grad_v_x = np.zeros((num_particles, 2))
    grad_v_y = np.zeros((num_particles, 2))

    for i in range(num_particles):
        grid_weights_x = np.outer(Ny_half[:, i], Nx[:, i])
        v[i, 0] = np.sum(u_x * grid_weights_x)
        grid_weights_y = np.outer(Ny[:, i], Nx_half[:, i])
        v[i, 1] = np.sum(u_y * grid_weights_y)
        
        flat_wx = grid_weights_x.ravel()
        flat_wy = grid_weights_y.ravel()
        dx = (pos_x.reshape(-1, 2) - pos[i])
        dy = (pos_y.reshape(-1, 2) - pos[i])

        Bx = np.sum(u_x.ravel()[:, None] * flat_wx[:, None] * dx, axis=0)
        By = np.sum(u_y.ravel()[:, None] * flat_wy[:, None] * dy, axis=0)

        # for cubic splines, Dx = Dy = 1/3 dh^2 I
        grad_v_x[i] = 3 / (dh * dh) * Bx
        grad_v_y[i] = 3 / (dh * dh) * By
    
    return v, grad_v_x, grad_v_y

def compute_weights(pos):
    x_list = enumerate(dh * np.arange(0, x_cols))
    y_list_half = enumerate(0.5 * dh + dh * np.arange(0, x_rows))

    y_list =  enumerate(dh * np.arange(0, y_rows))
    x_list_half = enumerate(0.5 * dh + dh * np.arange(0, y_cols))

    Nx = np.zeros((x_cols, num_particles))
    Ny_half = np.zeros((x_rows, num_particles))
    Ny = np.zeros((y_rows, num_particles))
    Nx_half = np.zeros((y_cols, num_particles))

    for i, x in x_list:
        Nx[i] = N(one_by_dh * (pos[:, 0] - x))
    for i, y in y_list_half:
        Ny_half[i] = N(one_by_dh * (pos[:, 1] - y))

    for i, y in y_list:
        Ny[i] = N(one_by_dh * (pos[:, 1] - y))
    for i, x in x_list_half:
        Nx_half[i] = N(one_by_dh * (pos[:, 0] - x))
    
    return Nx, Ny, Nx_half, Ny_half

def solve_advection(pos, v):
    k1 = v
    
    pos_temp = pos + 0.5 * k1 * dt
    u_x, u_y, _ = P2G(k1, pos_temp)
    k2 = G2P(u_x, u_y, pos_temp)

    pos_temp = pos + 0.5 * k2 * dt
    u_x, u_y, _ = P2G(k2, pos_temp)
    k3 = G2P(u_x, u_y, pos_temp)

    pos_temp = pos + k3 * dt
    u_x, u_y, _ = P2G(k3, pos_temp)
    k4 = G2P(u_x, u_y, pos_temp)

    pos += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    pos[:, 0] = np.clip(pos[:, 0], eps, n*dh - eps)
    pos[:, 1] = np.clip(pos[:, 1], eps, n*dh - eps)
    return pos

def solve_advection_simple(pos, v):
    pos += dt * v

    pos[:, 0] = np.clip(pos[:, 0], eps, n*dh - eps)
    pos[:, 1] = np.clip(pos[:, 1], eps, n*dh - eps)
    return pos

def update_state(pos):
    # 0 -> empty, 1-> water
    grid_state = np.zeros((n, n))
    for position in pos:
        grid_idx = np.floor(position / np.array([dh, dh]))
        grid_state[int(grid_idx[1]), int(grid_idx[0])] = 1

    return grid_state
