import sys

sys.path.append('./')
from src.solver import *
from src.utils import *

if __name__ == "__main__":
    pos, v, v_grad_x, v_grad_y = initialize()
    grid_pos_x, grid_pos_y = precompute_grid_positions()

    for i in range(num_iter):
        # advection step + render
        pos = solve_advection_simple(pos, v)

        # update state grid
        grid_state = update_state(pos)

        render(pos, v, i, "APIC Method", "apic")

        # transfer to grid and apply gravity
        u_x, u_y = affine_P2G(v, v_grad_x, v_grad_y, pos)

        # pressure project
        u_x, u_y = resolve_forces(u_x, u_y, grid_state)

        # transfer to particles
        v, v_grad_x, v_grad_y = affine_G2P(u_x, u_y, pos, grid_pos_x, grid_pos_y)
    
    # make video
    make_video('apic')

    print("simulation saved to apic.mp4")