import sys

sys.path.append('./')
from src.solver import *
from src.utils import *

if __name__ == "__main__":
    pos, v, _, _ = initialize()

    for i in range(num_iter):
        # advection step + render
        pos = solve_advection_simple(pos, v)

        # update state grid
        grid_state = update_state(pos)

        render(pos, v, i, "PIC Method", "pic")

        # transfer to grid and apply gravity
        u_x, _, u_y = P2G(v, pos)

        # pressure project
        u_x, u_y = resolve_forces(u_x, u_y, grid_state)

        # transfer to particles
        v = G2P(u_x, u_y, pos)
    
    # make video
    make_video('pic')

    print("simulation saved to pic.mp4")