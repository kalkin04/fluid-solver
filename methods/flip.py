import sys

sys.path.append('./')
from src.solver import *
from src.utils import *

if __name__ == "__main__":
    pos, v, _, _ = initialize()

    for i in range(num_iter):
        # advection step + render
        pos = solve_advection(pos, v)

        # update state grid
        grid_state = update_state(pos)
        
        render(pos, v, i, "FLIP Method", 'flip')

        # transfer to grid
        u_x, u_y, u_y_g = P2G(v, pos)

        # pressure project
        u_xp, u_yp = resolve_forces(u_x, u_y_g, grid_state)

        # transfer to particles
        v += G2P(u_xp - u_x, u_yp - u_y, pos)

    # make video
    make_video('flip')

    print("simulation saved to flip.mp4")