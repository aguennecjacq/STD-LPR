from numba import float64, jit, void, int64
from numba.types import UniTuple
import numpy as np
from proximal_operators import D, svt, prox_tv
from P_operator import get_weights_map, P, P_inv


@jit(float64(float64[:, :]), cache=True, nopython=True)
def l2_norm(input_array):
    return np.sqrt(np.sum(input_array ** 2))


@jit(UniTuple(float64[:, :], 2)(float64[:, :], int64, float64, float64, int64), cache=True, nopython=True)
def std_lpr(input_image, r=5, rho=5.0, mu=1.0, nb_iter=200):
    n, m = input_image.shape

    # init
    f = input_image
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    y = np.zeros_like(f)
    v_prev = np.zeros_like(f)

    # setup parameters
    step_r = (r // 2) + 1
    nb_patches = (n // step_r) * (m // step_r)
    weights_map = get_weights_map((n, m), r)  # weights map for the reconstruction of the patch_operator

    # memory allocations
    gamma = 1.0
    p_array = np.zeros((r ** 2, nb_patches))

    for iter_ in range(nb_iter):

        # update u
        u = f - v - y
        prox_tv(u, h=mu / rho, nb_iter=200)

        # update v
        tmp = f - u - y
        v = np.zeros_like(f)
        P(tmp, p_array, r)
        svt(p_array, gamma / rho)
        P_inv(p_array, v, r, weights_map)

        # update y
        y = y + (u + v - f)

        if iter_ % 10:
            if abs(l2_norm(v_prev) - l2_norm(v)) < 0.01:
                break
            else:
                v_prev = v

    return u, v
