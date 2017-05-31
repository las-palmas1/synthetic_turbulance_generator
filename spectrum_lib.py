import numpy as np
from scipy.fftpack import fftn
import typing


class SpatialSpectrum3d:
    def __init__(self, i_cnt: int, j_cnt: int, k_cnt: int, grid_step: float, u_arr: typing.List[float],
                 v_arr: typing.List[float], w_arr: typing.List[float]):
        self.i_cnt = i_cnt
        self.j_cnt = j_cnt
        self.k_cnt = k_cnt
        self.grid_step = grid_step
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr

    @classmethod
    def _get_array_3d(cls, list_1d: typing.List[float], i_cnt: int, j_cnt: int, k_cnt: int):
        result = np.array(list_1d).reshape([k_cnt, j_cnt, i_cnt])
        return result

    @classmethod
    def _get_wave_number_array(cls, i_cnt: int, j_cnt: int, k_cnt: int, grid_step):
        x_size = grid_step * (i_cnt - 1)
        y_size = grid_step * (j_cnt - 1)
        z_size = grid_step * (k_cnt - 1)
        result_i = np.zeros([k_cnt, j_cnt, i_cnt])
        result_j = np.zeros([k_cnt, j_cnt, i_cnt])
        result_k = np.zeros([k_cnt, j_cnt, i_cnt])
        ki_arr = np.linspace(1 / x_size, 1 / grid_step, i_cnt)
        kj_arr = np.linspace(1 / y_size, 1 / grid_step, j_cnt)
        kk_arr = np.linspace(1 / z_size, 1 / grid_step, k_cnt)
        for k in range(k_cnt):
            for j in range(j_cnt):
                for i in range(i_cnt):
                    result_i[k, j, i] = ki_arr[i]
                    result_j[k, j, i] = kj_arr[j]
                    result_k[k, j, i] = kk_arr[k]
        return result_i, result_j, result_k

if __name__ == '__main__':
    sp = SpatialSpectrum3d
    ar = [2, 3, 4, 5, 4, 1, 1.2, 2.2, 3.2, 1.3, 2.3, 3.3]
    print(sp._get_array_3d(ar, 3, 2, 2))
