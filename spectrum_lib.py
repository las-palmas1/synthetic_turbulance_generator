import numpy as np
from scipy.fftpack import fftn
import logging


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def read_velocity_file(filename):
    velocity_arr = []
    with open(filename, 'r') as file:
        while True:
            st = file.readline()
            if len(st) == 0:
                break
            velocity_arr.append([float(i) for i in st.split(sep=' ')])
    return np.array(velocity_arr)[:, 0], np.array(velocity_arr)[:, 1], np.array(velocity_arr)[:, 2]


class SpatialSpectrum3d:
    def __init__(self, i_cnt: int, j_cnt: int, k_cnt: int, grid_step: float, u_arr: np.ndarray,
                 v_arr: np.ndarray, w_arr: np.ndarray, num_point: int):
        self.i_cnt = i_cnt
        self.j_cnt = j_cnt
        self.k_cnt = k_cnt
        self.grid_step = grid_step
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.num_point = num_point
        self.k_abs_arr = None
        self.energy_sum_arr = None
        self.energy_u_arr = None
        self.energy_v_arr = None
        self.energy_w_arr = None

    @classmethod
    def _get_array_3d(cls, array_1d: np.ndarray, i_cnt: int, j_cnt: int, k_cnt: int):
        result = array_1d.reshape([k_cnt, j_cnt, i_cnt])
        return result

    @classmethod
    def _get_wave_number_array(cls, i_cnt: int, j_cnt: int, k_cnt: int, grid_step):
        logging.info('Wave number arrays calculating')
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

    @classmethod
    def _get_energy(cls, k_abs_arr: np.ndarray, velocity_fourier_arr: np.ndarray,
                    wave_number_abs: float):
        array_for_summation = 0.5 * velocity_fourier_arr * velocity_fourier_arr.conj()
        array_for_summation_filtered = np.abs(array_for_summation)[(k_abs_arr > wave_number_abs - 0.5) *
                                                                   (k_abs_arr < wave_number_abs + 0.5)]
        result = array_for_summation_filtered.sum()
        logging.debug('wave number = %.4f,   energy = %.4f' % (wave_number_abs, result))
        return result

    @classmethod
    def _get_spectrum(cls, ki_arr: np.ndarray, kj_arr: np.ndarray, kk_arr: np.ndarray, velocity_arr: np.ndarray,
                      num_point: int):
        logging.info('SPECTRUM CALCULATING\n')
        k_abs_arr = np.sqrt(ki_arr ** 2 + kj_arr ** 2 + kk_arr ** 2)
        k_arr_for_spectrum = np.linspace(k_abs_arr.min(), k_abs_arr.max(), num_point)
        velocity_fourier_arr = fftn(velocity_arr)
        energy_arr = np.zeros([num_point])
        for n, k in enumerate(k_arr_for_spectrum):
            energy_arr[n] = cls._get_energy(k_abs_arr, velocity_fourier_arr, k)
        return energy_arr, k_arr_for_spectrum

    def compute_spectrum(self):
        ki_arr, kj_arr, kk_arr = self._get_wave_number_array(self.i_cnt, self.j_cnt, self.k_cnt, self.grid_step)
        u_arr_3d = self._get_array_3d(self.u_arr, self.i_cnt, self.j_cnt, self.k_cnt)
        v_arr_3d = self._get_array_3d(self.v_arr, self.i_cnt, self.j_cnt, self.k_cnt)
        w_arr_3d = self._get_array_3d(self.w_arr, self.i_cnt, self.j_cnt, self.k_cnt)
        self.energy_u_arr, self.k_abs_arr = self._get_spectrum(ki_arr, kj_arr, kk_arr, u_arr_3d, self.num_point)
        self.energy_v_arr = self._get_spectrum(ki_arr, kj_arr, kk_arr, v_arr_3d, self.num_point)[0]
        self.energy_w_arr = self._get_spectrum(ki_arr, kj_arr, kk_arr, w_arr_3d, self.num_point)[0]
        self.energy_sum_arr = self.energy_u_arr + self.energy_v_arr + self.energy_w_arr


if __name__ == '__main__':
    vel = read_velocity_file(r'output\velocity.VEL')
    for i in range(len(vel[0])):
        print(vel[0][i], vel[1][i], vel[2][i])
