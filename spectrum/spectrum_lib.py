import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftn
import config
from core.lib import get_k_arr, get_tau, get_von_karman_spectrum, get_amplitude_arr, get_d_vector_theta_and_phase, \
    get_frequency, get_sigma_vector
from scipy.interpolate import interp1d
from scipy.integrate import quad

logging.basicConfig(format='%(levelname)s: %(message)s', level=config.log_level)


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
                 v_arr: np.ndarray, w_arr: np.ndarray, num_point: int, l_e: float =config.l_e,
                 alpha: float = config.alpha):
        self.i_cnt = i_cnt
        self.j_cnt = j_cnt
        self.k_cnt = k_cnt
        self.grid_step = grid_step
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.num_point = num_point
        self.l_e = l_e
        self.alpha = alpha
        self.l_cut = 2 * self.grid_step
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
    def _get_k_abs(cls, l_e: float, l_cut: float, alpha):
        k_abs_arr = get_k_arr(l_e, l_cut, alpha)
        k_abs_min = k_abs_arr.min()
        k_abs_max = k_abs_arr.max()
        return k_abs_min, k_abs_max

    @classmethod
    def _get_wave_number_array(cls, i_cnt: int, j_cnt: int, k_cnt: int, grid_step: float, k_abs_min: float,
                               k_abs_max: float):
        logging.info('Wave number arrays calculating')
        x_size = grid_step * (i_cnt - 1)
        y_size = grid_step * (j_cnt - 1)
        z_size = grid_step * (k_cnt - 1)
        result_i = np.zeros([k_cnt, j_cnt, i_cnt])
        result_j = np.zeros([k_cnt, j_cnt, i_cnt])
        result_k = np.zeros([k_cnt, j_cnt, i_cnt])
        # k_max = k_abs_max / np.sqrt(3)
        # norm_coef = k_abs_min / np.sqrt(1 / x_size ** 2 + 1 / y_size ** 2 + 1 / z_size ** 2)
        # ki_min = norm_coef / x_size
        # kj_min = norm_coef / y_size
        # kk_min = norm_coef / z_size
        # ki_arr = np.linspace(ki_min, k_max, i_cnt)
        # kj_arr = np.linspace(kj_min, k_max, j_cnt)
        # kk_arr = np.linspace(kk_min, k_max, k_cnt)
        # ki_arr = np.linspace(ki_min, k_max, i_cnt)
        # kj_arr = np.linspace(kj_min, k_max, j_cnt)
        # kk_arr = np.linspace(kk_min, k_max, k_cnt)
        ki_min = 1 / x_size
        kj_min = 1 / y_size
        kk_min = 1 / z_size
        ki_arr = np.linspace(ki_min, i_cnt * ki_min, i_cnt)
        kj_arr = np.linspace(kj_min, j_cnt * kj_min, j_cnt)
        kk_arr = np.linspace(kk_min, k_cnt * kk_min, k_cnt)
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
        volume = velocity_fourier_arr.shape[0] * velocity_fourier_arr.shape[1] * velocity_fourier_arr.shape[2]
        array_for_summation = 0.5 * (np.abs(velocity_fourier_arr) / volume) ** 2
        array_for_summation_filtered = np.abs(array_for_summation)[(k_abs_arr > wave_number_abs - 0.5) *
                                                                   (k_abs_arr < wave_number_abs + 0.5)]
        result = array_for_summation_filtered.sum()
        logging.debug('wave number = %.4f,   energy = %.4f' % (wave_number_abs, result))
        return result

    @classmethod
    def _get_spectrum(cls, ki_arr: np.ndarray, kj_arr: np.ndarray, kk_arr: np.ndarray, velocity_arr: np.ndarray,
                      num_point: int):
        logging.info('')
        logging.info('SPECTRUM CALCULATING\n')
        k_abs_arr = np.sqrt(ki_arr ** 2 + kj_arr ** 2 + kk_arr ** 2)
        k_arr_for_spectrum = np.linspace(k_abs_arr.min(), k_abs_arr.max(), num_point)
        velocity_fourier_arr = fftn(velocity_arr)
        energy_arr = np.zeros([num_point])
        for n, k in enumerate(k_arr_for_spectrum):
            energy_arr[n] = cls._get_energy(k_abs_arr, velocity_fourier_arr, k)
        return energy_arr, k_arr_for_spectrum

    def compute_spectrum(self):
        k_abs_min, k_abs_max = self._get_k_abs(self.l_e, self.l_cut, self.alpha)
        ki_arr, kj_arr, kk_arr = self._get_wave_number_array(self.i_cnt, self.j_cnt, self.k_cnt, self.grid_step,
                                                             k_abs_min, k_abs_max)
        u_arr_3d = self._get_array_3d(self.u_arr, self.i_cnt, self.j_cnt, self.k_cnt)
        v_arr_3d = self._get_array_3d(self.v_arr, self.i_cnt, self.j_cnt, self.k_cnt)
        w_arr_3d = self._get_array_3d(self.w_arr, self.i_cnt, self.j_cnt, self.k_cnt)
        self.energy_u_arr, self.k_abs_arr = self._get_spectrum(ki_arr, kj_arr, kk_arr, u_arr_3d, self.num_point)
        self.energy_v_arr = self._get_spectrum(ki_arr, kj_arr, kk_arr, v_arr_3d, self.num_point)[0]
        self.energy_w_arr = self._get_spectrum(ki_arr, kj_arr, kk_arr, w_arr_3d, self.num_point)[0]
        self.energy_sum_arr = self.energy_u_arr + self.energy_v_arr + self.energy_w_arr

    def get_turb_kinetic_energy(self):
        energy_interp = interp1d(self.k_abs_arr, self.energy_sum_arr)
        result = quad(lambda k: energy_interp(k), min(self.k_abs_arr), max(self.k_abs_arr))[0]
        return result


def plot_spectrum_with_predefined(k_arr, energy_arr, filename, l_cut, l_e, l_cut_min, l_e_max,
                                  viscosity=config.viscosity, dissipation_rate=config.dissipation_rate,
                                  alpha=config.alpha, u0=config.u0, r_vector=np.array([0., 0., 0.]), t=config.time):
    logging.info('')
    logging.info('Plotting spectrum')
    plt.figure(figsize=(9, 7))
    k_arr_predef = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr_von_karman = get_von_karman_spectrum(k_arr_predef, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr_predef, energy_arr_von_karman)
    d_vector, theta, phase = get_d_vector_theta_and_phase(k_arr_predef.shape[0])
    frequency = get_frequency(k_arr_predef.shape[0])
    v_arr = []
    for i in range(k_arr_predef.shape[0]):
        sigma_vector = get_sigma_vector(d_vector[i], theta[i])
        v_arr.append(2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector *
                     np.cos(k_arr_predef[i] * np.dot(d_vector[i], r_vector) + phase[i] + frequency[i] * t / tau))
    energy_arr_predef = [np.linalg.norm(v) ** 2 for v in v_arr]
    plt.plot(k_arr_predef, energy_arr_von_karman, color='blue', lw=1, label=r'$Модифицированный\ спектр\ фон\ Кармана$')
    plt.plot(k_arr_predef, energy_arr_predef, color='red', lw=0.5,
             label=r'$Заданный\ спектр\ синтетического\ поля$')
    plt.plot(k_arr, energy_arr, color='green', lw=1, label=r'$Вычисленный\ спектр$')
    plt.plot([2 * np.pi / l_cut_min, 2 * np.pi / l_cut_min], [0, 2 * max(energy_arr_predef)], lw=3, color='black',
             label=r'$k_{max}$')
    plt.ylim(10e-5, 10e-1)
    plt.xlim(1, 10e1)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both')
    plt.legend(fontsize=12, loc=3)
    plt.xlabel(r'$k$', fontsize=20)
    plt.ylabel(r'$E$', fontsize=20)
    plt.savefig(filename)


if __name__ == '__main__':
    vel = read_velocity_file(r'output\velocity.VEL')
    for i in range(len(vel[0])):
        print(vel[0][i], vel[1][i], vel[2][i])
