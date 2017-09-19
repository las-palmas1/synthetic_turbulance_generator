import logging
import matplotlib.pyplot as plt
import numpy as np
import config
from core.les_inlet_ic_lib import get_k_arr, get_tau, get_von_karman_spectrum, get_amplitude_arr, get_d_vector, get_phase, get_z, \
    get_theta, get_frequency, get_sigma_vector
from core.diht_ic_lib import make_fft_grid, make_fft

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
    def __init__(self, num: int, grid_step: float, u_arr: np.ndarray, v_arr: np.ndarray, w_arr: np.ndarray,
                 num_point: int):
        """
        :param num: количество узлов в на стороне генерируемого куба
        :param grid_step: шаг сетки
        :param u_arr:
        :param v_arr:
        :param w_arr:
        :param num_point: размер массива со значениями энергии
        """
        self.num = num
        self.grid_step = grid_step
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.num_point = num_point
        self.l_cut = 2 * self.grid_step
        self.u_hat = None
        self.v_hat = None
        self.w_hat = None
        self.k_mag: np.ndarray = None
        self.e_k_mag: np.ndarray = None

    @classmethod
    def _get_energy(cls, m_grid: np.ndarray, energy_arr: np.ndarray, m_mag):
        """Вычисляет величину энергии в шаровом слое единичной толщины в простанстве m"""
        energy_arr_filt = energy_arr[(m_grid > m_mag - 0.5) * (m_grid < m_mag + 0.5)]
        energy = energy_arr_filt.sum()
        return energy

    @classmethod
    def make_spectrum(cls, num: int, length, u_hat: np.ndarray, v_hat: np.ndarray, w_hat: np.ndarray, m: np.ndarray,
                      num_pnt=100):
        logging.info('START CALCULATING SPECTRUM')
        # вычисляем индексы узлов сетки в волновом пространстве
        m_i = np.zeros([num, num, num])
        m_j = np.zeros([num, num, num])
        m_k = np.zeros([num, num, num])
        for im in range(num):
            for jm in range(num):
                for km in range(num):
                    m_i[im, jm, km] = m[im]
                    m_j[im, jm, km] = m[jm]
                    m_k[im, jm, km] = m[km]
        # находим расстоняния от узлов до центра в условном пространстве индексов
        m_grid = np.sqrt(m_i ** 2 + m_j ** 2 + m_k ** 2)
        # вычисляем энергию поля скоростей
        energy = 0.5 * (1 / num) ** 6 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2 + np.abs(w_hat) ** 2)
        # вычисляем энергетический спектр в физическом волновом пространстве
        m_mag = np.linspace(0, m_grid.max(), num_pnt)
        e_k_mag = np.zeros(num_pnt)
        for i in range(num_pnt):
            e_k_mag[i] = cls._get_energy(m_grid, energy, m_mag[i]) * length / (2 * np.pi)
            logging.debug('e_k_mag[%s] = %.5f' % (i, e_k_mag[i]))
        k_mag = m_mag * 2 * np.pi / length
        logging.info('FINISH CALCULATING SPECTRUM')
        return k_mag, e_k_mag

    def compute_spectrum(self):
        u_3d = self.u_arr.reshape([self.num, self.num, self.num])
        v_3d = self.v_arr.reshape([self.num, self.num, self.num])
        w_3d = self.w_arr.reshape([self.num, self.num, self.num])
        self.u_hat, self.v_hat, self.w_hat = make_fft(u_3d, v_3d, w_3d)
        n, m, x, k = make_fft_grid(self.num, self.grid_step * (self.num - 1))
        self.k_mag, self.e_k_mag = self.make_spectrum(self.num, self.grid_step * (self.num - 1), self.u_hat,
                                                      self.v_hat, self.w_hat, m, self.num_point)

    def get_turb_kinetic_energy(self):
        energy: np.ndarray = 0.5 * (1 / self.num) ** 6 * (np.abs(self.u_hat) ** 2 + np.abs(self.v_hat) ** 2 +
                                                          np.abs(self.w_hat) ** 2)
        return energy.sum()


def plot_spectrum_von_karman(k_arr, energy_arr, filename, l_cut, l_e, l_cut_min, l_e_max,
                             viscosity=config.viscosity, dissipation_rate=config.dissipation_rate,
                             alpha=config.alpha, xlim=(1, 10e2), ylim=(10e-6, 10e-1)):
    logging.info('Plotting spectrum')
    plt.figure(figsize=(9, 7))
    k_arr_predef = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    energy_arr_von_karman = get_von_karman_spectrum(k_arr_predef, l_cut, l_e, viscosity, dissipation_rate)
    plt.plot(k_arr_predef, energy_arr_von_karman, color='blue', lw=1, label=r'$Модифицированный\ спектр\ фон\ Кармана$')
    plt.plot(k_arr, energy_arr, color='green', lw=1, label=r'$Вычисленный\ спектр$')
    plt.plot([2 * np.pi / l_cut_min, 2 * np.pi / l_cut_min], [0, 2 * max(energy_arr)], lw=3, color='black',
             label=r'$k_{max}$')
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both')
    plt.legend(fontsize=12, loc=3)
    plt.xlabel(r'$k$', fontsize=20)
    plt.ylabel(r'$E$', fontsize=20)
    plt.savefig(filename)
    plt.show()


def plot_spectrum_exp(k, energy, k_fit, energy_fit, filename, xlim=(1, 10e2), ylim=(10e-6, 10e-1)):
    logging.info('Plotting spectrum')
    plt.figure(figsize=(9, 7))
    plt.plot(k, energy, lw=1, label='Synthetic field')
    plt.plot(k_fit, energy_fit, lw=1, label='Experiment')
    plt.grid()
    plt.xlabel('k', fontsize=10)
    plt.ylabel('E', fontsize=10)
    plt.legend(fontsize=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
    plt.grid(which='both')
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    vel = read_velocity_file(r'output\velocity.VEL')
    for i in range(len(vel[0])):
        print(vel[0][i], vel[1][i], vel[2][i])
