import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import typing
import matplotlib.pyplot as plt
import logging
import numba as nb
import time


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def get_k_arr(l_e_max, l_cut_min, alpha=0.01) -> np.ndarray:
    """
    :param l_e_max: максимальная на всей расчетной области длина волны
        наиболее энергонесущих мод синтезированного поля пульсаций
    :param l_cut_min:  минимальная разрешаемая на всей расчетной области длина волны
    :param alpha: константа
    :return: массив модулей волновых чисел
    """
    k_max = 1.5 * 2 * np.pi / l_cut_min
    k_min = 0.5 * 2 * np.pi / l_e_max
    n = int(np.log(k_max / k_min) / np.log(1 + alpha))
    exponent_arr = np.array(np.linspace(1, n, n))
    result = k_min * (1 + alpha) ** exponent_arr
    return result


def get_tau(u0: np.ndarray, l_e_max):
    """
    :param u0: характерная скорость
    :param l_e_max: максимальная на всей расчетной области длина волны
           наиболее энергонесущих мод синтезированного поля пульсаций
    :return: временной масштаб
    """
    if linalg.norm(u0) != 0:
        return 2 * l_e_max / linalg.norm(u0)
    else:
        return np.inf


def get_energy_arr(k_arr: np.ndarray, l_cut, l_e, viscosity, dissipation_rate) -> np.ndarray:
    """
    :param k_arr:  массив модулей волновых чисел
    :param l_cut: инимальная длина волны в рассчитываемом узле
    :param l_e: максимальная в рассчитываемом узле длина волны
            наиболее энергонесущих мод синтезированного поля пульсаций
    :param viscosity: молекулярная вязкость
    :param dissipation_rate:  степень диссипации
    :return: энергетический спектр синтетического поля в данном узле
    """
    k_eta = 2 * np.pi * (viscosity ** 3 / dissipation_rate) ** (-0.25)
    k_e = 2 * np.pi / l_e
    k_cut = 2 * np.pi / l_cut
    f_eta = np.exp(-(12 * k_arr / k_eta) ** 2)
    arr = (k_arr - 0.9 * k_cut)
    f_cut = np.exp(-(4 * (arr > 0) * arr / k_cut) ** 3)
    result = (k_arr / k_e) ** 4 / (1 + 2.4 * (k_arr / k_e) ** 2) ** (17 / 6) * f_eta * f_cut
    return result


def get_amplitude_arr(k_arr: np.ndarray, energy_arr: np.ndarray) -> np.ndarray:
    """
    :param k_arr: массив модулей волновых чисел
    :param energy_arr: энергетический спектр синтетического поля
    :return: массив амплитуд мод поля пульсаций в данном узле
    """
    k_arr1 = k_arr[0: k_arr.shape[0] - 1]
    k_arr2 = k_arr[1: k_arr.shape[0]]
    delta_k = k_arr2 - k_arr1
    norm_coef = (energy_arr[0: energy_arr.shape[0] - 1] * delta_k).sum()
    result = np.zeros(k_arr.shape[0])
    result[0: k_arr.shape[0] - 1] = energy_arr[0: k_arr.shape[0] - 1] * delta_k / norm_coef
    return result


def get_rotation_matrix_3d(rotation_axis: np.ndarray, rotation_angle) -> np.ndarray:
    x_rotation_axis = rotation_axis[0]
    y_rotation_axis = rotation_axis[1]
    z_rotation_axis = rotation_axis[2]
    theta = rotation_angle
    result = np.zeros([3, 3])
    result[0, 0] = np.cos(theta) + (1 - np.cos(theta)) * x_rotation_axis ** 2
    result[0, 1] = (1 - np.cos(theta)) * x_rotation_axis * y_rotation_axis - np.sin(theta) * z_rotation_axis
    result[0, 2] = (1 - np.cos(theta)) * x_rotation_axis * z_rotation_axis + np.sin(theta) * y_rotation_axis
    result[1, 0] = (1 - np.cos(theta)) * y_rotation_axis * x_rotation_axis + np.sin(theta) * z_rotation_axis
    result[1, 1] = np.cos(theta) + (1 - np.cos(theta)) * y_rotation_axis ** 2
    result[1, 2] = (1 - np.cos(theta)) * y_rotation_axis * z_rotation_axis - np.sin(theta) * x_rotation_axis
    result[2, 0] = (1 - np.cos(theta)) * z_rotation_axis * x_rotation_axis - np.sin(theta) * y_rotation_axis
    result[2, 1] = (1 - np.cos(theta)) * z_rotation_axis * y_rotation_axis + np.sin(theta) * x_rotation_axis
    result[2, 2] = np.cos(theta) + (1 - np.cos(theta)) * z_rotation_axis ** 2
    return result


def get_d_vector_theta_and_phase(size=0) -> tuple:
    z = random.uniform(-1, 1, size)
    phi = random.uniform(0, 2 * np.pi, size)
    theta = np.arccos(z)
    result = np.zeros([size, 3])
    result[:, 0] = np.sin(theta) * np.cos(phi)
    result[:, 1] = np.sin(theta) * np.sin(phi)
    result[:, 2] = z
    return result, theta, phi


def get_z(size=0) -> np.ndarray:
    z = random.uniform(-1, 1, size)
    return z


def get_phase(size=0) -> np.ndarray:
    phi = random.uniform(0, 2 * np.pi, size)
    return phi


def get_theta(z: np.ndarray) -> np.ndarray:
    theta = np.arccos(z)
    return theta


def get_d_vector(z: np.ndarray, phase: np.ndarray, theta: np.ndarray) -> np.ndarray:
    result = np.zeros([z.shape[0], 3])
    result[:, 0] = np.sin(theta) * np.cos(phase)
    result[:, 1] = np.sin(theta) * np.sin(phase)
    result[:, 2] = z
    return result


def get_sigma_vector(d_vector: np.ndarray, theta) -> np.ndarray:
    z_rotation_axis = 0
    if (d_vector[1] >= 0) and (d_vector[0] > 0):
        x_rotation_axis = 1 / np.sqrt(1 + (d_vector[0] / d_vector[1]) ** 2)
        y_rotation_axis = -np.sqrt(1 - x_rotation_axis ** 2)
    elif (d_vector[0] <= 0) and (d_vector[1] > 0):
        x_rotation_axis = 1 / np.sqrt(1 + (d_vector[0] / d_vector[1]) ** 2)
        y_rotation_axis = np.sqrt(1 - x_rotation_axis ** 2)
    elif (d_vector[0] < 0) and (d_vector[1] <= 0):
        x_rotation_axis = -1 / np.sqrt(1 + (d_vector[0] / d_vector[1]) ** 2)
        y_rotation_axis = np.sqrt(1 - x_rotation_axis ** 2)
    else:
        x_rotation_axis = -1 / np.sqrt(1 + (d_vector[0] / d_vector[1]) ** 2)
        y_rotation_axis = -np.sqrt(1 - x_rotation_axis ** 2)
    rotation_axis = np.array([x_rotation_axis, y_rotation_axis, z_rotation_axis])
    rotation_matrix = get_rotation_matrix_3d(rotation_axis, theta)
    phi_prime = random.uniform(0, 2 * np.pi)
    vector_prime = np.zeros(3)
    vector_prime[0] = np.cos(phi_prime)
    vector_prime[1] = np.sin(phi_prime)
    vector_prime[2] = 0
    result = np.dot(linalg.inv(rotation_matrix), vector_prime)
    return result


def get_frequency(size=0):
    return random.normal(2 * np.pi, 2 * np.pi, size=size)


def get_sigma_vector_array(d_vector_arr: np.ndarray, theta_arr: np.ndarray) -> np.ndarray:
    sigma_vector_arr = np.zeros([d_vector_arr.shape[0], 3])
    for i in range(d_vector_arr.shape[0]):
        sigma_vector_arr[i] = get_sigma_vector(d_vector_arr[i], theta_arr[i])
    return sigma_vector_arr


def get_auxiliary_pulsation_velocity_parameters(l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                                                alpha=0.01, u0=0.):
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_energy_arr(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    d_vector_arr, theta_arr, phase_arr = get_d_vector_theta_and_phase(k_arr.shape[0])
    frequency_arr = get_frequency(k_arr.shape[0])
    sigma_vector_arr = get_sigma_vector_array(d_vector_arr, theta_arr)
    return tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr


def plot_spectrum(r_vector, t, filename, l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                  alpha=0.01, u0=0):
    plt.figure(figsize=(9, 7))
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_energy_arr(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    d_vector, theta, phase = get_d_vector_theta_and_phase(k_arr.shape[0])
    frequency = get_frequency(k_arr.shape[0])
    v_arr = []
    for i in range(k_arr.shape[0]):
        sigma_vector = get_sigma_vector(d_vector[i], theta[i])
        v_arr.append(2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector *
                     np.cos(k_arr[i] * np.dot(d_vector[i], r_vector) + phase[i] + frequency[i] * t / tau))
    energy_arr_new = [linalg.norm(v) ** 2 for v in v_arr]
    plt.plot(k_arr, energy_arr, color='blue', lw=1, label=r'$Спектр\ фон\ Кармана$')
    plt.plot(k_arr, energy_arr_new, color='red', lw=1, label=r'$Спектр\ синтетического\ поля$')
    plt.plot([2 * np.pi / l_cut_min, 2 * np.pi / l_cut_min], [0, 2 * max(energy_arr_new)], lw=3, color='black',
             label=r'$k_{max}$')
    plt.ylim(10e-10, 1.1 * max(energy_arr_new))
    plt.xlim(min(k_arr), max(k_arr))
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both')
    plt.legend(fontsize=16, loc=3)
    plt.xlabel(r'$k$', fontsize=20)
    plt.ylabel(r'$E$', fontsize=20)
    plt.savefig(filename)


@nb.jit(nb.double[:](nb.double[:], nb.double, nb.double, nb.double[:], nb.double[:], nb.double[:, :],
                     nb.double[:, :], nb.double[:], nb.double[:]))
def get_auxiliary_pulsation_velocity(r_vector, t, tau, k_arr: np.ndarray, amplitude_arr: np.ndarray,
                                     d_vector_arr: np.ndarray, sigma_vector_arr: np.ndarray,
                                     phase_arr: np.ndarray, frequency_arr: np.ndarray) -> np.ndarray:
    result = np.array([0., 0., 0.])
    for i in range(len(k_arr)):
        result += (2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector_arr[i] *
                np.cos(k_arr[i] * np.dot(d_vector_arr[i], r_vector) + phase_arr[i] + frequency_arr[i] * t / tau))
    return result


class HomogeneousIsotropicTurbulenceGenerator:
    """
    Предосталяет интерфейс для генерации на равномерной сетке в области в форме прямоугольного
        параллелепипеда поля однородной изотропоной турбулентности и сохранения данных о
        пульсациях и  завихренности в текстовых файлах.
    """
    def __init__(self, i_cnt: int, j_cnt: int, k_cnt: int, tec_filename, plot3d_filename, velocity_filename,
                 grid_step, l_e, viscosity, dissipation_rate, alpha=0.01, u0=np.array([0., 0., 0.]), time=0.):
        """
        :param i_cnt: количество ячеек в направлении орта i
        :param j_cnt: количество ячеек в направлении орта j
        :param k_cnt: количество ячеек в направлении орта k
        :param tec_filename: имя файла с выходными данными
        :param grid_step: шаг сетки
        :param l_e: длина волны наиболее энергонесущих мод синтезированного поля пульсаций
        :param viscosity: молекулярная вязкость
        :param dissipation_rate: степень диссипации
        :param alpha: константа для определения набора волновых чисел
        :param u0: характерная скорость
        :param time: параметр времени
        """
        self.i_cnt = i_cnt
        self.j_cnt = j_cnt
        self.k_cnt = k_cnt
        self.tec_filename = tec_filename
        self.plot3d_filename = plot3d_filename
        self.velocity_filename = velocity_filename
        self.grid_step = grid_step
        self.l_e = l_e
        self.viscosity = viscosity
        self.dissipation_rate = dissipation_rate
        self.alpha = alpha
        self.u0 = u0
        self.time = time
        self.l_cut = 2 * self.grid_step
        self._index_gen = None
        self._index_gen = None
        self._x_arr = []
        self._y_arr = []
        self._z_arr = []
        self._u_arr = []
        self._v_arr = []
        self._w_arr = []
        self._vorticity_x_arr = None
        self._vorticity_y_arr = None
        self._vorticity_z_arr = None

    Vector = typing.TypeVar('Vector', typing.Iterable[int], typing.Iterable[float])

    @classmethod
    def _get_index_generator(cls, i_cnt, j_cnt, k_cnt) -> typing.Iterator[Vector]:
        for k1 in range(k_cnt):
            for j1 in range(j_cnt):
                for i1 in range(i_cnt):
                    yield i1, j1, k1

    @classmethod
    def _get_coordinates_arrays(cls, index_gen: typing.Iterator[Vector], grid_step) -> \
            typing.Iterable[np.ndarray]:
        x_arr = []
        y_arr = []
        z_arr = []
        for index_vector in index_gen:
             x_arr.append(index_vector[0] * grid_step)
             y_arr.append(index_vector[1] * grid_step)
             z_arr.append(index_vector[2] * grid_step)
        return np.array(x_arr), np.array(y_arr), np.array(z_arr)

    def _create_tec_file(self, filename, index_gen: typing.Iterator[Vector], x_arr, y_arr, z_arr, u_arr, v_arr, w_arr,
                         vorticity_x_arr, vorticity_y_arr, vorticity_z_arr):
        logging.info('Creating TEC file')
        file = open(filename, 'w')
        file.write('VARIABLES = X Y Z I J K U V W VORT_X VORT_Y VORT_Z\n')
        file.write('ZONE I= %s J= %s K= %s\n' % (self.i_cnt, self.j_cnt, self.k_cnt))
        for index_vector, x, y, z, u, v, w, vort_x, vort_y, vort_z in zip(index_gen, x_arr, y_arr, z_arr, u_arr,
                                                                     v_arr, w_arr, vorticity_x_arr, vorticity_y_arr,
                                                                     vorticity_z_arr):
            file.write('%s %s %s %s %s %s %s %s %s %s %s %s\n' % (x, y, z, index_vector[0], index_vector[1],
                                                                  index_vector[2], u, v, w, vort_x, vort_y, vort_z))
        file.close()

    @classmethod
    def _get_index_shift(cls, parameter_arr: typing.List[float], number: int, j_cnt: int, i_cnt: int, delta_i: int = 1,
                         delta_j: int = 1,  delta_k: int = 1):
        return parameter_arr[number + delta_i + i_cnt * delta_j + i_cnt * j_cnt * delta_k]

    def _get_x_derivative(self, parameter_arr: typing.List[float], x_arr: typing.List[float], number: int):
        dp = self._get_index_shift(parameter_arr, number, self.j_cnt, self.i_cnt, 1, 0, 0) - \
            self._get_index_shift(parameter_arr, number, self.j_cnt, self.i_cnt, -1, 0, 0)
        dx = self._get_index_shift(x_arr, number, self.j_cnt, self.i_cnt, 1, 0, 0) - \
            self._get_index_shift(x_arr, number, self.j_cnt, self.i_cnt, -1, 0, 0)
        return dp / dx

    def _get_y_derivative(self, parameter_arr: typing.List[float], y_arr: typing.List[float], number: int):
        dp = self._get_index_shift(parameter_arr, number, self.j_cnt, self.i_cnt, 0, 1, 0) - \
             self._get_index_shift(parameter_arr, number, self.j_cnt, self.i_cnt, 0, -1, 0)
        dy = self._get_index_shift(y_arr, number, self.j_cnt, self.i_cnt, 0, 1, 0) - \
             self._get_index_shift(y_arr, number, self.j_cnt, self.i_cnt, 0, -1, 0)
        return dp / dy

    def _get_z_derivative(self, parameter_arr: typing.List[float], z_arr: typing.List[float], number: int):
        dp = self._get_index_shift(parameter_arr, number, self.j_cnt, self.i_cnt, 0, 0, 1) - \
             self._get_index_shift(parameter_arr, number, self.j_cnt, self.i_cnt, 0, 0, -1)
        dz = self._get_index_shift(z_arr, number, self.j_cnt, self.i_cnt, 0, 0, 1) - \
             self._get_index_shift(z_arr, number, self.j_cnt, self.i_cnt, 0, 0, -1)
        return dp / dz

    def _get_velocity_arrays(self, x_arr: np.ndarray, y_arr: np.ndarray,
                             z_arr: np.ndarray) -> np.ndarray:
        logging.info('Velocity calculation')
        u_arr = np.zeros(x_arr.shape[0])
        v_arr = np.zeros(x_arr.shape[0])
        w_arr = np.zeros(x_arr.shape[0])
        result = np.zeros([3, x_arr.shape[0]])
        tau = get_tau(self.u0, self.l_e)
        k_arr = get_k_arr(self.l_e, self.l_cut, self.alpha)
        energy_arr = get_energy_arr(k_arr, self.l_cut, self.l_e, self.viscosity, self.dissipation_rate)
        amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
        for i in range(x_arr.shape[0]):
            z = get_z(k_arr.shape[0])
            phase_arr = get_phase(k_arr.shape[0])
            theta_arr = get_theta(z)
            frequency_arr = get_frequency(k_arr.shape[0])
            d_vector_arr = get_d_vector(z, phase_arr, theta_arr)
            sigma_vector_arr = get_sigma_vector_array(d_vector_arr, theta_arr)
            v_vector = get_auxiliary_pulsation_velocity(np.array([x_arr[i], y_arr[i], z_arr[i]]), self.time, tau, k_arr,
                                                        amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr,
                                                        frequency_arr)
            logging.info('n = %s  ---  u = %.3f, v = %.3f, w = %.3f' %
                         (i, v_vector[0], v_vector[1], v_vector[2]))
            u_arr[i] = v_vector[0] + self.u0[0]
            v_arr[i] = v_vector[1] + self.u0[1]
            w_arr[i] = v_vector[2] + self.u0[2]
        result[0] = u_arr
        result[1] = v_arr
        result[2] = w_arr
        return result

    def _get_vorticity_arrays(self, index_gen: typing.Iterator[Vector],
                              x_arr: typing.List[float], y_arr: typing.List[float], z_arr: typing.List[float],
                              u_arr: typing.List[float], v_arr: typing.List[float], w_arr: typing.List[float]):
        logging.info('Vorticity calculation')
        result_x = np.zeros(self.i_cnt * self.j_cnt * self.k_cnt)
        result_y = np.zeros(self.i_cnt * self.j_cnt * self.k_cnt)
        result_z = np.zeros(self.i_cnt * self.j_cnt * self.k_cnt)
        n = 0
        for u, index_vector in zip(range(len(u_arr)), index_gen):
            if index_vector[0] != 0 and index_vector[1] != 0 and index_vector[2] != 0 and \
               index_vector[0] != self.i_cnt - 1 and \
               index_vector[1] != self.j_cnt - 1 and index_vector[2] != self.k_cnt - 1:
                vort_x = self._get_y_derivative(w_arr, y_arr, u) - self._get_z_derivative(v_arr, z_arr, u)
                vort_y = self._get_z_derivative(u_arr, z_arr, u) - self._get_x_derivative(w_arr, x_arr, u)
                vort_z = self._get_x_derivative(v_arr, x_arr, u) - self._get_y_derivative(u_arr, y_arr, u)
                result_x[u] = vort_x
                result_y[u] = vort_y
                result_z[u] = vort_z
            else:
                vort_x = 0
                vort_y = 0
                vort_z = 0
                result_x[u] = vort_x
                result_y[u] = vort_y
                result_z[u] = vort_z
            logging.info('n = %s  ---  vort_x = %.3f, vort_y = %.3f, vort_z = %.3f' %
                         (n, vort_x, vort_y, vort_z))
            n += 1
        return result_x, result_y, result_z

    @classmethod
    def _create_plot3d_file(cls, filename, i_cnt, j_cnt, k_cnt, x_arr, y_arr, z_arr):
        logging.info('Creating PLOT3D file')
        file = open(filename, mode='w', encoding='ascii')
        file.write('1 \n')
        file.write('%s %s %s \n' % (i_cnt, j_cnt, k_cnt))
        for x in x_arr:
            file.write('%s ' % x)
        for y in y_arr:
            file.write('%s ' % y)
        for z in z_arr:
            file.write('%s ' % z)
        file.close()

    @classmethod
    def _create_velocity_file(cls, filename, u_arr: typing.List[float], v_arr: typing.List[float],
                              w_arr: typing.List[float]):
        logging.info('Creating velocity file')
        file = open(filename, mode='w', encoding='utf-8')
        for u, v, w in zip(u_arr, v_arr, w_arr):
            file.write('%s %s %s \n' % (u, v, w))
        file.close()

    def commit(self):
        self._index_gen = self._get_index_generator(self.i_cnt, self.j_cnt, self.k_cnt)
        self._x_arr, self._y_arr, self._z_arr = self._get_coordinates_arrays(self._index_gen, self.grid_step)
        start = time.time()
        velocity_vector_arr = self._get_velocity_arrays(self._x_arr, self._y_arr, self._z_arr)
        end = time.time()
        self._u_arr = velocity_vector_arr[0]
        self._v_arr = velocity_vector_arr[1]
        self._w_arr = velocity_vector_arr[2]
        self._vorticity_x_arr, self._vorticity_y_arr, self._vorticity_z_arr = \
            self._get_vorticity_arrays(self._index_gen, self._x_arr, self._y_arr, self._z_arr,
                                       self._u_arr, self._v_arr, self._w_arr)
        self._create_tec_file(self.tec_filename, self._index_gen, self._x_arr, self._y_arr,
                              self._z_arr, self._u_arr, self._v_arr, self._w_arr, self._vorticity_x_arr,
                              self._vorticity_y_arr, self._vorticity_z_arr)
        self._create_plot3d_file(self.plot3d_filename, self.i_cnt, self.j_cnt, self.k_cnt, self._x_arr, self._y_arr,
                                 self._z_arr)
        self._create_velocity_file(self.velocity_filename, self._u_arr, self._v_arr, self._w_arr)
        logging.info('Finish')
        logging.info('Velocity calculation time is %.4f s' % (end - start))


if __name__ == '__main__':
    # plot_spectrum([0.1, 0.1, 0.1], 0, 'output\spectrum', 0.0005, 0.005, 0.0005, 0.005, 2e-5, 6e3, u0=0)
    turb_generator = HomogeneousIsotropicTurbulenceGenerator(3, 4, 5, 'output\Test.TEC',
                                                                         r'output\test_grid.PFD',
                                                                         r'output\velocity.VEL', 0.001, 0.005,
                                                             2e-5, 6e3)
    # turb_generator.commit()
    # t_arr = np.linspace(0, 3.0, 30000)
    # u_arr = []
    # v_arr = []
    # w_arr = []
    # uv_arr = []
    # vw_arr = []
    # uw_arr = []
    # k_u_arr = []
    # k_uw_arr = []
    # for i in range(10):
    #     u_arr = []
    #     v_arr = []
    #     w_arr = []
    #     uv_arr = []
    #     vw_arr = []
    #     uw_arr = []
    #     tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr = \
    #         get_auxiliary_pulsation_velocity_parameters(0.0002, 0.002, 0.0002, 0.002, 2e-5, 6e3, u0=2)
    #     for t in t_arr:
    #         print(i, '', t)
    #         v_vector = get_auxiliary_pulsation_velocity([0.01, 0.01, 0.01], t, tau, k_arr, amplitude_arr, d_vector_arr,
    #                                                     sigma_vector_arr, phase_arr, frequency_arr)
    #         u_arr.append(v_vector[0])
    #         v_arr.append(v_vector[1])
    #         w_arr.append(v_vector[2])
    #         uv_arr.append(v_vector[0] * v_vector[1])
    #         vw_arr.append(v_vector[1] * v_vector[2])
    #         uw_arr.append(v_vector[0] * v_vector[2])
    #     k_u_arr.append(sum(u_arr) / len(u_arr))
    #     k_uw_arr.append(sum(uw_arr) / len(uw_arr))
    # print(sum(k_u_arr) / len(k_u_arr))
    # print(sum(k_uw_arr) / len(k_uw_arr))


    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # for i in range(500):
    #     d_vector, theta, phase = get_d_vector_theta_and_phase(size=1)
    #     sigma_vector = get_sigma_vector(d_vector[0], theta[0])
    #     print(np.dot(d_vector[0], sigma_vector))
    #     # ax.plot(xs=[d_vector[0, 0]], ys=[d_vector[0, 1]], zs=[d_vector[0, 2]], marker='o', color='r')
    #     ax.plot(xs=[sigma_vector[0]], ys=[sigma_vector[1]], zs=[sigma_vector[2]], marker='o', color='g')
    # plt.show()


