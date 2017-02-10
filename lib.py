import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import typing
import matplotlib.pyplot as plt
import logging


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def get_k_arr(l_e_max, l_cut_min, alpha=0.01) -> typing.List[float]:
    result = []
    k_max = 1.5 * 2 * np.pi / l_cut_min
    k_min = 0.5 * 2 * np.pi / l_e_max
    k_cur = k_min
    while k_cur <= k_max:
        result.append(k_cur)
        k_cur = k_min * (1 + alpha) ** len(result)
    return result


def get_tau(u0, l_e_max):
    if u0 != 0:
        return 2 * l_e_max / u0
    else:
        return np.inf


def get_energy_arr(k_arr: typing.List[float], l_cut, l_e, viscosity, dissipation_rate) -> typing.List[float]:
    result = []
    k_eta = 2 * np.pi * (viscosity ** 3 / dissipation_rate) ** (-0.25)
    k_e = 2 * np.pi / l_e
    k_cut = 2 * np.pi / l_cut
    for k in k_arr:
        f_eta = np.exp(-(12 * k / k_eta) ** 2)
        f_cut = np.exp(-(4 * max(k - 0.9 * k_cut, 0) / k_cut) ** 3)
        energy_cur = (k / k_e) ** 4 / (1 + 2.4 * (k / k_e) ** 2) ** (17 / 6) * f_eta * f_cut
        result.append(energy_cur)
    return result


def get_amplitude_arr(k_arr: typing.List[float], energy_arr: typing.List[float]) -> typing.List[float]:
    result = []
    norm_coef = 0
    for i in range(len(k_arr) - 1):
        delta_k = k_arr[i + 1] - k_arr[i]
        energy = energy_arr[i]
        norm_coef += delta_k * energy
    for i in range(len(k_arr) - 1):
        delta_k = k_arr[i + 1] - k_arr[i]
        energy = energy_arr[i]
        result.append(energy * delta_k / norm_coef)
    result.append(0)
    return result


def get_rotation_matrix_3d(rotation_axis: typing.List[float], rotation_angle) -> np.ndarray:
    x_rotation_axis = rotation_axis[0]
    y_rotation_axis = rotation_axis[1]
    z_rotation_axis = rotation_axis[2]
    theta = rotation_angle
    result = np.array([
        [np.cos(theta) + (1 - np.cos(theta)) * x_rotation_axis ** 2,
         (1 - np.cos(theta)) * x_rotation_axis * y_rotation_axis - np.sin(theta) * z_rotation_axis,
         (1 - np.cos(theta)) * x_rotation_axis * z_rotation_axis + np.sin(theta) * y_rotation_axis],
        [(1 - np.cos(theta)) * y_rotation_axis * x_rotation_axis + np.sin(theta) * z_rotation_axis,
         np.cos(theta) + (1 - np.cos(theta)) * y_rotation_axis ** 2,
         (1 - np.cos(theta)) * y_rotation_axis * z_rotation_axis - np.sin(theta) * x_rotation_axis],
        [(1 - np.cos(theta)) * z_rotation_axis * x_rotation_axis - np.sin(theta) * y_rotation_axis,
         (1 - np.cos(theta)) * z_rotation_axis * y_rotation_axis + np.sin(theta) * x_rotation_axis,
         np.cos(theta) + (1 - np.cos(theta)) * z_rotation_axis ** 2]
        ])
    return result


def get_d_vector_theta_and_phase() -> tuple:
    z = random.uniform(-1, 1)
    phi = random.uniform(0, 2 * np.pi)
    theta = np.arccos(z)
    result = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), z])
    return result, theta, phi


def get_sigma_vector(d_vector: np.ndarray, theta: float) -> np.ndarray:
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
    rotation_axis = [x_rotation_axis, y_rotation_axis, z_rotation_axis]
    rotation_matrix = get_rotation_matrix_3d(rotation_axis, theta)
    z_prime = 0
    phi_prime = random.uniform(0, 2 * np.pi)
    x_prime = np.cos(phi_prime)
    y_prime = np.sin(phi_prime)
    result = np.dot(linalg.inv(rotation_matrix), np.array([x_prime, y_prime, z_prime]))
    return result


def get_frequency():
    return random.normal(2 * np.pi, 2 * np.pi)


def get_auxiliary_pulsation_velocity_parameters(l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                                                alpha=0.01, u0=0):
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_energy_arr(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    d_vector_arr = []
    sigma_vector_arr = []
    phase_arr = []
    frequency_arr = []
    for i in range(len(k_arr)):
        d_vector, theta, phase = get_d_vector_theta_and_phase()
        d_vector_arr.append(d_vector)
        phase_arr.append(phase)
        sigma_vector = get_sigma_vector(d_vector, theta)
        sigma_vector_arr.append(sigma_vector)
        frequency = get_frequency()
        frequency_arr.append(frequency)
    return tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr


def get_auxiliary_pulsation_velocity(r_vector, t, tau, k_arr: typing.List[float], amplitude_arr: typing.List[float],
                                     d_vector_arr: typing.List[float], sigma_vector_arr: typing.List[float],
                                     phase_arr: typing.List[float], frequency_arr: typing.List[float]) -> np.ndarray:
    result = np.array([0.0, 0.0, 0.0])
    for i in range(len(k_arr)):
        result += 2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector_arr[i] * \
                  np.cos(k_arr[i] * np.dot(d_vector_arr[i], r_vector) + phase_arr[i] + frequency_arr[i] * t / tau)
    return result


def plot_spectrum(r_vector, t, filename, l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                  alpha=0.01, u0=0):
    plt.figure(figsize=(9, 7))
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_energy_arr(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    v_arr = []
    for i in range(len(k_arr)):
        d_vector, theta, phase = get_d_vector_theta_and_phase()
        sigma_vector = get_sigma_vector(d_vector, theta)
        frequency = get_frequency()
        v_arr.append(2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector * \
                     np.cos(k_arr[i] * np.dot(d_vector, r_vector) + phase + frequency * t / tau))
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
    plt.show()


class UniformGridAuxiliaryPulsationVelocityFieldGenerator:
    def __init__(self, i_cnt: int, j_cnt: int, k_cnt: int, tec_filename, plot3d_filename, velocity_filename,
                 grid_step, l_e, viscosity, dissipation_rate, alpha=0.01, u0=0, time=0):
        """
        :param i_cnt: количество ячеек в направлении орта i
        :param j_cnt: количество ячеек в направлении орта j
        :param k_cnt: количество ячеек в направлении орта k
        :param tec_filename: имя файла с выходными данными
        :param grid_step: шаг сетки
        :param l_e: размер наиболее энергонесущих вихрей
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
        self._i_arr = []
        self._j_arr = []
        self._k_arr = []
        self._x_arr = []
        self._y_arr = []
        self._z_arr = []
        self._u_arr = []
        self._v_arr = []
        self._w_arr = []
        self._vorticity_x_arr = None
        self._vorticity_y_arr = None
        self._vorticity_z_arr = None

    @classmethod
    def _get_index_arrays(cls, i_cnt, j_cnt, k_cnt):
        i_arr = []
        j_arr = []
        k_arr = []
        for k1 in range(k_cnt):
            for j1 in range(j_cnt):
                for i1 in range(i_cnt):
                    i_arr.append(i1)
                    j_arr.append(j1)
                    k_arr.append(k1)
        return i_arr, j_arr, k_arr

    @classmethod
    def _get_coordinates_arrays(cls, i_arr, j_arr, k_arr, grid_step):
        x_arr = []
        y_arr = []
        z_arr = []
        for i, j, k in zip(i_arr, j_arr, k_arr):
            x_arr.append(i * grid_step)
            y_arr.append(j * grid_step)
            z_arr.append(k * grid_step)
        return x_arr, y_arr, z_arr

    def _get_velocity_arrays(self, x_arr, y_arr, z_arr):
        logging.info('Velocity calculation')
        u_arr = []
        v_arr = []
        w_arr = []
        for i, j, k in zip(range(len(x_arr)), range(len(y_arr)), range(len(z_arr))):
            tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr = \
                get_auxiliary_pulsation_velocity_parameters(self.l_cut, self.l_e, self.l_cut, self.l_e,
                                                            self.viscosity, self.dissipation_rate,
                                                            self.alpha, self.u0)
            v_vector = get_auxiliary_pulsation_velocity([x_arr[i], y_arr[j], z_arr[k]], self.time, tau, k_arr,
                                                        amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr,
                                                        frequency_arr)
            logging.info('i = %s, j = %s, k = %s  ---  u = %.3f, v = %.3f, w = %.3f' %
                         (i, j, k, v_vector[0], v_vector[1], v_vector[2]))
            u_arr.append(v_vector[0])
            v_arr.append(v_vector[1])
            w_arr.append(v_vector[2])
        return u_arr, v_arr, w_arr

    def _create_tec_file(self, filename, i_arr, j_arr, k_arr, x_arr, y_arr, z_arr, u_arr, v_arr, w_arr,
                         vorticity_x_arr, vorticity_y_arr, vorticity_z_arr):
        logging.info('Creating TEC file')
        file = open(filename, 'w')
        file.write('VARIABLES = X Y Z I J K U V W VORT_X VORT_Y VORT_Z\n')
        file.write('ZONE I= %s J= %s K= %s\n' % (self.i_cnt, self.j_cnt, self.k_cnt))
        for i, j, k, x, y, z, u, v, w, vort_x, vort_y, vort_z in zip(i_arr, j_arr, k_arr, x_arr, y_arr, z_arr, u_arr,
                                                                     v_arr, w_arr, vorticity_x_arr, vorticity_y_arr,
                                                                     vorticity_z_arr):
            file.write('%s %s %s %s %s %s %s %s %s %s %s %s\n' % (x, y, z, i, j, k, u, v, w, vort_x, vort_y, vort_z))
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

    def _get_vorticity_arrays(self, i_arr: typing.List[int], j_arr: typing.List[int], k_arr: typing.List[int],
                              x_arr: typing.List[float], y_arr: typing.List[float], z_arr: typing.List[float],
                              u_arr: typing.List[float], v_arr: typing.List[float], w_arr: typing.List[float]):
        logging.info('Vorticity calculation')
        result_x = np.zeros(self.i_cnt * self.j_cnt * self.k_cnt)
        result_y = np.zeros(self.i_cnt * self.j_cnt * self.k_cnt)
        result_z = np.zeros(self.i_cnt * self.j_cnt * self.k_cnt)
        for i in range(len(u_arr)):
            if i_arr[i] != 0 and j_arr[i] != 0 and k_arr[i] != 0 and i_arr[i] != self.i_cnt - 1 and \
               j_arr[i] != self.j_cnt - 1 and k_arr[i] != self.k_cnt - 1:
                vort_x = self._get_y_derivative(w_arr, y_arr, i) - self._get_z_derivative(v_arr, z_arr, i)
                vort_y = self._get_z_derivative(u_arr, z_arr, i) - self._get_x_derivative(w_arr, x_arr, i)
                vort_z = self._get_x_derivative(v_arr, x_arr, i) - self._get_y_derivative(u_arr, y_arr, i)
                result_x[i] = vort_x
                result_y[i] = vort_y
                result_z[i] = vort_z
            else:
                vort_x = 0
                vort_y = 0
                vort_z = 0
                result_x[i] = vort_x
                result_y[i] = vort_y
                result_z[i] = vort_z
            logging.info('i = %s, j = %s, k = %s  ---  vort_x = %.3f, vort_y = %.3f, vort_z = %.3f' %
                         (i_arr[i], j_arr[i], k_arr[i], vort_x, vort_y, vort_z))
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
        self._i_arr, self._j_arr, self._k_arr = self._get_index_arrays(self.i_cnt, self.j_cnt, self.k_cnt)
        self._x_arr, self._y_arr, self._z_arr = self._get_coordinates_arrays(self._i_arr, self._j_arr, self._k_arr,
                                                                             self.grid_step)
        self._u_arr, self._v_arr, self._w_arr = self._get_velocity_arrays(self._x_arr, self._y_arr, self._z_arr)
        self._vorticity_x_arr, self._vorticity_y_arr, self._vorticity_z_arr = \
            self._get_vorticity_arrays(self._i_arr, self._j_arr, self._k_arr,
                                       self._x_arr, self._y_arr, self._z_arr,
                                       self._u_arr, self._v_arr, self._w_arr)
        self._create_tec_file(self.tec_filename, self._i_arr, self._j_arr, self._k_arr, self._x_arr, self._y_arr,
                              self._z_arr, self._u_arr, self._v_arr, self._w_arr, self._vorticity_x_arr,
                              self._vorticity_y_arr, self._vorticity_z_arr)
        self._create_plot3d_file(self.plot3d_filename, self.i_cnt, self.j_cnt, self.k_cnt, self._x_arr, self._y_arr,
                                 self._z_arr)
        self._create_velocity_file(self.velocity_filename, self._u_arr, self._v_arr, self._w_arr)
        logging.info('Finish')


if __name__ == '__main__':
    # plot_spectrum([0.1, 0.1, 0.1], 0, 'output\spectrum', 0.0005, 0.005, 0.0005, 0.005, 2e-5, 6e3, u0=0)
    turb_generator = UniformGridAuxiliaryPulsationVelocityFieldGenerator(3, 4, 5, 'output\Test.TEC',
                                                                         r'output\test_grid.PFD',
                                                                         r'output\velocity.VEL', 0.001, 0.005,
                                                                         2e-5, 6e3)
    # turb_generator.commit()
    t_arr = np.linspace(0, 3.0, 30000)
    u_arr = []
    v_arr = []
    w_arr = []
    uv_arr = []
    vw_arr = []
    uw_arr = []
    k_u_arr = []
    k_uw_arr = []
    for i in range(10):
        u_arr = []
        v_arr = []
        w_arr = []
        uv_arr = []
        vw_arr = []
        uw_arr = []
        tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr = \
            get_auxiliary_pulsation_velocity_parameters(0.0002, 0.002, 0.0002, 0.002, 2e-5, 6e3, u0=2)
        for t in t_arr:
            print(i, '', t)
            v_vector = get_auxiliary_pulsation_velocity([0.01, 0.01, 0.01], t, tau, k_arr, amplitude_arr, d_vector_arr,
                                                        sigma_vector_arr, phase_arr, frequency_arr)
            u_arr.append(v_vector[0])
            v_arr.append(v_vector[1])
            w_arr.append(v_vector[2])
            uv_arr.append(v_vector[0] * v_vector[1])
            vw_arr.append(v_vector[1] * v_vector[2])
            uw_arr.append(v_vector[0] * v_vector[2])
        k_u_arr.append(sum(u_arr) / len(u_arr))
        k_uw_arr.append(sum(uw_arr) / len(uw_arr))
    print(sum(k_u_arr) / len(k_u_arr))
    print(sum(k_uw_arr) / len(k_uw_arr))
    # print(sum(u_arr) / len(u_arr))
    # print(sum(uv_arr) / len(uv_arr))
    # plt.plot(t_arr, u_arr, 'r')`
    # plt.plot(t_arr, uv_arr, 'b')
    # plt.grid()
    # plt.show()

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # for i in range(500):
    #     d_vector, theta = get_d_vector_and_theta()
    #     sigma_vector = get_sigma_vector(d_vector, theta)
    #     print(np.dot(d_vector, sigma_vector))
    #     ax.plot(xs=[d_vector[0]], ys=[d_vector[1]], zs=[d_vector[2]], marker='o', color='r')
    # plt.show()


