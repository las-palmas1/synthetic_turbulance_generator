from core import dhit_ic_lib as diht_lib, les_inlet_ic_lib as les_lib
import logging
import config
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import os
import typing
from abc import ABCMeta, abstractmethod

logging.basicConfig(format='%(levelname)s: %(message)s', level=config.log_level)
Vector = typing.TypeVar('Vector', typing.Iterable[int], typing.Iterable[float])


class HITGenerator(metaclass=ABCMeta):
    """
    Базовый класс для всех классов, генериующих начальные условия для задачи распада
    однородной изотропной турбулентности
    """
    def __init__(self, num, data_files_dir: str, grid_step):
        self.num = num
        self.data_files_dir = data_files_dir
        self.tec_filename = os.path.join(data_files_dir, 'synthetic_turbulence_field.TEC')
        self.cell_centered_grid_filename = os.path.join(data_files_dir, 'grid_cell_centered.PFG')
        self.node_grid_filename = os.path.join(data_files_dir, 'grid_node.PFG')
        self.velocity_filename = os.path.join(data_files_dir, 'velocity.VEL')
        self.velocity_num_cells_filename = os.path.join(data_files_dir, 'velocity_%s_cells.VEL' % self.num)
        self.grid_step = grid_step

    @abstractmethod
    def run(self, **kwargs):
        pass

    @classmethod
    def _get_cell_center_coordinates_arrays(cls, index_gen: typing.Iterator[Vector], grid_step) -> \
            typing.Iterable[np.ndarray]:
        """Возвращает координаты центров ячеек, в которых будут генерироваться скорости"""
        x_arr = []
        y_arr = []
        z_arr = []
        for index_vector in index_gen:
            x_arr.append(index_vector[0] * grid_step)
            y_arr.append(index_vector[1] * grid_step)
            z_arr.append(index_vector[2] * grid_step)
        return np.array(x_arr), np.array(y_arr), np.array(z_arr)

    @abstractmethod
    def _get_velocity_arrays(self):
        pass

    @classmethod
    def _get_index_generator(cls, num: int) -> typing.Iterator[Vector]:
        for k1 in range(num):
            for j1 in range(num):
                for i1 in range(num):
                    yield i1, j1, k1

    def _create_tec_file(self, filename, index_gen: typing.Iterator[Vector], x_arr, y_arr, z_arr, u_arr, v_arr, w_arr,
                         vorticity_x_arr, vorticity_y_arr, vorticity_z_arr):
        logging.info('Creating TEC file')
        file = open(filename, 'w')
        file.write('VARIABLES = X Y Z I J K U V W VORT_X VORT_Y VORT_Z\n')
        file.write('ZONE I= %s J= %s K= %s\n' % (self.num, self.num, self.num))
        for index_vector, x, y, z, u, v, w, vort_x, vort_y, vort_z in zip(index_gen, x_arr, y_arr, z_arr, u_arr,
                                                                          v_arr, w_arr, vorticity_x_arr,
                                                                          vorticity_y_arr, vorticity_z_arr):
            file.write('%s %s %s %s %s %s %s %s %s %s %s %s\n' % (x, y, z, index_vector[0], index_vector[1],
                                                                  index_vector[2], u, v, w, vort_x, vort_y, vort_z))
        file.close()

    @classmethod
    def _get_index_shift(cls, parameter_arr: np.ndarray, number: int, num: int, delta_i: int = 1,
                         delta_j: int = 1, delta_k: int = 1):
        return parameter_arr[number + delta_i + num * delta_j + num * num * delta_k]

    def _get_x_derivative(self, parameter_arr: np.ndarray, x_arr: np.ndarray, number: int):
        dp = self._get_index_shift(parameter_arr, number, self.num, 1, 0, 0) - \
             self._get_index_shift(parameter_arr, number, self.num, -1, 0, 0)
        dx = self._get_index_shift(x_arr, number, self.num, 1, 0, 0) - \
             self._get_index_shift(x_arr, number, self.num, -1, 0, 0)
        return dp / dx

    def _get_y_derivative(self, parameter_arr: np.ndarray, y_arr: np.ndarray, number: int):
        dp = self._get_index_shift(parameter_arr, number, self.num, 0, 1, 0) - \
             self._get_index_shift(parameter_arr, number, self.num, 0, -1, 0)
        dy = self._get_index_shift(y_arr, number, self.num, 0, 1, 0) - \
             self._get_index_shift(y_arr, number, self.num, 0, -1, 0)
        return dp / dy

    def _get_z_derivative(self, parameter_arr: np.ndarray, z_arr: np.ndarray, number: int):
        dp = self._get_index_shift(parameter_arr, number, self.num, 0, 0, 1) - \
             self._get_index_shift(parameter_arr, number, self.num, 0, 0, -1)
        dz = self._get_index_shift(z_arr, number, self.num, 0, 0, 1) - \
             self._get_index_shift(z_arr, number, self.num, 0, 0, -1)
        return dp / dz

    def _get_vorticity_arrays(self, index_gen: typing.Iterator[Vector],
                              x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray,
                              u_arr: np.ndarray, v_arr: np.ndarray, w_arr: np.ndarray):
        logging.info('Vorticity calculation')
        result_x = np.zeros(self.num**3)
        result_y = np.zeros(self.num**3)
        result_z = np.zeros(self.num**3)
        n = 0
        for u, index_vector in zip(range(len(u_arr)), index_gen):
            if index_vector[0] != 0 and index_vector[1] != 0 and index_vector[2] != 0 and \
               index_vector[0] != self.num - 1 and \
               index_vector[1] != self.num - 1 and index_vector[2] != self.num - 1:
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
            logging.debug('n = %s  ---  vort_x = %.3f, vort_y = %.3f, vort_z = %.3f' %
                          (n, vort_x, vort_y, vort_z))
            n += 1
        return result_x, result_y, result_z

    def _create_velocity_component_file(self, x_arr, y_arr, z_arr, vel_arr, component_name):
        """
        Создает .csv файл, в котором содержаться значения координат узлов и одной и компонент скорости

        :param x_arr: массив координат по x
        :param y_arr: массив координат по y
        :param z_arr: массив координат по z
        :param vel_arr: массив значений компоненты скорости
        :param component_name: имя компоненты скорости
        :return: None
        """
        logging.info('Creation of %s velocity component file' % component_name)
        frame = pd.DataFrame.from_records([[x, y, z, vel] for x, y, z, vel in zip(x_arr, y_arr, z_arr, vel_arr)])
        frame.to_csv(os.path.join(self.data_files_dir, '%s_velocity.txt' % component_name), header=False, index=False,
                     sep=',')

    @classmethod
    def _create_plot3d_file(cls, filename, num, x_arr, y_arr, z_arr):
        """Создает файл кубической сетки  с заданным количеством узлов на сторонев формате Plot3d,
        в котором сохраняются значения координат из заданных одномерных массивов"""
        logging.info('Creating PLOT3D file')
        file = open(filename, mode='w', encoding='ascii')
        file.write('1 \n')
        file.write('%s %s %s \n' % (num, num, num))
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
        """Создает файл с расширением .VEL, в котором будут содержаться начальные условия для Lazurit"""
        logging.info('Creating velocity file')
        file = open(filename, mode='w', encoding='utf-8')
        for u, v, w in zip(u_arr, v_arr, w_arr):
            file.write('%s %s %s\n' % (u, v, w))
        file.close()

    def _create_all_files(self, x_arr, y_arr, z_arr, u_arr, v_arr, w_arr,
                          vorticity_x_arr, vorticity_y_arr, vorticity_z_arr):
        index_gen = self._get_index_generator(self.num)
        self._create_tec_file(self.tec_filename, index_gen, x_arr, y_arr,
                              z_arr, u_arr, v_arr, w_arr, vorticity_x_arr,
                              vorticity_y_arr, vorticity_z_arr)
        self._create_plot3d_file(self.cell_centered_grid_filename, self.num, x_arr, y_arr, z_arr)
        self._create_velocity_file(self.velocity_filename, u_arr, v_arr, w_arr)
        self._create_velocity_file(self.velocity_num_cells_filename, u_arr, v_arr, w_arr)
        self._create_velocity_component_file(x_arr, y_arr, z_arr, u_arr, 'u')
        self._create_velocity_component_file(x_arr, y_arr, z_arr, v_arr, 'v')
        self._create_velocity_component_file(x_arr, y_arr, z_arr, w_arr, 'w')
        self._create_node_mesh()

    @classmethod
    def _get_nodes_coordinates_arrays(cls, index_gen: typing.Iterator[Vector], grid_step) -> \
            typing.Iterable[np.ndarray]:
        """Возвращает координаты узлов сетки"""
        x_arr = []
        y_arr = []
        z_arr = []
        for index_vector in index_gen:
            x_arr.append((index_vector[0] - 0.5) * grid_step)
            y_arr.append((index_vector[1] - 0.5) * grid_step)
            z_arr.append((index_vector[2] - 0.5) * grid_step)
        return np.array(x_arr), np.array(y_arr), np.array(z_arr)

    def _create_node_mesh(self):
        """Создает и сохраняет файл в формате Plot3d, содержащй координаты узлов"""
        index_gen = self._get_index_generator(self.num + 1)
        x_arr, y_arr, z_arr = self._get_nodes_coordinates_arrays(index_gen, self.grid_step)
        self._create_plot3d_file(self.node_grid_filename, self.num + 1, x_arr, y_arr, z_arr)


class HITGeneratorVonKarman(HITGenerator):
    """
        Предоставляет интерфейс для генерации на равномерной сетке в области в форме куба
        поля однородной изотропоной турбулентности со спектром, соответствующем формуле фон Кармана,
        и сохранения данных о пульсациях и  завихренности в текстовых файлах.
    """

    def __init__(self, num, data_files_dir: str, grid_step, l_e, viscosity, dissipation_rate,
                 alpha=0.01, u0=np.array([0., 0., 0.]), time=0.):
        """
        :param num: количество узлов в на стороне генерируемого куба
        :param data_files_dir: имя файла с выходными данными
        :param grid_step: шаг сетки
        :param l_e: длина волны наиболее энергонесущих мод синтезированного поля пульсаций
        :param viscosity: молекулярная вязкость
        :param dissipation_rate: степень диссипации
        :param alpha: константа для определения набора волновых чисел
        :param u0: характерная скорость
        :param time: параметр времени
        """
        HITGenerator.__init__(self, num, data_files_dir, grid_step)
        self.l_e = l_e
        self.viscosity = viscosity
        self.dissipation_rate = dissipation_rate
        self.alpha = alpha
        self.u0 = u0
        self.time = time
        self.l_cut = 2 * self.grid_step
        self._x_arr: np.ndarray = None
        self._y_arr: np.ndarray = None
        self._z_arr: np.ndarray = None
        self.u_arr = None
        self.v_arr = None
        self.w_arr = None
        self._vorticity_x_arr = None
        self._vorticity_y_arr = None
        self._vorticity_z_arr = None

    def _get_velocity(self, x, y, z, tau, k, amplitude, phase, d_vector, frequency, sigma_vector) -> tuple:
        """
        Возвращает значение скоростей в точке с заданными координатами
        :param x:
        :param y:
        :param z:
        :return: tuple
        """
        v_vector = les_lib.get_auxiliary_pulsation_velocity(np.array([x, y, z]), self.time, tau, k,
                                                            amplitude, d_vector, sigma_vector, phase,
                                                            frequency)
        logging.info(' u = %.3f, v = %.3f, w = %.3f' %
                     (v_vector[0], v_vector[1], v_vector[2]))
        u = v_vector[0] + self.u0[0]
        v = v_vector[1] + self.u0[1]
        w = v_vector[2] + self.u0[2]
        return u, v, w

    def _get_velocity_arrays_mp_pool(self, proc_num: int = 2):
        """
        Вычисление скоростей в многопроцессорном режиме
        """
        logging.info('Velocity calculation in mp mode')
        logging.info('Processes number: %s' % proc_num)
        start = time.time()
        tau = les_lib.get_tau(self.u0, self.l_e)
        k = les_lib.get_k_arr(self.l_e, self.l_cut, self.alpha)
        energy = les_lib.get_von_karman_spectrum(k, self.l_cut, self.l_e, self.viscosity, self.dissipation_rate)
        amplitude = les_lib.get_amplitude_arr(k, energy)
        z = les_lib.get_z(k.shape[0])
        phase = les_lib.get_phase(k.shape[0])
        theta = les_lib.get_theta(z)
        d_vector = les_lib.get_d_vector(z, phase, theta)
        frequency = les_lib.get_frequency(k.shape[0])
        sigma_vector = les_lib.get_sigma_vector_array(d_vector, theta)

        tau_arr = np.full(self.num**3, tau)
        k_arr = np.full([self.num**3, len(k)], k)
        amplitude_arr = np.full([self.num**3, len(amplitude)], amplitude)
        phase_arr = np.full([self.num**3, len(phase)], phase)
        d_vector_arr = np.full([self.num**3, len(d_vector), 3], d_vector)
        frequency_arr = np.full([self.num**3, len(frequency)], frequency)
        sigma_vector_arr = np.full([self.num**3, len(sigma_vector), 3], sigma_vector)

        with mp.Pool(processes=proc_num) as pool:
            velocities = pool.starmap(self._get_velocity,
                                      list(zip(self._x_arr, self._y_arr, self._z_arr, tau_arr, k_arr, amplitude_arr,
                                               phase_arr, d_vector_arr, frequency_arr, sigma_vector_arr)))
        end = time.time()
        logging.info('Velocity calculation time is %.4f s' % (end - start))
        u_arr = np.array(velocities)[:, 0]
        v_arr = np.array(velocities)[:, 1]
        w_arr = np.array(velocities)[:, 2]
        return u_arr, v_arr, w_arr

    def _get_velocity_arrays(self):
        logging.info('Velocity calculation')
        start = time.time()
        u_arr = np.zeros(self._x_arr.shape[0])
        v_arr = np.zeros(self._x_arr.shape[0])
        w_arr = np.zeros(self._x_arr.shape[0])
        tau = les_lib.get_tau(self.u0, self.l_e)
        k_arr = les_lib.get_k_arr(self.l_e, self.l_cut, self.alpha)
        energy_arr = les_lib.get_von_karman_spectrum(k_arr, self.l_cut, self.l_e, self.viscosity, self.dissipation_rate)
        amplitude_arr = les_lib.get_amplitude_arr(k_arr, energy_arr)
        z = les_lib.get_z(k_arr.shape[0])
        phase_arr = les_lib.get_phase(k_arr.shape[0])
        theta_arr = les_lib.get_theta(z)
        d_vector_arr = les_lib.get_d_vector(z, phase_arr, theta_arr)
        frequency_arr = les_lib.get_frequency(k_arr.shape[0])
        sigma_vector_arr = les_lib.get_sigma_vector_array(d_vector_arr, theta_arr)
        for i in range(self._x_arr.shape[0]):
            v_vector = les_lib.get_auxiliary_pulsation_velocity(np.array([self._x_arr[i], self._y_arr[i], self._z_arr[i]]),
                                                                self.time, tau, k_arr, amplitude_arr, d_vector_arr,
                                                                sigma_vector_arr, phase_arr, frequency_arr)
            logging.info('n = %s  ---  u = %.3f, v = %.3f, w = %.3f' %
                         (i, v_vector[0], v_vector[1], v_vector[2]))
            u_arr[i] = v_vector[0] + self.u0[0]
            v_arr[i] = v_vector[1] + self.u0[1]
            w_arr[i] = v_vector[2] + self.u0[2]
        end = time.time()
        logging.info('Velocity calculation time is %.4f s' % (end - start))
        return u_arr, v_arr, w_arr

    def run(self, **kwargs):
        """
        kwargs:
                1. mode: str, optional, 'single', 'mp'
                2. proc_num: int, optional
        """
        index_gen1 = self._get_index_generator(self.num)
        self._x_arr, self._y_arr, self._z_arr = self._get_cell_center_coordinates_arrays(index_gen1, self.grid_step)
        if 'proc_num' not in kwargs:
            proc_num = 2
        else:
            proc_num = kwargs['proc_num']
        if 'mode' in kwargs:
            if kwargs['mode'] == 'single':
                self.u_arr, self.v_arr, self.w_arr = self._get_velocity_arrays()
            elif kwargs['mode'] == 'mp':
                self.u_arr, self.v_arr, self.w_arr = self._get_velocity_arrays_mp_pool(proc_num=proc_num)
            else:
                raise KeyError('Incorrect mode value')
        else:
            self.u_arr, self.v_arr, self.w_arr = self._get_velocity_arrays()
        index_gen2 = self._get_index_generator(self.num)
        self._vorticity_x_arr, self._vorticity_y_arr, self._vorticity_z_arr = \
            self._get_vorticity_arrays(index_gen2, self._x_arr, self._y_arr, self._z_arr,
                                       self.u_arr, self.v_arr, self.w_arr)
        self._create_all_files(self._x_arr, self._y_arr, self._z_arr, self.u_arr, self.v_arr, self.w_arr,
                               self._vorticity_x_arr, self._vorticity_y_arr, self._vorticity_z_arr)
        logging.info('Finish')


class HITGeneratorGivenSpectrum(HITGenerator):
    """
        Предоставляет интерфейс для генерации на равномерной сетке в области в форме куба
        поля однородной изотропоной турбулентности со спектром, соответствующем заданному,
        и сохранения данных о пульсациях и  завихренности в текстовых файлах.
    """
    def __init__(self, num, data_files_dir: str, grid_step, E_k_fit, k_fit):
        """
        :param num: количество узлов в на стороне генерируемого куба
        :param data_files_dir: имя файла с выходными данными
        :param grid_step: шаг сетки
        :param E_k_fit: массив значений энергии заданного спектра
        :param k_fit: массив значений волновых чисел заданного спектра
        """
        HITGenerator.__init__(self, num, data_files_dir, grid_step)
        self.E_k_fit = E_k_fit
        self.k_fit = k_fit
        self._x_arr: np.ndarray = None
        self._y_arr: np.ndarray = None
        self._z_arr: np.ndarray = None
        self.u_arr = None
        self.v_arr = None
        self.w_arr = None
        self._vorticity_x_arr = None
        self._vorticity_y_arr = None
        self._vorticity_z_arr = None

    def _get_velocity_arrays(self):
        length = self.grid_step * (self.num - 1)
        n, m, x, k = diht_lib.make_fft_grid(self.num, length)
        velocities = diht_lib.make_field(self.num, length, m, k, self.k_fit, self.E_k_fit)
        u_hat = velocities[0]
        v_hat = velocities[1]
        w_hat = velocities[2]
        u, v, w = diht_lib.make_ifft(u_hat, v_hat, w_hat)
        return np.reshape(u, [self.num**3]), np.reshape(v, [self.num**3]), np.reshape(w, [self.num**3])

    def run(self, **kwargs):
        index_gen1 = self._get_index_generator(self.num)
        self._x_arr, self._y_arr, self._z_arr = self._get_cell_center_coordinates_arrays(index_gen1, self.grid_step)
        index_gen2 = self._get_index_generator(self.num)
        self.u_arr, self.v_arr, self.w_arr = self._get_velocity_arrays()
        self._vorticity_x_arr, self._vorticity_y_arr, self._vorticity_z_arr = \
            self._get_vorticity_arrays(index_gen2, self._x_arr, self._y_arr, self._z_arr,
                                       self.u_arr, self.v_arr, self.w_arr)
        self._create_all_files(self._x_arr, self._y_arr, self._z_arr, self.u_arr, self.v_arr, self.w_arr,
                               self._vorticity_x_arr, self._vorticity_y_arr, self._vorticity_z_arr)
        logging.info('Finish')

