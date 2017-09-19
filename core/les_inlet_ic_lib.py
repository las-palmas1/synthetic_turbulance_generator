import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import logging
import config
import numba as nb


logging.basicConfig(format='%(levelname)s: %(message)s', level=config.log_level)


def get_k_arr(l_e_max: float, l_cut_min: float, alpha=0.01) -> np.ndarray:
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


def get_von_karman_spectrum(k_arr: np.ndarray, l_cut, l_e, viscosity, dissipation_rate) -> np.ndarray:
    """
    :param k_arr:  массив модулей волновых чисел
    :param l_cut: инимальная длина волны в рассчитываемом узле
    :param l_e: максимальная в рассчитываемом узле длина волны
            наиболее энергонесущих мод синтезированного поля пульсаций
    :param viscosity: молекулярная вязкость
    :param dissipation_rate:  степень диссипации
    :return: модифицированный энергетический спектр фон Кармана синтетического поля в данном узле
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
    """
    :param rotation_axis: вектор оси вращения
    :param rotation_angle: угол поворота
    :return: матрицу поворота
    """
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


def get_z(size=0) -> np.ndarray:
    """
    Возвращает массив случайных величин, каждая из которых имеет равномерное распределение
    в интервале от -1 до 1. Каждая их низ также является координатой по оси Z единичного вектора, равномерно
    распределенного по сфере.
    """
    z = random.uniform(-1, 1, size)
    return z


def get_phase(size=0) -> np.ndarray:
    """
    Возвращает масиив размера size фаз. Являются также координатами (долгота) единичного вектора, равномерно
    распределенного по сфере.
    """
    phi = random.uniform(0, 2 * np.pi, size)
    return phi


def get_theta(z: np.ndarray) -> np.ndarray:
    """
    Возвращает третью координату единичного вектора (широту), равномерно
    распределенного по сфере.
    """
    theta = np.arccos(z)
    return theta


def get_d_vector(z: np.ndarray, phase: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Возвращает координаты в декартовой системе единичного вектора, равномерно
    распределенного по сфере.
    """
    result = np.zeros([z.shape[0], 3])
    result[:, 0] = np.sin(theta) * np.cos(phase)
    result[:, 1] = np.sin(theta) * np.sin(phase)
    result[:, 2] = z
    return result


def get_sigma_vector(d_vector: np.ndarray, theta) -> np.ndarray:
    """
    Возвращает координаты единичного вектора, лежащего в плоскости, нормальной к вектору d_vector.
    Его направление в данной плоскости задается случайным углом, равномерно распределенным в интервале от 0
    до 2 * pi. озвращаемый вектор определяет направление мод скорости
    """
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
    """
    Возврщает массив безразмерных круговых частот мод Фурье, являющихся случайными величинами с
    нормальным распределением и имеющих среднее значение и стандартное отклонение, равные 2*pi.
    """
    return random.normal(2 * np.pi, 2 * np.pi, size=size)


def get_sigma_vector_array(d_vector_arr: np.ndarray, theta_arr: np.ndarray) -> np.ndarray:
    sigma_vector_arr = np.zeros([d_vector_arr.shape[0], 3])
    for i in range(d_vector_arr.shape[0]):
        sigma_vector_arr[i] = get_sigma_vector(d_vector_arr[i], theta_arr[i])
    return sigma_vector_arr


def get_auxiliary_pulsation_velocity_parameters(l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                                                alpha=0.01, u0=np.array([0, 0, 0])):
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_von_karman_spectrum(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    z = get_z(k_arr.shape[0])
    theta_arr = get_theta(z)
    phase_arr = get_phase(k_arr.shape[0])
    d_vector_arr = get_d_vector(z, phase_arr, theta_arr)
    frequency_arr = get_frequency(k_arr.shape[0])
    sigma_vector_arr = get_sigma_vector_array(d_vector_arr, theta_arr)
    return tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr


@nb.jit(nb.double[:](nb.double[:], nb.double, nb.double, nb.double[:], nb.double[:], nb.double[:, :],
                     nb.double[:, :], nb.double[:], nb.double[:]))
def get_auxiliary_pulsation_velocity(r_vector, t, tau, k_arr: np.ndarray, amplitude_arr: np.ndarray,
                                     d_vector_arr: np.ndarray, sigma_vector_arr: np.ndarray,
                                     phase_arr: np.ndarray, frequency_arr: np.ndarray) -> np.ndarray:
    """
    Вектор вспомогательной пульсационной скорости в заданной точке и в заданный момент времени
    """
    result = np.array([0., 0., 0.])
    for i in range(len(k_arr)):
        result += (2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector_arr[i] *
                np.cos(k_arr[i] * np.dot(d_vector_arr[i], r_vector) + phase_arr[i] + frequency_arr[i] * t / tau))
    return result


if __name__ == '__main__':
    pass
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # for i in range(100):
    #     z = get_z(1)
    #     theta = get_theta(z)
    #     phase = get_phase(1)
    #     d_vector = get_d_vector(z, phase, theta)
    #     sigma_vector = get_sigma_vector(d_vector[0], theta[0])
    #     print(np.dot(d_vector[0], sigma_vector))
    #     ax.plot(xs=[d_vector[0, 0]], ys=[d_vector[0, 1]], zs=[d_vector[0, 2]], marker='o', color='r')
    #     ax.plot(xs=[sigma_vector[0]], ys=[sigma_vector[1]], zs=[sigma_vector[2]], marker='o', color='g')
    #     ax.set_xlim(-1, 1)
    #     ax.set_ylim(-1, 1)
    #     ax.set_zlim(-1, 1)
    # plt.show()


