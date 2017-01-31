import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import typing
import matplotlib.pyplot as plt


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


def get_d_vector_and_theta() -> tuple:
    z = random.uniform(-1, 1)
    phi = random.uniform(0, 2 * np.pi)
    theta = np.arccos(z)
    result = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), z])
    return result, theta


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


def get_phase():
    return random.uniform(0, 2 * np.pi)


def get_frequency():
    return random.normal(2 * np.pi, 2 * np.pi)


def get_auxiliary_pulsation_velocity(r_vector, t, l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                                     alpha=0.01, u0=0) -> np.ndarray:
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_energy_arr(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    result = np.array([0.0, 0.0, 0.0])
    for i in range(len(k_arr)):
        d_vector, theta = get_d_vector_and_theta()
        sigma_vector = get_sigma_vector(d_vector, theta)
        phase = get_phase()
        frequency = get_frequency()
        result += 2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector * \
                  np.cos(k_arr[i] * np.dot(d_vector, r_vector) + phase + frequency * t / tau)
    return result


def plot_spectrum(r_vector, t, l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                  alpha=0.01, u0=0):
    plt.figure(figsize=(9, 7))
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_energy_arr(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    v_arr = []
    for i in range(len(k_arr)):
        d_vector, theta = get_d_vector_and_theta()
        sigma_vector = get_sigma_vector(d_vector, theta)
        phase = get_phase()
        frequency = get_frequency()
        v_arr.append(2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector * \
                     np.cos(k_arr[i] * np.dot(d_vector, r_vector) + phase + frequency * t / tau))
    energy_arr_new = [linalg.norm(v) ** 2 for v in v_arr]
    plt.plot(k_arr, energy_arr, color='black', lw=1)
    plt.plot(k_arr, energy_arr_new, color='red', lw=1)
    plt.xlim(min(k_arr), 2 * np.pi / l_cut_min)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both')
    plt.xlabel('k', fontsize=16)
    plt.ylabel('E', fontsize=16)
    plt.show()


if __name__ == '__main__':
    plot_spectrum([0.1, 0.1, 0.1], 0, 0.0005, 0.005, 0.0005, 0.005, 2e-5, 6e3, u0=0)
    # t_arr = np.linspace(0, 0.2, 500)
    # u_arr = []
    # v_arr = []
    # w_arr = []
    # uv_arr = []
    # vw_arr = []
    # uw_arr = []
    # k_u_arr = []
    # k_uw_arr = []
    # for i in range(20):
    #     for t in t_arr:
    #         print(i, '', t)
    #         v_vector = get_auxiliary_pulsation_velocity([0.1, 0.1, 0.1], t, 0.0005, 0.005, 0.0005, 0.005,  2e-5, 6e3, u0=2)
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
    # print(sum(u_arr) / len(u_arr))
    # print(sum(uv_arr) / len(uv_arr))
    # plt.plot(t_arr, u_arr, 'r')
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
    #     ax.plot(xs=[sigma_vector[0]], ys=[sigma_vector[1]], zs=[sigma_vector[2]], marker='o', color='r')
    # plt.show()


