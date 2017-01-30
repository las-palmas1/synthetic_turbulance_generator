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


def get_d_vector_and_theta() -> tuple:
    phi = random.uniform(0, 2 * np.pi)
    theta = random.uniform(0, 2 * np.pi)
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]), theta


def get_sigma_vector(d_vector: np.ndarray, theta: float) -> np.ndarray:
    z_rotation_axis = 0
    x_rotation_axis = 1 / np.sqrt(1 + (d_vector[0] / d_vector[1]) ** 2)
    y_rotation_axis = np.sqrt(1 - x_rotation_axis ** 2)
    rotation_matrix = np.array([
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
                                     alpha=0.025, u0=0) -> np.ndarray:
    k_arr = get_k_arr(l_e_max, l_cut_min, alpha=alpha)
    tau = get_tau(u0, l_e_max)
    energy_arr = get_energy_arr(k_arr, l_cut, l_e, viscosity, dissipation_rate)
    amplitude_arr = get_amplitude_arr(k_arr, energy_arr)
    result = np.array([0, 0, 0])
    for i in range(len(k_arr)):
        d_vector, theta = get_d_vector_and_theta()
        sigma_vector = get_sigma_vector(d_vector, theta)
        phase = get_phase()
        frequency = get_frequency()
        result += 2 * np.sqrt(3 / 2) * np.sqrt(amplitude_arr[i]) * sigma_vector * \
                  np.cos(k_arr[i] * np.dot(d_vector, r_vector) + phase + frequency * t / tau)
    return result


def plot_spectrum(r_vector, t, l_cut, l_e, l_cut_min, l_e_max, viscosity, dissipation_rate,
                  alpha=0.025, u0=0):
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


plot_spectrum([0.1, 0.1, 0.1], 1, 0.0005, 0.005, 0.0005, 0.005, 5e-6, 0.01)
