import numpy as np


def make_fft_grid(num: int, length):
        dx = length / num
        dk = 2 * np.pi / length
        n = np.array(np.linspace(0, num - 1, num))
        m = np.array(np.linspace(-num / 2, num / 2 - 1, num))
        x = n * dx
        k = m * dk
        return n, m, x, k


def make_spectrum(u_fft: np.ndarray, v_fft: np.ndarray, w_fft: np.ndarray, num: int, length, m: np.ndarray,
                          res: int):
    u_fft_phys = (length / num) ** 3 * u_fft
    v_fft_phys = (length / num) ** 3 * v_fft
    w_fft_phys = (length / num) ** 3 * w_fft

    dr = 1 / res
    grid_im = np.zeros([res, res, res])
    grid_jm = np.zeros([res, res, res])
    grid_km = np.zeros([res, res, res])

    for im in range(res):
        for jm in range(res):
            for km in range(res):
                grid_im[im, jm, km] = (-1 / 2 + (im + 1) * dr - dr / 2)
                grid_jm[im, jm, km] = (-1 / 2 + (jm + 1) * dr - dr / 2)
                grid_km[im, jm, km] = (-1 / 2 + (km + 1) * dr - dr / 2)

    e_phys = 0.5 * (np.abs(u_fft_phys) ** 2 + np.abs(v_fft_phys) ** 2 + np.abs(w_fft_phys))

    m_max = int(np.ceil(np.sqrt(3) * num / 2))
    e_k_mag = np.zeros([m_max + 1])

    for im in range(res):
        for jm in range(res):
            for km in range(res):
                mgrid = np.reshape(np.round(np.sqrt((grid_im + m[im]) ** 2 + (grid_jm + m[jm]) ** 2 +
                                                    (grid_km + m[km]) ** 2)), [res ** 3])
                e_k_per_cell = e_phys[im, jm, km] / (res ** 3)
                for ii in range(res ** 3):
                    e_k_mag[int(mgrid[ii])] = e_k_mag[int(mgrid[ii])] + e_k_per_cell

    k_mag = 2 * np.pi / length * np.linspace(0, m_max, m_max + 1)
    e_k_mag = e_k_mag * length / (2 * np.pi)
    return k_mag, e_k_mag