import numpy as np
from scipy.fftpack import fftn, ifftn
import logging
import config
import time
import numba as nb

logging.basicConfig(format='%(levelname)s: %(message)s', level=config.log_level)


def make_fft_grid(num: int, length):
        dx = length / num
        dk = 2 * np.pi / length
        n = np.array(np.linspace(0, num - 1, num))
        m = np.array(np.linspace(-num / 2, num / 2 - 1, num))
        x = n * dx
        k = m * dk
        return n, m, x, k


def make_spectrum_alt_way(u_hat: np.ndarray, v_hat: np.ndarray, w_hat: np.ndarray, num: int, length, m: np.ndarray,
                          res: int):
    logging.info('START CALCULATING SPECTRUM')
    u_fft_phys = (length / num) ** 3 * u_hat
    v_fft_phys = (length / num) ** 3 * v_hat
    w_fft_phys = (length / num) ** 3 * w_hat

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

    e_phys = 0.5 * (np.abs(u_fft_phys) ** 2 + np.abs(v_fft_phys) ** 2 + np.abs(w_fft_phys) ** 2)

    m_max = int(np.ceil(np.sqrt(3) * num / 2))
    e_k_mag = np.zeros([m_max + 1])

    for im in range(num):
        for jm in range(num):
            for km in range(num):
                logging.info('im = %s, jm = %s, km = %s' % (im, jm, km))
                mgrid = np.reshape(np.round(np.sqrt((grid_im + m[im]) ** 2 + (grid_jm + m[jm]) ** 2 +
                                                    (grid_km + m[km]) ** 2)), [res ** 3])
                e_k_per_cell = e_phys[im, jm, km] / (res ** 3)
                for ii in range(res ** 3):
                    e_k_mag[int(mgrid[ii])] = e_k_mag[int(mgrid[ii])] + e_k_per_cell

    k_mag = 2 * np.pi / length * np.linspace(0, m_max, m_max + 1)
    e_k_mag = e_k_mag * length / (2 * np.pi)
    logging.info('FINISH CALCULATING SPECTRUM')
    return k_mag, e_k_mag


def _get_energy(m_grid: np.ndarray, energy_arr: np.ndarray, m_mag, res):
    energy_arr_filt = energy_arr[(m_grid > m_mag - 0.5 * res) * (m_grid < m_mag + 0.5 * res)]
    energy = energy_arr_filt.sum() / res
    return energy


def make_spectrum(num: int, length, u_hat: np.ndarray, v_hat: np.ndarray, w_hat: np.ndarray, m: np.ndarray,
                  num_pnt=100, res=3):
    logging.info('START CALCULATING SPECTRUM')
    m_i = np.zeros([num, num, num])
    m_j = np.zeros([num, num, num])
    m_k = np.zeros([num, num, num])
    for im in range(num):
        for jm in range(num):
            for km in range(num):
                m_i[im, jm, km] = m[im]
                m_j[im, jm, km] = m[jm]
                m_k[im, jm, km] = m[km]
    m_grid = np.sqrt(m_i**2 + m_j**2 + m_k**2)
    energy = 0.5 * (length / num)**6 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    m_mag = np.linspace(0, m_grid.max(), num_pnt)
    e_k_mag = np.zeros(num_pnt)
    for i in range(num_pnt):
        e_k_mag[i] = _get_energy(m_grid, energy, m_mag[i], res) * length / (2 * np.pi)
        logging.debug('e_k_mag[%s] = %.5f' % (i, e_k_mag[i]))
    k_mag = m_mag * 2 * np.pi / length
    logging.info('FINISH CALCULATING SPECTRUM')
    return k_mag, e_k_mag


@nb.jit(nb.double(nb.double, nb.double[:], nb.double[:]))
def energy_interp(k_mag, k_fit: np.ndarray, e_k_fit: np.ndarray):
    i = 0
    num = len(k_fit)
    e_k = ((k_mag - k_fit[0])/(k_fit[1] - k_fit[0]) * (e_k_fit[1] - e_k_fit[0]) + e_k_fit[0])
    while k_mag > k_fit[i]:
        i += 1
        if i == num - 1:
            e_k = np.exp((np.log(k_mag) - np.log(k_fit[num-2])) / (np.log(k_fit[num-1]) - np.log(k_fit[num-2])) *
                         (np.log(e_k_fit[num-1]) - np.log(e_k_fit[num-2])) + np.log(e_k_fit[num-2]))
            break
        e_k = np.exp((np.log(k_mag) - np.log(k_fit[i-1])) / (np.log(k_fit[i]) - np.log(k_fit[i-1])) *
                     (np.log(e_k_fit[i]) - np.log(e_k_fit[i-1])) + np.log(e_k_fit[i-1]))
    return e_k


def get_max_div(u: np.ndarray, v: np.ndarray, w: np.ndarray, num: int, length):
    divmax = 0
    dx = length / num
    for im in range(1, num):
        for jm in range(1, num):
            for km in range(1, num):
                divijk = (u[im, jm, km] - u[im-1, jm, km]) / dx + (v[im, jm, km] - v[im, jm-1, km]) / dx + \
                         (w[im, jm, km] - w[im, jm, km-1]) / dx
                if abs(divijk) > divmax:
                    divmax = abs(divijk)
    return divmax


def make_field(num: int, length: float, m: np.ndarray, k: np.ndarray, k_fit: np.ndarray, e_k_fit: np.ndarray):
    logging.info('START VELOCITY FIELD GENERATION')
    start = time.time()
    dx = length / num
    dk = 2 * np.pi / length
    u_hat = np.zeros_like(np.zeros([num, num, num]), dtype=np.complex)
    v_hat = np.zeros_like(np.zeros([num, num, num]), dtype=np.complex)
    w_hat = np.zeros_like(np.zeros([num, num, num]), dtype=np.complex)
    result = np.zeros_like(np.zeros([3, num, num, num]), dtype=np.complex)
    kmod = 2j * np.sin(np.pi * m / num) / dx

    for im in range(1, int(num / 2 + 1)):
        for jm in range(1, num):
            for km in range(1, num):
                k_mag = np.sqrt(k[im]**2 + k[jm]**2 + k[km]**2)
                u_hat_mag = 1 / (dx**3) * np.sqrt(2 * energy_interp(k_mag, k_fit, e_k_fit) *
                                                  dk / (4 * np.pi * (k_mag/dk)**2))
                if np.isnan(u_hat_mag):
                    u_hat_mag = 0
                kdiv = np.array([kmod[im], kmod[jm], kmod[km]])
                theta_rand = 2 * np.pi * np.random.rand(3)

                u1 = np.cross([1.123, -0.982, 1.182], kdiv)
                u1 = u1 / np.linalg.norm(u1)
                u2 = np.cross(u1, kdiv)
                u2 = u2 / np.linalg.norm(u2)

                u_hat_re = u_hat_mag * np.cos(theta_rand[0])
                u_hat_im = u_hat_mag * np.sin(theta_rand[0])

                r1 = np.cos(theta_rand[1])*u1 + np.sin(theta_rand[1])*u2
                r2 = np.cos(theta_rand[2])*u1 + np.sin(theta_rand[2])*u2

                u_hat[im, jm, km] = r1[0] * u_hat_re + 1j*r2[0]*u_hat_im
                v_hat[im, jm, km] = r1[1] * u_hat_re + 1j*r2[1]*u_hat_im
                w_hat[im, jm, km] = r1[2] * u_hat_re + 1j*r2[2]*u_hat_im

    for im in range(int(num/2 + 1), num):
        for jm in range(1, num):
            for km in range(1, num):
                u_hat[im, jm, km] = u_hat[num-im, num-jm, num-km].conj()
                v_hat[im, jm, km] = v_hat[num-im, num-jm, num-km].conj()
                w_hat[im, jm, km] = w_hat[num-im, num-jm, num-km].conj()

    im = int(num / 2)
    for jm in range(int(num / 2 + 1), num):
        for km in range(1, num):
            u_hat[im, jm, km] = u_hat[num - im, num - jm, num - km].conj()
            v_hat[im, jm, km] = v_hat[num - im, num - jm, num - km].conj()
            w_hat[im, jm, km] = w_hat[num - im, num - jm, num - km].conj()

    im = int(num / 2)
    jm = int(num / 2)
    for km in range(int(num / 2 + 1), num):
        u_hat[im, jm, km] = u_hat[num - im, num - jm, num - km].conj()
        v_hat[im, jm, km] = v_hat[num - im, num - jm, num - km].conj()
        w_hat[im, jm, km] = w_hat[num - im, num - jm, num - km].conj()

    im = int(num / 2)
    jm = int(num / 2)
    km = int(num / 2)
    u_hat[im, jm, km] = 0
    v_hat[im, jm, km] = 0
    w_hat[im, jm, km] = 0
    finish = time.time()
    logging.info('FINISH VELOCITY FIELD GENERATION')
    logging.info('Total time  =  %.3f' % (finish - start))
    result[0] = u_hat
    result[1] = v_hat
    result[2] = w_hat
    return result


def make_fft(u, v, w):
    u_hat = fftn(u)
    v_hat = fftn(v)
    w_hat = fftn(w)
    u_hat = np.fft.fftshift(u_hat)
    v_hat = np.fft.fftshift(v_hat)
    w_hat = np.fft.fftshift(w_hat)
    return u_hat, v_hat, w_hat


def make_ifft(u_hat, v_hat, w_hat):
    u_hat = np.fft.fftshift(u_hat)
    v_hat = np.fft.fftshift(v_hat)
    w_hat = np.fft.fftshift(w_hat)
    u = ifftn(u_hat)
    v = ifftn(v_hat)
    w = ifftn(w_hat)
    return u.real, v.real, w.real


def save_filed(u: np.ndarray, v: np.ndarray, w: np.ndarray, filename: str):
    ar = np.zeros([3] + list(u.shape))
    ar[0] = u
    ar[1] = v
    ar[2] = w
    np.save(filename, ar)


def load_field(filename: str):
    result = np.fromfile(filename)
    return result[0], result[1], result[2]

