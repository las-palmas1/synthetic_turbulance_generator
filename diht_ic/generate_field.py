from diht_ic.lib import make_field, make_fft_grid, save_filed, make_ifft, make_spectrum_alt_way, get_max_div, make_spectrum
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


exp_data = loadmat(r'C:\Users\User\Documents\tasks\others\ic_gen\CBC_exp.mat')

num = 20
length = 1

M = 5.08
U0 = 1000
L_ref = 11*M
u_ref = np.sqrt(3/2)*22.2
k_42 = exp_data['k_42'][0] * L_ref
E_42 = exp_data['E_42'][0] / (u_ref**2 *L_ref)
k_98 = exp_data['k_98'][0] * L_ref
E_98 = exp_data['E_98'][0] / (u_ref**2 *L_ref)
k_171 = exp_data['k_171'][0] * L_ref
E_171 = exp_data['E_171'][0] / (u_ref**2 *L_ref)

if __name__ == '__main__':
    n, m, x, k = make_fft_grid(num, length)
    u_hat, v_hat, w_hat = make_field(num, length, m, k, k_42, E_42)
    u, v, w = make_ifft(u_hat, v_hat, w_hat)
    divmax = get_max_div(u, v, w, num, length)
    print(divmax)
    save_filed(u, v, w, 'velocity_filed')

    k_mag, e_k_mag = make_spectrum(num, length, u_hat, v_hat, w_hat, m, num_pnt=100, res=3)
    plt.figure(figsize=(8, 6))
    plt.plot(k_mag, e_k_mag, lw=0.5, label='Synthetic field')
    plt.plot(k_42, E_42, lw=0.5, label='Experiment')
    plt.grid()
    plt.xlabel('k', fontsize=10)
    plt.ylabel('E', fontsize=10)
    plt.legend(fontsize=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()