from matlab.lib import make_field, make_fft_grid, save_filed, make_ifft
from scipy.io import loadmat
import numpy as np


exp_data = loadmat(r'C:\Users\User\Documents\tasks\others\ic_gen\CBC_exp.mat')

num = 20
length = 1

M = 5.08
U0 = 1000
L_ref = 11*M
u_ref = np.sqrt(3/2)*22.2
k_42 = exp_data['k_42'][0] * L_ref
E_42 = exp_data['k_42'][0] / (u_ref**2 *L_ref)
k_98 = exp_data['k_98'][0] * L_ref
E_98 = exp_data['k_98'][0] / (u_ref**2 *L_ref)
k_171 = exp_data['k_171'][0] * L_ref
E_171 = exp_data['k_171'][0] / (u_ref**2 *L_ref)

if __name__ == '__main__':
    n, m, x, k = make_fft_grid(num, length)
    u_hat, v_hat, w_hat = make_field(num, length, m, k, k_42, E_42)
    u, v, w = make_ifft(u_hat, v_hat, w_hat)
    save_filed(u, v, w, 'velocity_filed.txt')
