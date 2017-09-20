import config
import numpy as np
import os
import matplotlib.pyplot as plt


def read_cfx_monitor_data(filename, start_line=6):
    with open(filename, 'r') as file:
        st = ' '
        st_arr = []
        line = 1
        while st:
            st = file.readline()
            if line >= start_line and st:
                st_arr.append(st)
            line += 1
        result = []
        for n, i in enumerate(st_arr):
            result.append(list(np.fromstring(i, sep=',')))
        return np.array(result)


def get_kinetic_energy(u_arr: np.ndarray, v_arr: np.ndarray, w_arr: np.ndarray):
    vel_squared = u_arr**2 + v_arr**2 + w_arr**2
    result = np.zeros([len(u_arr)])
    for i in range(len(u_arr)):
        result[i] = 0.5 * (vel_squared[0: (i + 1)].sum() / (i + 1))
    return result

base_dir = os.path.dirname(os.path.dirname(__file__))

cfx_data = read_cfx_monitor_data(os.path.join(config.monitor_data_dir, 'cfx.csv'))
# lazurit_data = np.fromfile(os.path.join(config.monitor_data_dir,
#                                             'lazurit.dat'), sep=' ').reshape([config.max_time_step + 1, 7])
time_arr = config.time_step * np.linspace(1, config.max_time_step, config.max_time_step)
cfx_u_arr = cfx_data[:, 1]
cfx_v_arr = cfx_data[:, 2]
cfx_w_arr = cfx_data[:, 3]
cfx_kinetic_energy = get_kinetic_energy(cfx_u_arr, cfx_v_arr, cfx_w_arr)
# lazurit_u_arr = lazurit_data[0: config.max_time_step, 2]
# lazurit_v_arr = lazurit_data[0: config.max_time_step, 3]
# lazurit_w_arr = lazurit_data[0: config.max_time_step, 4]
# lazurit_kinetic_energy = get_kinetic_energy(lazurit_u_arr, lazurit_v_arr, lazurit_w_arr)

if __name__ == '__main__':
    plt.figure(figsize=(8, 6))
    plt.plot(time_arr, cfx_u_arr, lw=1, label='cfx')
    # plt.plot(time_arr, lazurit_u_arr, lw=1, label='lazurit')
    plt.legend(fontsize=12)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.grid()
    plt.xlim(time_arr[0], time_arr[len(time_arr) - 1])
    plt.savefig(os.path.join(base_dir, config.monitor_plots_dir, 'u_vel.png'))

    plt.figure(figsize=(8, 6))
    plt.plot(time_arr, cfx_v_arr, lw=1, label='cfx')
    # plt.plot(time_arr, lazurit_v_arr, lw=1, label='lazurit')
    plt.legend(fontsize=12)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('v', fontsize=12)
    plt.grid()
    plt.xlim(time_arr[0], time_arr[len(time_arr) - 1])
    plt.savefig(os.path.join(base_dir, config.monitor_plots_dir, 'v_vel.png'))

    plt.figure(figsize=(8, 6))
    plt.plot(time_arr, cfx_w_arr, lw=1, label='cfx')
    # plt.plot(time_arr, lazurit_w_arr, lw=1, label='lazurit')
    plt.legend(fontsize=12)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('w', fontsize=12)
    plt.grid()
    plt.xlim(time_arr[0], time_arr[len(time_arr) - 1])
    plt.savefig(os.path.join(base_dir, config.monitor_plots_dir, 'w_vel.png'))

    plt.figure(figsize=(8, 6))
    plt.plot(time_arr, cfx_kinetic_energy, lw=1, label='cfx')
    # plt.plot(time_arr, lazurit_kinetic_energy, lw=1, label='lazurit')
    plt.plot(time_arr, 0.005 / time_arr ** 1.2, lw=1, color='black', linestyle='--', label=r'$\sim t^{-1.2}$')
    plt.title('Turbulence kinetic energy from monitor data')
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('time', fontsize=12)
    plt.ylabel(r'$K_t$', fontsize=12)
    plt.grid()
    plt.ylim(0, 1)
    plt.xlim(time_arr[0], time_arr[len(time_arr) - 1])
    plt.savefig(os.path.join(base_dir, config.monitor_plots_dir, 'kinetic_energy.png'))
    plt.show()


