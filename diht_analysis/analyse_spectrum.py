from tecplot_lib import TextDataLoader
import config
from spectrum.spectrum_lib import SpatialSpectrum3d, read_velocity_file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing
from diht_analysis.analyse_monitor_data import cfx_kinetic_energy as mon_cfx_energy_arr,\
    lazurit_kinetic_energy as mon_lazurit_energy_arr, time_arr as mon_time_arr


base_dir = os.path.dirname(os.path.dirname(__file__))


def set_plot(title: str, legend: bool=False):
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both')
    plt.xlabel(r'$k$', fontsize=20)
    plt.ylabel(r'$E$', fontsize=20)
    plt.title(title, fontsize=14)
    if legend:
        plt.legend(fontsize=8, loc=0)
    plt.ylim(10e-5, 10e-1)
    plt.xlim(1, 10e1)


def plot_initial_spectrum():
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spectrum = SpatialSpectrum3d(config.i_cnt, config.j_cnt, config.k_cnt,
                                 config.grid_step, u_arr, v_arr, w_arr, 200)
    spectrum.compute_spectrum()
    plt.plot(spectrum.k_abs_arr, spectrum.energy_sum_arr, lw=1.2, color='black', label=r'$t = 0$')


def plot_spectrum(i_cnt, j_cnt, k_cnt, frames: typing.List[pd.DataFrame], subplot_num: int, title: str,
                  sol_time_arr=None):
    plt.subplot(1, 2, subplot_num)
    plot_initial_spectrum()
    for frame, sol_time in zip(frames, sol_time_arr):
        u_arr = np.array(frame['U'])
        v_arr = np.array(frame['V'])
        w_arr = np.array(frame['W'])
        spectrum = SpatialSpectrum3d(i_cnt, j_cnt, k_cnt,
                                     config.grid_step, u_arr, v_arr, w_arr, 200)
        spectrum.compute_spectrum()
        plt.plot(spectrum.k_abs_arr, spectrum.energy_sum_arr, lw=0.5, label=r'$t = %s $' % round(sol_time, 3))
    set_plot(title, legend=bool(list(sol_time_arr)))


def get_kinetic_energy_arr(i_cnt, j_cnt, k_cnt, frames: typing.List[pd.DataFrame]):
    kinetic_energy_arr = []
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spectrum = SpatialSpectrum3d(config.i_cnt, config.j_cnt, config.k_cnt,
                                 config.grid_step, u_arr, v_arr, w_arr, 200)
    spectrum.compute_spectrum()
    kinetic_energy_arr.append(spectrum.get_turb_kinetic_energy())
    for frame in frames:
        u_arr = np.array(frame['U'])
        v_arr = np.array(frame['V'])
        w_arr = np.array(frame['W'])
        spectrum = SpatialSpectrum3d(i_cnt, j_cnt, k_cnt,
                                     config.grid_step, u_arr, v_arr, w_arr, 200)
        spectrum.compute_spectrum()
        kinetic_energy_arr.append(spectrum.get_turb_kinetic_energy())
    return np.array(kinetic_energy_arr)


def make_comparison_plot(cfx_frames: typing.List[pd.DataFrame], lazurit_frames: typing.List[pd.DataFrame],
                         sol_time_arr=None):
    plt.figure(figsize=(13, 7))
    plot_spectrum(config.i_cnt, config.j_cnt, config.k_cnt, cfx_frames, 1, 'CFX', sol_time_arr)
    plot_spectrum(config.i_cnt + 1, config.j_cnt + 1, config.k_cnt + 1, lazurit_frames, 2, 'Lazurit', sol_time_arr)
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, 'spectrum_history.png'))
    plt.show()


def sort_frames(frames: typing.List[pd.DataFrame], sol_time_arr):
    fr = pd.DataFrame.from_dict({'frames': frames, 'sol_time': sol_time_arr})
    sort_fr = fr.sort_values(by='sol_time')
    return list(sort_fr['frames']), np.array(list(sort_fr['sol_time']))

if __name__ == '__main__':
    cfx_frames: typing.List[pd.DataFrame] = []
    lazurit_frames: typing.List[pd.DataFrame] = []
    cfx_sol_time_arr = []
    lazurit_sol_time_arr = []
    for fname in os.listdir(os.path.join(config.cfx_data_dir, 'txt')):
        sol_time = TextDataLoader.get_solution_time(os.path.join(config.cfx_data_dir, 'txt', fname))
        cfx_sol_time_arr.append(sol_time)
        frame = TextDataLoader.get_frame(os.path.join(config.cfx_data_dir, 'txt', fname))
        filter_series = pd.Series([not i for i in np.array(np.isnan(frame.U))])
        cfx_frames.append(frame.ix[filter_series])

    for fname in os.listdir(os.path.join(config.lazurit_data_dir, 'txt')):
        sol_time = TextDataLoader.get_solution_time(os.path.join(config.lazurit_data_dir, 'txt', fname))
        lazurit_sol_time_arr.append(sol_time)
        frame = TextDataLoader.get_frame(os.path.join(config.lazurit_data_dir, 'txt', fname))
        lazurit_frames.append(frame)

    cfx_frames, cfx_sol_time_arr = sort_frames(cfx_frames, cfx_sol_time_arr)
    lazurit_frames, lazurit_sol_time_arr = sort_frames(lazurit_frames, lazurit_sol_time_arr)

    make_comparison_plot(cfx_frames, lazurit_frames, cfx_sol_time_arr)

    cfx_energy_arr = get_kinetic_energy_arr(config.i_cnt, config.j_cnt, config.k_cnt, cfx_frames)
    lazurit_energy_arr = get_kinetic_energy_arr(config.i_cnt + 1, config.j_cnt + 1, config.k_cnt + 1, lazurit_frames)

    cfx_sol_time_arr = np.array([0] + list(cfx_sol_time_arr))
    lazurit_sol_time_arr = np.array([0] + list(lazurit_sol_time_arr))
    plt.figure(figsize=(8, 6))
    plt.plot(cfx_sol_time_arr, cfx_energy_arr, lw=1, label='cfx')
    plt.plot(lazurit_sol_time_arr, lazurit_energy_arr, lw=1, label='lazurit')
    plt.plot(cfx_sol_time_arr, 0.005 / cfx_sol_time_arr ** 1.2, lw=1, color='black', linestyle='--',
             label=r'$\sim t^{-1.2}$')
    plt.legend(fontsize=12)
    plt.title('Turbulence kinetic energy from spectrum')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('time', fontsize=12)
    plt.ylabel(r'$K_t$', fontsize=12)
    plt.grid()
    plt.ylim(0, 1)
    plt.xlim(cfx_sol_time_arr[0], cfx_sol_time_arr[len(cfx_sol_time_arr) - 1])
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, 'kinetic_energy.png'))

    plt.figure(figsize=(8, 6))
    plt.plot(cfx_sol_time_arr, cfx_energy_arr, lw=1, label='cfx, from spectrum', linestyle='-', color='red')
    plt.plot(lazurit_sol_time_arr, lazurit_energy_arr, lw=1, label='lazurit, from spectrum', linestyle='-',
             color='blue')
    plt.plot(mon_time_arr, mon_cfx_energy_arr, lw=1, label='cfx, from monitor', linestyle=':', color='red')
    plt.plot(mon_time_arr, mon_lazurit_energy_arr, lw=1, label='lazurit, from monitor', linestyle=':', color='blue')
    plt.plot(cfx_sol_time_arr, 0.005 / cfx_sol_time_arr ** 1.2, lw=1, color='black', linestyle='--',
             label=r'$\sim t^{-1.2}$')
    plt.legend(fontsize=12)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('time', fontsize=12)
    plt.ylabel(r'$K_t$', fontsize=12)
    plt.grid()
    plt.ylim(0, 1)
    plt.xlim(cfx_sol_time_arr[0], cfx_sol_time_arr[len(cfx_sol_time_arr) - 1])
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, 'kinetic_energy_monitor_spectrum_comparison.png'))
    plt.show()