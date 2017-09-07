from tecplot_lib import TextDataLoader
import config
from spectrum.spectrum_lib import SpatialSpectrum3d, read_velocity_file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing

base_dir = os.path.dirname(os.path.dirname(__file__))


def set_plot(title: str, legend: bool=False):
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both')
    plt.xlabel(r'$k$', fontsize=20)
    plt.ylabel(r'$E$', fontsize=20)
    plt.title(title, fontsize=14)
    if legend:
        plt.legend(fontsize=9, loc=3)
    plt.ylim(10e-5, 10e-1)
    plt.xlim(1, 10e1)


def plot_initial_spectrum():
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spectrum = SpatialSpectrum3d(config.i_cnt, config.j_cnt, config.k_cnt,
                                 config.grid_step, u_arr, v_arr, w_arr, 200)
    spectrum.compute_spectrum()
    plt.plot(spectrum.k_abs_arr, spectrum.energy_sum_arr, lw=1.2, color='black')


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
        plt.plot(spectrum.k_abs_arr, spectrum.energy_sum_arr, lw=0.5, label=round(sol_time, 4))
    set_plot(title, legend=bool(sol_time_arr))


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
    return list(sort_fr['frames']), list(sort_fr['sol_time'])

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
    lazurit_frames, lazurit_sol_time_arr =sort_frames(lazurit_frames, lazurit_sol_time_arr)

    make_comparison_plot(cfx_frames, lazurit_frames, cfx_sol_time_arr)
