from tecplot_lib import TextDataLoader
import config
from spectrum.spectrum_lib import SpatialSpectrum3d, read_velocity_file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing
from dhit_ic.gen_ic import k_42, k_98, k_171, E_42, E_98, E_171


base_dir = os.path.dirname(os.path.dirname(__file__))


def set_plot(title: str, xlim, ylim, legend: bool=False):
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both')
    plt.xlabel(r'$k$', fontsize=20)
    plt.ylabel(r'$E$', fontsize=20)
    plt.title(title, fontsize=14)
    if legend:
        plt.legend(fontsize=8, loc=0)
    plt.ylim(*ylim)
    plt.xlim(*xlim)


def plot_initial_spectrum():
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spectrum = SpatialSpectrum3d(config.num, config.grid_step, u_arr, v_arr, w_arr, 100)
    spectrum.compute_spectrum()
    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=1.4, color='black', label=r'$t = 0$')


def plot_spectrum(num, frames: typing.List[pd.DataFrame], subplot_index: typing.Tuple[int, int, int],
                  sol_time_arr=None):
    plt.subplot(*subplot_index)
    plot_initial_spectrum()
    for frame, sol_time in zip(frames, sol_time_arr):
        u_arr = np.array(frame['U'])
        v_arr = np.array(frame['V'])
        w_arr = np.array(frame['W'])
        spectrum = SpatialSpectrum3d(num, config.grid_step, u_arr, v_arr, w_arr, 100)
        spectrum.compute_spectrum()
        plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.5, label=r'$t = %s $' % round(sol_time, 3))


def get_kinetic_energy_arr(num, frames: typing.List[pd.DataFrame]):
    kinetic_energy_arr = []
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spectrum = SpatialSpectrum3d(config.num, config.grid_step, u_arr, v_arr, w_arr, 100)
    spectrum.compute_spectrum()
    kinetic_energy_arr.append(spectrum.get_turb_kinetic_energy())
    for frame in frames:
        u_arr = np.array(frame['U'])
        v_arr = np.array(frame['V'])
        w_arr = np.array(frame['W'])
        spectrum = SpatialSpectrum3d(num, config.grid_step, u_arr, v_arr, w_arr, 200)
        spectrum.compute_spectrum()
        kinetic_energy_arr.append(spectrum.get_turb_kinetic_energy())
    return np.array(kinetic_energy_arr)


def make_comparison_plot(frames_set1: typing.List[pd.DataFrame], name1: str, num1: int,
                         frames_set2: typing.List[pd.DataFrame],
                         name2: str, num2: int, sol_time_arr=None, xlim=(1e1, 1e3), ylim=(1e-6, 1e-3),
                         save_name='spectrum_history.png'):
    """Создание двойного графика истроии измения спектра"""
    plt.figure(figsize=(13, 7))
    if frames_set2:
        plot_spectrum(num1, frames_set1, (1, 2, 1), sol_time_arr)
        set_plot(name1, legend=bool(list(sol_time_arr)), xlim=xlim, ylim=ylim)
        plot_spectrum(num2, frames_set2, (1, 2, 2), sol_time_arr)
        set_plot(name2, legend=bool(list(sol_time_arr)), xlim=xlim, ylim=ylim)
    else:
        plot_spectrum(num1, frames_set1, (1, 1, 1), sol_time_arr)
        set_plot(name1, legend=bool(list(sol_time_arr)), xlim=xlim, ylim=ylim)
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, save_name))


def make_plot_with_exp(frames_set1: typing.List[pd.DataFrame], name1: str, num1: int,
                       frames_set2: typing.List[pd.DataFrame],
                       name2: str, num2: int, xlim=(1e1, 1e3), ylim=(1e-6, 1e-3),
                       save_name='spectrum_history_with_exp_data.png', theory_set=1e-1):
    """
    Создание графика истории изменения спектра в ходе расчета и эксперимента
    """
    t_show = [0.28, 0.66]
    plt.figure(figsize=(10, 8))
    plot_initial_spectrum()
    num_t_show = 0
    k_theory = np.linspace(1e1, 1e3, 100)
    e_k_theory = theory_set * k_theory ** (-5 / 3)
    for num, frame in enumerate(frames_set1):
        u_arr = np.array(frame['U'])
        v_arr = np.array(frame['V'])
        w_arr = np.array(frame['W'])
        spectrum = SpatialSpectrum3d(num1, config.grid_step, u_arr, v_arr, w_arr, 100)
        spectrum.compute_spectrum()
        if round(lazurit_sol_time_arr1[num], 2) in t_show:
            if num_t_show == 0:
                plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color='red', label=name1)
            else:
                plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color='red')
            num_t_show += 1
    if frames_set2:
        num_t_show = 0
        for num, frame in enumerate(frames_set2):
            u_arr = np.array(frame['U'])
            v_arr = np.array(frame['V'])
            w_arr = np.array(frame['W'])
            spectrum = SpatialSpectrum3d(num2, config.grid_step, u_arr, v_arr, w_arr, 100)
            spectrum.compute_spectrum()
            if round(lazurit_sol_time_arr2[num], 2) in t_show:
                if num_t_show == 0:
                    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color='blue', label=name2)
                else:
                    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color='blue')
                num_t_show += 1
    plt.plot(k_42, E_42, linestyle='', marker='s', ms=6, color='black', mew=1, label='Exp', mfc='white')
    plt.plot(k_98, E_98, linestyle='', marker='s', ms=6, color='black', mew=1, mfc='white')
    plt.plot(k_171, E_171, linestyle='', marker='s', ms=6, color='black', mew=1, mfc='white')
    plt.plot(k_theory, e_k_theory, linestyle=':', lw=1, color='green', label=r'$\sim k^{-5/3}$')
    set_plot('История изменения спектра', xlim, ylim, True)
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, save_name))


def sort_frames(frames: typing.List[pd.DataFrame], sol_time_arr):
    fr = pd.DataFrame.from_dict({'frames': frames, 'sol_time': sol_time_arr})
    sort_fr = fr.sort_values(by='sol_time')
    return list(sort_fr['frames']), np.array(list(sort_fr['sol_time']))


def get_frames_set_and_sol_time(data_dir):
    frames: typing.List[pd.DataFrame] = []
    sol_time_arr = []
    for fname in os.listdir(os.path.join(data_dir, 'txt')):
        sol_time = TextDataLoader.get_solution_time(os.path.join(data_dir, 'txt', fname))
        sol_time_arr.append(sol_time)
        frame = TextDataLoader.get_frame(os.path.join(data_dir, 'txt', fname))
        filter_series = pd.Series([not i for i in np.array(np.isnan(frame.U))])
        frames.append(frame.ix[filter_series])
    return frames, sol_time_arr


def make_kinetic_energy_plot(frames_set1: typing.List[pd.DataFrame], name1: str, num1: int, sol_time_arr1,
                             frames_set2: typing.List[pd.DataFrame], name2: str, num2: int, sol_time_arr2,
                             ylim=(0, 0.05), scale='linear', save_name='kinetic_energy.png', theory_set=0.004):
    """
    Создание графика изменения кинетической энергии турбулентности
    """
    time = np.linspace(0, 1.0, 100)
    energy_arr1 = get_kinetic_energy_arr(num1, frames_set1)
    energy_arr2 = get_kinetic_energy_arr(num2, frames_set2)

    plt.figure(figsize=(8, 6))
    sol_time_arr1 = np.array([0] + list(sol_time_arr1))
    plt.plot(sol_time_arr1, energy_arr1, lw=1, label=name1)
    if frames_set2:
        sol_time_arr2 = np.array([0] + list(sol_time_arr2))
        plt.plot(sol_time_arr2, energy_arr2, lw=1, label=name2)
    plt.plot(time, theory_set / time ** 1.2, lw=1, color='black', linestyle='--',
             label=r'$\sim t^{-1.2}$')
    plt.legend(fontsize=12)
    plt.title('Turbulence kinetic energy from spectrum')
    plt.xlabel('time', fontsize=12)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.ylabel(r'$K_t$', fontsize=12)
    plt.grid()
    plt.ylim(*ylim)
    plt.xlim(sol_time_arr1[0], sol_time_arr1[len(sol_time_arr1) - 1])
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, save_name))

if __name__ == '__main__':
    lazurit_frames1, lazurit_sol_time_arr1 = get_frames_set_and_sol_time(os.path.join(config.lazurit_data_dir, 'continuity_test'))
    lazurit_frames2, lazurit_sol_time_arr2 = [], None  # get_frames_set_and_sol_time(os.path.join(config.lazurit_data_dir, 'low_re_number'))

    lazurit_frames1, lazurit_sol_time_arr1 = sort_frames(lazurit_frames1, lazurit_sol_time_arr1)
    # lazurit_frames2, lazurit_sol_time_arr2 = sort_frames(lazurit_frames2, lazurit_sol_time_arr2)

    make_comparison_plot(lazurit_frames1, 'Lazurit, continuity_test', config.num + 1,
                         lazurit_frames2, 'Lazurit, Re = 290', config.num + 1, lazurit_sol_time_arr1,
                         save_name='spectrum_history_lazurit_32cells_continuity_test.png', ylim=(1e-6, 1e-3))

    # ---------------------------------------------------------------------------
    # Создание графика истории изменения спектра в ходе расчета и эксперимента
    # ---------------------------------------------------------------------------
    make_plot_with_exp(lazurit_frames1, 'Lazurit, continuity_test', config.num+1,
                       lazurit_frames2, 'Lazurit, Re = 290', config.num+1,
                       save_name='spectrum_history_with_exp_data_lazurit_32cells_continuity_test.png',
                       ylim=(1e-6, 1e-3), theory_set=1e-1)
    # --------------------------------------------------------------------------------
    # создание графика изменения кинетической энергии турбулентности
    # --------------------------------------------------------------------------------
    make_kinetic_energy_plot(lazurit_frames1, 'Lazurit, continuity_test', config.num+1, lazurit_sol_time_arr1,
                             lazurit_frames2, 'Lazurit, Re = 290', config.num + 1, lazurit_sol_time_arr2,
                             save_name='kinetic_energy_lazurit_32cells_continuity_test.png', ylim=(0, 0.05),
                             theory_set=0.0005, scale='log')
    plt.show()
