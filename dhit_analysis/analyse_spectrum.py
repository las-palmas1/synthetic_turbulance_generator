from tecplot_lib import TextDataLoader
import config
from spectrum.spectrum_lib import SpatialSpectrum3d, read_velocity_file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing
from scipy.integrate import quad
from scipy.interpolate import interp1d
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
        plt.legend(fontsize=9, loc=0)
    plt.ylim(*ylim)
    plt.xlim(*xlim)


def plot_initial_spectrum():
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spectrum = SpatialSpectrum3d(config.num, config.grid_step, u_arr, v_arr, w_arr, 100)
    spectrum.compute_spectrum()
    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=1.4, color='black', label=r'$t = 0$')


def plot_spectrum(num, frames: typing.List[pd.DataFrame], subplot_index: typing.Tuple[int, int, int],
                  truncate_fiction_cells: bool, sol_time_arr=None):
    plt.subplot(*subplot_index)
    plot_initial_spectrum()
    for frame, sol_time in zip(frames, sol_time_arr):
        u_arr = np.array(frame['U'])
        v_arr = np.array(frame['V'])
        w_arr = np.array(frame['W'])
        spectrum = SpatialSpectrum3d(num, config.grid_step, u_arr, v_arr, w_arr, 100, truncate_fiction_cells)
        spectrum.compute_spectrum()
        plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.5, label=r'$t = %s $' % round(sol_time, 5))


def get_kinetic_energy_arr(num, frames: typing.List[pd.DataFrame], truncate_fiction_cells: bool,):
    kinetic_energy_arr = []
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spectrum = SpatialSpectrum3d(config.num, config.grid_step, u_arr, v_arr, w_arr, 100, truncate_fiction_cells)
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


def make_comparison_plot(frames_set1: typing.List[pd.DataFrame], name1: str, num1: int, truncate_fiction_cells1: bool,
                         frames_set2: typing.List[pd.DataFrame], name2: str, num2: int, truncate_fiction_cells2: bool,
                         sol_time_arr=None, xlim=(1e1, 1e3), ylim=(1e-6, 1e-3),
                         save_name='spectrum_history.png'):
    """
    Создание двойного графика истроии измения спектра
    NOTE: работает только в том случае, если в обоих случях величина шага сетки была одинакова
    """
    plt.figure(figsize=(13, 7))
    if frames_set2:
        plot_spectrum(num1, frames_set1, (1, 2, 1), truncate_fiction_cells1, sol_time_arr)
        set_plot(name1, legend=bool(list(sol_time_arr)), xlim=xlim, ylim=ylim)
        plot_spectrum(num2, frames_set2, (1, 2, 2), truncate_fiction_cells2, sol_time_arr)
        set_plot(name2, legend=bool(list(sol_time_arr)), xlim=xlim, ylim=ylim)
    else:
        plot_spectrum(num1, frames_set1, (1, 1, 1), truncate_fiction_cells1, sol_time_arr)
        set_plot(name1, legend=bool(list(sol_time_arr)), xlim=xlim, ylim=ylim)
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, save_name))


def plot_theory_and_exp_data(theory_set=1e-1):
    k_theory = np.linspace(1e1, 1e3, 100)
    e_k_theory = theory_set * k_theory ** (-5 / 3)
    plt.plot(k_42, E_42, linestyle='', marker='s', ms=6, color='black', mew=1, label='Exp', mfc='white')
    plt.plot(k_98, E_98, linestyle='', marker='s', ms=6, color='black', mew=1, mfc='white')
    plt.plot(k_171, E_171, linestyle='', marker='s', ms=6, color='black', mew=1, mfc='white')
    plt.plot(k_theory, e_k_theory, linestyle=':', lw=1, color='green', label=r'$\sim k^{-5/3}$')


def plot_spectrum_from_frame(frame, num, label, color, truncate_fiction_cells: bool=False):
    u_arr = np.array(frame['U'])
    v_arr = np.array(frame['V'])
    w_arr = np.array(frame['W'])
    spectrum = SpatialSpectrum3d(num, config.length / (num - 1), u_arr, v_arr, w_arr, 100, truncate_fiction_cells)
    spectrum.compute_spectrum()
    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color=color, label=label)


def plot_initial_spectrum_num_cells(num, label, color):
    """График начального спектра по файлу со скоростями, сгенерированными для сетки с заданным количество ячеек"""
    name_temp = 'velocity_%s_cells.VEL'
    u_arr, v_arr, w_arr = read_velocity_file(os.path.join(base_dir, config.data_files_dir, name_temp % num))
    spectrum = SpatialSpectrum3d(num, config.length / (num - 1), u_arr, v_arr, w_arr, 100)
    spectrum.compute_spectrum()
    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color=color, label=label)


def make_different_grids_plot(frames_set1: typing.List[pd.DataFrame], label1: str, num1: int,
                              truncate_fiction_cells1: bool,
                              frames_set2: typing.List[pd.DataFrame], label2: str, num2: int,
                              truncate_fiction_cells2: bool,
                              frames_set3: typing.List[pd.DataFrame], label3: str, num3: int,
                              truncate_fiction_cells3: bool, sol_time_arr, num_init1, num_init2, num_init3,
                              xlim=(1e1, 1e3), ylim=(1e-6, 1e-3),
                              save_name='spectrum_history_diff_grids.png', theory_set=1e-1
                              ):
    """
    Создание графиков для разных сеток
    NOTE: работает только в том случае, если во всех случях величина шага сетки была одинакова
    """
    plt.figure(figsize=(20, 8))
    plt.subplot(131)
    plot_initial_spectrum_num_cells(num_init1, label1, 'blue')
    plot_initial_spectrum_num_cells(num_init2, label2, 'red')
    plot_initial_spectrum_num_cells(num_init3, label3, 'green')
    plot_theory_and_exp_data(theory_set)
    set_plot('time = 0.00 sec', xlim, ylim, True)
    for num in range(len(frames_set1)):
        if round(sol_time_arr[num], 2) == 0.28:
            plt.subplot(132)
            if not frames_set2 and not frames_set3:
                plot_spectrum_from_frame(frames_set1[num], num1, label1, 'blue', truncate_fiction_cells1)
            elif not frames_set3:
                plot_spectrum_from_frame(frames_set1[num], num1, label1, 'blue', truncate_fiction_cells1)
                plot_spectrum_from_frame(frames_set2[num], num2, label2, 'red', truncate_fiction_cells2)
            else:
                plot_spectrum_from_frame(frames_set1[num], num1, label1, 'blue', truncate_fiction_cells1)
                plot_spectrum_from_frame(frames_set2[num], num2, label2, 'red', truncate_fiction_cells2)
                plot_spectrum_from_frame(frames_set3[num], num3, label3, 'green', truncate_fiction_cells3)
            plot_theory_and_exp_data(theory_set)
            set_plot('time = 0.28 sec', xlim, ylim, True)
        if round(sol_time_arr[num], 2) == 0.66:
            plt.subplot(133)
            if not frames_set2 and not frames_set3:
                plot_spectrum_from_frame(frames_set1[num], num1, label1, 'blue', truncate_fiction_cells1)
            elif not frames_set3:
                plot_spectrum_from_frame(frames_set1[num], num1, label1, 'blue', truncate_fiction_cells1)
                plot_spectrum_from_frame(frames_set2[num], num2, label2, 'red', truncate_fiction_cells2)
            else:
                plot_spectrum_from_frame(frames_set1[num], num1, label1, 'blue', truncate_fiction_cells1)
                plot_spectrum_from_frame(frames_set2[num], num2, label2, 'red', truncate_fiction_cells2)
                plot_spectrum_from_frame(frames_set3[num], num3, label3, 'green', truncate_fiction_cells3)
            plot_theory_and_exp_data(theory_set)
            set_plot('time = 0.66 sec', xlim, ylim, True)
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, save_name))


def make_plot_with_exp(frames_set1: typing.List[pd.DataFrame], name1: str, num1: int, truncate_fiction_cells1: bool,
                       frames_set2: typing.List[pd.DataFrame], name2: str, num2: int, truncate_fiction_cells2: bool,
                       sol_time_arr, xlim=(1e1, 1e3), ylim=(1e-6, 1e-3),
                       save_name='spectrum_history_with_exp_data.png', theory_set=1e-1):
    """
    Создание графика истории изменения спектра в ходе расчета и эксперимента
    NOTE: работает только в том случае, если в обоих случях величина шага сетки была одинакова
    """
    t_show = [0.28, 0.66]
    plt.figure(figsize=(10, 8))
    plot_initial_spectrum()
    num_t_show = 0
    for num, frame in enumerate(frames_set1):
        u_arr = np.array(frame['U'])
        v_arr = np.array(frame['V'])
        w_arr = np.array(frame['W'])
        spectrum = SpatialSpectrum3d(num1, config.grid_step, u_arr, v_arr, w_arr, 100, truncate_fiction_cells1)
        spectrum.compute_spectrum()
        if round(sol_time_arr[num], 2) in t_show:
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
            spectrum = SpatialSpectrum3d(num2, config.grid_step, u_arr, v_arr, w_arr, 100, truncate_fiction_cells2)
            spectrum.compute_spectrum()
            if round(sol_time_arr[num], 2) in t_show:
                if num_t_show == 0:
                    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color='blue', label=name2)
                else:
                    plt.plot(spectrum.k_mag, spectrum.e_k_mag, lw=0.8, color='blue')
                num_t_show += 1
    plot_theory_and_exp_data(theory_set)
    set_plot('Сравнение изменения спектра при расчете \nс экспериментальными данными', xlim, ylim, True)
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
                             truncate_fiction_cells1: bool,
                             frames_set2: typing.List[pd.DataFrame], name2: str, num2: int, sol_time_arr2,
                             truncate_fiction_cells2: bool,
                             ylim=(0, 0.05), xlim: tuple=None, scale='linear', save_name='kinetic_energy.png',
                             theory_set=0.004):
    """
    Создание графика изменения кинетической энергии турбулентности
    """
    time = np.linspace(0, 1.0, 300)
    energy_arr1 = get_kinetic_energy_arr(num1, frames_set1, truncate_fiction_cells1)
    energy_arr2 = get_kinetic_energy_arr(num2, frames_set2, truncate_fiction_cells2)

    exp_time = [0, 0.28, 0.66]
    # кинетическая энергия турбулентности по экспериментальным даныым
    E_42_interp = interp1d(k_42, E_42)
    k_t42 = quad(E_42_interp, min(k_42), max(k_42))[0]
    E_98_interp = interp1d(k_98, E_98)
    k_t98 = quad(E_98_interp, min(k_98), max(k_98))[0]
    E_171_interp = interp1d(k_171, E_171)
    k_t171 = quad(E_171_interp, min(k_171), max(k_171))[0]

    plt.figure(figsize=(8, 6))
    sol_time_arr1 = np.array([0] + list(sol_time_arr1))
    plt.plot(sol_time_arr1, energy_arr1, lw=1, label=name1)
    if frames_set2:
        sol_time_arr2 = np.array([0] + list(sol_time_arr2))
        plt.plot(sol_time_arr2, energy_arr2, lw=1, label=name2)
    plt.plot(time, theory_set / time ** 1.2, lw=1, color='black', linestyle='--',
             label=r'$\sim t^{-1.2}$')
    plt.plot(exp_time, [k_t42, k_t98, k_t171], linestyle='', marker='s', ms=6, mew=1, mfc='white', label='Exp',
             color='black')
    plt.legend(fontsize=12)
    plt.title('Кинетическая энергия турбулентности')
    plt.xlabel('time', fontsize=12)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.ylabel(r'$K_t$', fontsize=12)
    plt.grid()
    plt.ylim(*ylim)
    if xlim:
        plt.xlim(*xlim)
    else:
        plt.xlim(sol_time_arr1[0], sol_time_arr1[len(sol_time_arr1) - 1])
    plt.savefig(os.path.join(base_dir, config.spectrum_plots_dir, save_name))

if __name__ == '__main__':
    # frames1, sol_time_arr1 = get_frames_set_and_sol_time(os.path.join(config.cfx_data_dir,
    #                                                      '0,5_length'))
    # frames2, sol_time_arr2 = get_frames_set_and_sol_time(os.path.join(config.lazurit_data_dir, '0,5_length'))
    #
    # frames1, sol_time_arr1 = sort_frames(frames1, sol_time_arr1)
    # frames2, sol_time_arr2 = sort_frames(frames2, sol_time_arr2)
    #
    # make_comparison_plot(frames1, 'CFX, 46 cells', config.num, False,
    #                      frames2, 'Lazurit, 46 cells', config.num+2, True, sol_time_arr1,
    #                      save_name='spectrum_history_lazurit_cfx_46cells.png', ylim=(1e-6, 1e-3))
    #
    # # ---------------------------------------------------------------------------
    # # Создание графика истории изменения спектра в ходе расчета и эксперимента
    # # ---------------------------------------------------------------------------
    # make_plot_with_exp(frames1, 'CFX, 46 cells', config.num, False,
    #                    frames2, 'Lazurit, 46 cells', config.num+2, True, sol_time_arr1,
    #                    save_name='spectrum_history_with_exp_data_lazurit_cfx_46cells.png',
    #                    ylim=(1e-6, 1e-3), theory_set=1e-1)
    # # --------------------------------------------------------------------------------
    # # создание графика изменения кинетической энергии турбулентности
    # # --------------------------------------------------------------------------------
    # make_kinetic_energy_plot(frames1, 'CFX, 46 cells', config.num, sol_time_arr1, False,
    #                          frames2, 'Lazurit, 46 cells', config.num+2, sol_time_arr2, True,
    #                          save_name='kinetic_energy_lazurit_cfx_46cells.png', ylim=(0, 0.1),
    #                          theory_set=0.0025, scale='linear', xlim=(0.0, 0.66))
    # plt.show()

    # --------------------------------------------------------------------------------------
    # создание графиков для разных сеток
    # --------------------------------------------------------------------------------------

    grid1_dir = os.path.join(config.main_data_dir, '46_cells', config.lazurit_rel_data_dir, '0,5_length')
    grid2_dir = os.path.join(config.main_data_dir, '64_cells', config.lazurit_rel_data_dir, 'low_re')
    grid3_dir = os.path.join(config.main_data_dir, '90_cells', config.lazurit_rel_data_dir, 'low_re')

    frames_grid1, sol_time_arr_grid1 = get_frames_set_and_sol_time(grid1_dir)
    frames_grid2, sol_time_arr_grid2 = get_frames_set_and_sol_time(grid2_dir)
    frames_grid3, sol_time_arr_grid3 = get_frames_set_and_sol_time(grid3_dir)

    frames_grid1, sol_time_arr_grid1 = sort_frames(frames_grid1, sol_time_arr_grid1)
    frames_grid2, sol_time_arr_grid2 = sort_frames(frames_grid2, sol_time_arr_grid2)
    frames_grid3, sol_time_arr_grid3 = sort_frames(frames_grid3, sol_time_arr_grid3)

    make_different_grids_plot(frames_grid1, 'Lazurit, 46 cells', 46 + 2, True,
                              frames_grid2, 'Lazurit, 64 cells', 64 + 2, True,
                              frames_grid3, 'Lazurit, 90 cells', 90 + 2, True,
                              sol_time_arr_grid1, 46, 64, 90,
                              save_name='spectrum_history_diff_grids.png',
                              theory_set=1e-1)
    plt.show()
