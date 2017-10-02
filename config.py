import numpy as np
import logging
# ------------------------------------------------------------------------------------------
# настройки для генерации поля вспомогательной пульсационной скорости на равномерной сетке
# ------------------------------------------------------------------------------------------
ic_type = 'exp'    # тип начальных условий, 'exp' или 'von_karman'
num = 32  # количество узлов в на стороне генерируемого куба
length = 1   # длина стороны куба в м
# NOTE: генерируемый спектр получается безразмерным только в случае, если length = 1; в противном случае
# энергия будет иметь размерноть м, а волновые числа - 1/м. В первом случае скорость по умолчанию генерируется в
# безразмерном виде, поэтому энергия не имеет в составе своей размерности множителя м^2/с^2. Во втором случае
# размерность генерируемой скорости как от величины параметра length, так и от размерности заданного спектра (при
# length = 1 зависит только от спектра). Если он обезразмерен, то скорость получается безразмерной, если - нет, то
# скорость будет размерной (по умолчанию в экспериментальных данных энергия в см^3/с^2, а волновое число в 1/см).
grid_step = length / (num - 1)   # шаг сетки в м
l_e = 1.5   # длина волны наиболее энергонесущих мод синтезированного поля пульсаций
viscosity = 2e-5   # молекулярная вязкость
dissipation_rate = 6e1   # степень диссипации
alpha = 0.01   # константа для определения набора волновых чисел
u0 = np.array([0., 0., 0.])  # характерная скорость
time = 0.  # параметр времени
log_level = logging.INFO  # уровень логгирования
run_mode = 'single'    # способ вычисления скоростей (многопроцессорный "single" или простой "mp_pool")
proc_num = 4    # число процессов
data_files_dir = 'output\data_files'
spectrum_plots_dir = 'output\spectrum_plots'
monitor_plots_dir = 'output\monitor_data_plots'
cfx_data_dir = r'C:\Users\User\Documents\tasks\computitions_and post_processing\diht\32_cells\data_for_analysis\cfx'
lazurit_data_dir = 'C:\\Users\\User\\Documents\\tasks\computitions_and post_processing\\diht\\32_cells\\' \
                   'data_for_analysis\\lazurit'
monitor_data_dir = 'C:\\Users\\User\\Documents\\tasks\\computitions_and post_processing\\diht\\' \
                   '32_cells\\data_for_analysis\\monitor'
exp_data = 'dhit_ic\CBC_exp.mat'
# -----------------------------------------------------------------------
# параметры для обезразмеривания экспериментального спектра
# -----------------------------------------------------------------------
M = 5.08  # в см
U0 = 1000  # в см/с
L_ref = 11*M  # в см
u_ref = np.sqrt(3/2)*22.2  # в см/с
# ------------------------------------------------------------------------------
#  некоторые параметры решателя
# ------------------------------------------------------------------------------
time_step = 1e-4   # шаг по времени
time_step_int = 700     # интервал, через который происходит сохранение нестационарных данных
max_time_step = 6600
