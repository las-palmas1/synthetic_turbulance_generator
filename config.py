import numpy as np
import logging
# ------------------------------------------------------------------------------------------
# настройки для генерации поля вспомогательной пульсационной скорости на равномерной сетке
# ------------------------------------------------------------------------------------------
i_cnt = 20   # количество узлов в направлении орта i
j_cnt = 20   # количество узлов в направлении орта j
k_cnt = 20   # количество узлов в направлении орта k
grid_step = 0.095   # шаг сетки в м
l_e = 1.5   # длина волны наиболее энергонесущих мод синтезированного поля пульсаций
viscosity = 2e-5   # молекулярная вязкость
dissipation_rate = 6e1   # степень диссипации
alpha = 0.01   # константа для определения набора волновых чисел
u0 = np.array([0., 0., 0.])  # характерная скорость
time = 0.  # параметр времени
log_level = logging.INFO  # уровень логгирования
run_mode = 'mp_pool'    # способ вычисления скоростей (многопроцессорный "single" или простой "mp_pool")
proc_num = 4    # число процессов
data_files_dir = 'output\data_files'
spectrum_plots_dir = 'output\spectrum_plots'
monitor_plots_dir = 'output\monitor_data_plots'
cfx_data_dir = r'C:\Users\User\Documents\tasks\computitions_and post_processing\diht\20_cells\data_for_analysis\cfx'
lazurit_data_dir = 'C:\\Users\\User\\Documents\\tasks\computitions_and post_processing\\diht\\20_cells\\' \
                   'data_for_analysis\\lazurit'
monitor_data_dir = 'C:\\Users\\User\\Documents\\tasks\\computitions_and post_processing\\diht\\' \
                   '20_cells\\data_for_analysis\\monitor'

# ------------------------------------------------------------------------------
#  некоторые параметры решателя
# ------------------------------------------------------------------------------
time_step = 1e-3   # шаг по времени
time_step_int = 20     # интервал, через который происходит сохранение нестационарных данных
max_time_step = 200
