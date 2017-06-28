import numpy as np
# ------------------------------------------------------------------------------------------
# настройки для генерации поля вспомогательной пульсационной скорости на равномерной сетке
# ------------------------------------------------------------------------------------------
i_cnt = 40   # количество узлов в направлении орта i
j_cnt = 40   # количество узлов в направлении орта j
k_cnt = 40   # количество узлов в направлении орта k
grid_step = 0.01   # шаг сетки в м
l_e = 0.06   # длина волны наиболее энергонесущих мод синтезированного поля пульсаций
viscosity = 2e-5   # молекулярная вязкость
dissipation_rate = 6e1   # степень диссипации
alpha = 0.01   # константа для определения набора волновых чисел
u0 = np.array([0., 0., 0.])  # характерная скорость
time = 0.  # параметр времени


