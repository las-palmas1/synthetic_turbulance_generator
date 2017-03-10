import numpy as np
# -------------------------------------------------------------------------------------
# настройки для тестирования спектрального метода вычислением одноточечных корреляций
# -------------------------------------------------------------------------------------
time = 3    # промежуток времени, на котором будут вычислено поле пульсаций
ts_cnt = 30000  # число шагов по времени
iter_cnt = 2  # число итераций
l_e = 2e-3   # наибольший размер вихрей
l_cut = 2e-4   # наименьший размер вихрей
vector = np.array([0.01, 0.01, 0.01])  # радиус-вектор точки вычисления пульсаций
viscosity = 2e-5   # молекулярная вязкость
dissipation_rate = 6e3   # степень диссипации
alpha = 0.01   # константа для определения набора волновых чисел
u0 = np.array([1., 0., 0.])  # характерная скорость
places = 2   # номер разряда округления при тестировании


