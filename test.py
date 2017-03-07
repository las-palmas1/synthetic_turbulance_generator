import lib
import unittest as ut
import typing
import numpy as np
import test_config


class _SpectralMethodOnePointCorrelations:
    def __init__(self, l_cut, l_e, u0, vector: typing.Iterable[float], time=3., ts_cnt=3000, iter_cnt: int=10,
                 viscosity=2e-5, dissipation_rate=6e3, alpha=0.01):
        """
        :param l_cut: минимальный размер вихря
        :param l_e: максимальный размер вихря
        :param u0: характерная скорость
        :param vector: [x, y, z] радиус-вектор, характеризующий положение точки, в кторой производится
        вычисление корреляций
        :param time: величина временного интервала, в течение которого производится расчет пульсаций
        :param ts_cnt:  число шагов по времени
        :param iter_cnt: число повторных вычислений
        :param viscosity: вязкость
        :param dissipation_rate: степень дисспации
        :param alpha:
        """
        self.l_cut = l_cut
        self.l_e = l_e
        self.u0 = u0
        self.vector = vector
        self.time = time
        self.ts_cnt = ts_cnt
        self.iter_cnt = iter_cnt
        self.viscosity = viscosity
        self.dissipation_rate = dissipation_rate
        self.alpha = alpha
        self._u_av_arr = []
        self._v_av_arr = []
        self._w_av_arr = []
        self._uv_av_arr = []
        self._uw_av_arr = []
        self._vw_av_arr = []
        self.u_rel = 0.
        self.v_rel = 0.
        self.w_rel = 0.
        self.uv_rel = 0.
        self.uw_rel = 0.
        self.vw_rel = 0.
        self.average_velocity_vector_arr = np.zeros((self.iter_cnt, 6))
        self.max_abs_velocity_vector_arr = np.zeros((self.iter_cnt, 6))
        self.rel_velocity_vector_arr = np.zeros((self.iter_cnt, 6))
        self.rel_velocity_vector = np.zeros(6)
        self._t_arr = None

    Vector = typing.TypeVar('Vector', typing.Iterable[int], typing.Iterable[float])

    def _get_velocity_generator(self, iter_number, time_arr: np.ndarray, tau, k_arr, amplitude_arr, d_vector_arr,
                                sigma_vector_arr, phase_arr, frequency_arr) -> typing.Iterator[Vector]:
        for time in time_arr:
            print('iter_number = %s, time = %.4f' % (iter_number, time))
            velocity_vector = lib.get_auxiliary_pulsation_velocity(self.vector, time, tau, k_arr, amplitude_arr,
                                                                   d_vector_arr, sigma_vector_arr, phase_arr,
                                                                   frequency_arr)
            u = velocity_vector[0]
            v = velocity_vector[1]
            w = velocity_vector[2]
            uv = u * v
            uw = u * w
            vw = v * w
            yield u, v, w, uv, uw, vw

    def commit(self):
        self._t_arr = np.linspace(0, self.time, self.ts_cnt)
        for i in range(self.iter_cnt):
            tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr = \
                lib.get_auxiliary_pulsation_velocity_parameters(self.l_cut, self.l_e, self.l_cut, self.l_e,
                                                                self.viscosity, self.dissipation_rate, self.alpha,
                                                                self.u0)
            velocities_gen = self._get_velocity_generator(i, self._t_arr, tau, k_arr, amplitude_arr, d_vector_arr,
                                                                  sigma_vector_arr, phase_arr, frequency_arr)
            velocity_sum = np.zeros(6)
            for n, velocity_vector in enumerate(velocities_gen):
                velocity_sum += np.array(velocity_vector)
                abs_vel_vector = abs(np.array(velocity_vector))
                if n == 0:
                    self.max_abs_velocity_vector_arr[i] = abs_vel_vector
                else:
                    self.max_abs_velocity_vector_arr[i] = abs_vel_vector * \
                                                          (abs_vel_vector > self.max_abs_velocity_vector_arr[i]) + \
                                                          self.max_abs_velocity_vector_arr[i] * \
                                                          (abs_vel_vector < self.max_abs_velocity_vector_arr[i])

            average_velocity_vector = velocity_sum / self.ts_cnt
            self.average_velocity_vector_arr[i] = average_velocity_vector
            self.rel_velocity_vector_arr[i] = self.average_velocity_vector_arr[i] / self.max_abs_velocity_vector_arr[i]
        for j in range(self.iter_cnt):
            print('iter = %s, <u>_rel = %.6f, <v>_rel = %.6f,  <w>_rel = %.6f' %
                  (j, self.rel_velocity_vector_arr[j][0],  self.rel_velocity_vector_arr[j][1],
                   self.rel_velocity_vector_arr[j][2]))
            print('iter = %s, <uv>_rel = %.6f, <vw>_rel = %.6f,  <vw>_rel = %.6f' %
                  (j, self.rel_velocity_vector_arr[j][3], self.rel_velocity_vector_arr[j][4],
                   self.rel_velocity_vector_arr[j][5]))
        self.u_rel = sum(self.rel_velocity_vector_arr[:, 0]) / self.iter_cnt
        self.v_rel = sum(self.rel_velocity_vector_arr[:, 1]) / self.iter_cnt
        self.w_rel = sum(self.rel_velocity_vector_arr[:, 2]) / self.iter_cnt
        self.uv_rel = sum(self.rel_velocity_vector_arr[:, 3]) / self.iter_cnt
        self.uw_rel = sum(self.rel_velocity_vector_arr[:, 4]) / self.iter_cnt
        self.vw_rel = sum(self.rel_velocity_vector_arr[:, 5]) / self.iter_cnt


class SpectralMethodTestCase(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.correlations = _SpectralMethodOnePointCorrelations(test_config.l_cut, test_config.l_e,
                                                               test_config.u0, test_config.vector,
                                                               test_config.time, test_config.ts_cnt,
                                                               test_config.iter_cnt, test_config.viscosity,
                                                               test_config.dissipation_rate, test_config.alpha)
        cls.correlations.commit()

    def test_average_u_rel(self):
        self.assertAlmostEqual(0., self.correlations.u_rel, places=3, msg='<u>_rel = %.6f' % self.correlations.u_rel)

    def test_average_v_rel(self):
        self.assertAlmostEqual(0., self.correlations.v_rel, places=3, msg='<v>_rel = %.6f' % self.correlations.v_rel)

    def test_average_w_rel(self):
        self.assertAlmostEqual(0., self.correlations.w_rel, places=3, msg='<w>_rel = %.6f' % self.correlations.w_rel)

    def test_average_uv_rel(self):
        self.assertNotAlmostEqual(0., self.correlations.uv_rel, places=3, msg='<uv>_rel = %.6f' %
                                                                              self.correlations.uv_rel)

    def test_average_uw_rel(self):
        self.assertNotAlmostEqual(0., self.correlations.uw_rel, places=3, msg='<uw>_rel = %.6f' %
                                                                              self.correlations.uv_rel)

    def test_average_vw_rel(self):
        self.assertNotAlmostEqual(0., self.correlations.vw_rel, places=3, msg='<vw>_rel = %.6f' %
                                                                              self.correlations.vw_rel)

if __name__ == '__main__':
    loader = ut.TestLoader()
    ut.TextTestRunner(stream=open('test.txt', 'w'),
                      verbosity=2).run(loader.loadTestsFromTestCase(SpectralMethodTestCase))
