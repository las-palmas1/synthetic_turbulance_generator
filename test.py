import lib
import unittest as ut
import typing
import numpy as np
import test_config
import matplotlib.pyplot as plt


class _SpectralMethodPulsationAndReynoldsStressHistory:
    def __init__(self, l_cut, l_e, u0, vector: typing.Iterable[float], time=3., ts_cnt=3000, iter_cnt: int=10,
                 viscosity=2e-5, dissipation_rate=6e3, alpha=0.01):
        """
        :param l_cut: минимальный размер вихря
        :param l_e: максимальный размер вихря
        :param u0: характерная скорость
        :param vector: [x, y, z] радиус-вектор, характеризующий положение точки, в кторой производится
        вычисление пульсаций скорости и напряжений Рейнольдса
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
        self.u_rel_av = 0.
        self.v_rel_av = 0.
        self.w_rel_av = 0.
        self.uv_rel_av = 0.
        self.uw_rel_av = 0.
        self.vw_rel_av = 0.
        self.uu_rel_av = 0.
        self.vv_rel_av = 0.
        self.ww_rel_av = 0.
        self.average_velocity_vector_arr = np.zeros((self.iter_cnt, 9))
        self.max_abs_velocity_vector_arr = np.zeros((self.iter_cnt, 9))
        self.rel_av_velocity_vector_arr = np.zeros((self.iter_cnt, 9))
        self.rel_av_velocity_vector = np.zeros(9)
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
            yield u, v, w, uv, uw, vw, u * u, v * v, w * w

    def _get_rel_average_velocity_vector(self, velocity_gen: typing.Iterator[Vector], i, ts_cnt) -> np.ndarray:
        result = np.zeros(9)
        velocity_sum = np.zeros(9)
        for n, velocity_vector in enumerate(velocity_gen):
            velocity_sum += np.array(velocity_vector)
            abs_vel_vector = abs(np.array(velocity_vector))
            if n == 0:
                self.max_abs_velocity_vector_arr[i] = abs_vel_vector
            else:
                self.max_abs_velocity_vector_arr[i] = abs_vel_vector * \
                                                (abs_vel_vector > self.max_abs_velocity_vector_arr[i]) + \
                                                self.max_abs_velocity_vector_arr[i] * \
                                                (abs_vel_vector < self.max_abs_velocity_vector_arr[i])
            average_velocity_vector = velocity_sum / ts_cnt
            self.average_velocity_vector_arr[i] = average_velocity_vector
            result = self.average_velocity_vector_arr[i] / self.max_abs_velocity_vector_arr[i]
        return result

    @classmethod
    def _get_arrays_for_plotting(cls, velocity_gen: typing.Iterator[Vector], ts_cnt):
        velocity_arr = np.zeros([ts_cnt, 3])
        av_velocity_arr = np.zeros([ts_cnt, 9])
        velocity_sum = np.zeros(9)
        max_abs_velocity_vector_arr = np.zeros(9)
        for n, velocity_vector in enumerate(velocity_gen):
            print('plotting arrays computing, iter =  %s' % n)
            velocity_sum += np.array(velocity_vector)
            abs_vel_vector = abs(np.array(velocity_vector))
            velocity_arr[n, 0] = velocity_vector[0]
            velocity_arr[n, 1] = velocity_vector[1]
            velocity_arr[n, 2] = velocity_vector[2]
            if n == 0:
                max_abs_velocity_vector_arr = abs_vel_vector
            else:
                max_abs_velocity_vector_arr = abs_vel_vector * \
                                                (abs_vel_vector > max_abs_velocity_vector_arr) + \
                                                max_abs_velocity_vector_arr * \
                                                (abs_vel_vector < max_abs_velocity_vector_arr)
            av_velocity_arr[n] = (velocity_sum / n)

        return velocity_arr, av_velocity_arr

    @classmethod
    def _plot_velocity_component_history(cls, vel_component_arr, t_arr, component_name, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(t_arr, vel_component_arr, color='red', lw=0.7)
        plt.grid()
        plt.xlabel(r'$t, s$', fontsize=16)
        plt.ylabel(r'$%s$' % component_name, fontsize=16)
        plt.xlim(0, t_arr[len(t_arr) - 1])
        ylim = max(abs(vel_component_arr)) * 1.1
        plt.ylim(-ylim, ylim)
        plt.savefig(filename)

    @classmethod
    def _plot_average_pulsation_history(cls, av_velocity_arr: np.ndarray, t_arr, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(t_arr, av_velocity_arr[:, 0], label=r'$<u>$', color='blue', lw=0.7)
        plt.plot(t_arr, av_velocity_arr[:, 1], label=r'$<v>$', color='red', lw=0.7)
        plt.plot(t_arr, av_velocity_arr[:, 2], label=r'$<w>$', color='green', lw=0.7)
        plt.grid()
        plt.xlabel(r'$t, s$', fontsize=16)
        plt.ylabel(r'$<u_i>$', fontsize=16)
        plt.ylabel('')
        plt.ylim(-1, 1)
        plt.xlim(0, t_arr[len(t_arr) - 1])
        plt.legend(fontsize=12)
        plt.savefig(filename)

    @classmethod
    def _plot_stress_history(cls, av_velocity_arr: np.ndarray, t_arr, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(t_arr, av_velocity_arr[:, 3], label=r'$<uv>$', color='blue', lw=0.7)
        plt.plot(t_arr, av_velocity_arr[:, 4], label=r'$<uw>$', color='red', lw=0.7)
        plt.plot(t_arr, av_velocity_arr[:, 5], label=r'$<vw>$', color='green', lw=0.7)
        plt.plot(t_arr, av_velocity_arr[:, 6], label=r'$<uu>$', color='blue', lw=0.7, linestyle=':')
        plt.plot(t_arr, av_velocity_arr[:, 7], label=r'$<vv>$', color='red', lw=0.7, linestyle=':')
        plt.plot(t_arr, av_velocity_arr[:, 8], label=r'$<ww>$', color='green', lw=0.7, linestyle=':')
        plt.grid()
        plt.xlabel(r'$t, s$', fontsize=16)
        plt.ylabel(r'$<u_i u_j>$', fontsize=16)
        plt.ylim(-2, 3)
        plt.xlim(0, t_arr[len(t_arr) - 1])
        plt.legend(fontsize=12)
        plt.savefig(filename)

    @classmethod
    def _plot_history(cls, velocity_arr: np.ndarray, rel_av_velocity_arr: np.ndarray, t_arr):
        name_template = 'output/history_plots/spectral_method_%s.png'
        cls._plot_velocity_component_history(velocity_arr[:, 0], t_arr, 'u', name_template % 'u')
        cls._plot_velocity_component_history(velocity_arr[:, 1], t_arr, 'v', name_template % 'v')
        cls._plot_velocity_component_history(velocity_arr[:, 2], t_arr, 'w', name_template % 'w')
        cls._plot_average_pulsation_history(rel_av_velocity_arr, t_arr, name_template % 'average_pulsation')
        cls._plot_stress_history(rel_av_velocity_arr, t_arr, name_template % 'stress')

    @classmethod
    def _log_iterations_results(cls, rel_av_velocity_vector_arr: np.ndarray):
        for j in range(rel_av_velocity_vector_arr.shape[0]):
            print('\niter = %s, <u>_rel = %.6f, <v>_rel = %.6f,  <w>_rel = %.6f' %
                  (j, rel_av_velocity_vector_arr[j][0], rel_av_velocity_vector_arr[j][1],
                   rel_av_velocity_vector_arr[j][2]))
            print('iter = %s, <uv>_rel = %.6f, <uw>_rel = %.6f,  <vw>_rel = %.6f' %
                  (j, rel_av_velocity_vector_arr[j][3], rel_av_velocity_vector_arr[j][4],
                   rel_av_velocity_vector_arr[j][5]))
            print('iter = %s, <uu>_rel = %.6f, <vv>_rel = %.6f,  <ww>_rel = %.6f' %
                  (j, rel_av_velocity_vector_arr[j][6], rel_av_velocity_vector_arr[j][7],
                   rel_av_velocity_vector_arr[j][8]))

    def _compute_average_results(self, rel_av_velocity_vector_arr: np.ndarray, iter_cnt):
        self.u_rel_av = sum(rel_av_velocity_vector_arr[:, 0]) / iter_cnt
        self.v_rel_av = sum(rel_av_velocity_vector_arr[:, 1]) / iter_cnt
        self.w_rel_av = sum(rel_av_velocity_vector_arr[:, 2]) / iter_cnt
        self.uv_rel_av = sum(rel_av_velocity_vector_arr[:, 3]) / iter_cnt
        self.uw_rel_av = sum(rel_av_velocity_vector_arr[:, 4]) / iter_cnt
        self.vw_rel_av = sum(rel_av_velocity_vector_arr[:, 5]) / iter_cnt
        self.uu_rel_av = sum(rel_av_velocity_vector_arr[:, 6]) / iter_cnt
        self.vv_rel_av = sum(rel_av_velocity_vector_arr[:, 7]) / iter_cnt
        self.ww_rel_av = sum(rel_av_velocity_vector_arr[:, 8]) / iter_cnt

    def commit(self):
        self._t_arr = np.linspace(0, self.time, self.ts_cnt)
        for i in range(self.iter_cnt):
            tau, k_arr, amplitude_arr, d_vector_arr, sigma_vector_arr, phase_arr, frequency_arr = \
                lib.get_auxiliary_pulsation_velocity_parameters(self.l_cut, self.l_e, self.l_cut, self.l_e,
                                                                self.viscosity, self.dissipation_rate, self.alpha,
                                                                self.u0)
            velocity_gen = self._get_velocity_generator(i, self._t_arr, tau, k_arr, amplitude_arr, d_vector_arr,
                                                        sigma_vector_arr, phase_arr, frequency_arr)

            if i == 0:
                velocity_arr, av_velocity_arr = self._get_arrays_for_plotting(velocity_gen, self.ts_cnt)
                self._plot_history(velocity_arr, av_velocity_arr, self._t_arr)

            self.rel_av_velocity_vector_arr[i] = self._get_rel_average_velocity_vector(velocity_gen, i, self.ts_cnt)

        self._log_iterations_results(self.rel_av_velocity_vector_arr)
        self._compute_average_results(self.rel_av_velocity_vector_arr, self.iter_cnt)


class SpectralMethodTestCase(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.correlations = _SpectralMethodPulsationAndReynoldsStressHistory(test_config.l_cut, test_config.l_e,
                                                                            test_config.u0, test_config.vector,
                                                                            test_config.time, test_config.ts_cnt,
                                                                            test_config.iter_cnt, test_config.viscosity,
                                                                            test_config.dissipation_rate,
                                                                            test_config.alpha)
        cls.correlations.commit()

    def test_u_rel_av(self):
        self.assertAlmostEqual(0., self.correlations.u_rel_av, places=test_config.places,
                               msg='<u>_rel = %.6f' % self.correlations.u_rel_av)

    def test_v_rel_av(self):
        self.assertAlmostEqual(0., self.correlations.v_rel_av, places=test_config.places,
                               msg='<v>_rel = %.6f' % self.correlations.v_rel_av)

    def test_w_rel_av(self):
        self.assertAlmostEqual(0., self.correlations.w_rel_av, places=test_config.places,
                               msg='<w>_rel = %.6f' % self.correlations.w_rel_av)

    def test_uv_rel_av(self):
        self.assertAlmostEqual(0., self.correlations.uv_rel_av, places=test_config.places,
                               msg='<uv>_rel = %.6f' % self.correlations.uv_rel_av)

    def test_uw_rel_av(self):
        self.assertAlmostEqual(0., self.correlations.uw_rel_av, places=test_config.places,
                               msg='<uw>_rel = %.6f' % self.correlations.uw_rel_av)

    def test_vw_rel_av(self):
        self.assertAlmostEqual(0., self.correlations.vw_rel_av, places=test_config.places,
                               msg='<vw>_rel = %.6f' % self.correlations.vw_rel_av)

    def test_uu_rel_av(self):
        self.assertNotAlmostEqual(0., self.correlations.uu_rel_av, places=test_config.places,
                                  msg='<uu>_rel = %.6f' % self.correlations.uu_rel_av)

    def test_vv_rel_av(self):
        self.assertNotAlmostEqual(0., self.correlations.vv_rel_av, places=test_config.places,
                                  msg='<vv>_rel = %.6f' % self.correlations.vv_rel_av)

    def test_ww_rel_av(self):
        self.assertNotAlmostEqual(0., self.correlations.ww_rel_av, places=test_config.places,
                                  msg='<ww>_rel = %.6f' % self.correlations.ww_rel_av)

if __name__ == '__main__':
    loader = ut.TestLoader()
    ut.TextTestRunner(stream=open('test.txt', 'w'),
                      verbosity=2).run(loader.loadTestsFromTestCase(SpectralMethodTestCase))
