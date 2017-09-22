import os
import config
from spectrum import spectrum_lib
from dhit_ic.gen_ic import k_42, k_98, k_171, E_42, E_98, E_171

base_dir = os.path.dirname(os.path.dirname(__file__))

if __name__ == '__main__':
    u_arr, v_arr, w_arr = spectrum_lib.read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spatial_spectrum_3d = spectrum_lib.SpatialSpectrum3d(config.num, config.grid_step,
                                                         u_arr, v_arr, w_arr, 100)
    spatial_spectrum_3d.compute_spectrum()
    if config.ic_type == 'exp':
        spectrum_lib.plot_spectrum_exp(spatial_spectrum_3d.k_mag, spatial_spectrum_3d.e_k_mag, k_42, E_42,
                                       os.path.join(base_dir, config.spectrum_plots_dir, 'init_spectrum_exp.png'),
                                       xlim=(1e1, 1e4), ylim=(1e-5, 1e-3))
        print('k_t = %.4f' % spatial_spectrum_3d.get_turb_kinetic_energy())
    else:
        spectrum_lib.plot_spectrum_von_karman(spatial_spectrum_3d.k_mag, spatial_spectrum_3d.e_k_mag,
                                              os.path.join(base_dir, config.spectrum_plots_dir,
                                                           'init_spectrum_von_karman.png'),
                                              2 * config.grid_step, config.l_e, 2 * config.grid_step, config.l_e)

        print('k_t = %.4f' % spatial_spectrum_3d.get_turb_kinetic_energy())

