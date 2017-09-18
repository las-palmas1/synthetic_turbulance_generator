import os
import config
from spectrum import spectrum_lib

base_dir = os.path.dirname(os.path.dirname(__file__))

if __name__ == '__main__':
    u_arr, v_arr, w_arr = spectrum_lib.read_velocity_file(os.path.join(base_dir, config.data_files_dir, 'velocity.VEL'))
    spatial_spectrum_3d = spectrum_lib.SpatialSpectrum3d(config.num, config.grid_step,
                                                         u_arr, v_arr, w_arr, 100)
    spatial_spectrum_3d.compute_spectrum()

    spectrum_lib.plot_spectrum_with_predefined(spatial_spectrum_3d.k_mag, spatial_spectrum_3d.e_k_mag,
                                               os.path.join(base_dir, config.spectrum_plots_dir,
                                                            'spatial_spectrum_3d.png'),
                                               2 * config.grid_step, config.l_e, 2 * config.grid_step, config.l_e)
    print(spatial_spectrum_3d.get_turb_kinetic_energy())