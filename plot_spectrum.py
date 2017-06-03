import spectrum_lib
from generate_synthetic_field import data_files_dir
import config
import os

spectrum_plots_dir = 'output\spectrum_plots'

if __name__ == '__main__':
    u_arr, v_arr, w_arr = spectrum_lib.read_velocity_file(os.path.join(data_files_dir, 'velocity.VEL'))
    spatial_spectrum_3d = spectrum_lib.SpatialSpectrum3d(config.i_cnt, config.j_cnt, config.k_cnt, config.grid_step,
                                                         u_arr, v_arr, w_arr, 200)
    spatial_spectrum_3d.compute_spectrum()

    spectrum_lib.plot_spectrum_with_predefined(spatial_spectrum_3d.k_abs_arr, spatial_spectrum_3d.energy_sum_arr,
                                               os.path.join(spectrum_plots_dir, 'spatial_spectrum_3d.png'),
                                               2 * config.grid_step, config.l_e, 2 * config.grid_step, config.l_e)