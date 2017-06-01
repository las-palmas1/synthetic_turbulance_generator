import config
import lib
import os


data_files_dir = 'output\data_files'

if __name__ == '__main__':
    turbulence_generator = lib.HomogeneousIsotropicTurbulenceGenerator(config.i_cnt, config.j_cnt,
                                                                       config.k_cnt,
                                                                       os.path.join(data_files_dir,
                                                                                    'synthetic_turbulence_field.TEC'),
                                                                       os.path.join(data_files_dir, 'grid.PFG'),
                                                                       os.path.join(data_files_dir, 'velocity.VEL'),
                                                                       config.grid_step, config.l_e,
                                                                       config.viscosity,
                                                                       config.dissipation_rate, config.alpha,
                                                                       config.u0, config.time)
    turbulence_generator.commit()
