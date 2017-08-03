import config
import lib

if __name__ == '__main__':
    turbulence_generator = lib.HomogeneousIsotropicTurbulenceGenerator(config.i_cnt, config.j_cnt,
                                                                       config.k_cnt,
                                                                       config.data_files_dir,
                                                                       config.grid_step, config.l_e,
                                                                       config.viscosity,
                                                                       config.dissipation_rate, config.alpha,
                                                                       config.u0, config.time)
    turbulence_generator.commit()
