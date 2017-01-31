import config
import lib

lib.plot_spectrum([0.01, 0.01, 0.01], 0, r'output\synthetic_turbulence_field', 2 * config.grid_step, config.l_e,
                  2 * config.grid_step, config.l_e, config.viscosity, config.dissipation_rate, config.alpha,
                  config.u0)
turbulence_generator = lib.UniformGridAuxiliaryPulsationVelocityFieldGenerator(config.i_cnt, config.j_cnt,
                                                                               config.k_cnt,
                                                                               r'output\synthetic_turbulence_field.TEC',
                                                                               config.grid_step, config.l_e,
                                                                               config.viscosity,
                                                                               config.dissipation_rate, config.alpha,
                                                                               config.u0, config.time)
# turbulence_generator.commit()
