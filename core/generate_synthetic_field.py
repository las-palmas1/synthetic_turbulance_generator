import config
from core import les_inlet_ic_lib
import os

base_dir = os.path.dirname(os.path.dirname(__file__))

if __name__ == '__main__':
    turbulence_generator = les_inlet_ic_lib.HomogeneousIsotropicTurbulenceGenerator(config.num,
                                                                                    os.path.join(base_dir, config.data_files_dir),
                                                                                    config.grid_step, config.l_e,
                                                                                    config.viscosity,
                                                                                    config.dissipation_rate, config.alpha,
                                                                                    config.u0, config.time)
    turbulence_generator.run(mode=config.run_mode, proc_num=config.proc_num)
