import config
from core import lib
import os

base_dir = os.path.basename(os.path.basename(__file__))

if __name__ == '__main__':
    turbulence_generator = lib.HomogeneousIsotropicTurbulenceGenerator(config.i_cnt, config.j_cnt,
                                                                       config.k_cnt,
                                                                       os.path.join(base_dir, config.data_files_dir),
                                                                       config.grid_step, config.l_e,
                                                                       config.viscosity,
                                                                       config.dissipation_rate, config.alpha,
                                                                       config.u0, config.time)
    turbulence_generator.run(mode=config.run_mode, proc_num=config.proc_num)
