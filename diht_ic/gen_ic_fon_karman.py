import config
from diht_ic.lib import HITGeneratorVonKarman
import os

base_dir = os.path.dirname(os.path.dirname(__file__))

if __name__ == '__main__':
    turbulence_generator = HITGeneratorVonKarman(config.num,
                                                 os.path.join(base_dir, config.data_files_dir),
                                                 config.grid_step, config.l_e,
                                                 config.viscosity,
                                                 config.dissipation_rate, config.alpha,
                                                 config.u0, config.time)
    turbulence_generator.run(mode=config.run_mode, proc_num=config.proc_num)
