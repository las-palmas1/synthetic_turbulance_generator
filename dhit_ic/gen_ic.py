import logging
from scipy.io import loadmat
import config
import os
from dhit_ic.lib import HITGeneratorGivenSpectrum, HITGeneratorVonKarman

base_dir = os.path.dirname(os.path.dirname(__file__))
logging.basicConfig(format='%(levelname)s: %(message)s', level=config.log_level)

exp_data = loadmat(os.path.join(base_dir, config.exp_data))

# обезразмеривание экспериментальных данных (закомментировано)
# перевод размерностей k в 1/м, E в м^3/с^2
k_42 = exp_data['k_42'][0] * 1e2  # config.L_ref
E_42 = config.increase_degree * exp_data['E_42'][0] / 1e6   # (config.u_ref**2 * config.L_ref)
k_98 = exp_data['k_98'][0] * 1e2  # config.L_ref
E_98 = config.increase_degree * exp_data['E_98'][0] / 1e6  # (config.u_ref**2 * config.L_ref)
k_171 = exp_data['k_171'][0] * 1e2  # config.L_ref
E_171 = config.increase_degree * exp_data['E_171'][0] / 1e6  # (config.u_ref**2 * config.L_ref)

if __name__ == '__main__':
    if config.ic_type == 'exp':
        hit_generator = HITGeneratorGivenSpectrum(config.num, os.path.join(base_dir, config.data_files_dir),
                                                  config.grid_step, E_42, k_42)
    else:
        hit_generator = HITGeneratorVonKarman(config.num,
                                              os.path.join(base_dir, config.data_files_dir),
                                              config.grid_step, config.l_e,
                                              config.viscosity,
                                              config.dissipation_rate, config.alpha,
                                              config.u0, config.time)
    hit_generator.run()

