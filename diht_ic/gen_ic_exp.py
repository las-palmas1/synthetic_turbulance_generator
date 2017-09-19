import logging
from scipy.io import loadmat
import config
import os
from diht_ic.lib import HITGeneratorGivenSpectrum

base_dir = os.path.dirname(os.path.dirname(__file__))
logging.basicConfig(format='%(levelname)s: %(message)s', level=config.log_level)

exp_data = loadmat(os.path.join(base_dir, config.exp_data))

# обезразмеривание экспериментальных данных
k_42 = exp_data['k_42'][0] * config.L_ref
E_42 = exp_data['E_42'][0] / (config.u_ref**2 * config.L_ref)
k_98 = exp_data['k_98'][0] * config.L_ref
E_98 = exp_data['E_98'][0] / (config.u_ref**2 * config.L_ref)
k_171 = exp_data['k_171'][0] * config.L_ref
E_171 = exp_data['E_171'][0] / (config.u_ref**2 * config.L_ref)

if __name__ == '__main__':
    hit_generator = HITGeneratorGivenSpectrum(config.num, os.path.join(base_dir, config.data_files_dir),
                                              config.grid_step, E_42, k_42)
    hit_generator.run()

