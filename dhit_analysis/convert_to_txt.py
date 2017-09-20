import config
from tecplot_lib import get_write_data_set_command, wrap_macro, execute_macro, \
    get_open_data_file_command, LoaderType, create_macro_file
import os
import numpy as np

"""
Создание макроса tecplot для конвертации файлов с результатами в текстовый формат
"""

if __name__ == '__main__':
    cfx_files = os.listdir(os.path.join(config.cfx_data_dir, 'bin'))
    macro = ''
    for n, file in enumerate(cfx_files):
        open_file = get_open_data_file_command(os.path.join(config.cfx_data_dir, 'bin', file), LoaderType.CFX)
        write_data = get_write_data_set_command(os.path.join(config.cfx_data_dir, 'txt', 'cfx_%s.dat' % n),
                                                binary=False, var_list=list(np.linspace(1, 15, 15, dtype=np.int)),
                                                zone_list=[1])
        macro += open_file + write_data
    # lazurit_files = os.listdir(os.path.join(config.lazurit_data_dir, 'bin'))
    # for n, file in enumerate(lazurit_files):
    #     open_file = get_open_data_file_command(os.path.join(config.lazurit_data_dir, 'bin', file), LoaderType.TECPLOT)
    #     write_data = get_write_data_set_command(os.path.join(config.lazurit_data_dir, 'txt', 'lazurit_%s.dat' % n),
    #                                             binary=False)
    #     macro += open_file + write_data
    macro = wrap_macro(macro)
    create_macro_file(macro, 'convert_to_txt.mcr')