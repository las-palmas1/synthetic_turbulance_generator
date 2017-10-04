import config
from tecplot_lib import get_write_data_set_command, wrap_macro, execute_macro, \
    get_open_data_file_command, LoaderType, create_macro_file
import os
import numpy as np

"""
Создание макроса tecplot для конвертации файлов с результатами в текстовый формат
"""


def get_conversation_macro(data_dir: str, type: LoaderType, var_list=None, zone_list=None) -> str:
    result = ''
    files = os.listdir(os.path.join(data_dir, 'bin'))
    for n, file in enumerate(files):
        open_file = get_open_data_file_command(os.path.join(data_dir, 'bin', file), type)
        write_data = get_write_data_set_command(os.path.join(data_dir, 'txt', 'data_%s.dat' % n),
                                                binary=False, var_list=var_list, zone_list=zone_list)
        result += open_file + write_data
    return result

if __name__ == '__main__':
    macro = ''
    macro += get_conversation_macro(config.cfx_data_dir, LoaderType.CFX, list(np.linspace(1, 15, 15, dtype=np.int)),
                                    zone_list=[1])
    macro += get_conversation_macro(os.path.join(config.lazurit_data_dir, '1e-4'), LoaderType.TECPLOT)
    macro += get_conversation_macro(os.path.join(config.lazurit_data_dir, '5e-5'), LoaderType.TECPLOT)
    macro = wrap_macro(macro)
    create_macro_file(macro, 'convert_to_txt.mcr')