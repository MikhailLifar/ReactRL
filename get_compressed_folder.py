import os
import shutil


def get_compressed_folder(folder_path, names_to_copy: tuple):
    parent_f, child_f = os.path.split(folder_path)
    os.makedirs(f'{parent_f}/{child_f}_compressed')
    for name in names_to_copy:
        full_name = f'{parent_f}/{child_f}/{name}'
        assert os.path.exists(full_name)
        if os.path.isfile(full_name):
            shutil.copy2(full_name, f'{parent_f}/{child_f}_compressed/{name}')
        else:
            shutil.copytree(full_name, f'{parent_f}/{child_f}_compressed/{name}')


def get_compressed_along_folder(folder_path, names_to_copy: tuple,
                                key: callable = lambda s: True) -> None:
    dirs = os.listdir(folder_path)
    for d in dirs:
        if key(d):
            try:
                get_compressed_folder(f'{folder_path}/{d}', names_to_copy=names_to_copy)
            except AssertionError:
                print(f'Error! directory: {d}')


def choose_from_folder(folder_path, subfolder_path,
                       valid_file_key: callable = lambda s: True,
                       choose_typo: str = 'max',
                       take_key: callable = lambda s: s):
    if subfolder_path is None:
        subfolder_path = ''
    if subfolder_path != '':
        subfolder_path = '/' + subfolder_path
    in_dir = [name for name in os.listdir(f'{folder_path}{subfolder_path}')
              if valid_file_key(name)]
    if choose_typo == 'max':
        outfile = max(in_dir, key=take_key)
        return f'{folder_path}{subfolder_path}/{outfile}'
    elif choose_typo == 'one_file':
        outfile, = in_dir
        return f'{folder_path}{subfolder_path}/{outfile}'
    else:
        raise NotImplementedError


def collect_along_folder(folder_path,
                         subfolder_path=None,
                         choose_folder_key: callable = lambda s: True,
                         choose_file_key: callable = lambda s: True,
                         take_key: callable = lambda s: s,
                         dest_name: str = 'collected'):
    if not os.path.exists(f'{folder_path}/{dest_name}'):
        os.makedirs(f'{folder_path}/{dest_name}')
    else:
        print('Collected folder already exists.')
        add = input('Do you want add new files? If yes, type "yes" or "y" ')
        if add.lower()[0] == 'y':
            pass
        else:
            print('Collection canceled')
            return
    dirs = [name for name in os.listdir(folder_path) if choose_folder_key(name)]
    for d in dirs:
        try:
            file_path = choose_from_folder(f'{folder_path}/{d}', subfolder_path,
                                           choose_file_key, take_key=take_key)
            if os.path.isfile(file_path):
                shutil.copy2(file_path, f'{folder_path}/{dest_name}/{d}!{os.path.split(file_path)[1]}')
        except FileNotFoundError:
            print(f'Error with folder: {d}')


if __name__ == '__main__':
    #get_compressed_along_folder('run_RL_out/train_greed/220708_diff_state',
                                #names_to_copy=('agent', 'testing_deterministic',
                                         #'output_by_step.png', 'integral_by_step.csv'),
                                #key=lambda s: s[0] == '_')
    collect_along_folder('run_RL_out/train_greed/220726_limited_conditions',
                         'testing_deterministic',
                         choose_folder_key=lambda s: s[0] == '_' and s[1].isdigit(),
                         choose_file_key=lambda s: s[0].isdigit() and '_all_data' in s,
                         take_key=lambda s: float(s[:s.find('c')]))
    collect_along_folder('run_RL_out/train_greed/220726_limited_conditions',
                         'testing_deterministic',
                         choose_folder_key=lambda s: s[0] == '_' and s[1].isdigit(),
                         choose_file_key=lambda s: s[0].isdigit() and '_in' in s,
                         take_key=lambda s: float(s[:s.find('c')]))
    collect_along_folder('run_RL_out/train_greed/220726_limited_conditions',
                         'testing_deterministic',
                         choose_folder_key=lambda s: s[0] == '_' and s[1].isdigit(),
                         choose_file_key=lambda s: s[0].isdigit() and '_out' in s,
                         take_key=lambda s: float(s[:s.find('c')]))
    #collect_along_folder('run_RL_out/train_greed/220708_diff_state',
                         #choose_folder_key=lambda s: s[0] == '_' and s[1].isdigit(),
                         #choose_file_key=lambda s: 'by_step' in s and '.csv' in s,
                         #dest_name='collected_integral_curves')
    pass
