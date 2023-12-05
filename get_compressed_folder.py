import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_metric_stabilization_time(foldpath, quantile=0.99, col='smooth_1000_step'):
    assert os.path.exists(f'{foldpath}/integral_by_step.csv')
    df = pd.read_csv(f'{foldpath}/integral_by_step.csv', index_col=None, sep=';',)
    R_max = df[col].max()
    
    idx = np.where(df[col] > quantile * R_max)[0]
    assert len(idx)
    idx = idx[0]
    if 'smooth' in col:
        idx *= 20
    return idx


def get_time_along_many(foldpath, quantile=0.99, choose_subfold_key: callable = lambda s: True):
    
    df = pd.DataFrame(columns=['test-id', 'training_time', 'average_time', 'q1', 'q2', 'q3'])
    dirs = [name for name in os.listdir(foldpath) if choose_subfold_key(name)]
    for d in dirs:
        try:
            training_time = get_metric_stabilization_time(f'{foldpath}/{d}')
            test_id = int(d[1:])
            df.loc[test_id, ['test-id', 'training_time']] = [test_id, training_time]
        except (FileNotFoundError, AssertionError):
            print(f'Error with folder: {d}')
    df.loc[0, ['average_time', 'q1', 'q2', 'q3']] = [np.mean(df['training_time']),
                                 df['training_time'].quantile(0.25),
                                 df['training_time'].quantile(0.5),
                                 df['training_time'].quantile(0.75),]
    
    df.to_csv(f'{foldpath}/training_time.csv', index=False)
    
    fig, ax = plt.subplots()
    ax.hist(df['training_time'], bins=int(df['training_time'].max() / 1000))
    ax.set_title('training time histogram')
    ax.set_xlabel('training episodes before max')
    ax.set_ylabel('number of tests')
    fig.savefig(f'{foldpath}/training_time_hist.png')
    plt.close(fig)


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
    in_dir.sort()
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


def collect_training_results(folder, subfolder_path='testing_deterministic', dest_name='collected'):
    get_time_along_many(folder, quantile=0.98,
                        choose_subfold_key=lambda s: s[0] == '_' and s[1].isdigit(),)
    for name_part in (
            '_all_data',
            #'_in',
            '_input',
            #'_out',
            '_add',
            '_output',
            #'_target',
                      ):
        collect_along_folder(folder,
                             subfolder_path,
                             choose_folder_key=lambda s: s[0] == '_' and s[1].isdigit(),
                             choose_file_key=lambda s: s[0].isdigit() and name_part in s,
                             take_key=lambda s: float(s[:s.find('c')]),
                             dest_name=dest_name)


if __name__ == '__main__':
    # get_compressed_along_folder('run_RL_out/train_greed/220708_diff_state',
    #                             names_to_copy=('agent', 'testing_deterministic',
    #                                      'output_by_step.png', 'integral_by_step.csv'),
    #                             key=lambda s: s[0] == '_')

    folds = [
             #'220805_Libuda_more_CO_allowed',
             #'221103_Pd_CO2_sub_outs_I_01',
             #'221107_Pd_CO2_sub_outs_I_05',
             #'221223_turn_off_control',
             #'LibudaG/230707_vary_O2_CO_is_arbitrary',
             #'LibudaG/230713_CO(t)_RTP_shorter_period',
             #'LibudaG/230725_different_rates',
             #'LibudaG/230928_T1(440)_T2(560)_ignorant_agent',
             #'LibudaG/REFERENCE/231002_low_des_react_rates/',
             #'LibudaG/REFERENCE/230725_different_rates/',
             'LibudaG/231120_CORTP_NoSampling_dynrates/',
             ]
    for fold in folds:
        collect_training_results(f'run_RL_out/{fold}')

    # collect_along_folder('run_RL_out/train_greed/220708_diff_state',
    #                      choose_folder_key=lambda s: s[0] == '_' and s[1].isdigit(),
    #                      choose_file_key=lambda s: 'by_step' in s and '.csv' in s,
    #                      dest_name='collected_integral_curves')
    pass
