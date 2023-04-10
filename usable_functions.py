# import \
#     copy
import \
    os

import datetime


from math import pi, sqrt
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main_graph(frame, x_col_s, y_cols, name_out_file_s, num_files=1,
               minor_loc: tuple = (100, 1e-5),
               major_loc: tuple = (1000, 5e-5)
               ):
    # assert isinstance(minor_loc, tuple) and isinstance(major_loc, tuple), 'minor_loc and major_loc should be tuples'
    # assert len(y_cols) < 7, 'Not more than 6 graphs'
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    if num_files > 1:
        num_rows = frame.shape[0]
        step = num_rows // num_files + 1
        name_out_file_s = name_out_file_s.split('.')
        for i in range(num_files):
            main_graph(frame.loc[i * step:min((i+1) * step, num_rows)],
                       x_col_s, y_cols,
                       name_out_file_s=name_out_file_s[0] + '_' + str(i) + '.' + name_out_file_s[1],
                       minor_loc=minor_loc, major_loc=major_loc)
    else:
        fix, ax = plt.subplots(1, figsize=(15, 8))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_loc[0]))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_loc[0]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(major_loc[1]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_loc[1]))
        for i, col in enumerate(y_cols):
            # print(frame[col].dtypes)
            if '+' in col:
                add_cols = col.split('+')
                sum_col = np.zeros(frame.shape[0])
                for add_col in add_cols:
                    if add_col == '':
                        continue
                    if '*' in add_col:
                        pair = add_col.split('*')
                        k = float(pair[0])
                        add_col = pair[1]
                        sum_col += frame[add_col] * k
                    else:
                        sum_col += frame[add_col]  # Andrei wrote '283', but, according to followed text and article, I did replace to 383
                if isinstance(x_col_s, str):
                    ax.plot(frame[x_col_s], sum_col, label=col)
                else:
                    ax.plot(frame[x_col_s[i]], sum_col, label=col)
            else:
                if isinstance(x_col_s, str):
                    ax.plot(frame[x_col_s], frame[col], label=col)
                else:
                    ax.plot(frame[x_col_s[i]], frame[col], label=col)
        ax.legend()
        # plt.show()
        plt.savefig(name_out_file_s)


def get_float_from_invalid(frame, cols):
    for col in cols:
        for i, elem in enumerate(frame[col]):
            if isinstance(elem, str) and (',' in elem):
                elem = elem.replace(',', '.')
                frame.loc[i, col] = np.float64(elem)
        frame[col] = frame[col].astype('float64')


# OUTDATED
# def process(model_obj, data, delta_t, prepair_time=0, DEBUG=True):
#     if (not isinstance(data, np.ndarray)) or (len(data.shape) != 2):
#         assert False, 'invalid data for process method'
#     time_arr = data[:, -1]
#     process_length = time_arr.size
#     process_res_arr = np.empty(process_length)
#     model_obj.t = -prepair_time + time_arr[0]
#     while model_obj.t < time_arr[0]:
#         model_obj.update(
#             data[
#             0,
#             :-1],
#             delta_t)
#     end = time_arr[-1]
#     i = 0
#     while model_obj.t < end:
#         controlled_values = np.empty(data.shape[1] - 1)
#         for i1 in range(data.shape[1] - 1):
#             controlled_values[i1] = np.interp(model_obj.t, time_arr, data[:, i1])
#         model_out = model_obj.update(
#             controlled_values,
#             delta_t)
#         if model_obj.t > time_arr[i]:
#             process_res_arr[i] = model_out
#             i += 1
#     return process_res_arr


# OUTDATED
# def create_func_to_approximate(model_obj, df, label_name):
#     exp_data = model_obj.data_from_df_to_numpy(df)
#     labels = np.array(df[label_name])
#
#     def func(alphas, DEBUG=False, folder=None):
#         process_data_arr = exp_data
#         true_values_arr = labels
#
#         model_obj.set_params(alphas)
#         model_res = process(model_obj, data=process_data_arr, delta_t=0.1, prepair_time=1500, DEBUG=False)
#         res = np.mean(np.abs(model_res - true_values_arr))
#         if DEBUG:
#             fig, ax = plt.subplots(1, figsize=(15, 8))
#             # ax.title('%.2g', %alphas)
#             title = f'model: {model_obj.model_name}\n'
#             for name in model_obj.params:
#                 title += '%s=%.3g ' % (name, alphas[name])
#             ax.set_title(wrap(title, 100))
#             ax.plot(process_data_arr[:, -1].reshape(-1, 1), true_values_arr)
#             ax.plot(process_data_arr[:, -1].reshape(-1, 1), model_res)
#             fig.savefig(f'{folder}/{res}.png')
#             plt.close(fig)
#
#         return res
#
#     return func


def make_subdir_return_path(origin_path,
                            prefix='', postfix='', name='',
                            unique=True, with_date=True):
    assert (name == '') or not with_date, 'You cannot assign name parameter when with_date = True'
    if with_date:
        current_time = datetime.datetime.now()
        subdir_name = f'{current_time.year}_{current_time.month}_{current_time.day}__'
    else:
        subdir_name = name
    subdir_name = prefix + subdir_name
    if unique:
        ind = 0
        while os.path.exists(f'{origin_path}/{subdir_name}{ind}{postfix}/'):
            ind += 1
        subdir_name = f'{subdir_name}{ind}{postfix}'
    os.makedirs(f'{origin_path}/{subdir_name}/', exist_ok=False)
    return f'{origin_path}/{subdir_name}'


def make_unique_filename(filepath):
    root, ext = os.path.splitext(filepath)
    i = 0
    while os.path.exists(f'{root}_{i}{ext}'):
        i += 1
    return f'{root}_{i}{ext}'

