import copy
import sys
import os
import itertools
from time import sleep

import numpy as np
import pandas as pd

# from lib import *
from usable_functions import make_subdir_return_path


def fill_dict(values: list, values_names: tuple, values_groups: tuple = ()):
    variable_dict = dict()
    # for name in (val_name[:val_name.find(':')]
    #              for val_name in values_names if ':' in val_name):
    #     variable_dict[name] = dict()
    for g in values_groups:
        variable_dict[g] = dict()
    for i, name in enumerate(values_names):
        find = False
        for subset_name in values_groups:
            if f'{subset_name}:' in name:
                find = True
                sub_name = name[name.find(':') + 1:]
                variable_dict[subset_name][sub_name] = values[i]
        # assert find, f'Error! Invalid name: {name}'
        if not find:
            variable_dict[name] = values[i]

    return variable_dict


def safe_merge_nested_dicts(dict1, dict2):
    # TODO check whether is the function really safe
    ret = dict()
    for dicts in (dict1, dict2), (dict2, dict1):
        for k, v in dicts[0].items():
            if k in dicts[1] and not isinstance(v, dict):
                raise ValueError
            elif (k in dicts[1]) and (k not in ret):
                ret[k] = safe_merge_nested_dicts(dicts[0][k], dicts[1][k])
            elif (k not in ret) and isinstance(v, (dict, list)):
                ret[k] = copy.copy(v)
            elif k not in ret:
                ret[k] = v
    return ret


def flatten_dict_with_only_simple_types(d):
    # given nested dict returns flatten dict with only simple types values
    ret = dict()
    for k0, v0 in d.items():
        if isinstance(v0, dict):
            sub_dict = flatten_dict_with_only_simple_types(v0)
            for k1, v1 in sub_dict.items():
                ret[f'{k0}::{k1}'] = v1
        elif isinstance(v0, (int, float, bool, str)):
            ret[k0] = v0
    return ret


def run_jobs_list(
        iteration_function,
        params_variants: list,
        names: tuple,
        names_groups: tuple,
        PC,
        out_fold_path: str,
        separate_folds: bool = True,
        repeat: int = 1,
        const_params: dict = None,
        sort_iterations_by: str = None,
        summarize_function: callable = None,
        unique_folder=False,
        python_interpreter='venv/bin/python',
        cluster_command_ops=None,
        at_same_time: int = 30):

    import argparse
    # import pickle
    import json
    assert len(names) == len(params_variants[0]), 'Error: lengths mismatch'

    # get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('iter', type=int, help='optimize conditions from conditions list')
    # FAILED ATTEMPT TO BUILD QUEUE
    # parser.add_argument('--start_pos', type=int, default=0, required=False,
    #                     help='starting position in the list of parameters\nDo not touch it (only internal usage)!')
    args = parser.parse_args()

    # expanding the list to repeat iterations
    if repeat > 1:
        assert isinstance(params_variants, list), f'params_variants should be list,' \
                                                  f'got {type(params_variants)} instead'
        new_list = []
        for values_set in params_variants:
            new_list += [values_set] * repeat
        params_variants = new_list

    if unique_folder and (args.start_pos == 0):
        # TODO multiple folder creation may occur
        out_fold_path = make_subdir_return_path(out_fold_path, with_date=True, unique=True)
    elif not os.path.exists(out_fold_path):
        os.makedirs(out_fold_path)

    iterations_number = len(params_variants)
    iter_arg = args.iter
    # start_pos = args.start_pos
    if iter_arg == -1:
        # -1 iteration launches all the iterations

        # for i in range(start_pos, min(iterations_number, start_pos + at_same_time)):
        assert iterations_number <= at_same_time, f'Maximum number of iterations is {at_same_time}, unable to do {iterations_number}'
        file_to_execute_path = sys.argv[0]
        for i in range(iterations_number):
            if cluster_command_ops and (cluster_command_ops is not None):
                if isinstance(cluster_command_ops, dict):
                    ops = cluster_command_ops
                elif cluster_command_ops is True:
                    ops = {'m': 3000, 'n': 1}
                else:
                    raise ValueError(f'Wrong argument value for cluster_command_ops: {cluster_command_ops}')
                os.system(f'run-cluster -m {ops["m"]} -n {ops["n"]} "{python_interpreter} {file_to_execute_path} {i}"')
            else:
                os.system(f'{python_interpreter} {file_to_execute_path} {i}')
        if cluster_command_ops and (cluster_command_ops is not None):
            os.system(f'run-cluster -m 2000 -n 1 "{python_interpreter} {file_to_execute_path} -2"')
        else:
            os.system(f'{python_interpreter} {file_to_execute_path} -2')

    elif iter_arg == -2:
        # -2 iteration waits for each iteration completion
        # and then collects results of the iterations

        # Cycle which run as long as not all the iterations in a chunk are completed
        # and not all the files are created. Cycle does nothing
        path_variants = [f'{out_fold_path}/result_#.json',
                         f'{out_fold_path}/result_#_error.json']
        file_not_found = True
        while file_not_found:
            file_not_found = False
            for i in range(iterations_number):
                existence_flags = map(lambda p: os.path.exists(p.replace('#', str(i))), path_variants)
                if not any(existence_flags):
                    # print(f'It is bad! Bad index: {i}')
                    file_not_found = True
        # delay after cycle exit
        sleep(10)

        # if start_pos + at_same_time < iterations_number:
        #     # queue realization
        #     # if not all the variants from params_set were tried
        #     # launch another chunk
        #
        #     # QUESTION: what will happen if -1 iteration calls -2 and then -2 iteration calls -1?
        #     # QUESTION: what will happen if -2 iteration was launched by run-cluster command?
        #     os.system(f'{python_interpreter} {sys.argv[0]} -1 --start_pos {start_pos + at_same_time}')
        # else:

        if summarize_function is not None:
            try:
                summarize_function(out_fold_path)
            except Exception as e:
                with open(f'{out_fold_path}/summary_failed.txt', 'w') as fwrite:
                    fwrite.write(str(e))

        # The results of different iterations should be collected
        df = pd.DataFrame(index=np.arange(iterations_number))
        for i in range(iterations_number):
            res_path = next(filter(os.path.exists, map(lambda p: p.replace('#', str(i)), path_variants)))
            with open(res_path, 'r') as jfile:
                d = json.load(jfile)
                for k, subd in d.items():
                    d[k] = flatten_dict_with_only_simple_types(subd)
                for k, v in {**d['variables'], **d['return']}.items():
                    assert isinstance(v, (int, float, str, bool))
                    df.loc[i, k] = v
            # os.remove(res_path)
        if (sort_iterations_by is not None) and (sort_iterations_by in df.columns):
            df.sort_values(by=sort_iterations_by, inplace=True, ascending=False)
        df.to_excel(f'{out_fold_path}/results.xlsx', index_label='iteration')

    elif iter_arg >= 0:
        # nonegative iteration corresponds to one variant of the parameters
        # in the list of parameters

        variable_dict = fill_dict(params_variants[iter_arg], names, names_groups)
        if const_params is None:
            const_params = dict()
        for name in names_groups:
            if name not in const_params:
                const_params[name] = dict()

        if separate_folds:
            iter_folder = make_subdir_return_path(out_fold_path, name=f'_{iter_arg}', with_date=False, unique=False)
        else:
            iter_folder = out_fold_path

        # iteration function should return dictionary
        ret = {'Error': None}
        error_indicator = '_error'
        try:
            ret = iteration_function(PC, safe_merge_nested_dicts(variable_dict, const_params),
                                     iter_folder, iter_arg)
            error_indicator = ''
        except Exception as e:
            ret['Error'] = str(type(e))
            raise e
        finally:
            with open(f'{out_fold_path}/result_{iter_arg}{error_indicator}.json', 'w') as handle:
                json.dump({'variables': variable_dict, 'return': ret}, handle)


def jobs_list_from_grid(*value_sets,
                        names: tuple):
    assert len(names) == len(value_sets), 'Error: lengths mismatch'
    params_variants = list(itertools.product(*value_sets))
    contains_tuple = False
    # if tuple names contains subtuple of names
    for it in names:
        if isinstance(it, tuple):
            contains_tuple = True
            break
    if contains_tuple:
        # realisation of grid not for the single parameter,
        # but for the sets of parameters,
        # i. e. creation grid of the form
        # [
        #  [a11, a12, a13..., b11, b12..., ...], [a11, a12, a13..., b21, b22..., ...], [a11, a12, a13..., b31, b32.., ...],
        #  [a21, a22, a23..., b11, b12..., ...], [a21, a22, a23..., b21, b22..., ...], [a21, a22, a23..., b31, b32.., ...],
        #  ]
        for i, _ in enumerate(params_variants):
            new_params_set = []
            for j, it in enumerate(names):
                if isinstance(it, tuple):
                    for k, _ in enumerate(it):
                        new_params_set.append(params_variants[i][j][k])
                else:
                    new_params_set.append(params_variants[i][j])
            params_variants[i] = new_params_set
        new_names = []
        for it in names:
            if isinstance(it, tuple):
                for name in it:
                    new_names.append(name)
            else:
                new_names.append(it)
        names = tuple(new_names)
    return {'params_variants': params_variants, 'names': names}


def O2_CO_from_CO_x(x):
    return 1.e+5 * (1 - x), 1.e+5 * x


def one_turn_search_iteration(PC, params: dict, foldpath, it_arg):

    x0, x1 = params['x0'], params['x1']

    PC.reset()
    PC.set_controlled(O2_CO_from_CO_x(x0))
    PC.time_forward(params['t0'])
    PC.set_controlled(O2_CO_from_CO_x(x1))
    PC.time_forward(params['t1'])

    PC.get_and_plot(f'{foldpath}/OTS_{it_arg}_x0({x0:.4g})_x1({x1:.4g}).png',
                    plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                 'plot_mode': 'separately', 'out_names': ['CO2_count']})
    R = PC.integrate_along_history(out_name='CO2_count')

    return {'Total_CO2_Count': R}


def get_for_SBP_iteration(episode_time, first_to_turn: str, ziff_model: bool = False,
                          out_name_to_observe='CO2_count', ):

    def switch_between_pure_iteration(PC, params: dict, foldpath, it_arg):
        O2_max, CO_max = params['O2_max'], params['CO_max']
        if ('t0' in params) and ('t1' in params):
            t0, t1 = params['t0'], params['t1']
        else:
            t0, t1 = params['first_part'] * params['total'], (1 - params['first_part']) * params['total']

        pressures_to_set = [(O2_max, 0.), (0, CO_max)]
        if first_to_turn == 'CO':
            pressures_to_set = pressures_to_set[::-1]
        if ziff_model:
            # O2, CO -> x_CO
            pressures_to_set = [(pair[1] / sum(pair), ) for pair in pressures_to_set]

        PC.reset()
        while PC.get_current_time() < episode_time:
            PC.set_controlled(pressures_to_set[0])
            PC.time_forward(t0)
            PC.set_controlled(pressures_to_set[1])
            PC.time_forward(t1)

        _, output_history = PC.get_and_plot(f'{foldpath}/SBP_{it_arg}_t0({t0:.4g})_t1({t1:.4g}).png',
                                            plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                                         'plot_mode': 'separately', 'out_names': out_name_to_observe})
        R = PC.integrate_along_history(out_name=out_name_to_observe)

        return {'CO2': R}

    return {'iteration_function': switch_between_pure_iteration}


def get_for_Ziff_iterations(pressure_unit: float, episode_time, CO2_output_column=3,
                            out_names_to_plot=('CO2_count',),
                            take_from_the_end=0.5):
    import lib
    import json

    def Ziff_iteration(PC, params: dict, foldpath, it_arg):
        if params.get('x', None) is not None:
            O2, CO = (1 - params['x']) * 10 * pressure_unit, params['x'] * 10 * pressure_unit
        else:
            O2, CO = params['O2'], params['CO']
        CO2_and_covs = [0] * 3
        PC.reset()
        PC.set_controlled(params)
        PC.time_forward(episode_time)
        PC.get_and_plot(f'{foldpath}/Ziff_O2({O2 / pressure_unit:.2f})_CO({CO / pressure_unit:.2f})_{it_arg}.png',
                        plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                     'plot_mode': 'separately', 'out_names': out_names_to_plot})
        CO2_and_covs[0] = PC.get_process_output()[1][:, CO2_output_column]  # should be CO2 output column
        CO2_and_covs[1] = PC.additional_graph['thetaO'][:CO2_and_covs[0].size]
        CO2_and_covs[2] = PC.additional_graph['thetaCO'][:CO2_and_covs[0].size]

        for j, v in enumerate(CO2_and_covs):
            CO2_and_covs[j] = np.mean(v[int(v.size * (1. - take_from_the_end)):])

        return {'CO2': CO2_and_covs[0],
                'thetaO': CO2_and_covs[1],
                'thetaCO': CO2_and_covs[2], }

    def Ziff_summarize(foldapth):
        x_arr, avgs = [], []
        for fname in filter(lambda name: name.endswith('.json'), sorted(os.listdir(foldapth))):
            with open(f'{foldapth}/{fname}', 'r') as fread:
                d = json.load(fread)
                ret_d = d['return']
                # complicated lists expansion due to possible duplicates of x values
                if d['variables'].get('x', None) is not None:
                    x_ax_value = 10 * d['variables']['x']
                else:
                    x_ax_value = d['variables']['CO'] / pressure_unit
                idx = next(filter(lambda i: abs(x_ax_value - x_arr[i]) < 1.e-5, range(len(x_arr))), None)
                if idx is None:
                    avgs.append([[ret_d['CO2'], ret_d['thetaO'], ret_d['thetaCO']]])
                    x_arr.append(x_ax_value)
                else:
                    avgs[idx].append([ret_d['CO2'], ret_d['thetaO'], ret_d['thetaCO']])

        for i, matr in enumerate(avgs):
            avgs[i] = np.mean(avgs[i], axis=0)
        x_arr, avgs = np.array(x_arr), np.array(avgs)
        idxs = np.argsort(x_arr)
        x_arr, avgs = x_arr[idxs], avgs[idxs]
        lib.plot_to_file(x_arr, avgs[:, 0], {'label': 'Average CO2 prod. rate', 'linestyle': 'solid',
                                             'marker': 'h', 'c': 'purple',
                                             'twin': True,
                                             },
                         x_arr, avgs[:, 1], {'label': 'Average O2 coverage', 'linestyle': (0, (1, 1)),
                                             'marker': 'x', 'c': 'blue'},
                         x_arr, avgs[:, 2], {'label': 'Average CO coverage', 'linestyle': (0, (5, 5)),
                                             'marker': '+', 'c': 'red'},
                         fileName=f'{foldapth}/Ziff_summarize_CO2.png',
                         xlabel=f'CO, {pressure_unit:.4g} Pa', ylabel='coverages', title='Summarize across benchmark',
                         twin_params={'ylabel': 'CO2 prod. rate'}, )

    return {'iteration_function': Ziff_iteration, 'summarize_function': Ziff_summarize}

