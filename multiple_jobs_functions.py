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
        unique_folder=False,
        python_interpreter='venv/bin/python',
        on_cluster=False,
        at_same_time: int = 100):

    import argparse
    # import pickle
    import json
    assert len(names) == len(params_variants[0]), 'Error: lengths mismatch'

    # expanding the list to repeat iterations
    if repeat > 1:
        assert isinstance(params_variants, list), f'params_variants should be list,' \
                                                  f'got {type(params_variants)} instead'
        new_list = []
        for values_set in params_variants:
            new_list += [values_set] * repeat
        params_variants = new_list

    if unique_folder:
        out_fold_path = make_subdir_return_path(out_fold_path, with_date=True, unique=True)
    elif not os.path.exists(out_fold_path):
        os.makedirs(out_fold_path)

    # get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('iter', type=int, help='optimize conditions from conditions list')
    args = parser.parse_args()

    iterations_number = len(params_variants)
    iter_arg = args.iter
    if iter_arg == -1:
        file_to_execute_path = sys.argv[0]
        assert iterations_number <= at_same_time, f'{iterations_number} jobs were passed, but the maximum is {at_same_time}'  # TODO TODO TODO
        for i in range(iterations_number):
            if on_cluster:
                os.system(f'run-cluster -m 3000 -n 1 "{python_interpreter} {file_to_execute_path} {i}"')
            else:
                os.system(f'{python_interpreter} {file_to_execute_path} {i}')
        if on_cluster:
            os.system(f'run-cluster -m 2000 -n 1 "{python_interpreter} {file_to_execute_path} -2"')
        else:
            os.system(f'{python_interpreter} {file_to_execute_path} -2')

    elif iter_arg == -2:
        # Cycle which run as long as not all the iterations are completed
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

        # The results of different iterations should be collected
        df = pd.DataFrame(index=np.arange(iterations_number))
        for i in range(iterations_number):
            res_path = next(filter(os.path.exists, map(lambda p: p.replace('#', str(i)), path_variants)))
            with open(res_path, 'r') as jfile:
                d = json.load(jfile)
                for k, v in {**d['variables'], **d['return']}.items():
                    assert isinstance(v, (int, float, str, bool))
                    df.loc[i, k] = v
        if (sort_iterations_by is not None) and (sort_iterations_by in df.columns):
            df.sort_values(by=sort_iterations_by, inplace=True, ascending=False)
        df.to_excel(f'{out_fold_path}/results.xlsx', index_label='iteration')

    elif iter_arg >= 0:
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
            raise Exception
        finally:
            with open(f'{out_fold_path}/result_{iter_arg}{error_indicator}.json', 'w') as handle:
                json.dump({'variables': variable_dict, 'return': ret}, handle)

        # for name in optimize_bounds:
        #     if optimize_bounds[name] == 'model_lims':
        #         warnings.warn('CRUTCH HERE!')
        #         prefix = name[:name.find('_')]
        #         dict_with_lims = None
        #         if f'{prefix}_top' in variable_dict['model']:
        #             dict_with_lims = variable_dict['model']
        #         elif f'{prefix}_top' in const_params:
        #             dict_with_lims = const_params['model']
        #         if f'{prefix}_bottom' in dict_with_lims:
        #             optimize_bounds[name] = [dict_with_lims[f'{prefix}_bottom'], dict_with_lims[f'{prefix}_top']]
        #         else:
        #             optimize_bounds[name] = [0., dict_with_lims[f'{prefix}_top']]

        # model_obj = PC_obj.process_to_control
        # model_obj.reset()
        # model_obj.assign_and_eval_values(**(variable_dict['model']), **(const_params['model']))

        # limit_names = [name for name in variable_dict['model'] if '_top' in name]
        # max_top = max([variable_dict['model'][name] for name in limit_names])
        # limit_names = [name for name in const_params['model'] if '_top' in name]
        # max_top = max([const_params['model'][name] for name in limit_names] + [max_top])
        # if max_top > 0:
        #     PC_obj.set_plot_params(input_lims=[-1.e-1 * max_top, 1.1 * max_top])
        # else:
        #     # PC_obj.set_plot_params(input_lims=None)
        #     raise NotImplementedError
        #
        # for d in [variable_dict, const_params]:
        #     for attr_name in ['target_func', 'long_term_target']:
        #         if attr_name in d:
        #             setattr(PC_obj, attr_name, d[attr_name])
        #             PC_obj.target_func_name = d['target_func_name']

        # to_func_to_optimize = dict()
        # for name in ('episode_len', 'time_step', 'to_plot', 'expand_description'):
        #     if name in variable_dict:
        #         to_func_to_optimize[name] = variable_dict[name]
        #     elif name in const_params:
        #         to_func_to_optimize[name] = const_params[name]
        #     else:
        #         raise RuntimeError

        # iter_optimize(func_to_optimize_policy(PC_obj, policy_type(dict()), **to_func_to_optimize),
        #               optimize_bounds=optimize_bounds,
        #               **(variable_dict['iter_optimize']), **(const_params['iter_optimize']),
        #               out_folder=make_subdir_return_path(out_path, name=f'_{iter_arg}', with_date=False, unique=False),
        #               unique_folder=False)


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


def one_turn_search_iteration(PC, params: dict, foldpath, it_arg):

    def O2_CO_from_CO_x(x):
        return 1.e+5 * (1 - x), 1.e+5 * x

    x0, x1 = params['x0'], params['x1']

    PC.reset()
    PC.set_controlled(O2_CO_from_CO_x(x0))
    PC.time_forward(params['t0'])
    PC.set_controlled(O2_CO_from_CO_x(x1))
    PC.time_forward(params['t1'])

    PC.get_and_plot(f'{foldpath}/OPS_{it_arg}_x0({x0:.4g})_x1({x1:.4g}).png',
                    plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                 'plot_mode': 'separately', 'out_names': ['CO2_count']})
    R = PC.integrate_along_history(out_name='CO2_count')

    return {'Total_CO2_Count': R}
