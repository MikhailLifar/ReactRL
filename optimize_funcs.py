import copy
import itertools
import os
import warnings

import numpy as np
import scipy.optimize as optimize

from usable_functions import make_subdir_return_path
from ProcessController import ProcessController, func_to_optimize_policy


def iter_optimize(func_for_optimize, optimize_bounds, try_num=10, method=None, optimize_options=None, call_after_opt_params=None,
                  out_folder='.', unique_folder=True,
                  cut_left=True, cut_right=True):
    """
    Запускает несколько раз scipy.minimize и сортирует результаты по убыванию качества

    :param cut_right:
    :param cut_left:
    :param unique_folder:
    :param func_for_optimize: Функция пользователя, принимающая на вход dict{имяПараметра: значение}
    :param optimize_bounds: границы изменения параметров dict{имяПараметра: [min,max]}
    :param try_num:
    :param method: method в функции minimize (None - использовать по умолчанию)
    :param optimize_options: опции minimize
    :param call_after_opt_params: если не None, тогда после каждой оптимизации будет вызвана func_for_optimize(min, **call_after_opt_params)
    :param out_folder:
    :return:
    """

    import itertools

    if unique_folder:
        out_folder = make_subdir_return_path(out_folder)
    if call_after_opt_params is not None:
        if 'folder' in call_after_opt_params:
            if call_after_opt_params['folder'] == 'auto':
                call_after_opt_params['folder'] = out_folder
    rng = np.random.default_rng(0)
    dim = len(optimize_bounds)
    param_names = sorted(list(optimize_bounds.keys()))
    optimize_bounds = [optimize_bounds[pn] for pn in param_names]
    max_arr = np.zeros(dim)
    min_arr = np.zeros(dim)
    for i in range(dim):
        min_arr[i] = optimize_bounds[i][0]
        max_arr[i] = optimize_bounds[i][1]
    range_arr = max_arr - min_arr

    def convert_to_dict(vector):
        return {el: vector[i] for i, el in enumerate(param_names)}

    def func_for_optimize1(vector):
        return func_for_optimize(convert_to_dict(vector))

    min_fun = np.inf
    with open(f'{out_folder}/results.txt', 'w') as fout:
        points = np.array(list(itertools.product(*([[0., 1.]] * dim))))
        points[1], points[-1] = points[-1], points[1]
        perm = rng.permutation(len(points)-2)
        points[2:] = points[2 + perm, :]
        eps = 0.01
        success_rate = 0
        if cut_left:
            points[points == 0] += eps
        if cut_right:
            points[points == 1] -= eps
        for try_ind in range(try_num):
            if try_ind == 2:
                pass
            if (try_ind % 2 == 0) and (try_ind//2 < points.shape[0]):
                init_point = points[try_ind // 2] * range_arr + min_arr
            else:
                init_point = rng.random(dim) * range_arr + min_arr
            res = optimize.minimize(func_for_optimize1, init_point,
                                    method=method, options=optimize_options, bounds=optimize_bounds)
            s = f'min={res.fun:.4f} params: {convert_to_dict(res.x)}' \
                f'\nsuccess: {res.success}\nstatus: {res.status}\nmessage: {res.message}\n'
            s = '--Optimize iteration results--\n' + s
            print(s)
            fout.write(s)
            fout.flush()
            if call_after_opt_params is not None:
                if 'ind_picture' in call_after_opt_params:
                    call_after_opt_params['ind_picture'] = try_ind
                func_for_optimize(convert_to_dict(res.x), **call_after_opt_params)

            success_rate += res.success * 1. / try_num
            min_fun = min(res.fun, min_fun)

    return min_fun, success_rate


def optimize_different_methods(func_for_optimize, optimize_bounds,
                               methods: list = None,
                               try_num=10, optimize_options=None,
                               debug_params=None, out_folder='.',
                               cut_ends=(False, False)):
    scipy_implemented_meths = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                               'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA',
                               'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg',
                               'trust-exact', 'trust-krylov']
    common_folder = make_subdir_return_path(out_folder)
    if methods is None:
        methods = scipy_implemented_meths
    if try_num * len(methods) > 300:
        print('WARNING! It will be long optimization here!')
    for method_name in methods:
        method_out_folder = common_folder + f'/{method_name}'
        os.makedirs(method_out_folder, exist_ok=False)
        try:
            iter_optimize(func_for_optimize, optimize_bounds, try_num=try_num,
                          method=method_name, optimize_options=optimize_options,
                          call_after_opt_params=copy.deepcopy(debug_params),
                          out_folder=method_out_folder,
                          unique_folder=False,
                          cut_left=cut_ends[0], cut_right=cut_ends[1])
        except Exception as e:
            print(e)
            print(f'{method_name} is working incorrectly!')
        # except NotImplementedError as e:  # DEBUG statement
        #     pass


def get_for_repeated_opt_iterations(func_for_optimize, optimize_bounds, constrains=lambda d: None,
                                    cut_left=True, cut_right=True,
                                    method=None, try_num=30,
                                    optimize_options=None, call_after_opt_params=None):
    """

    :param func_for_optimize:
    :param optimize_bounds:
    :param constrains: constrains(d: dict). Returns None, but changes dict
    :param cut_left:
    :param cut_right:
    :param method:
    :param try_num:
    :param optimize_options:
    :param call_after_opt_params:
    :return:
    """

    rng = np.random.default_rng(0)
    dim = len(optimize_bounds)
    param_names = sorted(list(optimize_bounds.keys()))
    optimize_bounds = [optimize_bounds[pn] for pn in param_names]
    max_arr = np.zeros(dim)
    min_arr = np.zeros(dim)
    for i in range(dim):
        min_arr[i] = optimize_bounds[i][0]
        max_arr[i] = optimize_bounds[i][1]
    range_arr = max_arr - min_arr

    def convert_to_dict(vector):
        ret = {el: vector[j] for j, el in enumerate(param_names)}
        constrains(ret)
        return ret

    def func_for_optimize1(vector):
        return func_for_optimize(convert_to_dict(vector))

    points = np.array(list(itertools.product(*([[0., 1.]] * dim))))
    points[1], points[-1] = points[-1], points[1]
    perm = rng.permutation(len(points)-2)
    points[2:] = points[2 + perm, :]
    eps = 0.01
    if cut_left:
        points[points == 0] += eps
    if cut_right:
        points[points == 1] -= eps

    def repeated_optimization_iteration(PC, params: dict, foldpath, it_arg):
        try_ind = it_arg
        if (try_ind % 2 == 0) and (try_ind//2 < points.shape[0]):
            init_point = points[try_ind // 2] * range_arr + min_arr
        else:
            init_point = rng.random(dim) * range_arr + min_arr
        res = optimize.minimize(func_for_optimize1, init_point,
                                method=method, options=optimize_options, bounds=optimize_bounds)
        s = f'iteration: {it_arg}\n' \
            f'min={res.fun:.4f} params: {convert_to_dict(res.x)}' \
            f'\nsuccess: {res.success}\nstatus: {res.status}\nmessage: {res.message}\n'

        with open(f'{foldpath}/_optim_result_{try_ind}.txt', 'w') as fout:
            fout.write(s)
            fout.flush()
        if call_after_opt_params is not None:
            local_params = dict()
            local_params['ind_picture'] = str(try_ind)
            if ('folder' in call_after_opt_params) and (call_after_opt_params['folder'] == 'auto'):
                local_params['folder'] = foldpath
            func_for_optimize(convert_to_dict(res.x), **{**call_after_opt_params, **local_params})

        return {'fvalue': res.fun, 'value_at': convert_to_dict(res.x), 'success': res.success}

    def repeated_optimize_summarize(foldpath):
        with open(f'{foldpath}/optim_results.txt', 'w') as fout:
            for fname in filter(lambda name: name.startswith('_optim') and name.endswith('.txt'), os.listdir(foldpath)):
                with open(f'{foldpath}/{fname}', 'r') as f:
                    info = f.read()
                os.remove(f'{foldpath}/{fname}')
                fout.write(info)
                fout.write('\n')

    return {'iteration_function': repeated_optimization_iteration,
            'summarize_function': repeated_optimize_summarize,
            'params_variants': [(0, )] * try_num, 'names': ('___', ),  'names_groups': (),  # TODO crutch here (many fictive parameters)
            }


def get_for_param_opt_iterations(func_to_optimize, optimize_bounds):

    def parametric_optimization_iteration(PC, params: dict, foldpath, it_arg):
        for name in optimize_bounds:
            if optimize_bounds[name] == 'model_lims':
                warnings.warn('CRUTCH HERE!')
                prefix = name[:name.find('_')]
                dict_with_lims = None
                if f'{prefix}_top' in params['model']:
                    dict_with_lims = params['model']
                if f'{prefix}_bottom' in dict_with_lims:
                    optimize_bounds[name] = [dict_with_lims[f'{prefix}_bottom'], dict_with_lims[f'{prefix}_top']]
                else:
                    optimize_bounds[name] = [0., dict_with_lims[f'{prefix}_top']]

        model_obj = PC.process_to_control
        model_obj.reset()
        model_obj.assign_and_eval_values(**params['model'])

        for d in params:
            for attr_name in ['target_func', 'long_term_target']:
                if attr_name in d:
                    setattr(PC, attr_name, d[attr_name])
                    PC.target_func_name = d['target_func_name']

        min_fun, success_rate = iter_optimize(func_to_optimize(**(params['to_func_to_optimize'])),
                                optimize_bounds=optimize_bounds,
                                **(params['to_iter_optimize']),
                                out_folder=foldpath,
                                unique_folder=False)

        return {'min_fun': min_fun, 'success_rate': success_rate}

    return {'iteration_function': parametric_optimization_iteration,
            'names_groups': ('model', 'to_func_to_optimize', 'to_iter_optimize'),
            'separate_folds': True}
