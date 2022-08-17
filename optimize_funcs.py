import \
    copy
import itertools
import \
    os
from time import sleep

import numpy as np
import scipy.optimize as optimize

from usable_functions import make_subdir_return_path


def iter_optimize(func_for_optimize, optimize_bounds, try_num=10, method=None, optimize_options=None, debug_params=None,
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
    :param debug_params: если не None, тогда после каждой оптимизации будет вызвана func_for_optimize(min, **debug_params)
    :param out_folder:
    :return:
    """

    import itertools

    if unique_folder:
        out_folder = make_subdir_return_path(out_folder)
    if debug_params is not None:
        if 'folder' in debug_params:
            if debug_params['folder'] == 'auto':
                debug_params['folder'] = out_folder
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

    with open(f'{out_folder}/results.txt', 'w') as fout:
        points = np.array(list(itertools.product(*([[0., 1.]] * dim))))
        points[1], points[-1] = points[-1], points[1]
        perm = rng.permutation(len(points)-2)
        points[2:] = points[2 + perm, :]
        eps = 0.01
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
            if debug_params is not None:
                if 'ind_picture' in debug_params:
                    debug_params['ind_picture'] = try_ind
                func_for_optimize(convert_to_dict(res.x), **debug_params)


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
                          debug_params=copy.deepcopy(debug_params),
                          out_folder=method_out_folder,
                          unique_folder=False,
                          cut_left=cut_ends[0], cut_right=cut_ends[1])
        except Exception as e:
            print(e)
            print(f'{method_name} is working incorrectly!')
        # except NotImplementedError as e:  # DEBUG statement
        #     pass


def iter_optimize_cluster(func_for_optimize, optimize_bounds, try_num=10,
                          method=None, optimize_options=None, debug_params=None,
                          python_interpreter='venv/bin/python',
                          on_cluster=False,
                          out_path='.', unique_folder=True,
                          cut_left=True, cut_right=True,
                          file_to_execute_path='not_exists.py',
                          at_same_moment=50):
    """
    Запускает несколько раз scipy.minimize и сортирует результаты по убыванию качества

    :param on_cluster:
    :param python_interpreter:
    :param file_to_execute_path:
    :param at_same_moment:
    :param cut_right:
    :param cut_left:
    :param unique_folder:
    :param func_for_optimize: User defined function to optimize, receiving as an input dict{parameterName: value}
    :param optimize_bounds: bounds for the parameters dict{parameterName: [min,max]}
    :param try_num:
    :param method: optimization method name, argument to scipy minimize function, default - None
    :param optimize_options: argument to scipy minimize function
    :param debug_params: if is not None, then after each optimization try will be called func_for_optimize(min, **debug_params)
    :param out_path:
    :return:
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('iteration', type=int,
                        help='iteration of the optimization process',)
    args = parser.parse_args()

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if unique_folder:
        if args.iteration == -1:
            with open(f'{out_path}/helper.txt', 'w') as f:
                out_path = make_subdir_return_path(out_path)
                f.write(f'{out_path}\n')
        else:
            with open(f'{out_path}/helper.txt', 'r') as f:
                sleep(10)
                out_path = f.readline()[:-1]
    if debug_params is not None:
        if 'folder' in debug_params:
            if debug_params['folder'] == 'auto':
                debug_params['folder'] = out_path

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

    points = np.array(list(itertools.product(*([[0., 1.]] * dim))))
    points[1], points[-1] = points[-1], points[1]
    perm = rng.permutation(len(points)-2)
    points[2:] = points[2 + perm, :]
    eps = 0.01
    if cut_left:
        points[points == 0] += eps
    if cut_right:
        points[points == 1] -= eps

    if args.iteration == -1:
        try_ind = 0
        # DELAY = 20
        assert try_num <= at_same_moment, 'too many parallelization'
        # TODO improve the code allowing try_num > at_same_moment
        while try_ind < try_num:
            if on_cluster:
                os.system(f'run-cluster -n 1 -m 3000 "{python_interpreter} {file_to_execute_path} {try_ind}"')
            else:
                os.system(f'{python_interpreter} {file_to_execute_path} {try_ind}')
            # sleep(DELAY)
            try_ind += 1
        if on_cluster:
            os.system(f'run-cluster -n 1 -m 2000 "{python_interpreter} {file_to_execute_path} -2"')
        else:
            os.system(f'{python_interpreter} {file_to_execute_path} -2')
    elif args.iteration == -2:
        # Cycle which run so far so not all the iterations are completed
        # and not all the files are created. Cycle does nothing
        file_not_found = True
        while file_not_found:
            file_not_found = False
            for i in range(try_num):
                if not os.path.exists(f'{out_path}/result_{i}.txt'):
                    file_not_found = True
        # delay after cycle exit
        sleep(10)
        # summarize outputs of different iterations in one file
        # and remove individual file for each iteration
        with open(f'{out_path}/results.txt', 'w') as _:
            pass
        for i in range(try_num):
            with open(f'{out_path}/result_{i}.txt', 'r') as f:
                info = f.read()
            with open(f'{out_path}/results.txt', 'a') as f:
                f.write(info)
            os.remove(f'{out_path}/result_{i}.txt')
    else:
        try_ind = args.iteration
        # if try_ind == 2:  # DEBUG statement
        #     pass
        if (try_ind % 2 == 0) and (try_ind//2 < points.shape[0]):
            init_point = points[try_ind // 2] * range_arr + min_arr
        else:
            init_point = rng.random(dim) * range_arr + min_arr
        res = optimize.minimize(func_for_optimize1, init_point,
                                method=method, options=optimize_options, bounds=optimize_bounds)
        s = f'min={res.fun:.4f} params: {convert_to_dict(res.x)}' \
            f'\nsuccess: {res.success}\nstatus: {res.status}\nmessage: {res.message}\n'
        s = '--Optimize iteration results--\n' + s
        with open(f'{out_path}/result_{try_ind}.txt', 'w') as fout:
            fout.write(s)
            fout.flush()
        if debug_params is not None:
            debug_params['ind_picture'] = str(try_ind)
            func_for_optimize(convert_to_dict(res.x), **debug_params)

        # with open(f'{out_folder}/check.txt', 'a') as f:
        #     f.write(f'yes{try_ind}')
