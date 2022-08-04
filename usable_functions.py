import \
    copy
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
                        sum_col += frame[add_col] # Andrei wrote '283', but, according to followed text and article, I did replace to 383
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


def process(model_obj, data, delta_t, prepair_time=0, DEBUG=True):
    if (not isinstance(data, np.ndarray)) or (len(data.shape) != 2):
        assert False, 'invalid data for process method'
    time_arr = data[:, -1]
    process_length = time_arr.size
    process_res_arr = np.empty(process_length)
    model_obj.t = -prepair_time + time_arr[0]
    while model_obj.t < time_arr[0]:
        model_obj.update(
            data[
            0,
            :-1],
            delta_t)
    end = time_arr[-1]
    i = 0
    while model_obj.t < end:
        controlled_values = np.empty(data.shape[1] - 1)
        for i1 in range(data.shape[1] - 1):
            controlled_values[i1] = np.interp(model_obj.t, time_arr, data[:, i1])
        model_out = model_obj.update(
            controlled_values,
            delta_t)
        if model_obj.t > time_arr[i]:
            process_res_arr[i] = model_out
            i += 1
    return process_res_arr


def wrap(s, maxLineLen):
    if len(s) <= maxLineLen:
        return s
    i_last = 0
    i = i_last + maxLineLen
    res = ''
    while i < len(s):
        i1 = s.rfind(' ', i_last, i)
        if i1 == -1:
            i1 = i
        res += '\n' + s[i_last:i1]
        i_last = i1
        i = i_last + maxLineLen
    res += '\n' + s[i_last:]
    return res


def create_func_for_optimize(model_obj, df, label_name):
    exp_data = model_obj.data_from_df_to_numpy(df)
    labels = np.array(df[label_name])

    def func(alphas, DEBUG=False, folder=None):
        process_data_arr = exp_data
        true_values_arr = labels

        model_obj.set_params(alphas)
        model_res = process(model_obj, data=process_data_arr, delta_t=0.1, prepair_time=1500, DEBUG=False)
        res = np.mean(np.abs(model_res - true_values_arr))
        if DEBUG:
            fig, ax = plt.subplots(1, figsize=(15, 8))
            # ax.title('%.2g', %alphas)
            title = f'model: {model_obj.model_name}\n'
            for name in model_obj.params:
                title += '%s=%.3g ' % (name, alphas[name])
            ax.set_title(wrap(title, 100))
            ax.plot(process_data_arr[:, -1].reshape(-1, 1), true_values_arr)
            ax.plot(process_data_arr[:, -1].reshape(-1, 1), model_res)
            fig.savefig(f'{folder}/{res}.png')
            plt.close(fig)

        return res

    return func


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
    import scipy.optimize as optimize

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
        return {param_names[i]: vector[i] for i in range(dim)}

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


# from pyfitit section
def stableMean(ar, throwCount=1):
    if throwCount <= 0:
        return np.mean(ar)
    if len(ar) <= 2 * throwCount:
        return np.median(ar)
    sort_ar = np.sort(ar)
    return np.mean(sort_ar[throwCount:-throwCount])


def expandByReflection(X, Y, side='both', reflType='odd', stableReflMeanCount=1):
    """
    :param Y:
    :param X:
    :param side: 'left', 'right', 'both'
    :param reflType: 'odd' or 'even'
    :param stableReflMeanCount: int, for odd reflection - reflect using mean of edge instead of one edge value
    """
    assert reflType in ['odd', 'even']
    assert side in ['left', 'right', 'both']
    e0, xanes0 = X, Y
    assert np.all(X[1:] - X[:-1] >= 0)
    rx = np.flip(xanes0)
    re = np.flip(e0)
    if side in ['left', 'both']:
        X = np.concatenate((e0[0] - (re[:-1] - re[-1]), X))
        if reflType == 'even':
            Y = np.concatenate((rx[:-1], Y))
        else:
            mean = stableMean(xanes0[:stableReflMeanCount])
            Y = np.concatenate((2 * mean - rx[:-1], Y))
    if side in ['right', 'both']:
        X = np.concatenate((X, e0[-1] - (re[1:] - re[0])))
        if reflType == 'even':
            Y = np.concatenate((Y, rx[1:]))
        else:
            mean = stableMean(rx[:stableReflMeanCount])
            Y = np.concatenate((Y, 2 * mean - rx[1:]))
    return X, Y


def kernelCauchy(x, a, sigma):
    return sigma/2/pi/((x-a)**2+sigma**2/4)


def kernelGauss(x, a, sigma):
    return 1/sigma/sqrt(2*pi)*np.exp(-(x-a)**2/2/sigma**2)


def simpleSmooth(X, Y, sigma, kernel='Cauchy', new_x=None, sigma2percent=0.1, gaussWeight=0.2, assumeZeroInGaps=False, expandParams=None):
    """
    Smoothing
    :param gaussWeight: 
    :param sigma2percent: 
    :param new_x: 
    :param kernel: 
    :param sigma: 
    :param Y: 
    :param X: 
    :param assumeZeroInGaps: whether to assume, that spectrum = 0 between points (i.e. adf type smoothing)
    :param expandParams: params of utils.expandByReflection except e, xanes
    """

    from lib import integral

    assert len(X.shape) == 1
    assert len(Y.shape) == 1
    if expandParams is None:
        expandParams = {}
    x0, y0 = X, Y
    X, Y = expandByReflection(X, Y, **expandParams)
    # plotting.plotToFile(X, Y, 'expand', x0, y0, 'init', fileName=f'debug.png')
    if new_x is None:
        new_x = x0
    new_y = np.zeros(new_x.shape)
    for i in range(new_x.size):
        if kernel == 'Cauchy':
            kern = kernelCauchy(X, new_x[i], sigma)
        elif kernel == 'Gauss':
            kern = kernelGauss(X, new_x[i], sigma)
        elif kernel == 'C+G':
            kern = kernelCauchy(X, new_x[i], sigma) + gaussWeight * kernelGauss(X, new_x[i], sigma * sigma2percent)
        else:
            assert False, 'Unknown kernel name'
        norm = 1 if assumeZeroInGaps else integral(X, kern)
        if norm == 0:
            norm = 1
        new_y[i] = integral(X, Y * kern) / norm
    return new_y

