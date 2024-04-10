import copy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

matplotlib.rcParams.update(
    {'mathtext.default': 'regular'}
)


"""
    ====== JUPYTER =====
"""


def isJupyterNotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


"""
    ====== Plotting =====
"""


def createfig(interactive=False, figsize=None, figdpi=None, **kwargs):
    if not interactive:
        plt.ioff()
    else:
        if isJupyterNotebook():
            plt.ion()
    if interactive:
        if figsize is None:
            figsize = (16 * 0.5, 9 * 0.5)
        if figdpi is None:
            figdpi = 100
    else:
        if figsize is None:
            figsize = (16 / 3 * 2, 9 / 3 * 2)
        if figdpi is None:
            figdpi = 300
    fig, ax = plt.subplots(figsize=figsize, dpi=figdpi, **kwargs)
    return fig, ax


def savefig(fileName, fig, figdpi=None):
    if figdpi is None:
        figdpi = 300
    folder = os.path.split(fileName)[0]
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    fig.savefig(fileName, dpi=figdpi)


def closefig(fig):
    if matplotlib.get_backend() == 'nbAgg':
        return
    # if not utils.isJupyterNotebook(): plt.close(fig)  - notebooks also have limit - 20 figures
    if fig.number in plt.get_fignums():
        if isJupyterNotebook():
            plt.show(block=False)  # Even if plt.isinteractive() == True jupyter notebook doesn't show graph if in past plt.ioff/ion was called
        plt.close(fig)
    else:
        print('Warning: can\'t close not existent figure')
    if isJupyterNotebook():
        plt.ion()


def wrap(s, n):
    if len(s) <= n:
        return s
    words = s.split(' ')
    s = ''
    line = ''
    for w in words:
        if line != '':
            line += ' '
        line += w
        if len(line) > 80:
            if s != '':
                s += "\n"
            s += line
            line = ''
    if line != '':
        if s != '':
            s += "\n"
        s += line
    return s


def getPlotLim(z, gap=0.1):
    m = np.min(z)
    M = np.max(z)
    d = M-m
    return [m-d*gap, M+d*gap]


def updateYLim(ax):
    xlim = ax.get_xlim()
    ybounds = []
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        ybounds.append(getPlotLim(y[(xlim[0] <= x) & (x <= xlim[1])]))
    ybounds = np.array(ybounds)
    ylim = [np.min(ybounds[:, 0]), np.max(ybounds[:, 1])]
    ax.set_ylim(ylim)


def save_to_file(*p,  fileName=None,):
    assert len(p) % 3 == 0, f'Number of parameters {len(p)} is not multiple of 3'
    n = len(p)//3
    labels = {}

    for i in range(n):
        if isinstance(p[i*3+2], str):
            labels[i] = p[i*3+2]
        else:
            params = p[i*3+2]
            assert isinstance(params, dict)
            if 'label' in params:
                labels[i] = params['label']

    if fileName is None:
        fileName = 'data.csv'
    folder = os.path.split(os.path.expanduser(fileName))[0]
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    def save(file, obj):
        if not isinstance(obj, np.ndarray):
            obj = np.array(obj)
        obj = obj.reshape(1, -1)
        np.savetxt(file, obj, delimiter=',')
    with open(fileName, 'w') as f:
        for i in range(n):
            label = labels[i] if i in labels else str(i)
            f.write(label+' x: ')
            save(f, p[i*3])
            f.write(label+' y: ')
            save(f, p[i * 3 + 1])


def plot_in_axis(*p, ax, title='', xlabel='', ylabel='',
                 xlim=None, ylim=None,
                 twin_params: dict = None,
                 plotMoreFunction=None, ploMoreTwinFunction=None, yscale=None, grid=False):
    """
    Simple plot multiple graphs to file
    :param ax:
    :param twin_params:
    :param grid:
    :param yscale:
    :param p: sequence of triads x1,y1,label1,x2,y2,label2,....
    :param title:
    :param xlabel:
    :param ylabel:
    :param xlim:
    :param ylim:
    :param plotMoreFunction: function(ax)
    :param ploMoreTwinFunction:
    :return:
    """

    assert len(p) % 3 == 0, f'Number of parameters {len(p)} is not multiple of 3'

    right_ax = None
    if twin_params is not None:
        right_ax = ax.twinx()
    else:
        twin_params = {}

    n = len(p)//3
    labels = {}
    if yscale is not None:
        ax.set_yscale(yscale)
    for i in range(n):
        if isinstance(p[i*3+2], str):
            labels[i] = p[i*3+2]
            ax.plot(p[i*3], p[i*3+1], label=labels[i])
        else:
            params = p[i*3+2]
            assert isinstance(params, dict)
            params = copy.deepcopy(params)
            ax_to_plot = ax
            if 'twin' in params:
                del params['twin']
                ax_to_plot = right_ax
            if 'format' in params:
                fmt = params['format']
                del params['format']
                ax_to_plot.plot(p[i * 3], p[i * 3 + 1], fmt, **params)
            else:
                ax_to_plot.plot(p[i * 3], p[i * 3 + 1], **params)
            if 'label' in params:
                labels[i] = params['label']
    if title != '':
        title = wrap(title, 100)
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        if ylim is None:
            updateYLim(ax)
        # if ylim is None:
        #     # pyplot doesn't change ylim after xlim change
        #     ybounds = np.array([getPlotLim(p[i * 3 + 1][(xlim[0]<=p[i * 3]) & (p[i * 3]<=xlim[1])]) for i in range(n)])
        #     ylim = [np.min(ybounds[:,0]), np.max(ybounds[:,1])]
    if ylim is not None:
        if ylim[1] is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            ax.set_ylim(bottom=ylim[0])
    if twin_params.get('ylim', None) is not None:
        y0, y1 = twin_params['ylim']
        if y1 is not None:
            right_ax.set_ylim(y0, y1)
        else:
            right_ax.set_ylim(y0)

    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)
    if 'ylabel' in twin_params:
        right_ax.set_ylabel(twin_params['ylabel'])
    if plotMoreFunction is not None:
        plotMoreFunction(ax)
    if ploMoreTwinFunction is not None:
        ploMoreTwinFunction(right_ax)

    if any(labels[op] and labels[op] is not None for op in labels):
        ax.legend(loc=2, prop={'size': 12})
        if right_ax is not None:
            right_ax.legend(loc=1, prop={'size': 12})

    if grid:
        ax.grid()

    return ax, right_ax, labels  # TODO crutch with labels return


def plot_to_file(*p, fileName=None, save_csv=True, tight_layout=True,
                 figsize=None,
                 **plot_in_ax_ops):
    """

    :param p:
    :param fileName:
    :param save_csv:
    :param tight_layout:
    :param plot_in_ax_ops:
    :return:
    """
    fig, ax = createfig(figsize=figsize)

    _, _, labels = plot_in_axis(*p, ax=ax, **plot_in_ax_ops)

    if fileName is None:
        fileName = 'graph.png'
    folder = os.path.split(os.path.expanduser(fileName))[0]
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    savefig(fileName, fig)
    closefig(fig)
    # plt.clf()
    # plt.close(fig)

    n = len(p)//3
    if save_csv:
        def save(file, obj):
            if not isinstance(obj, np.ndarray):
                obj = np.array(obj)
            obj = obj.reshape(1, -1)
            np.savetxt(file, obj, delimiter=',')
        with open(os.path.splitext(fileName)[0]+'.csv', 'w') as f:
            for i in range(n):
                label = labels[i] if i in labels else str(i)
                f.write(label+' x: ')
                save(f, p[i*3])
                f.write(label+' y: ')
                save(f, p[i * 3 + 1])


def read_plottof_csv(datapath, ret_ops=False, ret_df=False, create_standard: bool = False):
    plot_ops = []
    df = pd.read_csv(datapath, header=None)
    df = df.T
    cols = []
    for i, elem in enumerate(df.loc[0, :]):
        cols.append(elem[:elem.find(':')])
        df.loc[0, i] = elem[elem.rfind(' ') + 1:]
    df.columns = cols
    df = df.astype('float64')

    if ret_ops:
        for i, _ in enumerate(cols[::2]):
            plot_ops += [df[cols[2 * i]].to_numpy(), df[cols[2 * i+1]].to_numpy(), cols[2 * i][:-2]]

    if create_standard:
        dir_part, file_part = os.path.split(datapath)
        f_name, ext = os.path.splitext(file_part)
        df.to_csv(f'{dir_part}/{f_name}_new{ext}',
                  sep=';', index=False)

    if not ret_df:
        df = None

    return plot_ops, df


def read_control_from_plottof(plottoffile, control_names, t_start=None, t_end=None,):
    options, _ = read_plottof_csv(plottoffile, ret_ops=True)
    opIndex = dict()
    for name in control_names:
        opIndex[name] = options[2::3].index(name) * 3 + 2

    times = options[opIndex[control_names[0]] - 2]
    contorlSeqs = dict()
    for name in control_names:
        contorlSeqs[name] = options[opIndex[name] - 1]

    # limit times
    if t_start is None:
        t_start = times[0]
    if t_end is None:
        t_end = times[-1]
    idx = (times >= t_start) & (times <= t_end)
    times = times[idx] - t_start
    for name in control_names:
        contorlSeqs[name] = contorlSeqs[name][idx]

    return times, contorlSeqs


def plot_from_file(*lbls_fmts, csvFileName: str,
                   chose_labels: list = None,
                   transforms=None, x_transform=None,
                   **plottof_ops):
    ops, _ = read_plottof_csv(csvFileName, True, False, False)

    # filter chosen ops
    if chose_labels is not None:
        new_ops = []
        for i, l in enumerate(ops[2::3]):
            if isinstance(l, str) and (l in chose_labels):
                new_ops += ops[3*i:3*(i+1)]
            elif isinstance(l, dict) and (l['label'] in chose_labels):
                new_ops += ops[3*i:3*(i+1)]
        ops = new_ops

    if x_transform is not None:
        for i in range(len(ops) // 3):
            ops[3 * i] = x_transform(ops[3 * i])
    # transform y values
    if transforms is not None:
        for trans_dict in transforms:
            label_idx = ops.index(trans_dict['old_label'])
            ops[label_idx - 1] = trans_dict['transform'](ops[label_idx - 1])
            if trans_dict.get('new_label', ''):
                ops[label_idx] = trans_dict['new_label']
            else:
                ops[label_idx] = trans_dict['old_label']

    # replace some labels with format dicts
    for i, l in enumerate(ops[2::3]):
        if l not in lbls_fmts:
            for lf in lbls_fmts:
                if isinstance(lf, dict) and (l == lf['label']):
                    ops[3 * i + 2] = lf

    if 'fileName' not in plottof_ops:
        plottof_ops['fileName'] = os.path.splitext(csvFileName)[0] + '.png'
    plot_to_file(*ops, **plottof_ops)


def plot_show_save_map(data, filepath, xticks=None, yticks=None, show: bool = False, save_data=True,
                       xlabel='?', ylabel='?', cbounds=None, **kwargs):

    if cbounds is None:
        cbounds = [None, None]

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', None))

    ybounds = xbounds = None
    if isinstance(xticks, dict):
        xbounds = [xticks['min'], xticks['max']]
        xticks = None
    if isinstance(yticks, dict):
        ybounds = [yticks['min'], yticks['max']]
        yticks = None

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data=data,
                            index=yticks,
                            columns=xticks,
                            )
    fname, ext = os.path.splitext(filepath)
    if save_data:
        data.to_csv(f'{fname}.csv', index=False)

    sns.heatmap(data, ax=ax, vmin=cbounds[0], vmax=cbounds[1],
                cbar_kws={'label': kwargs.get('color_ax_label', None)},
                **(kwargs['map_kwargs']))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    if xbounds is not None:
        xmin, xmax = xbounds
        ax.xaxis.set_major_locator(ticker.MultipleLocator(data.shape[1] / 10))
        ax.xaxis.set_major_formatter(lambda x, pos: f'{x / data.shape[1] * (xmax - xmin) + xmin:.2f}')
    if ybounds is not None:
        ymin, ymax = ybounds
        ax.yaxis.set_major_locator(ticker.MultipleLocator(data.shape[0] / 10))
        ax.yaxis.set_major_formatter(lambda y, pos: f'{y / data.shape[0] * (ymax - ymin) + ymin:.2f}')

    if show:
        plt.show()
    fig.savefig(filepath, dpi=400, bbox_inches='tight')
    plt.close(fig)



"""
    ====== PYTHON ITERABLES =====
"""





"""
    ====== ARRAYS AND SERIES =====
"""


def integral(x, y):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    assert len(x.shape) == 1
    dx = x[1:] - x[:-1]
    if len(y.shape) == 1:
        my = (y[1:]+y[:-1])/2
    else:
        assert len(y.shape) == 2
        assert y.shape[1] == len(x)
        # one spectrum - one row
        my = (y[:, 1:] + y[:, :-1]) / 2
    return np.dot(my, dx)


def extend_arr_ax0(x: np.ndarray, target_capacity: int, fill='NO'):
    # early return if no need to extend
    if target_capacity < x.shape[0]:
        return x

    # default new space filling
    if fill == 'NO':
        fill_new = lambda arr: np.empty_like(arr)
    else:
        fill_new = lambda arr: np.full_like(arr, fill)

    if len(x.shape) == 1:
        while target_capacity >= x.shape[0]:
            x = np.hstack((x, np.tile(fill_new(x), 9)))
    elif len(x.shape) == 2:
        while target_capacity >= x.shape[0]:
            x = np.vstack((x, np.tile(fill_new(x), (9, 1))))
    else:
        raise ValueError('Implemented only for 1d, 2d arrays')

    return x


"""
    ====== FROM PYFITIT =====
"""


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
    return sigma/2/np.pi/((x-a)**2+sigma**2/4)


def kernelGauss(x, a, sigma):
    return 1/sigma/np.sqrt(2*np.pi)*np.exp(-(x-a)**2/2/sigma**2)


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

