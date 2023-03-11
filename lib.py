import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {'mathtext.default': 'regular'}
)


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


def plot_to_file(*p, fileName=None, save_csv=True,
                 title='', xlabel='', ylabel='',
                 xlim=None, ylim=None,
                 twin_params: dict = None,
                 plotMoreFunction=None, yscale=None, tight_layout=True, grid=False):
    """
    Simple plot multiple graphs to file
    :param twin_params:
    :param grid:
    :param yscale:
    :param tight_layout:
    :param p: sequence of triads x1,y1,label1,x2,y2,label2,....
    :param fileName:
    :param save_csv: True/False
    :param title:
    :param xlabel:
    :param ylabel:
    :param xlim:
    :param ylim:
    :param plotMoreFunction: function(ax)
    :return:
    """
    assert len(p) % 3 == 0, f'Number of parameters {len(p)} is not multiple of 3'
    fig, ax = createfig()

    right_ax = None
    if twin_params is not None:
        right_ax = ax.twinx()
        if 'ylabel' in twin_params:
            right_ax.set_ylabel(twin_params['ylabel'])
        if 'ylim' in twin_params:
            right_ax.set_ylim(*twin_params['ylim'])

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
        # print(title)
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
            ax.set_ylim(ylim[0])
    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)
    if plotMoreFunction is not None:
        plotMoreFunction(ax)

    ax.legend(loc=2)
    if right_ax is not None:
        right_ax.legend(loc=1)

    if fileName is None:
        fileName = 'graph.png'
    folder = os.path.split(os.path.expanduser(fileName))[0]
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if grid:
        ax.grid()
    if tight_layout:
        fig.tight_layout()
    savefig(fileName, fig)
    closefig(fig)
    # plt.clf()
    # plt.close(fig)

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

    # print(df.shape)
    # print(df.dtypes)
    # print(df.columns)
    # print(df)
    if create_standard:
        dir_part, file_part = os.path.split(datapath)
        f_name, ext = os.path.splitext(file_part)
        df.to_csv(f'{dir_part}/{f_name}_new{ext}',
                  sep=';', index=False)

    if not ret_df:
        df = None

    return plot_ops, df


def plot_from_file(*lbls_fmts, csvFileName: str, **plottof_ops):
    ops, _ = read_plottof_csv(csvFileName, True, False, False)
    for i, l in enumerate(ops[2::3]):
        if l not in lbls_fmts:
            for lf in lbls_fmts:
                if isinstance(lf, dict) and (l == lf['label']):
                    ops[3 * i + 2] = lf
    if 'fileName' not in plottof_ops:
        plottof_ops['fileName'] = os.path.splitext(csvFileName)[0] + '.png'
    plot_to_file(*ops, **plottof_ops)


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

