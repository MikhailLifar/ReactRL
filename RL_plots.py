from itertools import product
import copy
import os.path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter as FormatObj
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import seaborn

import lib
from lib import read_plottof_csv

from typing import List


# FIG_SIZE_MAIN = (16 / 3 * 2, 9 / 3 * 2)
FIG_SIZE_MAIN = (16, 12)
# FIG_SIZE_WPLUS = (18, 12)
FIG_SIZE_TALL = (9 / 3 * 2, 9 / 3 * 2)
FIG_SIZE_SMALL = (16 / 2, 9 / 2)
# FIG_SIZE_ADJUSTED = (1.5 * FIG_SIZE_MAIN[0], 1.8 * FIG_SIZE_MAIN[1])
MAIN_PLOT_FONT_SIZE = 14
MAIN_LABEL_SIZE = 14  # 12, 16
MAIN_DPI = 600

matplotlib.rcParams.update(
    {'font.size': MAIN_PLOT_FONT_SIZE,
     # 'axes.labelsize': MAIN_PLOT_FONT_SIZE,
     'legend.fontsize': MAIN_LABEL_SIZE,
     'axes.titlesize': MAIN_LABEL_SIZE,
     'axes.labelsize': MAIN_LABEL_SIZE,
     'xtick.labelsize': MAIN_LABEL_SIZE,
     'ytick.labelsize': MAIN_LABEL_SIZE,
     'mathtext.default': 'regular'})

PLOT_FOLDER = './ARTICLE/article_figures'
DATA_DIR = './ARTICLE/data'

LABEL_A = 'CO'
LABEL_B = '$O_2$'
LABEL_B_VAR2 = 'O'


def savefig(fig, filepath):
    fig.savefig(filepath, bbox_inches='tight', dpi=MAIN_DPI)


def color_arr_from_arr(arr, bottom_color=(0.6, 0, 0.9), top_color=(0.9, 0.9, 0.5), bottom=0., up=0.):
    assert isinstance(arr, np.ndarray), 'arr should be np.ndarray'
    if not (bottom or up):
        bottom = np.min(arr)
        up = np.max(arr)
    else:
        bottom = np.min([bottom, np.min(arr)])
        up = np.max([up, np.max(arr)])
    new_arr = (arr - bottom)/(up - bottom)
    assert (np.max(new_arr) <= 1 + 1e-5) and (np.min(new_arr) >= - 1e-5), 'Error!'
    bottom_color = np.array(bottom_color)
    top_color = np.array(top_color)
    difference = top_color - bottom_color
    out_arr = np.zeros((new_arr.size, 3))
    for j in range(3):
        out_arr[:, j] = bottom_color[j] + difference[j] * new_arr
    return out_arr


def plot_learning_curve(data: List[np.ndarray], outpath: str,
                        xlim: int = None,
                        plottype: str = 'plot', **kwargs) -> None:
    xdata, ydata = data

    if xlim is not None:
        idx = xdata <= xlim
        xdata = xdata[idx]
        ydata = ydata[idx]

    # for n_integral only
    def brightness(x: np.ndarray):
        range_to_compute_mean_len = 10
        out = copy.deepcopy(x)
        for i in range(1, range_to_compute_mean_len):
            out[:out.size - i] += x[i:]
        out[out.size - range_to_compute_mean_len + 1:] = \
            x[out.size - range_to_compute_mean_len]
        out = np.abs((-1 * out / range_to_compute_mean_len) + x)
        out = (out - np.min(out)) / (np.max(out) - np.min(out))  # normalization
        out = out ** 0.27  # to more contrast
        return out

    color_arr = color_arr_from_arr(brightness(ydata),
                                   bottom_color=(0, 0, 0),
                                   top_color=(0.85, 0.85, 0.85),
                                   bottom=0, up=1.)
    fig, ax = plt.subplots(1, figsize=FIG_SIZE_SMALL)
    if plottype == 'scatter':
        ax.scatter(xdata,
                   ydata,
                   color=color_arr)  # color='#9922ee'
    elif plottype == 'plot':
        ax.plot(xdata,
                ydata,
                color='#9922ee')  # variants: '#9922ee', '#909090'
    else:
        raise ValueError
    ax.set_xlim(np.min(xdata), np.max(xdata))
    lims = kwargs.get('ylim', None)
    if lims is None:
        lims = (min(1.e-2, ydata.min()), ydata.max())
    ax.set_ylim(*lims)
    ax.yaxis.set_major_formatter('{x:.2g}')

    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1e-1 * lims[1]))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(5e-2 * lims[1]))
    # ax.tick_params(which='both', direction='in')

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1e-2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1e-2))

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    ax.set_xlabel('Episode number')  # alternative: 'номер эпизода'
    # ax.set_ylabel('Cumm. output divide on episode length')  # alternative: 'сумарный выход деленный на длину эпизода'
    ax.set_ylabel('Mean reaction rate')
    fig.savefig(outpath, dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_policy(data, ax, text_labels: bool = True, units: str = 'Pa',
                to_O2_style: dict = None, to_CO_style: dict = None):
    # TODO there should be problems with negative values

    O2_style = {'c': '#5588ff', 'linestyle': 'solid', 'label': '$O_{2}$'}  # c='#5588ff', '#555555'
    CO_style = {'c': '#ffaa77', 'linestyle': 'solid', 'label': 'CO'}  # c='#ffaa77', '#989898'
    if to_O2_style is not None:
        O2_style.update(to_O2_style)
    if to_CO_style is not None:
        CO_style.update(to_CO_style)

    xdata, O2_data, CO_data = data

    ax_bottom, ax_top = 0., None
    if units == 'normal':
        coef = pow(10., -int(np.log10(max(O2_data.max(), CO_data.max())) + 1.e-3))
        O2_data *= coef
        CO_data *= coef
        ax_top = 11.
    else:
        ax_top = 1.1 * pow(10., int(np.log10(max(O2_data.max(), CO_data.max())) + 1 - 1.e-3))

    l1 = ax.plot(xdata, O2_data, **O2_style)
    l2 = ax.plot(xdata, CO_data, **CO_style)
    ax.set_ylim(bottom=ax_bottom, top=ax_top)
    ax.yaxis.set_major_locator(MultipleLocator(ax_top / 10))
    ax.yaxis.set_minor_locator(MultipleLocator(ax_top / 20))

    print(CO_data[CO_data.size * 3 // 4] / O2_data[CO_data.size * 3 // 4])
    if text_labels:
        ax.set_title('Policy')
        # ax.set_title('Политика')
        ax.legend(loc='upper right', fontsize=12)
    return l1, l2


def plot_output(data, ax, text_labels: bool = True, ax_top=None):
    # TODO there should be problems with negative values

    xdata, ydata = data
    l1 = ax.plot(xdata, ydata, label='$CO_{2}$', c='#44bb44')  # '#ababab', '#44bb44'
    if ax_top is None:
        ax_top = 1.1 * pow(10., int(np.log10(ydata.max()) + 1 - 1.e-3))
    ax.set_ylim(bottom=0, top=ax_top)
    ax.yaxis.set_major_locator(MultipleLocator(ax_top / 10))
    ax.yaxis.set_minor_locator(MultipleLocator(ax_top / 20))
    if text_labels:
        ax.set_title('Out')
        # ax.set_title('Выход')
        ax.legend(loc='upper right', fontsize=12)
    return l1


def policy_out_graphs(datapath, outpath, together=False, units='Pa'):
    # data reading
    _, df = read_plottof_csv(datapath, ret_df=True, create_standard=False)
    policy_data = [df['inputB x'], df['inputB y'], df['inputA y']]
    out_data = [df['reaction rate x'], df['reaction rate y']]

    base_name, ext = os.path.splitext(outpath)

    if together:
        fig, axs = plt.subplots(1, 2, figsize=(FIG_SIZE_MAIN[0], FIG_SIZE_MAIN[1] * 0.7))
        # plotting policy
        plot_policy(policy_data, axs[0], units=units)
        ylabel = 'Pressure'
        if units != 'normal':
            ylabel += f', {units}'
        axs[0].set(xlabel='Time, s', ylabel=ylabel)
        # axs[0].set(xlabel='время, сек', ylabel='давление, Па')

        # plotting output
        plot_output(out_data, axs[1], ax_top=0.1)
        axs[1].set(xlabel='Time, s', ylabel='$CO_{2}$ formation rate')
        # axs[1].set(xlabel='время, сек', ylabel='скорость образования CO2')

        for ax in axs:
            ax.yaxis.set_major_formatter(FormatObj('%.2g'))
            ax.tick_params(which='both', direction='in')

        plt.tight_layout()
        fig.savefig(f'{base_name}_policy_with_out_{ext}',
                    dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)

    else:
        # plotting policy
        fig, ax = plt.subplots(1, figsize=FIG_SIZE_MAIN)
        plot_policy(policy_data, ax)
        fig.savefig(f'{base_name}_policy{ext}',
                    dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)

        # plotting output
        fig, ax = plt.subplots(1, figsize=FIG_SIZE_MAIN)
        plot_output(out_data, ax)
        fig.savefig(f'{base_name}_output{ext}',
                    dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


def plot_two_epochs(datapath1, datapath2, outpath):
    # data reading
    df1 = read_plottof_csv(datapath1, create_standard=False)
    df2 = read_plottof_csv(datapath2, create_standard=False)

    fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE_MAIN)

    # plotting policy
    l1, l2 = plot_policy(df1, axs[0, 0], text_labels=False)
    plot_policy(df2, axs[1, 0], text_labels=False)
    # plotting output
    l3 = plot_output(df1, axs[0, 1], text_labels=False)
    plot_output(df2, axs[1, 1], text_labels=False)

    for ax in axs[1, :]:
        ax.set(xlabel='time, s')
    for ax in axs[:, 0]:
        ax.set(ylabel='pressure, Pa')
        ax.yaxis.set_major_formatter('{x:.3g}')
    for ax in axs[:, 1]:
        ax.set(ylabel='$CO_{2}$ formation rate')

    # ax = fig.add_subplot(212)
    # labels = ['O2', 'CO', 'CO2']
    # ax.legend([l1, l2, l3], labels=labels)
    # fig.legend([l1, l2, l3], labels=labels,
    #            loc='lower center', borderaxespad=0.05,
    #            title='')

    plt.tight_layout()
    fig.savefig(outpath,
                dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_many_on_same(plots_data: list, titles: list,
                      outpath: str, in_row: int = 3,
                      plt_params: dict = None):
    plots_count = len(plots_data)
    assert plots_count == len(titles)
    assert plots_count % in_row == 0, 'invalid list'

    fig, axs = plt.subplots(plots_count // in_row, in_row,
                            figsize=(plots_count // in_row * 3 / 2, in_row * 2))

    if plt_params is None:
        plt_params = dict()
    for row in range(plots_count // in_row):
        for col in range(in_row):
            data_slice = plots_data[in_row * row + col]
            ax = axs[row][col]
            add_len = len(plt_params)
            chunck_len = 2 + add_len
            assert len(data_slice) % chunck_len == 0
            for i in range(len(data_slice) // chunck_len):
                for prop_pair in data_slice[2 + i * chunck_len:(i + 1) * chunck_len]:
                    plt_params[prop_pair[:prop_pair.find(':')]] = prop_pair[prop_pair.find(':') + 1:]
                ax.plot(*data_slice[chunck_len * i: chunck_len * i + 2], **plt_params)
            ax.set_ylim(bottom=0, top=1.1e-4)
            ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
            ax.set_title(titles[in_row * row + col], fontsize=8)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(8)

    plt.tight_layout()
    fig.savefig(outpath,
                dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_many_procedure(data_folder, out_folder):
    files = dict()
    prefixes = ['L2001', 'LPlus']
    subfolders = ['thetamax_variation', 'CTs_variation']
    for pref in prefixes:
        files[pref] = []
    for subf in subfolders:
        files_list = os.listdir(f'{data_folder}/{subf}')
        for pref in prefixes:
            files[pref] += [f'{subf}/{name}' for name in files_list if pref in name]

    for pref in prefixes:
        files[pref] = sorted(files[pref])
        titles = [name[name.find('_', name.rfind('/')) + 1:name.rfind('.')] for name in files[pref]]
        titles = [t.replace('_', ' = ') for t in titles]
        data = []
        for name in files[pref]:
            df = read_plottof_csv(f'{data_folder}/{name}', create_standard=False)
            data += [[df['CO x'].to_numpy(), df['CO y'].to_numpy(), 'color:#555555', 'label:O2',
                      df['O2 x'].to_numpy(), df['O2 y'].to_numpy(), 'color:#989898', 'label:CO']]
        plot_many_on_same(data, titles=titles,
                          outpath=f'{out_folder}/{pref}_graph.png',
                          plt_params={'label': None, 'color': None})


def compute_ratio(df: pd.DataFrame,
                  where: [list, float] = None,
                  numen: str = 'CO') -> list:
    """

    :param numen:
    :param df:
    :param where: value between 0 and 1
    Defines in which "part" of dataframe the ratio should be computed
    :return:
    """
    if isinstance(where, list):
        out = []
        for val in where:
            out.append(compute_ratio(df, val, numen)[0])
        return out
    else:
        if numen == 'O2':
            denom = 'CO'
        elif numen == 'CO':
            denom = 'O2'
        else:
            raise ValueError(f'Invalid value for numen: {numen}')
        ind = int((df.shape[0] - 1) * where)
        numen_arr = df[f'{numen} y'].to_numpy()
        denom_arr = df[f'{denom} y'].to_numpy()
        if denom_arr[ind] == 0:
            try_ind = ind - 5
            try_ind = max(min(try_ind, df.shape[0] - 1), 0)
            while (try_ind < min(ind + 5, df.shape[0] - 1))\
                    and (denom_arr[try_ind] == 0):
                try_ind += 1
            ind = try_ind
        return [numen_arr[ind] / denom_arr[ind]]


def compute_in_folder(folder_path, choose_key=lambda s: True,
                      target_name='target'):
    from lib import integral

    files = [f for f in os.listdir(folder_path) if choose_key(f)]
    files = sorted(files)  # ordered by name
    for name in files:
        df = read_plottof_csv(f'{folder_path}/{name}', create_standard=False)
        ratios = compute_ratio(df, where=[0.75 + (i - 10) * 0.01 for i in range(25)], numen='O2')
        ratios = np.array(ratios)
        if np.std(ratios) < 0.02 * np.mean(ratios):
            sol_type = 'const'
        else:
            sol_type = 'dynamic'
        integral_output = integral(df[f'{target_name} x'], df[f'{target_name} y'])
        s = f'{name} ratio: {np.mean(ratios):.3f}, integral: {integral_output:.3f},'\
              f' \n\tsolution type: {sol_type}'
        print(s)


def all_lines_in_folder_same_plot(folder_path, out_path, kind: str = 'return',
                                  to_legend: [str, list] = None, colors: list = None):
    from lib import integral

    # files = [name for name in os.listdir(folder_path)
    #          if ('O2_' in name) and ('CO_' in name)]
    files = os.listdir(folder_path)
    files = sorted(files)
    fig, ax = plt.subplots(1, figsize=FIG_SIZE_MAIN)
    if kind == 'both':
        ax2 = ax.twinx()
    else:
        ax2 = None
    maxY = None
    if colors is None:
        # color of the line depending on integral
        integrals = []
        for f in files:
            df = read_plottof_csv(f'{folder_path}/{f}', create_standard=False)
            integrals.append(integral(df['CO2 x'], df['CO2 y']))
        colors = color_arr_from_arr(np.array(integrals),
                                    bottom_color=(0.4, 0.4, 0.4), top_color=(0., 0., 0.))
    elif len(colors) != len(files):
        raise ValueError('Error! Length mismatch')
    if to_legend is None:
        to_legend = ''
    lns = None
    for i, f in enumerate(files):
        df = read_plottof_csv(f'{folder_path}/{f}', create_standard=False)
        O2_CO_ratio, = compute_ratio(df, 0.8, numen='O2')
        if isinstance(to_legend, str):
            if to_legend == 'ratios':
                line_label = f'O2/CO={O2_CO_ratio:.1f}'
            elif to_legend == 'model_id':
                model_id = f[:f.rfind('.')].split('_')[-1]
                line_label = f'{model_id}, O2/CO={O2_CO_ratio:.1f}'
            else:
                line_label = ''
        elif isinstance(to_legend, list):
            line_label = to_legend[i]
        else:
            raise ValueError
        if kind == 'policy':
            ax.plot(df['CO x'], df['CO y'], color=colors[i],
                    linestyle='dashed', label='CO')
            ax.plot(df['O2 x'], df['O2 y'], color=colors[i],
                    linestyle='dotted', label='$O_{2}$')
        elif kind == 'return':
            maxY = max(df['CO2 y'])
            ax.plot(df['CO2 x'], df['CO2 y'], color=colors[i],
                    label=line_label)  # {O2_CO_ratio:.1f}
        elif kind == 'both':
            l1, l2 = plot_policy(df, ax, units='normal',
                                 to_CO_style={'linestyle': 'dashed', 'c': colors[i]},
                                 to_O2_style={'linestyle': 'dotted', 'c': colors[i]},)
            l3 = ax2.plot(df['CO2 x'], df['CO2 y'], color=colors[i],
                label='$CO_{2}$')
            lns = l1 + l2 + l3
        else:
            raise ValueError
    if kind == 'policy':
        ax.set(ylabel='Pressure')
    elif kind == 'both':
        ax.set(ylabel='Pressure')
        ax2.set(ylabel='$CO_{2}$ formation rate')
        ax2.set_ylim(bottom=0., top=0.015)  # for fig1
        # ax2.set_ylim(bottom=0., top=0.017)  # for opt_vs_RL
        # ax2.set_ylim(bottom=0., top=0.05)  # for L2001_quazidynamic
    elif kind == 'return':
        ax.set(ylabel='$CO_{2}$ formation rate')
        ax.set_ylim(bottom=0., top=max(0.033, 1.15 * maxY))
    ax.set(xlabel='Time, s')
    ax.yaxis.set_major_formatter('{x:.2g}')
    if kind == 'both':
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', fontsize=12)
    else:
        ax.legend(loc='upper right', fontsize=12)
    fig.savefig(out_path, dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def multiple_curves(datapath, outpath, end_plot: int = None):
    files = os.listdir(datapath)
    files = sorted(files)
    fig, ax = plt.subplots(1, figsize=FIG_SIZE_MAIN)
    if end_plot is not None:
        end_plot //= 20
    for i, f in enumerate(files):
        data_to_plot = pd.read_csv(f'{datapath}/{f}', sep=';')['smooth_1000_step'].to_numpy()
        if end_plot is None:
            end_plot = data_to_plot.size
        else:
            end_plot = min(data_to_plot.size, end_plot)
        data_to_plot = data_to_plot[:end_plot]
        line_name = os.path.splitext(f)[0]
        x_arr = 20 * np.arange(end_plot)
        ax.plot(x_arr, data_to_plot, label=line_name)
    ax.set_ylim(bottom=0., top=0.031)
    ax.set(xlabel='Episode number', ylabel='Cumm. output divide on episode length')
    ax.legend(loc='lower right')
    fig.savefig(outpath, dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


# def opt_const_vs_rl_fig(data_path):
#     df_policy =
#     fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
#
#     ax2 = ax.twinx()
#     plot_policy(df, ax, units='normal',
#                         CO_style={'linestyle': 'dashed', 'c': colors[i]},
#                         O2_style={'linestyle': 'dotted', 'c': colors[i]},)


# def for_pres_2305():
#
#     def blue_red_white_instead_gray(filepath):
#         fold, fname = os.path.split(filepath)
#         fname, _ = os.path.splitext(fname)
#         lib.plot_from_file({'label': 'O2', 'color': 'blue', 'linewidth': 2.},
#                            {'label': 'CO', 'color': 'red', 'linewidth': 2.},
#                            transforms=({'old_label': 'O2', 'transform': lambda x: x * 1.e+5},
#                                        {'old_label': 'CO', 'transform': lambda x: x * 1.e+5},
#                                        ),
#                            chose_labels=('O2', 'CO'),
#                            csvFileName=filepath,
#                            save_csv=False,
#                            fileName=f'{fold}/{fname}_input_brw.png',
#                            xlabel='Time, s', ylabel='gas pressure',
#                            xlim=[0, 500], ylim=[-1.e-2, 1.1e+1])
#
#     def rewrite_summary(filepath):
#         fold, fname = os.path.split(filepath)
#         # fname, _ = os.path.splitext(fname)
#
#         def plot_more(ax):
#             for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#               ax.get_xticklabels() + ax.get_yticklabels()):
#                 item.set_fontsize(12)
#             # ax.set_xlabel('', fontsize=10)
#             # ax.set_ylabel('', fontsize=10)
#
#         lib.plot_from_file({'label': 'reaction rate', 'linestyle': 'solid',
#                                              'marker': 'h', 'c': 'purple',
#                                              'twin': True,
#                                              },
#                            {'label': 'O2 coverage', 'linestyle': (0, (1, 1)),
#                                              'marker': 'x', 'c': 'blue'},
#                            {'label': 'CO coverage', 'linestyle': (0, (5, 5)),
#                                              'marker': '+', 'c': 'red'},
#                            x_transform=lambda x: x / 10,
#                            transforms=({'old_label': 'Average CO2 prod. rate', 'transform': lambda x: x, 'new_label': 'reaction rate'},
#                                        {'old_label': 'Average O2 coverage', 'transform': lambda x: x, 'new_label': 'O2 coverage'},
#                                        {'old_label': 'Average CO coverage', 'transform': lambda x: x, 'new_label': 'CO coverage'},
#                                        ),
#                            chose_labels=None,
#                            csvFileName=filepath,
#                            save_csv=False,
#                            fileName=f'{fold}/rewrite_summary.png',
#                            xlabel=f'CO proportion', ylabel='coverages', title='Steady-state solutions',
#                            twin_params={'ylabel': 'reaction rate'},
#                            plotMoreFunction=plot_more)
#
#     # blue_red_white_instead_gray('temp/for_english/RL_const.csv')
#     # blue_red_white_instead_gray('temp/for_english/RL_periodic.csv')
#     # blue_red_white_instead_gray('temp/for_english/opt_L2001.csv')
#     # blue_red_white_instead_gray('temp/for_english/opt_LD.csv')
#
#     rewrite_summary('temp/for_english/L2001_benchmark/Ziff_summarize_CO2.csv')


# 2307 NEW ARTICLE figures
def plot_several_lines_uni(data, outpath=None, ax=None, to_styles: dict = None, **plot_proc_params):
    from functools import reduce

    styles = {
        1: {'c': '#aeaeae', 'linestyle': 'dashed', },
        2: {'c': '#989898', 'linestyle': 'dashed', },
        3: {'c': '#555555', 'linestyle': 'solid',
            'twin': True,
            }
    }

    if to_styles is not None:
        for k in to_styles:
            if k in styles:
                styles[k].update(to_styles[k])
            else:
                styles[k] = to_styles[k]

    xdata = data[0]

    # if plot_proc_params.get('twin_params', False) is False:
    #     plot_proc_params.update({'twin_params': {}})

    right_ax = None

    if ax is None:
        assert outpath is not None
        lib.plot_to_file(*list(reduce(lambda ret, i: ret + [xdata, data[i], styles[i]], range(1, len(data)), [])),
                         fileName=outpath,
                         save_csv=False,
                         **plot_proc_params)
    else:
        _, right_ax, _ = lib.plot_in_axis(*list(reduce(lambda ret, i: ret + [xdata, data[i], styles[i]], range(1, len(data)), [])),
                                          ax=ax,
                                          **plot_proc_params)

    return ax, right_ax


def input_plot(data, ax=None, outpath=None):
    plot_several_lines_uni(data,
                           outpath=outpath,
                           ax=ax,
                           # to_styles={1: {'label': 'inputB', 'linestyle': 'solid'}, 2: {'label': 'inputA', 'linestyle': 'solid'}},  # grey
                           to_styles={1: {'label': 'inputB', 'c': '#5588ff', 'linestyle': 'solid'},
                                      2: {'label': 'inputA', 'c': '#ffaa77', 'linestyle': 'solid'}},  # color
                           ylabel='Pressure', ylim=[0., 1.05],
                           title='input pressures',
                           twin_params=None,
                           )


def output_plot(data, ax=None, outpath=None):
    plot_several_lines_uni(data,
                           outpath=outpath,
                           ax=ax,
                           # to_styles={1: {'label': 'thetaB'}, 2: {'label': 'thetaA'}, 3: {'label': 'reaction rate'}},
                           to_styles={1: {'label': 'thetaB', 'c': '#9922ee'}, 2: {'label': 'thetaA', 'c': 'r'}, 3: {'label': 'reaction rate', 'c': '#44bb44'}},  # color
                           ylabel='Coverage', ylim=[0., 0.5],
                           title='rates & thetas',
                           # twin_params={'ylim': [0., 0.1], 'ylabel': '$CO_{2}$ formation rate'},
                           twin_params={'ylim': [0., 0.1], 'ylabel': 'Reaction rate'},
                           )


def input_output_plot(ax_input, ax_output, df_NM, df_RL,
                      xtop, cov_top=0.5,
                      twin_label=True, twin_top=0.1, update_colors=None):

    data_RL_in = [df_RL['inputB x'], df_RL['inputB y'], df_RL['inputA y']]

    colors = {'O2': '#5588ff', 'CO': '#ffaa77', 'CO2': '#44bb44'}
    if update_colors is None:
        update_colors = {}
    colors.update(update_colors)

    if df_NM is not None:
        data_NM_in = [df_NM['inputB x'], df_NM['inputB y'], df_NM['inputA y']]
        plot_several_lines_uni(data_NM_in,
                               outpath=None,
                               ax=ax_input,
                               # to_styles={1: {'label': 'inputB', 'linestyle': 'solid'}, 2: {'label': 'inputA', 'linestyle': 'solid'}},  # grey
                               to_styles={1: {'label': None, 'c': colors['O2'], 'linestyle': 'dashed'},
                                          2: {'label': None, 'c': colors['CO'], 'linestyle': 'dashed'}},  # color
                               twin_params=None,
                               )

    plot_several_lines_uni(data_RL_in,
                           outpath=None,
                           ax=ax_input,
                           # to_styles={1: {'label': 'inputB', 'linestyle': 'solid'}, 2: {'label': 'inputA', 'linestyle': 'solid'}},  # grey
                           to_styles={1: {
                                           # 'label': f'{LABEL_B}',  # давление $O_2$, {LABEL_B}
                                           'c': colors['O2'], 'linestyle': 'solid'},
                                      2: {
                                          # 'label': f'{LABEL_A}',  # давление CO, {LABEL_A}
                                          'c': colors['CO'], 'linestyle': 'solid'}},  # color
                           title='Политика агента',  # 'input pressures', 'model inputs', 'управляемые параметры', 'RL policy'
                           xlim=[0., xtop], ylim=[0., 1.05],
                           twin_params=None,
                           )

    def get_target_col(df):
        return 'outputC y' if 'outputC y' in df.columns else 'reaction rate y'

    data_RL_out = [df_RL['thetaB x'], df_RL['thetaB y'], df_RL['thetaA y'], df_RL[get_target_col(df_RL)]]

    if df_NM is not None:
        data_NM_out = [df_NM['thetaB x'], df_NM[get_target_col(df_NM)]]
        plot_several_lines_uni(data_NM_out,
                               outpath=None,
                               ax=ax_output,
                               to_styles={1: {
                                              # 'label': 'reaction rate',
                                              'c': colors['CO'], 'linestyle': 'dashed',
                                              'twin': True}},  # color
                               twin_params={'ylim': [0., twin_top]},
                               )
    _, right_ax = plot_several_lines_uni(data_RL_out,
                                       outpath=None,
                                       ax=ax_output,
                                       # to_styles={1: {'label': 'thetaB'}, 2: {'label': 'thetaA'}, 3: {'label': 'reaction rate'}},
                                       to_styles={1: {
                                                      # 'label': '$\\theta_O$',  # заселенность O, theta{LABEL_B_VAR2}
                                                      'c': colors['O2'], 'linestyle': 'solid'},
                                                  2: {
                                                      # 'label': '$\\theta_{CO}$',  # заселенность CO, theta{LABEL_A}
                                                      'c': colors['CO'], 'linestyle': 'solid'},
                                                  3: {
                                                      # 'label': 'reaction rate',  # скорость образования \n$CO_2$, reaction rate
                                                      'c': colors['CO2']}},  # color
                                       xlim=[0., xtop], ylim=[0., cov_top],
                                       title='Поведение модели',  # 'rate & thetas', 'model outputs', 'выходные параметры', 'Environment response'
                                       # twin_params={'ylim': [0., 0.1], 'ylabel': '$CO_{2}$ formation rate'},
                                       twin_params={'ylim': [0., twin_top], 'ylabel': 'Скорость реакции' if twin_label else None},  # 'Reaction rate'
                                       )

    right_ax.spines['right'].set_color(colors['CO2'])
    right_ax.tick_params(axis='y', colors=colors['CO2'])
    right_ax.yaxis.label.set_color(colors['CO2'])


def exp1_steady_state_map(exp_id):
    datapath = f'{DATA_DIR}/{exp_id}/analytic_steady_state.npy'
    data = np.load(datapath)

    lib.plot_show_save_map(data, (0., 1.), (0., 1.), f'{PLOT_FOLDER}/{exp_id}/exp1_steady_state_map.png',
                           save_data=False, xlabel='p$_{O_2}$', ylabel='p$_{CO}$',
                           color_ax_label='reaction rate')


def exp2_reverse_steady_state_maps(exp_id):
    pathB = f'{DATA_DIR}/{exp_id}/press_from_covs_pB.npy'
    pathA = f'{DATA_DIR}/{exp_id}/press_from_covs_pA.npy'
    pB, pA = np.load(pathB), np.load(pathA)

    mask = ~((pB >= 0.) & (pB <= 1.) & (pA >= 0.) & (pA <= 1.))

    pB[mask] = -0.1
    pA[mask] = -0.1
    pB, pA = pB[::-1, :], pA[::-1, :]

    lib.plot_show_save_map(pB, (0., 0.25), (0.5, 0.), f'{PLOT_FOLDER}/{exp_id}/exp2_pB_from_covs.png',
                           save_data=False, xlabel='thetaO', ylabel='thetaCO',
                           color_ax_label='p$_{O_2}$')
    lib.plot_show_save_map(pA, (0., 0.25), (0.5, 0.), f'{PLOT_FOLDER}/{exp_id}/exp2_pA_from_covs.png',
                           save_data=False, xlabel='thetaO', ylabel='thetaCO',
                           color_ax_label='p$_{CO}$')


def fig_2_learning_curve():
    # _, df = read_plottof_csv(datapath, ret_df=True)
    # data = [df['agent metric x'].to_numpy(), df['agent metric y'].to_numpy()]
    datapath = f'{DATA_DIR}/fig2/fig2_curve.csv'
    plotname = 'fig2.png'
    df = pd.read_csv(datapath, index_col=False, sep=';')
    data = [np.arange(df.shape[0]) + 1, df['n_integral'].to_numpy()]
    plot_learning_curve(data, f'{PLOT_FOLDER}/{plotname}', ylim=(0., 0.04), xlim=15_000)


# def fig_3_two_rate_sets():
#     data_folder = f'{DATA_FOLDER}/fig3_two_rate_sets_v1'
#     fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE_MAIN)  # (1.5 * FIG_SIZE_MAIN[0], 1.8 * FIG_SIZE_MAIN[1])
#
#     list_RL = sorted(os.listdir(f'{data_folder}/RL'))
#     list_NM = sorted(os.listdir(f'{data_folder}/NM'))
#
#     for i, fname_RL, fname_NM in zip(range(len(list_RL)), list_RL, list_NM):
#         _, df_RL = read_plottof_csv(f'{data_folder}/RL/{fname_RL}', ret_df=True)
#         _, df_NM = read_plottof_csv(f'{data_folder}/NM/{fname_NM}', ret_df=True)
#         input_output_plot(axs[0, i], axs[1, i], df_NM, df_RL,  xtop=60., twin_label=(i == 1),
#                           twin_top=1.5 * np.max(df_RL['outputC y']), cov_top=0.75)
#
#     axs[0, 0].set_ylabel('Pressure')
#     axs[1, 0].set_ylabel('Coverage')
#     axs[1, 0].set_xlabel('Time, s')
#     axs[1, 1].set_xlabel('Time, s')
#
#     savefig(fig, f'{PLOT_FOLDER}/fig3_two_rate_sets_v1.png')


def fig_3_one_rate_sets():
    data_folder = f'{DATA_DIR}/fig3_one_rate_set'
    fig, axs = plt.subplots(2, 1, figsize=(FIG_SIZE_MAIN[0] * 0.6, FIG_SIZE_MAIN[1]))  # (1.5 * FIG_SIZE_MAIN[0], 1.8 * FIG_SIZE_MAIN[1])

    list_NM = sorted(os.listdir(f'{data_folder}/NM'))
    list_RL = sorted(os.listdir(f'{data_folder}/RL'))

    for i, fname_RL, fname_NM in zip(range(len(list_RL)), list_RL, list_NM):
        _, df_RL = read_plottof_csv(f'{data_folder}/RL/{fname_RL}', ret_df=True)
        _, df_NM = read_plottof_csv(f'{data_folder}/NM/{fname_NM}', ret_df=True)
        input_output_plot(axs[0], axs[1], df_NM, df_RL,  xtop=60., twin_label=True,
                          twin_top=1.5 * np.max(df_RL['outputC y']), cov_top=0.75)

    axs[0].set_ylabel('Partial pressure, norm.units')  # Pressure
    axs[1].set_ylabel('Coverage $\\theta$')
    axs[1].set_xlabel('Time, s')

    savefig(fig, f'{PLOT_FOLDER}/fig3_one_rate_set.png')


# def fig4_covs_axis_plot():
#     pathB = f'{DATA_FOLDER}/libuda_react_div60/press_from_covs_pB.npy'
#     pathA = f'{DATA_FOLDER}/libuda_react_div60/press_from_covs_pA.npy'
#     path_dynamic = f'{DATA_FOLDER}/dynamic_advantage_rates/dynamic_sol.csv'
#     path_stationary_NM = f'{DATA_FOLDER}/dynamic_advantage_rates/NM_sol.csv'
#
#     pB, pA = np.load(pathB), np.load(pathA)
#     mask = ~((pB >= 0.) & (pB <= 1.) & (pA >= 0.) & (pA <= 1.))
#     pB[mask] = -0.1
#     pA[mask] = -0.1
#     # pB, pA = pB[::-1, :], pA[::-1, :]
#
#     idxs = np.apply_along_axis(lambda x: None if not np.where(x < 0)[0].size
#         else np.where(x < 0)[0][0], 0, pB)
#     thetaO = np.linspace(0., 0.25, pB.shape[0])
#     thetaCO = np.linspace(0., 0.5, pB.shape[1])[idxs % pB.shape[0]]
#
#     _, df = lib.read_plottof_csv(path_stationary_NM, ret_df=True, )
#     stationary_max = df.loc[df.index.max(), ['thetaB y', 'thetaA y']].to_numpy()
#
#     _, df = lib.read_plottof_csv(path_dynamic, ret_df=True, )
#     df = df.loc[df['thetaB x'] >= 120., ['thetaB x', 'thetaB y', 'thetaA x', 'thetaA y']]
#
#     fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(16, 9))
#     ax0.fill_between(thetaO, thetaCO, hatch='/', alpha=0., label='steady-state coverages')
#     lib.plot_in_axis(thetaO, thetaCO, {'c': 'black', 'ls': 'dashed', 'label': None},
#                      df['thetaB y'], df['thetaA y'], {'label': 'periodic regime'},
#                      ax=ax0,
#                      xlabel='$\\theta_O$', ylabel='$\\theta_{CO}$',
#                      xlim=[0., 0.3], ylim=[0., 0.5])
#     ax0.plot(stationary_max[0], stationary_max[1], 'ro', label='steady-state maximum', markersize=8)
#     ax0.legend()
#
#     lib.plot_in_axis(df['thetaB x'], df['thetaB y'], {'label': '$\\theta_O$', 'c': 'b'},
#                      df['thetaA x'], df['thetaA y'], {'label': '$\\theta_{CO}$', 'c': 'r'},
#                      df['thetaB x'], np.full_like(df['thetaB x'], stationary_max[0]), {'c': 'b', 'ls': 'dashed'},
#                      df['thetaA x'], np.full_like(df['thetaA x'], stationary_max[1]), {'c': 'r', 'ls': 'dashed'},
#                      ax=ax1,
#                      xlabel='Time, s', ylabel='Coverages',
#                      ylim=[0., 0.5]
#                      )
#
#     savefig(fig, f'{PLOT_FOLDER}/fig4_covs_axis_plot.png')
#     plt.close(fig)


def fig4_covs_axis_plot_v3():
    pathB = f'{DATA_DIR}/libuda_react_div60/press_from_covs_pB.npy'
    pathA = f'{DATA_DIR}/libuda_react_div60/press_from_covs_pA.npy'
    path_dynamic = f'{DATA_DIR}/dynamic_advantage_rates/dynamic_sol.csv'
    path_stationary_NM = f'{DATA_DIR}/dynamic_advantage_rates/NM_sol.csv'

    pB, pA = np.load(pathB), np.load(pathA)
    mask = ~((pB >= 0.) & (pB <= 1.) & (pA >= 0.) & (pA <= 1.))
    pB[mask] = -0.1
    pA[mask] = -0.1
    # pB, pA = pB[::-1, :], pA[::-1, :]

    idxs = np.apply_along_axis(lambda x: None if not np.where(x < 0)[0].size
        else np.where(x < 0)[0][0], 0, pB)
    thetaO = np.linspace(0., 0.25, pB.shape[0])
    thetaCO = np.linspace(0., 0.5, pB.shape[1])[idxs % pB.shape[0]]

    _, df = lib.read_plottof_csv(path_stationary_NM, ret_df=True, )
    stationary_max = df.loc[df.index.max(), ['thetaB y', 'thetaA y', 'outputC y']].to_numpy()

    _, df = lib.read_plottof_csv(path_dynamic, ret_df=True, )
    df = df.loc[df['thetaB x'] >= 120., ['thetaB x', 'thetaB y', 'thetaA x', 'thetaA y',
                                         'outputC x', 'outputC y']]

    fig, axs = plt.subplot_mosaic([['left', 'top'], ['left', 'bottom']], figsize=(16, 9))
    axs['left'].fill_between(thetaO, thetaCO, hatch='/', alpha=0., label='steady-state coverages')
    lib.plot_in_axis(thetaO, thetaCO, {'c': 'black', 'ls': 'dashed', 'label': None},
                     df['thetaB y'], df['thetaA y'], {'label': 'periodic regime'},
                     ax=axs['left'],
                     xlabel='$\\theta_O$', ylabel='$\\theta_{CO}$',
                     xlim=[0., 0.3], ylim=[0., 0.5])
    axs['left'].plot(stationary_max[0], stationary_max[1], 'ro', label='steady-state maximum', markersize=8)
    axs['left'].legend()

    lib.plot_in_axis(df['thetaB x'], df['thetaB y'], {'label': '$\\theta_O$', 'c': 'b'},
                     df['thetaA x'], df['thetaA y'], {'label': '$\\theta_{CO}$', 'c': 'r'},
                     df['thetaB x'], np.full_like(df['thetaB x'], stationary_max[0]), {'c': 'b', 'ls': 'dashed'},
                     df['thetaA x'], np.full_like(df['thetaA x'], stationary_max[1]), {'c': 'r', 'ls': 'dashed'},
                     ax=axs['bottom'],
                     xlabel='Time, s', ylabel='Coverage $\\theta$',
                     ylim=[0., 0.4]
                     )
    lib.plot_in_axis(df['outputC x'], df['outputC y'], {'label': 'reaction rate', 'c': 'purple'},
                     df['outputC x'], np.full_like(df['thetaB x'], stationary_max[2]), {'c': 'purple', 'ls': 'dashed'},
                     ax=axs['top'],
                     xlabel='Time, s', ylabel='Reaction rate',
                     ylim=[0., 0.005]
                     )

    savefig(fig, f'{PLOT_FOLDER}/fig4v3_covs_axis_plot.png')
    plt.close(fig)


# def fig_4_6_sol_example(fig_number):
#     assert fig_number in (4, 6)
#     # fig, axs = plt.subplots(1, 2, figsize=(16 / 3 * 2 + 4, 9 / 3 * 2))
#     fig, axs = plt.subplots(2, 1, figsize=(1.1 * FIG_SIZE_MAIN[0], 2 * FIG_SIZE_MAIN[1]))
#
#     _, df = read_plottof_csv(f'{DATA_FOLDER}/fig{fig_number}.csv', ret_df=True)
#     input_output_plot(axs[0], axs[1], None, df, xtop=30., twin_label=True,
#                       cov_top=0.5 if fig_number == 6 else 1.,
#                       twin_top=0.1 if fig_number == 6 else 0.006)
#
#     axs[0].set_ylabel('Pressure')
#     axs[1].set_ylabel('Coverage')
#
#     fig.savefig(f'{PLOT_FOLDER}/fig{fig_number}.png')


# def fig_5_dyndemolrg():
#     plots_num = 2
#     fig, axs = plt.subplots(2, plots_num, figsize=FIG_SIZE_MAIN)  # (27, 13)
#     foldpath = f'{DATA_FOLDER}/fig5'
#     for i, fname in enumerate(sorted(os.listdir(foldpath))):
#         _, df = read_plottof_csv(f'{foldpath}/{fname}', ret_df=True)
#         input_output_plot(axs[0, i], axs[1, i], None, df, xtop=240., twin_label=(i == 1),
#                           cov_top=1., twin_top=0.006)
#         if i == plots_num - 1:
#             break
#
#     axs[0, 0].set_ylabel('Pressure')
#     axs[1, 0].set_ylabel('Coverage')
#     for i in range(plots_num):
#         axs[1, i].set_xlabel('Time, s')
#
#     savefig(fig, f'{PLOT_FOLDER}/fig5.png')


def fig_5_dyn_demo():
    plots_num = 2
    fig, axs = plt.subplots(2, plots_num, figsize=FIG_SIZE_MAIN)  # (27, 13)
    foldpath = f'{DATA_DIR}/fig5_v2'

    fnames = sorted(os.listdir(foldpath))

    _, df = read_plottof_csv(f'{foldpath}/{fnames[0]}', ret_df=True)
    _, df_NM = read_plottof_csv(f'{foldpath}/{fnames[1]}', ret_df=True)
    input_output_plot(axs[0, 0], axs[1, 0], df_NM, df, xtop=60., twin_label=False,
                      cov_top=1., twin_top=0.006)

    _, df = read_plottof_csv(f'{foldpath}/{fnames[2]}', ret_df=True)
    input_output_plot(axs[0, 1], axs[1, 1], None, df, xtop=240., twin_label=True,
                      cov_top=1., twin_top=0.006)

    axs[0, 0].set_ylabel('Partial pressure, norm.units')  # Pressure
    axs[1, 0].set_ylabel('Coverage $\\theta$')
    for i in range(plots_num):
        axs[1, i].set_xlabel('Time, s')

    savefig(fig, f'{PLOT_FOLDER}/fig5_v2.png')


def fig_6_stchdemolrg():
    plots_num = 2
    fig, axs = plt.subplots(2, plots_num, figsize=FIG_SIZE_MAIN)

    foldpath = f'{DATA_DIR}/fig6'
    for i, fname in enumerate(sorted(os.listdir(foldpath))):
        _, df = read_plottof_csv(f'{foldpath}/{fname}', ret_df=True)
        input_output_plot(axs[0, i], axs[1, i], None, df, xtop=100., twin_label=(i == 1),
                          update_colors={'CO': 'gray'})
        if i == plots_num - 1:
            break

    axs[0, 0].set_ylabel('Partial pressure, norm.units')  # Pressure
    axs[1, 0].set_ylabel('Coverage $\\theta$')
    for i in range(plots_num):
        axs[1, i].set_xlabel('Time, s')

    savefig(fig, f'{PLOT_FOLDER}/fig6.png')


def fig_n1_k2k5_grid():
    dataNM = pd.read_excel(f'{DATA_DIR}/K2K5_grid_res/NM_rates.xlsx')
    dataRL = pd.read_excel(f'{DATA_DIR}/K2K5_grid_res/RL_rates.xlsx')
    variants = np.sort(dataNM['model::rate_des_A'].unique()).tolist()
    n = len(variants)

    ratesNM = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ratesNM[i, j] = dataNM.loc[(dataNM['model::rate_react'] == variants[i]) \
                               & (dataNM['model::rate_des_A'] == variants[j]), 'reaction_rate']
    ratesNM = ratesNM[::-1]

    ratesRL = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ratesRL[i, j] = np.max(dataRL.loc[(dataRL['model::rate_react'] == variants[i]) \
                                       & (dataRL['model::rate_des_A'] == variants[j]), 'reaction_rate'])
    ratesRL = ratesRL[::-1]

    lib.plot_show_save_map(ratesNM, f'{PLOT_FOLDER}/fign1_nm_rates.png',
                           xticks=variants, yticks=variants[::-1],
                           xlabel='K2', ylabel='K5',
                           color_ax_label='reaction rate',
                           title='NM obtained steady-state regimes',
                           map_kwargs={},
                           save_data=False,
                           figsize=FIG_SIZE_MAIN)

    lib.plot_show_save_map(ratesRL, f'{PLOT_FOLDER}/fign1_rl_rates.png',
                           xticks=variants, yticks=variants[::-1],
                           xlabel='K2', ylabel='K5',
                           color_ax_label='reaction rate',
                           title='RL obtained steady-state regimes',
                           map_kwargs={},
                           save_data=False,
                           figsize=FIG_SIZE_MAIN)

    ratio = ratesRL / ratesNM

    lib.plot_show_save_map(ratio, f'{PLOT_FOLDER}/fign1_ratio.png',
                           xticks=variants, yticks=variants[::-1],
                           xlabel='K2', ylabel='K5',
                           color_ax_label='RL/NM',
                           title='ratio RL obtained to NM obtained',
                           cbounds=[0.5, 2.],
                           map_kwargs={
                               # 'cmap': 'seismic',
                               'cmap': lib.shiftedColorMap(matplotlib.cm.seismic, midpoint=1./3),
                           },
                           save_data=False,
                           figsize=FIG_SIZE_MAIN
                           )

    for pcnt in 0.01, 0.02, 0.03, 0.05:
        ternary_ratio = (ratio > 1. - pcnt).astype('float') + (ratio > 1. + pcnt).astype('float') - 1.
        lib.plot_show_save_map(ternary_ratio, f'{PLOT_FOLDER}/fign1_ratio_ternarized_{pcnt:.2f}.png',
                               xticks=variants, yticks=variants[::-1],
                               xlabel='K2', ylabel='K5',
                               color_ax_label=f'(RL/NM > 1-{pcnt:.2f}) + (RL/NM > 1+{pcnt:.2f})',
                               title='ratio RL obtained to NM obtained, ternarized',
                               cbounds=[-1, 1],
                               map_kwargs={},
                               save_data=False,
                               figsize=FIG_SIZE_MAIN)

        lib.plot_show_save_map(ratio * (ratio > 1. + pcnt), f'{PLOT_FOLDER}/fign1_ratio_{pcnt:.2f}.png',
                               xticks=variants, yticks=variants[::-1],
                               xlabel='K2', ylabel='K5',
                               color_ax_label=f'RL/NM * [RL/NM > 1+{pcnt:.2f}]',
                               title=f'ratio RL obtained to NM obtained, non-zero if RL is adv > 1+{pcnt:.2f}',
                               map_kwargs={},
                               save_data=False,
                               figsize=FIG_SIZE_MAIN)


def fig_n2_integral_curves():
    # _, dataRL = lib.read_plottof_csv(f'{DATA_DIR}/dynamic_advantage_rates/dynamic_sol_1000.csv', ret_df=True)
    _, dataRL = lib.read_plottof_csv(f'{DATA_DIR}/dynamic_advantage_rates/dynamic_sol_2304_1000.csv', ret_df=True)
    _, dataNM = lib.read_plottof_csv(f'{DATA_DIR}/dynamic_advantage_rates/NM_sol_1000.csv', ret_df=True)

    # _, dataRL = lib.read_plottof_csv(f'{DATA_DIR}/dynamic_advantage_rates/RL_lowest_rates_1000.csv', ret_df=True)
    # _, dataNM = lib.read_plottof_csv(f'{DATA_DIR}/dynamic_advantage_rates/NM_lowest_rates_1000.csv', ret_df=True)

    time_RL = dataRL['outputC x'].to_numpy()
    rate_RL = dataRL['outputC y'].to_numpy()
    integral_output_RL = np.cumsum( (rate_RL[1:] + rate_RL[:-1]) * (time_RL[1:] - time_RL[:-1]) / 2 )
    time_1_RL = (time_RL[1:] + time_RL[:-1]) / 2

    time_NM = dataNM['outputC x'].to_numpy()
    rate_NM = dataNM['outputC y'].to_numpy()
    integral_output_NM = np.cumsum( (rate_NM[1:] + rate_NM[:-1]) * (time_NM[1:] - time_NM[:-1]) / 2 )
    time_1_NM = (time_NM[1:] + time_NM[:-1]) / 2

    fig, ax = plt.subplots(figsize=FIG_SIZE_MAIN)

    plot_several_lines_uni([time_1_RL, integral_output_RL], ax=ax,
                           to_styles={1: {'c': 'g', 'label': 'integral output, RL', 'linestyle': 'solid'}})
    plot_several_lines_uni([time_1_NM, integral_output_NM], ax=ax,
                           to_styles={1: {'c': 'b', 'label': 'integral output, NM', 'linestyle': 'solid'}})

    ax.set_title('Nelder-Mead vs RL integral CO2 return')
    ax.set_xlabel('Time, s')
    ax.set_ylabel('Integral CO2 return')

    savefig(fig, f'{PLOT_FOLDER}/fig_n2_dynamic_sol_2304.png')


# def fig_8():
#     foldpath = f'{DATA_FOLDER}/steady_state_comparison'
#     _, df_libuda = read_plottof_csv(f'{foldpath}/Libuda_CORTP.csv', ret_df=True)
#     _, df_zgb = read_plottof_csv(f'{foldpath}/ZGB.csv', ret_df=True)
#
#     fig, axs = plt.subplots(2, 1, figsize=(1.2 * FIG_SIZE_MAIN[0], 2.25 * FIG_SIZE_MAIN[1]))
#
#     data_libuda = [df_libuda['mean_reaction_rate x'], df_libuda['thetaB y'], df_libuda['thetaA y'],
#                    df_libuda['mean_reaction_rate y'],
#                    ]
#     color1 = '#aaaaff'
#     plot_several_lines_uni(data_libuda,
#                            ax=axs[0],
#                            to_styles={1: {'label': 'thetaO',  'linestyle': 'dashed', 'c': color1},
#                                       2: {'label': 'thetaCO',  'linestyle': 'dotted', 'c': color1},
#                                       3: {'label': 'reaction rate', 'c': color1}
#                                       },
#                            xlabel='x$_{CO}$', xlim=[0., 1.],
#                            ylabel='Coverage', ylim=[0., 0.5],
#                            title='DE model',
#                            twin_params={'ylabel': 'Reaction rate'},
#                            )
#
#     data_zgb = [df_zgb['reaction_rate x'], df_zgb['thetaO y'], df_zgb['thetaCO y'],
#                 df_zgb['reaction_rate y'],
#                 ]
#     color2 = 'purple'
#     plot_several_lines_uni(data_zgb,
#                            ax=axs[1],
#                            to_styles={1: {'label': 'thetaO', 'linestyle': 'dashed', 'c': color2},
#                                       2: {'label': 'thetaCO', 'linestyle': 'dotted', 'c': color2},
#                                       3: {'label': 'reaction rate', 'c': color2}
#                                       },
#                            xlabel='x$_{CO}$', xlim=[0., 1.],
#                            ylabel='Coverage', ylim=[0., 1.02],
#                            title='ZGB model',
#                            # twin_params={'ylim': [0., 0.1], 'ylabel': '$CO_{2}$ formation rate'},
#                            twin_params={'ylabel': 'Reaction rate'},
#                            )
#
#     fig.savefig(f'{PLOT_FOLDER}/fig8.png')


def fig_S1():
    data_path = f'{DATA_DIR}/figS1/figS1.csv'
    _, df = lib.read_plottof_csv(data_path, ret_df=True)

    data = [df['outputC x'], df['outputC y']]
    color2 = 'purple'

    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
    plot_several_lines_uni(data,
                           ax=ax,
                           to_styles={1: {'c': color2, 'linestyle': 'solid'}
                                      },
                           xlabel='Time, s',
                           ylabel='Reaction rate', ylim=[0., 0.07],
                           title='Libuda data reproduction',
                           )

    savefig(fig, f'{PLOT_FOLDER}/figS1.png')


def ananikov_thesis():
    data_folder = f'{DATA_DIR}/ananikov'
    fig0, ax0 = plt.subplots(figsize=(12 / 3 * 2, 12 / 3 * 2))
    fig1, ax1 = plt.subplots(figsize=(12 / 3 * 2, 12 / 3 * 2))

    _, df = lib.read_plottof_csv(f'{data_folder}/dynamic_advantage_rates/dynamic_sol.csv', ret_df=True)
    input_output_plot(ax0, ax1, df_NM=None, df_RL=df, xtop=250., twin_top=0.006, cov_top=1.05)

    # axs[0, 0].set_ylabel('Pressure')
    # axs[1, 0].set_ylabel('Coverage')
    ax0.set_xlabel('Время, с')  # 'Время, с', 'Time, s'
    ax1.set_xlabel('Время, с')  # 'Время, с', 'Time, s'

    fig0.savefig(f'{PLOT_FOLDER}/ananikov_in.png')
    fig1.savefig(f'{PLOT_FOLDER}/ananikov_out.png')


def ananikov_sol_plot(datapath, foldpath, xtop):
    fig, axs = plt.subplots(2, 1, figsize=(FIG_SIZE_MAIN[0] * 0.6, FIG_SIZE_MAIN[1]))  # (1.5 * FIG_SIZE_MAIN[0], 1.8 * FIG_SIZE_MAIN[1])

    _, df_RL = read_plottof_csv(datapath, ret_df=True)
    input_output_plot(axs[0], axs[1], None, df_RL,  xtop=xtop, twin_label=True,
                      twin_top=1.5 * np.max(df_RL['outputC y' if 'outputC y' in df_RL.columns else 'reaction rate y']), cov_top=0.75,
                      update_colors={'CO': 'gray'})

    axs[0].set_xlabel('Время, с')
    axs[1].set_xlabel('Время, с')

    _, fname = os.path.split(datapath)
    fname = os.path.splitext(fname)[0]

    savefig(fig, f'{foldpath}/{fname}.png')


def main() -> None:
    # fig_2_learning_curve()
    # fig_3_one_rate_sets()
    # fig4_covs_axis_plot_v3()
    # fig_5_dyn_demo()
    # fig_6_stchdemolrg()
    # fig_S1()

    # data_folder = f'{PLOT_FOLDER}/ananikov/task1_data'
    # for fname in os.listdir(data_folder):
    #     ananikov_sol_plot(f'{data_folder}/{fname}', f'{PLOT_FOLDER}/ananikov/task1', xtop=240.)

    # data_folder = f'{PLOT_FOLDER}/ananikov/task2_data'
    # for fname in os.listdir(data_folder):
    #     ananikov_sol_plot(f'{data_folder}/{fname}', f'{PLOT_FOLDER}/ananikov/task2', xtop=100.)

    # fig_n1_k2k5_grid()
    fig_n2_integral_curves()

    # exp1_steady_state_map('exp_libuda_react_div60')
    # exp2_reverse_steady_state_maps('exp_libuda_react_div60')

    # ananikov_thesis()

    # fig3.1
    # df = pd.read_csv(filepath, sep=';')
    # _, data = read_plottof_csv(f'{DATA_FOLDER}/agent_metric_fig31.csv', ret_df=True)
    # data = [data['agent metric x'].to_numpy(), data['agent metric y'].to_numpy()]
    # plot_learning_curve(data, f'{PLOT_FOLDER}/fig31.png', ylim=(-1.e-2, 1.3))

    # fig3.2
    # policy_out_graphs(f'{DATA_FOLDER}/test_episode_fig32_RL.csv',
    #                   f'{PLOT_FOLDER}/fig32.png',
    #                   together=True)

    # fig3.2 var 2
    # _, data = read_plottof_csv(f'{DATA_FOLDER}/test_episode_fig32_RL.csv', ret_df=True)
    # data = [data['inputB x'], data['inputB y'], data['inputA y'], data['reaction rate y']]
    # plot_several_lines_uni(data, f'{PLOT_FOLDER}/fig32_var2.png',
    #                      xlabel='Time, s', ylabel='Pressure', ylim=[0., 2.5],
    #                      twin_params={'ylim': [0., 0.1], 'ylabel': '$CO_{2}$ formation rate'},)

    # # this works
    # # fig3 (_ variant)
    # plot_learning_curve(datapath=f'{work_folder}/data/fig3/0708_diff_state_5.csv',
    #                 outpath=f'{work_folder}/n_output_scatter.png',
    #                 end_plot=10000, plottype='scatter')

    # # this works
    # # fig4
    # policy_out_graphs(datapath=f'{work_folder}/figures_final_versions_with_data/fig4/fig4__6_1_1.csv',
    #                   outpath=f'{work_folder}/new_plots_here/4_1.png',
    #                   together=True, units='normal')

    # plot_many_procedure(f'{work_folder}/data/220628_variation', out_folder=work_folder)

    # # suplymentary figures
    # multiple_curves('run_RL_out/important_results/220711_diff_states_rews_curves/group_6_1',
    #           f'{work_folder}/sup_fig_group_61.png')

    # # L2001_converge_to_const
    # all_lines_in_folder_same_plot(f'{work_folder}/figures_final_versions_with_data/fig_L2001_converge_to_const',
    #                               f'{work_folder}/fig_L2001_converge_to_const.png',
    #                               to_legend=[''], kind='both',
    #                               colors=['#676767'])

    # for group in ('with_entropy', 'with_exploration', 'with_variable_noise'):
    #     print('\n\n\n')
    #     print(f'group: {group}')
    #     compute_in_folder(f'run_RL_out/important_results/220811_small_exploration/{group}',
    #                   choose_key=lambda s: '.csv' in s)
    #     print('\n\n\n')

    # for group in ('border1', 'border2', 'border3', 'border4', ):
    #     print('\n\n\n')
    #     print(f'group: {group}')
    #     compute_in_folder(f'run_RL_out/important_results/220811_O2_2_border_decrease/{group}',
    #                   choose_key=lambda s: '.csv' in s)
    #     print('\n\n\n')

    # compute_in_folder(f'run_RL_out/important_results/220809_L2001_diff_lims/O2_lim_100',
    #                   choose_key=lambda s: '.csv' in s)

    # for_pres_2305()

    pass


if __name__ == '__main__':
    main()
