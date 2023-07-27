import copy
import os.path

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter as FormatObj
from matplotlib.ticker import MultipleLocator

import matplotlib.pyplot as plt

import pandas as pd

import lib
from lib import read_plottof_csv

from typing import List


MAIN_FIG_SIZE = (16 / 3 * 2, 9 / 3 * 2)
MAIN_PLOT_FONT_SIZE = 16
MAIN_DPI = 300

matplotlib.rcParams.update(
    {'font.size': MAIN_PLOT_FONT_SIZE, 'mathtext.default': 'regular'})

PLOT_FOLDER = './ARTICLE/article_figures'


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
    """
    Plots return/another agent metric vs episode number

    :param data:
    :param outpath:
    :param xlim:
    :param plottype:
    :param kwargs:
    :return:
    """
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
    fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
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
    lims = kwargs.get('ylim', None)
    if lims is None:
        lims = (min(1.e-2, ydata.min()), ydata.max())
    ax.set_ylim(*lims)
    ax.yaxis.set_major_formatter('{x:.2g}')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1e-1 * lims[1]))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5e-2 * lims[1]))
    ax.tick_params(which='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Episode number')  # alternative: 'номер эпизода'
    ax.set_ylabel('Cumm. output divide on episode length')  # alternative: 'сумарный выход деленный на длину эпизода'
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
        fig, axs = plt.subplots(1, 2, figsize=(MAIN_FIG_SIZE[0], MAIN_FIG_SIZE[1] * 0.7))
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
        fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
        plot_policy(policy_data, ax)
        fig.savefig(f'{base_name}_policy{ext}',
                    dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)

        # plotting output
        fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
        plot_output(out_data, ax)
        fig.savefig(f'{base_name}_output{ext}',
                    dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


def plot_two_epochs(datapath1, datapath2, outpath):
    # data reading
    df1 = read_plottof_csv(datapath1, create_standard=False)
    df2 = read_plottof_csv(datapath2, create_standard=False)

    fig, axs = plt.subplots(2, 2, figsize=MAIN_FIG_SIZE)

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
    fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
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


def sup_fig_0(datapath, outpath, end_plot: int = None):
    files = os.listdir(datapath)
    files = sorted(files)
    fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
    if end_plot is not None:
        end_plot //= 20
    # color_arr = color_arr_from_arr(np.arange(len(files)),
    #                                top_color=(0., 0., 0.),
    #                                bottom_color=(0.75, 0.75, 0.75))
    styles = ['solid', (0, (1, 1)), (0, (3, 3)), (0, (5, 1)), (0, (3, 5, 1, 5))]
    for i, f in enumerate(files):
        data_to_plot = pd.read_csv(f'{datapath}/{f}', sep=';')['smooth_1000_step'].to_numpy()
        if end_plot is None:
            end_plot = data_to_plot.size
        else:
            end_plot = min(data_to_plot.size, end_plot)
        data_to_plot = data_to_plot[:end_plot]
        line_name = f[f.find('_', 2)+1:f.find('!')]
        x_arr = 20 * np.arange(end_plot)
        ax.plot(x_arr, data_to_plot, c=(0., 0., 0.), linestyle=styles[i], label=line_name)
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
        1: {'c': '#aeaeae', 'linestyle': 'dashed', 'label': '$O_{2}$', },
        2: {'c': '#989898', 'linestyle': 'dashed', 'label': 'CO', },
        3: {'c': '#555555', 'linestyle': 'solid', 'label': '$CO_{2}$',
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

    if plot_proc_params.get('twin_params', False) is False:
        plot_proc_params.update({'twin_params': {}})

    if ax is None:
        assert outpath is not None
        lib.plot_to_file(*list(reduce(lambda ret, i: ret + [xdata, data[i], styles[i]], range(1, len(data)), [])),
                         fileName=outpath,
                         save_csv=False,
                         **plot_proc_params)
    else:
        lib.plot_in_axis(*list(reduce(lambda ret, i: ret + [xdata, data[i], styles[i]], range(1, len(data)), [])),
                         ax=ax,
                         **plot_proc_params)

    return ax


def fig_learning_curve():
    # _, df = read_plottof_csv(f'{PLOT_FOLDER}/data/curve_fig11.csv', ret_df=True)
    _, df = read_plottof_csv(f'{PLOT_FOLDER}/data/curve_fig31.csv', ret_df=True)
    data = [df['agent metric x'].to_numpy(), df['agent metric y'].to_numpy()]
    # plot_learning_curve(data, f'{PLOT_FOLDER}/fig11.png', ylim=(-1.e-2, 1.3), xlim=8000)
    plot_learning_curve(data, f'{PLOT_FOLDER}/fig31.png', ylim=(-1.e-2, 1.3), xlim=8000)


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


def fig_1_2():
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # RL
    _, df = read_plottof_csv(f'{PLOT_FOLDER}/data/RL_policy_fig12.csv', ret_df=True)

    data = [df['thetaB x'], df['inputB y'], df['inputA y']]

    input_plot(data, axs[0, 0], outpath=f'{PLOT_FOLDER}/fig12_RL_in.png')

    data = [df['thetaB x'], df['thetaB y'], df['thetaA y'], df['reaction rate y']]
    output_plot(data, axs[0, 1], outpath=f'{PLOT_FOLDER}/fig12_RL_thetas.png')

    # NM
    _, df = read_plottof_csv(f'{PLOT_FOLDER}/data/NelderMead_fig12.csv', ret_df=True)

    data = [df['thetaB x'], df['inputB y'], df['inputA y']]
    input_plot(data, axs[1, 0], outpath=f'{PLOT_FOLDER}/fig12_NM_in.png')

    data = [df['thetaB x'], df['thetaB y'], df['thetaA y'], df['reaction rate y']]
    output_plot(data, axs[1, 1], outpath=f'{PLOT_FOLDER}/fig12_NM_thetas.png')

    fig.savefig(f'{PLOT_FOLDER}/fig12_whole.png')


def fig_1_2_v2():
    fig, axs = plt.subplots(1, 2, figsize=MAIN_FIG_SIZE)

    _, df_NM = read_plottof_csv(f'{PLOT_FOLDER}/data/NelderMead_fig12.csv', ret_df=True)
    _, df_RL = read_plottof_csv(f'{PLOT_FOLDER}/data/fig12_RL_50s.csv', ret_df=True)

    data_NM_in = [df_NM['inputB x'], df_NM['inputB y'], df_NM['inputA y']]
    data_RL_in = [df_RL['inputB x'], df_RL['inputB y'], df_RL['inputA y']]

    plot_several_lines_uni(data_NM_in,
                           outpath=None,
                           ax=axs[0],
                           # to_styles={1: {'label': 'inputB', 'linestyle': 'solid'}, 2: {'label': 'inputA', 'linestyle': 'solid'}},  # grey
                           to_styles={1: {'label': None, 'c': '#5588ff', 'linestyle': 'dashed'},
                                      2: {'label': None, 'c': '#ffaa77', 'linestyle': 'dashed'}},  # color
                           ylabel='Pressure', ylim=[0., 1.05],
                           title='input pressures',
                           twin_params=None,
                           )

    plot_several_lines_uni(data_RL_in,
                           outpath=None,
                           ax=axs[0],
                           # to_styles={1: {'label': 'inputB', 'linestyle': 'solid'}, 2: {'label': 'inputA', 'linestyle': 'solid'}},  # grey
                           to_styles={1: {'label': 'inputB', 'c': '#5588ff', 'linestyle': 'solid'},
                                      2: {'label': 'inputA', 'c': '#ffaa77', 'linestyle': 'solid'}},  # color
                           ylabel='Pressure', ylim=[0., 1.05],
                           title='input pressures',
                           twin_params=None,
                           )

    data_RL_out = [df_RL['thetaB x'], df_RL['thetaB y'], df_RL['thetaA y'], df_RL['outputC y']]

    plot_several_lines_uni(data_RL_out,
                           outpath=None,
                           ax=axs[1],
                           # to_styles={1: {'label': 'thetaB'}, 2: {'label': 'thetaA'}, 3: {'label': 'reaction rate'}},
                           to_styles={1: {'label': 'thetaB', 'c': '#5588ff', 'linestyle': 'solid'},
                                      2: {'label': 'thetaA', 'c': '#ffaa77', 'linestyle': 'solid'},
                                      3: {'label': 'reaction rate', 'c': '#44bb44'}},  # color
                           ylabel='Coverage', ylim=[0., 0.5],
                           title='rates & thetas',
                           # twin_params={'ylim': [0., 0.1], 'ylabel': '$CO_{2}$ formation rate'},
                           twin_params={'ylim': [0., 0.1], 'ylabel': 'Reaction rate'},
                           )

    fig.savefig(f'{PLOT_FOLDER}/fig12_v2.png')


def fig_3_2():
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))

    foldpath = f'{PLOT_FOLDER}/data/fig32'
    for i, fname in enumerate(os.listdir(foldpath)):
        _, df = read_plottof_csv(f'{foldpath}/{fname}', ret_df=True)
        input_plot([df['inputB x'], df['inputB y'], df['inputA y']], ax=axs[0, i])
        output_plot([df['thetaB x'], df['thetaB y'], df['thetaA y'], df['reaction rate y']], ax=axs[1, i])

    fig.savefig(f'{PLOT_FOLDER}/fig32_whole.png')


def main_func() -> None:
    # fig_learning_curve()
    # fig_1_2()
    fig_1_2_v2()
    # fig_3_2()

    # fig3.1
    # df = pd.read_csv(filepath, sep=';')
    # _, data = read_plottof_csv(f'{PLOT_FOLDER}/data/agent_metric_fig31.csv', ret_df=True)
    # data = [data['agent metric x'].to_numpy(), data['agent metric y'].to_numpy()]
    # plot_learning_curve(data, f'{PLOT_FOLDER}/fig31.png', ylim=(-1.e-2, 1.3))

    # fig3.2
    # policy_out_graphs(f'{PLOT_FOLDER}/data/test_episode_fig32_RL.csv',
    #                   f'{PLOT_FOLDER}/fig32.png',
    #                   together=True)

    # fig3.2 var 2
    # _, data = read_plottof_csv(f'{PLOT_FOLDER}/data/test_episode_fig32_RL.csv', ret_df=True)
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
    # sup_fig_0('run_RL_out/important_results/220711_diff_states_rews_curves/group_6_1',
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
    main_func()