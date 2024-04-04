import copy
import os
import warnings

import scipy.interpolate
import numpy as np
import pandas as pd
# import time

import lib
from test_models import BaseModel

# from usable_functions import wrap
from predefined_policies import AbstractPolicy


class ProcessController:
    """
    ProcessController - very abstract class controlling some chemical or physical process
    General public methods: set_controlled, time_forward, get_process_output,
     integrate_along_history, all plotters, reset
    Additional public methods: process, const_preprocess, get_current_time
    """

    # class Plotter:
    #     """
    #     Implementing very flexible plotter
    #     It should be capable to plot any combination of any names (no taking into account groups)
    #     with any plot name
    #     It should implement graphs with twin axis
    #     """
    #
    #     def __init__(self, *plots):
    #         """
    #
    #         :param plots:
    #         list of tuples
    #         each tuple is of form (names_to_plot: tuple, twin_name, styles: tuple, twin_style
    #          plot_more_func, plot_name ...)
    #         """
    #         pass
    #
    #     pass

    def __init__(self,
                 process_to_control_obj: BaseModel,
                 analyser_dt: float = 1.,
                 RESOLUTION: int = 10,
                 controlled_names: list = None, output_names: list = None,
                 target_func_to_maximize=None, long_term_target_to_maximize=None, target_func_name='target',
                 real_exp=False,
                 memory_init: int = 1_000, memory_limit: int = 1_000_000,
                 target_int_or_sum: str = 'int'):
        assert (target_func_to_maximize is not None) + (long_term_target_to_maximize is not None) < 2, 'Can specify target or long_term_target, not both'

        self.rng = np.random.default_rng(seed=0)

        # self.analyser_dt = min(analyzer_dt, 0.25 * supposed_exp_time)  # analyser period, seconds
        self.analyser_dt = analyser_dt
        self.analyser_delay = 0.

        self.memory_init = memory_init
        self.memory_limit = memory_limit  # depends on available RAM

        self.RESOLUTION = RESOLUTION

        if controlled_names is None:
            self.controlled_names = process_to_control_obj.names['input']
            if not self.controlled_names:  # if there are not input names...
                for i in range(process_to_control_obj.parameters_count['input']):
                    self.controlled_names.append(f'in{i+1}')  # ... generate names. I am not sure that
                    # it is the best solution
        else:
            assert controlled_names and (controlled_names == process_to_control_obj.names['input'])
            self.controlled_names = controlled_names

        self.controlled_ind = {name: i for i, name in enumerate(self.controlled_names)}
        self.controlled = np.zeros(len(self.controlled_names))
        # self.gas_max_values = 100  # percent
        # gas flow values (one for each constant interval)
        self.controlling_signals_history = np.full((self.memory_init, len(self.controlled_names)), -1.)
        # duration of each constant region in gas_flow_history
        self.controlling_signals_history_dt = np.full(self.memory_init, -1.)
        self.time = 0.

        self.step_count = 0  # current step in flow history

        if output_names is None:
            self.output_names = process_to_control_obj.names['output']
            if not self.output_names:  # if there are not input names...
                for i in range(process_to_control_obj.parameters_count['output']):
                    self.output_names.append(f'out{i+1}')  # ... generate names. I am not sure that
                    # it is the best solution
        else:
            assert output_names and (output_names == process_to_control_obj.names['output'])
            self.output_names = output_names
        self.output_ind = {name: i for i, name in enumerate(self.output_names)}
        self.output_history = np.full((self.memory_init, len(self.output_names)), -1.)
        self.output_history_dt = np.full(self.memory_init, -1.)
        self.target_history = None
        if target_func_to_maximize is not None:
            self.target_history = np.full(self.memory_init, np.nan, dtype=np.float)
        self.long_term_target = long_term_target_to_maximize

        self.controlling_signals_history[0] = self.controlled

        self.process_to_control = process_to_control_obj
        self.additional_graph = {name: np.zeros(self.output_history.shape[0])
                                 for name in self.process_to_control.plot}

        # for execution time measurement
        # self.times = [0, 0, 0]

        self.target_func = target_func_to_maximize
        self.target_int_or_sum = target_int_or_sum
        self.real_exp = real_exp

        # old plot
        # self.plot_lims = dict()
        self.plot_axes_names = dict()
        self.target_func_name = target_func_name
        # self._initialize_plot_params()

        self.metrics_put_on_plot = None

        # 5 23 new plot
        self.plot_specs = None

    def set_controlled(self, new_values):
        """
        :param new_values: array of new values for each controlled parameter
         in the same order as in self.controlled_names,
         or dictionary which keys are controlled parameters
        """
        if isinstance(new_values, dict):
            for name in new_values:
                self.controlled[self.controlled_ind[name]] = new_values[name]
        else:
            for name in self.controlled_names:
                self.controlled[self.controlled_ind[name]] = new_values[self.controlled_ind[name]]

    def time_forward(self, dt=1.):
        """
        Makes time step, dt is measured in seconds
        """
        assert dt > 0
        assert self.step_count < self.memory_limit

        # extend memory if ran out
        self.controlling_signals_history_dt = lib.extend_arr_ax0(self.controlling_signals_history_dt, target_capacity=self.step_count,
                                                                 fill=-1.)
        self.controlling_signals_history = lib.extend_arr_ax0(self.controlling_signals_history, target_capacity=self.step_count)

        i = self.step_count
        if i > 0 and np.all(self.controlling_signals_history[i - 1] == self.controlled):
            self.controlling_signals_history_dt[i - 1] += dt
        else:
            self.controlling_signals_history[i] = self.controlled
            self.controlling_signals_history_dt[i] = dt
            self.step_count += 1
        self.time += dt

    def process(self, time_step_seq: np.ndarray, in_data_df: pd.DataFrame):
        """
        Executes process use sequence of controlling signals from DataFrame

        :param time_step_seq:
        :param in_data_df:
        :return:
        """
        assert time_step_seq.size == in_data_df.shape[0] - 1, 'Error! Length mismatch'
        self.set_controlled({name: in_data_df.loc[0, name] for name in in_data_df.columns})
        for i in range(time_step_seq.size):
            self.time_forward(time_step_seq[i])
            self.set_controlled({name: in_data_df.loc[i + 1, name] for name in in_data_df.columns})

    def process_by_policy_objs(self, policies, episode_time, policy_step):
        time_seq = np.arange(0., episode_time, policy_step)
        controlled_to_pass = np.array([p(time_seq) for p in policies]).transpose()

        for i in range(time_seq.size):
            self.set_controlled(controlled_to_pass[i])
            self.time_forward(policy_step)
        while self.get_current_time() < episode_time:
            self.time_forward(policy_step)

    def const_preprocess(self, in_values, time=300, dt=None):
        """
        Makes updates for the inner process without saving data from this updates in history.
        Method can be used to bring inner process in a desirable state
        before starting experiment/simulation

        :param dt: 
        :param in_values:
        :param time:
        :return:
        """

        if dt is None:
            dt = self.analyser_dt * self.RESOLUTION
        dt = min(time, dt)
        count = int(time / dt)
        if isinstance(in_values, dict):
            in_values_arr = np.array([in_values[name] for name in self.controlled_names])
        else:
            in_values_arr = np.array(in_values)
        for i in range(count):
            self.process_to_control.update(in_values_arr, dt, False)
        return self.process_to_control.model_output

    def get_current_time(self):
        self.time = np.sum(self.controlling_signals_history_dt[:self.step_count])
        return self.time

    def get_current_state(self):
        return self.process_to_control.model_output

    def get_process_output(self):
        """
        Calculates new and returns list of ALL output values from analyzer!
        """
        # t0 = time.time()
        current_time = self.get_current_time()
        measurements_count = int(np.floor(current_time / self.analyser_dt))

        # extend memory if ran out of it
        if measurements_count >= self.output_history_dt.size:
            assert measurements_count < self.memory_limit
            self.output_history_dt = lib.extend_arr_ax0(self.output_history_dt, measurements_count, fill=-1)
            self.output_history = lib.extend_arr_ax0(self.output_history, measurements_count)
            for name, arr in self.additional_graph.items():
                self.additional_graph[name] = lib.extend_arr_ax0(arr, measurements_count, fill=0)
            if self.target_history is not None:
                self.target_history = lib.extend_arr_ax0(self.target_history, measurements_count, fill=np.nan)

        last_ind = np.where(self.output_history_dt == -1)[0][0]
        # last_time = max(0, last_ind-2) * self.analyser_dt
        time_history = np.cumsum(self.controlling_signals_history_dt[:self.step_count])
        # time_history = np.insert(time_history, 0, 0)
        interp_funcs = []
        if time_history.shape[0] > 1:
            for name in self.controlled_names:
                f = scipy.interpolate.interp1d(time_history,
                                               self.controlling_signals_history[:self.step_count, self.controlled_ind[name]],
                                               kind='next', fill_value='extrapolate')
                interp_funcs.append(f)
        else:

            def create_f(name_in_inputs):
                def func(x):
                    return self.controlling_signals_history[self.step_count - 1,
                                                            self.controlled_ind[name_in_inputs]]
                return func

            for name in self.controlled_names:
                interp_funcs.append(create_f(name))

        # t1 = time.time()
        RESOLUTION = self.RESOLUTION

        if not self.real_exp:
            for i in range(last_ind, measurements_count):
                for j in range(RESOLUTION):
                    t = (i-1) * self.analyser_dt + (j + 1) * self.analyser_dt / RESOLUTION - self.analyser_delay
                    self.process_to_control.update(np.array([func(t) for func in interp_funcs]).flatten(), self.analyser_dt / RESOLUTION, True)
                self.output_history[i] = self.process_to_control.model_output
                self.output_history_dt[i] = i * self.analyser_dt
                for name in self.additional_graph:
                    self.additional_graph[name][i] = self.process_to_control.plot[name]
                # debug statements
                # if (i - 60) % 100 == 0:
                #     pass
        else:
            for i in range(last_ind, measurements_count):
                t = (i-1) * self.analyser_dt - self.analyser_delay
                self.process_to_control.update([func(t) for func in interp_funcs], self.analyser_dt, True)
                self.output_history[i] = self.process_to_control.model_output
                self.output_history_dt[i] = i * self.analyser_dt
                for name in self.additional_graph:
                    self.additional_graph[name][i] = self.process_to_control.plot[name]

        # t2 = time.time()
        res = self.output_history_dt[:measurements_count], self.output_history[:measurements_count]
        # t3 = time.time()
        # self.times[0] += t1-t0
        # self.times[1] += t2-t1
        # self.times[2] += t3-t2

        #     "easy" way for get_process_output
        #     if time_history.size > 1:
        #         O2 = scipy.interpolate.interp1d(time_history, self.gas_flow_history[self.step_count-5:self.step_count, self.gas_ind['O2']], kind='next', fill_value='extrapolate')
        #         CO = scipy.interpolate.interp1d(time_history, self.gas_flow_history[self.step_count-5:self.step_count, self.gas_ind['CO']], kind='next', fill_value='extrapolate')

        return res

    def get_target_for_passed(self):
        inds_to_compute = (self.output_history_dt >= 0) & np.isnan(self.target_history)
        if any(inds_to_compute):
            self.target_history[inds_to_compute] = \
                np.apply_along_axis(self.target_func, 1, self.output_history[inds_to_compute])
        idxs = np.isfinite(self.target_history)
        return self.output_history_dt[idxs], self.target_history[idxs]

    def integrate_along_history(self, time_segment=None, out_name=None,
                                target_mode=False,
                                ):
        output_time_stamp, output = self.get_process_output()
        if target_mode:
            output = np.apply_along_axis(self.target_func, 1, output)
            assert output.shape[0] == output_time_stamp.shape[0]
        else:
            if output.shape[1] > 1:  # integral only for one output name at same time allowed
                assert out_name is not None
                new_output = None
                for i, name in enumerate(self.output_names):
                    if name == out_name:
                        new_output = output[:, i]
                        break
                output = new_output
                assert output is not None
            else:
                output = output[:, 0]
        if time_segment is None:
            if self.target_int_or_sum == 'int':
                return lib.integral(output_time_stamp, output)
            else:
                assert self.target_int_or_sum == 'sum'
                return np.sum(output)
        assert len(time_segment) == 2, 'Time segment should has the form: [beg, end], no another options'
        inds = (time_segment[0] <= output_time_stamp) & (output_time_stamp <= time_segment[1])
        if self.target_int_or_sum == 'int':
            return lib.integral(output_time_stamp[inds], output[inds])
        else:
            assert self.target_int_or_sum == 'sum'
            return np.sum(output[inds])

    def get_long_term_target(self, time_segment=None):
        output_dt, output = self.get_process_output()
        if time_segment is None:
            return self.long_term_target(output_dt, output)
        inds = (time_segment[0] <= output_dt) & (output_dt <= time_segment[1])
        return self.long_term_target(output_dt[inds], output[inds])

    def get_cumulative_target(self, time_segment=None):
        if self.target_func is not None:
            return self.integrate_along_history(time_segment, target_mode=True)
        elif self.long_term_target is not None:
            return self.get_long_term_target(time_segment)

    # def _initialize_plot_lims(self):
    #     for kind in ('input', 'output'):
    #         self.plot_lims[kind] = [np.min(self.process_to_control.get_bounds('min', kind)),
    #                                 np.max(self.process_to_control.get_bounds('max', kind))]
    #         self.plot_lims[kind][0] = self.plot_lims[kind][0] * (1. - np.sign(self.plot_lims[kind][0]) * 0.03)
    #         self.plot_lims[kind][1] = self.plot_lims[kind][1] * (1. + np.sign(self.plot_lims[kind][1]) * 0.03)
    #     self.plot_lims['additional'] = None
    #
    # def _initialize_plot_params(self):
    #     self._initialize_plot_lims()
    #     for kind in ('input', 'output', 'additional', 'target'):
    #         self.plot_axes_names[kind] = '?'
    #
    # def set_plot_params(self, **kwargs):
    #     for kind in ('input', 'output', 'additional'):
    #         if f'{kind}_lims' in kwargs:
    #             self.plot_lims[kind] = kwargs[f'{kind}_lims']
    #             del kwargs[f'{kind}_lims']
    #     for kind in ('input', 'output', 'additional', 'target'):
    #         if f'{kind}_ax_name' in kwargs:
    #             self.plot_axes_names[kind] = kwargs[f'{kind}_ax_name']
    #             del kwargs[f'{kind}_ax_name']
    #     if 'target_name' in kwargs:
    #         self.target_func_name = kwargs['target_name']
    #         del kwargs['target_name']
    #     if len(kwargs):
    #         raise ValueError('At least one name contains error')

    def set_metrics(self, *args):
        """
        args: args should have the form: (name1, func1), (name2, func2), ...
        """
        self.metrics_put_on_plot = dict()
        for p in args:
            self.metrics_put_on_plot[p[0]] = p[1]

    def get_info(self):
        s = ''
        for name in ('analyser_dt', 'RESOLUTION', 'target_func_name'):
            s += f'{name}: {getattr(self, name)}\n'
        s += 'controlled names: '
        for name in self.controlled_names:
            s += f'{name} '
        s += '\noutput names: '
        for name in self.output_names:
            s += f'{name} '
        s += '\n'
        return s

    def set_plot_specs(self, *plot_specs):
        self.plot_specs = plot_specs

    def plot_from_spec(self, spec, filepath):
        """

        :param filepath:
        :param spec: {names: , groups: , styles: , to_fname:, to_plot_to_f: {}, with_metrics: ()}
        :return:
        """

        idxs = self.output_history_dt > -1
        t_output = self.output_history_dt[idxs]
        t_controlled = np.cumsum(self.controlling_signals_history_dt[:self.step_count])

        plot_list = []

        names = spec['names']
        len_names = len(names)

        names_group = spec['groups']
        if isinstance(names_group, str):
            names_group = [names_group for _ in names]

        styles = spec['styles']
        if styles is None:
            styles = [None] * len_names
        colors = iter(['b', 'r', 'y', 'g'])
        styles = list(styles)
        for i, s in enumerate(styles):
            if s is None:
                styles[i] = {}
            if not (styles[i].get('c', False) or styles[i].get('color', False)):
                try:
                    styles[i].update({'c': next(colors)})
                except StopIteration:
                    pass

        for name, g, style in zip(names, names_group, styles):
            if style is None:
                style = {}
            if 'label' not in style:
                style.update({'label': name})
            if g == 'input':
                # interp controlled
                if t_controlled.shape[0] > 1:
                    f = scipy.interpolate.interp1d(t_controlled, self.controlling_signals_history[:self.step_count, self.controlled_ind[name]], kind='next', fill_value='extrapolate')
                    plot_list += [t_output, f(t_output), style]
                else:
                    value = self.controlling_signals_history[self.step_count - 1, self.controlled_ind[name]]
                    plot_list += [t_output, np.full_like(t_output, value), style]
            elif g == 'output':
                plot_list += [t_output, self.output_history[idxs, self.output_ind[name]], style]
            elif g == 'add':
                plot_list += [t_output, self.additional_graph[name][idxs], style]
            elif g == 'target':
                target_history = np.apply_along_axis(self.target_func, 1, self.output_history[idxs]).reshape(-1, 1)
                plot_list += [t_output, target_history, style]
            else:
                raise ValueError(f'unknown plot group: {g}')

        filepath, ext = os.path.splitext(filepath)
        lib.plot_to_file(*plot_list, fileName=f'{filepath}{spec["to_fname"]}{ext}', **spec['to_plot_to_f'])
        return plot_list

    def plot(self, filepath):

        # save the information about model and PC object
        d, _ = os.path.split(filepath)
        if not os.path.exists(f'{d}/info.txt'):
            with open(f'{d}/info.txt', 'w') as f:
                f.write('-----Model-----\n')
                f.write(self.process_to_control.get_model_info())
                f.write('\n-----ProcessController-----\n')
                f.write(self.get_info())

        idxs = self.output_history_dt > -1
        common_title = ''
        if self.target_func is not None:
            integral = lib.integral(*self.get_target_for_passed())
            common_title = f'Integral {self.target_func_name}: {integral:.3g}\n'
        if self.long_term_target is not None:
            integral = self.long_term_target(self.output_history_dt[idxs], self.output_history[idxs])
            common_title = f'Target: {integral:.3g}\n'
        for name in self.metrics_put_on_plot:
            common_title += f'{name}: {self.metrics_put_on_plot[name](self.output_history_dt[idxs], self.output_history[idxs]):.3g}, '

        list_to_csv = []
        for spec in self.plot_specs:
            spec['to_plot_to_f']['save_csv'] = False
            spec['to_plot_to_f']['title'] = common_title
            list_to_csv += self.plot_from_spec(spec, filepath)
        for i, d in enumerate(list_to_csv[2::3]):
            list_to_csv[2 + 3 * i] = d['label']
        lib.save_to_file(*list_to_csv,
                         fileName=filepath[:filepath.rfind('.')] + '_all_data.csv', )

    def get_and_plot(self, file_name, plot_params=None):
        """

        :param file_name:
        :param plot_params: deprecated since new plot method
        :return:
        """

        # if isinstance(plot_params, dict):
        #     self.plot(file_name, **plot_params)
        # else:
        #     self.plot(file_name)

        ret = self.get_process_output()
        self.plot(file_name)
        return ret

    def reset(self):
        self.rng = np.random.default_rng(seed=0)

        self.controlled = np.zeros(len(self.controlled_names))
        # gas flow values (one for each constant interval)
        self.controlling_signals_history = np.full((self.memory_init, len(self.controlled_names)), -1.)
        # duration of each constant region in gas_flow_history
        self.controlling_signals_history_dt = np.full(self.memory_init, -1.)
        self.step_count = 0  # current step in flow history
        self.time = 0.

        self.output_history = np.full_like(self.output_history, -1.)
        self.output_history_dt = np.full(self.output_history.shape[0], -1.)
        if self.target_func is not None:
            self.target_history = np.full_like(self.target_history, np.nan, dtype=np.float)

        self.controlling_signals_history[0] = self.controlled

        self.additional_graph = {name: np.zeros(len(self.output_history)) for name in self.process_to_control.plot}

        self.process_to_control.reset()


def create_func_to_approximate(exp_df: pd.DataFrame, model_obj,
                               label_name, in_cols,
                               set_k: list = None, use_k: list = None,
                               rename_dict: dict = None,
                               conv_params: dict = None, ):
    if conv_params is None:
        conv_params = dict()
    df_to_process = exp_df[in_cols]
    if rename_dict is not None:
        df_to_process.rename(columns=rename_dict, inplace=True)
    labels = np.array(exp_df[label_name])
    norm_koef = np.mean(labels)
    time_step_seq = (exp_df.loc[1:, 'Time'].to_numpy()
                     - exp_df.loc[:exp_df.shape[0] - 2, 'Time'].to_numpy())
    labels_x = np.zeros(exp_df.shape[0])
    labels_x[1:] = time_step_seq.cumsum()
    PC_obj = ProcessController(model_obj)
    PC_obj.gas_analyser_delay = 0  # not sure about this

    def func(alphas, DEBUG=False, folder=None, ind_picture=None):

        PC_obj.reset()
        PC_obj.process_to_control.set_params(alphas)
        PC_obj.process(time_step_seq, df_to_process)
        x_conv, conv = PC_obj.get_process_output()
        model_res = np.interp(labels_x, x_conv, conv)
        # model_res *= np.mean(labels) / np.mean(model_res)  # bad way to make values comparable
        if isinstance(set_k, list):
            set_k[0] = np.max(labels) / np.max(model_res)
            model_res = model_res * set_k[0]  # TODO: fix this crutch
        elif isinstance(use_k, list):
            model_res = model_res * use_k[0]
        else:
            model_res = model_res * (np.max(labels) / np.max(model_res))
        res = np.mean(np.abs(model_res - labels)) / norm_koef

        if DEBUG:
            fig, ax = lib.plt.subplots(1, figsize=(15, 8))
            # ax.title('%.2g', %alphas)
            title = f'relative MAE: {res:.3f}\n'
            title += f'model: {PC_obj.process_to_control.model_name}\n'
            for name in PC_obj.process_to_control.params:
                title += '%s=%.3g ' % (name, alphas[name])
            ax.set_title(lib.wrap(title, 100))
            ax.plot(labels_x, labels, label='exp')
            ax.plot(labels_x, model_res, label='model')
            ax.legend()
            if np.isnan(res):
                fig.savefig(f'{folder}/{res}.png')
            else:
                if ind_picture is None:
                    fig.savefig(f'{folder}/{res:.5f}.png')
                else:
                    fig.savefig(f'{folder}/{res:.3f}_iter_{ind_picture}.png')
            lib.plt.close(fig)

        return res

    return func


def create_to_approximate_many_frames(dfs: list, model_obj, label_name, in_cols,
                                      ind_set_k: int = -1,
                                      koefs: np.ndarray = None,
                                      **kargs_glob):
    funcs = []
    koef_here = [-1.]

    assert ind_set_k < len(dfs), 'k is too big'
    for i, df in enumerate(dfs):
        if i == ind_set_k:
            funcs.append(create_func_to_approximate(
                df, model_obj, label_name, in_cols,
                set_k=koef_here, **kargs_glob))
        else:
            funcs.append(create_func_to_approximate(
                df, model_obj, label_name, in_cols,
                use_k=koef_here, **kargs_glob))

    if koefs is None:
        koefs = np.ones(len(funcs))
    koefs = koefs / np.sum(koefs)

    def func(alphas, **kargs_local):
        res = 0
        for k in range(koefs.size):
            if 'ind_picture' in kargs_local:
                new_args = copy.deepcopy(kargs_local)
                new_args['ind_picture'] += f'_exp{k}'
            else:
                new_args = kargs_local
            res += funcs[k](alphas, **new_args) * koefs[k]
        return res

    return func


def func_to_optimize_policy(PC_obj: ProcessController, policy_obj: AbstractPolicy, episode_len, time_step,
                            t_start_count_from: float = 0.,
                            expand_description: callable = None,
                            **kwargs):
    """
    Function-shell for the policy object using in optimization methods.

    :param t_start_count_from:
    :param PC_obj:
    :param policy_obj:
    :param episode_len:
    :param time_step:
    :param expand_description: Function that receives and transforms dict.
        The parameter allows to tide parameters between each other and, so,
        to reduce dimensionality.
        deprecated, replaced by constrains in get_for_repeated_opt_iterations function
    :param kwargs:
    :return:
    """

    import os.path
    time_step, episode_len = float(time_step), float(episode_len)

    def f_to_optimize(func_description: dict, DEBUG=False, folder=None, ind_picture=None):

        if expand_description is not None:
            expand_description(func_description)

        def create_f_per_name(controlled_name, name_ind_in_model):
            controlled_name_dict = dict()
            for param_name in func_description:
                if f'{controlled_name}_' in param_name:
                    controlled_name_dict[param_name.replace(f'{controlled_name}_', '')] = func_description[param_name]

            func = copy.deepcopy(policy_obj)
            func.update_policy(controlled_name_dict)
            func.set_limitations(
                *(PC_obj.process_to_control.limits['input'][name_ind_in_model]),
            )

            return func

        funcs = [create_f_per_name(name, i) for i, name in enumerate(PC_obj.controlled_names)]
        time_seq = np.arange(0., episode_len, time_step)
        controlled_to_pass = np.array([func(time_seq) for func in funcs]).transpose()

        PC_obj.reset()
        for i in range(time_seq.size):
            PC_obj.set_controlled(controlled_to_pass[i])
            PC_obj.time_forward(time_step)
        while PC_obj.get_current_time() < episode_len:
            PC_obj.time_forward(time_step)

        if PC_obj.target_func is not None:
            R = PC_obj.integrate_along_history(target_mode=True,
                                               time_segment=[t_start_count_from, episode_len])
        elif PC_obj.long_term_target is not None:
            R = PC_obj.get_long_term_target()
        else:
            raise ValueError

        if DEBUG:
            # if not os.path.exists(f'{folder}/model_info.txt'):
            #     with open(f'{folder}/model_info.txt', 'w') as f:
            #         f.write(PC_obj.process_to_control.add_info)

            # def ax_func(ax):
            #     # ax.set_title(f'integral: {R:.4g}')
            #     pass

            # PC_obj.plot(f'{folder}/try_{ind_picture}_return_{R:.3f}.png',
            #             plot_more_function=ax_func, plot_mode='separately',
            #             time_segment=[0., episode_len],
            #             **kwargs['to_plot'])  # SMALL CRUTCH HERE!

            PC_obj.plot(f'{folder}/try_{ind_picture}_return_{R:.3f}.png')

        return -R

    return f_to_optimize
