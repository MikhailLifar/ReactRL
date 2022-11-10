import copy

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

    def __init__(self, process_to_control_obj: BaseModel,
                 controlled_names: list = None, output_names: list = None,
                 target_func_to_maximize=None,
                 long_term_target_to_maximize=None,
                 real_exp=False,
                 supposed_step_count: int = None,
                 supposed_exp_time: float = None):
        self.rng = np.random.default_rng(seed=0)

        self.analyser_dt = 1.  # analyser period, seconds
        self.analyser_delay = 0

        if supposed_step_count is None:
            self.max_step_count = 1000000  # depends on available RAM
        else:
            self.max_step_count = supposed_step_count  # depends on available RAM
        if supposed_exp_time is None:
            self.max_exp_time = 1000000  # sec
        else:
            self.max_exp_time = supposed_exp_time  # sec

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
        self.controlling_signals_history = np.full((self.max_step_count, len(self.controlled_names)), -1.)
        # duration of each constant region in gas_flow_history
        self.controlling_signals_history_dt = np.full(self.max_step_count, -1.)

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
        size = np.ceil(self.max_exp_time / self.analyser_dt).astype('int32')
        self.output_history = np.full((size, len(self.output_names)), -1.)
        self.output_history_dt = np.full(size, -1.)
        self.target_history = None
        if target_func_to_maximize is not None:
            self.target_history = np.full(size, np.nan, dtype=np.float)
        if long_term_target_to_maximize is not None:
            self.long_term_target = long_term_target_to_maximize

        self.controlling_signals_history[0] = self.controlled

        self.process_to_control = process_to_control_obj
        self.additional_graph = {name: np.zeros_like(self.output_history) for name in self.process_to_control.plot}

        # for execution time measurement
        # self.times = [0, 0, 0]

        self.target_func = target_func_to_maximize
        self.real_exp = real_exp

        self.plot_lims = dict()
        self.plot_axes_names = dict()
        self.target_func_name = 'target'
        self._initialize_plot_params()

        self.metrics_put_on_plot = None

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
        assert self.step_count < self.max_step_count
        i = self.step_count
        if i > 0 and np.all(self.controlling_signals_history[i - 1] == self.controlled):
            self.controlling_signals_history_dt[i - 1] += dt
        else:
            self.controlling_signals_history[i] = self.controlled
            self.controlling_signals_history_dt[i] = dt
            self.step_count += 1

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

    def process_by_policy_objs(self, policies, episode_time, resolution):
        time_seq = np.arange(0., episode_time, resolution)
        controlled_to_pass = np.array([p(time_seq) for p in policies]).transpose()

        for i in range(time_seq.size):
            self.set_controlled(controlled_to_pass[i])
            self.time_forward(resolution)
        while self.get_current_time() < episode_time:
            self.time_forward(resolution)

    def const_preprocess(self, in_values, process_time=300, RESOLUTION=20):
        """
        Makes updates for the inner process without saving data from this updates in history.
        Method can be used to bring inner process in a desirable state
        before starting experiment/simulation

        :param in_values:
        :param process_time:
        :param RESOLUTION:
        :return:
        """
        dt = self.analyser_dt * RESOLUTION
        count = int(process_time // dt)
        if isinstance(in_values, dict):
            in_values_arr = np.array([in_values[name] for name in self.controlled_names])
        else:
            in_values_arr = np.array(in_values)
        for i in range(count):
            self.process_to_control.update(in_values_arr, dt, False)

    def get_current_time(self):
        return np.sum(self.controlling_signals_history_dt[:self.step_count])

    def get_process_output(self, RESOLUTION=2):
        """
        Calculates new and returns list of ALL output values from analyzer!
        """
        # t0 = time.time()
        current_time = self.get_current_time()
        measurements_count = int(np.floor(current_time / self.analyser_dt))
        last_ind = np.where(self.output_history == -1)[0][0]
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
        if not self.real_exp:
            for i in range(last_ind, measurements_count):
                for j in range(RESOLUTION):
                    t = (i-1) * self.analyser_dt + (j + 1) * self.analyser_dt / RESOLUTION - self.analyser_delay
                    self.process_to_control.update([func(t) for func in interp_funcs], self.analyser_dt / RESOLUTION, True)
                self.output_history[i] = self.process_to_control.model_output
                self.output_history_dt[i] = i * self.analyser_dt
                for name in self.additional_graph:
                    self.additional_graph[name][i] = self.process_to_control.plot[name]
                # # debug statements
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
        return self.target_history[np.isfinite(self.target_history)]

    def integrate_along_history(self, time_segment=None, out_name=None,
                                target_mode=False, **kwargs):
        if 'RESOLUTION' not in kwargs:
            kwargs['RESOLUTION'] = 2
        output_time_stamp, output = self.get_process_output(kwargs['RESOLUTION'])
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
            return lib.integral(output_time_stamp, output)
        assert len(time_segment) == 2, 'Time segment should has the form: [brg, end], no another options'
        inds = (time_segment[0] <= output_time_stamp) & (output_time_stamp <= time_segment[1])
        return lib.integral(output_time_stamp[inds], output[inds])

    def get_long_term_target(self, **kwargs):
        output_dt, output = self.get_process_output(**kwargs)
        return self.long_term_target(output_dt, output)

    def _initialize_plot_lims(self):
        for kind in ('input', 'output'):
            self.plot_lims[kind] = [np.min(self.process_to_control.get_bounds('min', kind)),
                                    np.max(self.process_to_control.get_bounds('max', kind))]
            self.plot_lims[kind][0] = self.plot_lims[kind][0] * (1. - np.sign(self.plot_lims[kind][0]) * 0.03)
            self.plot_lims[kind][1] = self.plot_lims[kind][1] * (1. + np.sign(self.plot_lims[kind][1]) * 0.03)

    def _initialize_plot_params(self):
        self._initialize_plot_lims()
        for kind in ('input', 'output', 'additional', 'target'):
            self.plot_axes_names[kind] = '?'

    def set_plot_params(self, **kwargs):
        for kind in ('input', 'output'):
            if f'{kind}_lims' in kwargs:
                self.plot_lims[kind] = kwargs[f'{kind}_lims']
                del kwargs[f'{kind}_lims']
        for kind in ('input', 'output', 'additional', 'target'):
            if f'{kind}_ax_name' in kwargs:
                self.plot_axes_names[kind] = kwargs[f'{kind}_ax_name']
                del kwargs[f'{kind}_ax_name']
        if 'target_name' in kwargs:
            self.target_func_name = kwargs['target_name']
            del kwargs['target_name']
        if len(kwargs):
            raise ValueError('At least one name contains error')

    def set_metrics(self, *args):
        """
        args: args should have the form: (name1, func1), (name2, func2), ...
        """
        self.metrics_put_on_plot = dict()
        for p in args:
            self.metrics_put_on_plot[p[0]] = p[1]

    def plot(self, file_name, time_segment=None, plot_more_function=None, additional_plot: [str, list] = None, plot_mode='together',
             out_names=None, with_metrics: bool = True):

        inds = self.output_history_dt > -1
        output_time_stamp, no_change_output = self.output_history_dt[inds], self.output_history[inds]
        output = no_change_output
        if isinstance(out_names, str):
            out_names = [out_names]
        if out_names is None:
            assert len(self.output_names) == 1
            out_names = self.output_names
        # sort as in self.output_names, if 'target' in out_names - target will be the last col
        out_names = [name for name in (self.output_names + ['target', 'long_term_target']) if name in out_names]
        # process out_names as a list
        idxs = [i for i, name in enumerate(self.output_names) if name in out_names]
        if idxs:
            output = output[:, idxs]
        else:
            output = None
        if 'target' in out_names:
            target_history = np.apply_along_axis(self.target_func, 1, no_change_output)
            target_history = target_history.reshape(-1, 1)
            if output is not None:
                output = np.hstack((output, target_history))
            else:
                output = target_history
        if output is None:
            raise ValueError('Cannot assign output names!')

        time_history = np.cumsum(self.controlling_signals_history_dt[:self.step_count])
        interp_funcs = []
        if time_history.shape[0] > 1:
            for name in self.controlled_names:
                f = scipy.interpolate.interp1d(time_history, self.controlling_signals_history[:self.step_count, self.controlled_ind[name]], kind='next', fill_value='extrapolate')
                interp_funcs.append(f)

        else:

            def create_f(name_in_inputs):
                def func(x):
                    value = self.controlling_signals_history[self.step_count - 1,
                                                             self.controlled_ind[name_in_inputs]]
                    return np.full_like(x, value)
                return func

            for name in self.controlled_names:
                interp_funcs.append(lambda x: create_f(name)(x))

        def plot_more_function2(ax):
            ax.set_facecolor('#cacaca')
            if plot_more_function is not None:
                plot_more_function(ax)

        additional = tuple()
        if isinstance(additional_plot, str):
            if additional_plot == 'all_mode':
                for name in self.additional_graph:
                    additional += (output_time_stamp, self.additional_graph[name][:output_time_stamp.shape[0]], name)
            elif additional_plot in self.additional_graph:
                additional += (output_time_stamp, self.additional_graph[additional_plot][:output_time_stamp.shape[0]], additional_plot)
            additional_plot = True
        elif isinstance(additional_plot, list):
            for name in additional_plot:
                if name in self.additional_graph:
                    additional += (output_time_stamp, self.additional_graph[name][:output_time_stamp.shape[0]], name)
            additional_plot = True
        else:
            additional_plot = False

        common_title = ''
        if 'target' in out_names:
            integral = lib.integral(output_time_stamp, self.get_target_for_passed())
            common_title = f'Integral {self.target_func_name}: {integral:.3g}\n'
        if 'long_term_target' in out_names:
            integral = self.long_term_target(output_time_stamp, no_change_output)
            common_title = f'Target: {integral:.3g}\n'
            out_names.remove('long_term_target')
        if with_metrics and self.metrics_put_on_plot is not None:
            for name in self.metrics_put_on_plot:
                common_title += f'{name}: {self.metrics_put_on_plot[name](no_change_output):.3g}, '

        if plot_mode == 'together':
            common_plot_list = []
            for i, name in enumerate(self.controlled_names):
                common_plot_list += [output_time_stamp, interp_funcs[i](output_time_stamp), name]
            for i, name in enumerate(out_names):
                common_plot_list += [output_time_stamp, output[:, i], name]
            lib.plot_to_file(*common_plot_list, *additional, title=common_title, xlabel='Time, s', ylabel='?',
                             fileName=file_name, xlim=time_segment, plotMoreFunction=plot_more_function2)
        else:
            input_plot_list = []
            for i, name in enumerate(self.controlled_names):
                input_plot_list += [output_time_stamp, interp_funcs[i](output_time_stamp), name]
            output_plot_list = []
            for i, name in enumerate(out_names):
                output_plot_list += [output_time_stamp, output[:, i], name]
                # # Check if limits for plotting are correct, it is computationally expensive!
                # if self.plot_lims['input'] is not None:
                #     if (np.min(input_plot_list[-1]) < self.plot_lims['input'][0]) or \
                #             (np.max(input_plot_list[-1]) > self.plot_lims['input'][1]):
                #         self.plot_lims['input'] = None

            # # Check if limits for plotting are correct, it is computationally expensive!
            # if self.plot_lims['output'] is not None:
            #     if (np.min(output) < self.plot_lims['output'][0]) or \
            #             (np.max(output) > self.plot_lims['output'][1]):
            #         self.plot_lims['output'] = None

            lib.save_to_file(*input_plot_list,
                             *output_plot_list,
                             *additional,
                             fileName=file_name[:file_name.rfind('.')] + '_all_data.csv', )
            lib.plot_to_file(*input_plot_list,
                             title=common_title,
                             xlabel='Time, s', ylabel=self.plot_axes_names['input'],
                             save_csv=False,
                             fileName=file_name[:file_name.rfind('.')] + '_in.png',
                             xlim=time_segment, ylim=self.plot_lims['input'],
                             plotMoreFunction=plot_more_function2)
            for i, name in enumerate(out_names):
                # if name == 'target:':
                #     title = common_title
                # else:
                #     integral = self.integrate_along_history(out_name=name)
                #     title = f'Integral {name} output: {integral}'
                lib.plot_to_file(*(output_plot_list[3 * i: 3 * (i + 1)]),
                                 title=common_title,
                                 xlabel='Time, s',
                                 ylabel=self.plot_axes_names['output'],
                                 save_csv=False,
                                 fileName=file_name[:file_name.rfind('.')] + f'_{name}_out.png',
                                 xlim=time_segment, ylim=self.plot_lims['output'],
                                 plotMoreFunction=plot_more_function2)
            if additional_plot and len(additional):
                lib.plot_to_file(*additional,
                                 title=common_title,
                                 xlabel='Time, s',
                                 ylabel=self.plot_axes_names['additional'],
                                 save_csv=False,
                                 fileName=file_name[:file_name.rfind('.')] + '_add.png',
                                 xlim=time_segment,
                                 plotMoreFunction=plot_more_function2)
        # if time_segment is not None:
        #     i = (time_segment[0] <= output_time_stamp) & (output_time_stamp <= time_segment[1])
        #     print('CO output integral =', lib.integral(output_time_stamp[i], output[i]))

    def get_and_plot(self,
                     file_name,
                     plot_params=None,
                     get_params=None):
        if isinstance(get_params, dict):
            self.get_process_output(**get_params)
        else:
            self.get_process_output()
        if isinstance(plot_params, dict):
            self.plot(file_name, **plot_params)
        else:
            self.plot(file_name)

    def reset(self):
        self.rng = np.random.default_rng(seed=0)

        self.controlled = np.zeros(len(self.controlled_names))
        # gas flow values (one for each constant interval)
        self.controlling_signals_history = np.full((self.max_step_count, len(self.controlled_names)), -1.)
        # duration of each constant region in gas_flow_history
        self.controlling_signals_history_dt = np.full(self.max_step_count, -1.)
        self.step_count = 0  # current step in flow history

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
        x_conv, conv = PC_obj.get_process_output(**conv_params)
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
                            **kwargs):
    import os.path
    time_step, episode_len = float(time_step), float(episode_len)

    def f_to_optimize(func_description: dict, DEBUG=False, folder=None, ind_picture=None):

        def create_f_per_name(controlled_name, name_ind_in_model):
            controlled_name_dict = dict()
            for param_name in func_description:
                if f'{controlled_name}_' in param_name:
                    controlled_name_dict[param_name.replace(f'{controlled_name}_', '')] = func_description[param_name]

            func = copy.deepcopy(policy_obj)
            func.set_policy(controlled_name_dict)
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
                                               time_segment=[0., episode_len])

        elif PC_obj.long_term_target is not None:
            R = PC_obj.get_long_term_target()

        if DEBUG:
            if not os.path.exists(f'{folder}/model_info.txt'):
                with open(f'{folder}/model_info.txt', 'w') as f:
                    f.write(PC_obj.process_to_control.add_info)

            def ax_func(ax):
                # ax.set_title(f'integral: {R:.4g}')
                pass

            PC_obj.plot(f'{folder}/try_{ind_picture}_return_{R:.3f}.png',
                        plot_more_function=ax_func, plot_mode='separately',
                        time_segment=[0., episode_len],
                        **kwargs['to_plot'])  # SMALL CRUTCH HERE!
        return -R

    return f_to_optimize


# def func_to_optimize_stationary_sol(PC_obj: ProcessController,
#                                     episode_len):
#     import os
#
#     def f_to_optimize(controlled, DEBUG=False, folder=None, ind_picture=None):
#         if not isinstance(controlled, dict):
#             raise ValueError('Error!')
#         PC_obj.reset()
#         PC_obj.set_controlled(controlled)
#         PC_obj.time_forward(episode_len)
#         R = PC_obj.integrate_along_history(target_mode=True,
#                                            time_segment=[0., episode_len])
#         if DEBUG:
#             if not os.path.exists(f'{folder}/model_info.txt'):
#                 with open(f'{folder}/model_info.txt', 'w') as f:
#                     f.write(PC_obj.process_to_control.add_info)
#
#             def ax_func(ax):
#                 ax.set_title(f'O2: {controlled["O2"]:.2g}, CO: {controlled["CO"]:.2g},\nintegral: {R:.4g}')
#
#             PC_obj.plot(f'{folder}/try_{ind_picture}_return_{R:.3f}.png',
#                         plot_more_function=ax_func, plot_mode='separately',
#                         time_segment=[0., episode_len], additional_plot=['thetaCO', 'thetaO'])
#         return -R
#
#     return f_to_optimize


# def func_to_optimize_sin_sol(PC_obj: ProcessController,
#                              episode_len, dt=1.):
#     import os
#
#     dt, episode_len = float(dt), float(episode_len)
#
#     def f_to_optimize(sin_description, DEBUG=False, folder=None, ind_picture=None):
#         if not isinstance(sin_description, dict):
#             raise ValueError('Error!')
#
#         def create_f_per_name(name, name_ind_in_model):
#             A = sin_description[f'{name}_A']
#             omega = sin_description[f'{name}_omega']
#             alpha = sin_description[f'{name}_alpha']
#             bias = sin_description[f'{name}_bias']
#
#             def func(t):
#                 res = A * np.sin(omega * t + alpha) + bias
#                 lower_bound = PC_obj.process_to_control.limits['input'][name_ind_in_model][0]
#                 upper_bound = PC_obj.process_to_control.limits['input'][name_ind_in_model][1]
#                 res[res < lower_bound] = lower_bound
#                 res[res > upper_bound] = upper_bound
#                 return res
#
#             return func
#
#         funcs = [create_f_per_name(name, i) for i, name in enumerate(PC_obj.controlled_names)]
#         time_seq = np.arange(0., episode_len, dt)
#         controlled_to_pass = np.array([func(time_seq) for func in funcs]).transpose()
#
#         PC_obj.reset()
#         for i in range(time_seq.size):
#             PC_obj.set_controlled(controlled_to_pass[i])
#             PC_obj.time_forward(dt)
#         while PC_obj.get_current_time() < episode_len:
#             PC_obj.time_forward(dt)
#
#         R = PC_obj.integrate_along_history(target_mode=True,
#                                            time_segment=[0., episode_len])
#         if DEBUG:
#             if not os.path.exists(f'{folder}/model_info.txt'):
#                 with open(f'{folder}/model_info.txt', 'w') as f:
#                     f.write(PC_obj.process_to_control.add_info)
#
#             def ax_func(ax):
#                 ax.set_title(f'integral: {R:.4g}')
#
#             PC_obj.plot(f'{folder}/try_{ind_picture}_return_{R:.3f}.png',
#                         plot_more_function=ax_func, plot_mode='separately',
#                         time_segment=[0., episode_len], additional_plot=['thetaCO', 'thetaO'])
#         return -R
#
#     return f_to_optimize

def func_to_optimize_two_step_sol(PC_obj: ProcessController,
                                  episode_len, min_step_len=10.,
                                  **kwargs):
    import os

    def f_to_optimize(two_step_description, DEBUG=False, folder=None, ind_picture=None):
        if not isinstance(two_step_description, dict):
            raise ValueError('Error!')

        controlled_set_1 = [two_step_description[f'{name}_1'] for name in
                            PC_obj.controlled_names]
        controlled_set_2 = [two_step_description[f'{name}_2'] for name in
                            PC_obj.controlled_names]
        t1, t2 = two_step_description['time_1'], two_step_description['time_2']
        t1 = max(t1, min_step_len)
        t2 = max(t2, min_step_len)

        PC_obj.reset()
        while PC_obj.get_current_time() <= episode_len:
            PC_obj.set_controlled(controlled_set_1)
            PC_obj.time_forward(t1)
            PC_obj.set_controlled(controlled_set_2)
            PC_obj.time_forward(t2)

        if PC_obj.target_func is not None:
            R = PC_obj.integrate_along_history(target_mode=True,
                                               time_segment=[0., episode_len])
        elif PC_obj.long_term_target is not None:
            R = PC_obj.get_long_term_target()

        if DEBUG:
            if not os.path.exists(f'{folder}/model_info.txt'):
                with open(f'{folder}/model_info.txt', 'w') as f:
                    f.write(PC_obj.process_to_control.add_info)

            def ax_func(ax):
                # ax.set_title(f'integral: {R:.4g}')
                pass

            PC_obj.plot(f'{folder}/try_{ind_picture}_return_{R:.3f}.png',
                        plot_more_function=ax_func, plot_mode='separately',
                        time_segment=[0., episode_len], **(kwargs['to_plot']))
        return -R

    return f_to_optimize
