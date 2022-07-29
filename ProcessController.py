# import \
#     copy

import scipy.interpolate
import numpy as np
import pandas as pd
# import time

import lib
from test_models import *

# from usable_functions import wrap


class ProcessController:
    """
    ProcessController - very abstract class controlling some chemical or physical process
    General public methods: set_controlled, time_forward, get_process_output,
     integrate_along_history, all plotters, reset
    Additional public methods: process, const_preprocess, get_current_time
    """

    def __init__(self, process_to_control_obj: BaseModel,
                 controlled_names: list = None, output_names: list = None,
                 target_func=None,
                 real_exp=False,
                 supposed_step_count: int = None,
                 supposed_exp_time: float = None):
        self.rng = np.random.default_rng(seed=0)

        self.analyser_dt = 1.  # analyser period, seconds
        self.analyser_delay = 0

        if supposed_step_count is None:
            self.max_step_count = 1000000
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
                    self.controlled_names.append(str(i + 1))  # ... generate names. I am not sure that
                    # it is the best solution
        else:
            assert controlled_names and (controlled_names == process_to_control_obj.names['input'])
            self.controlled_names = controlled_names
        self.controlled_ind = {self.controlled_names[i]: i for i in range(len(self.controlled_names))}
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
                    self.output_names.append(str(i + 1))  # ... generate names. I am not sure that
                    # it is the best solution
        else:
            assert output_names and (output_names == process_to_control_obj.names['output'])
            self.output_names = output_names
        size = np.ceil(self.max_exp_time / self.analyser_dt).astype('int32')
        self.output_history = np.full((size, len(self.output_names)), -1.)
        self.output_history_dt = np.full(size, -1.)

        self.controlling_signals_history[0] = self.controlled

        self.process_to_control = process_to_control_obj
        self.additional_graph = {name: np.zeros_like(self.output_history) for name in self.process_to_control.plot}

        # for execution time measurement
        # self.times = [0, 0, 0]

        self.target_func = target_func
        self.real_exp = real_exp

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
        return res

    #     "easy" way for get_process_output
    #     if time_history.size > 1:
    #         O2 = scipy.interpolate.interp1d(time_history, self.gas_flow_history[self.step_count-5:self.step_count, self.gas_ind['O2']], kind='next', fill_value='extrapolate')
    #         CO = scipy.interpolate.interp1d(time_history, self.gas_flow_history[self.step_count-5:self.step_count, self.gas_ind['CO']], kind='next', fill_value='extrapolate')

    def integrate_along_history(self, time_segment=None, out_name=None,
                                target_mode=False):
        output_time_stamp, output = self.get_process_output()
        if target_mode:
            output = np.apply_along_axis(self.target_func, 0, output)
            assert output.shape[0] == output_time_stamp.shape[0]
        else:
            if output.shape[1] > 1:  # integral only for one output at same time name allowed
                assert out_name is not None
                new_output = None
                for i, name in enumerate(self.output_names):
                    if name == out_name:
                        new_output = output[:, i]
                        break
                output = new_output
                assert output is not None
        if time_segment is None:
            return lib.integral(output_time_stamp, output)
        assert len(time_segment) == 2, 'Time segment should has the form: [brg, end], no another options'
        inds = (time_segment[0] <= output_time_stamp) & (output_time_stamp <= time_segment[1])
        return lib.integral(output_time_stamp[inds], output[inds])

    def plot(self, file_name, time_segment=None, plot_more_function=None, additional_plot=None, plot_mode='together',
             out_name=None):

        if len(self.output_names) == 1:
            out_name = self.output_names[0]
        assert out_name is not None
        inds = self.output_history_dt > -1
        output_time_stamp, output = self.output_history_dt[inds], self.output_history[inds]
        if len(self.output_names) > 1:
            assigned = False
            for i, name in enumerate(self.output_names):
                if name == out_name:
                    output = output[:, i]
                    assigned = True
            assert assigned

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
            # for i in range(self.step_count-1):
            #     if self.controlling_signals_history[i + 1, self.controlled_ind['O2']] > 0:
            #         ax.axvspan(time_history[i], time_history[i+1], facecolor='grey', alpha=0.3)
            # if self.controlling_signals_history[0, self.controlled_ind['O2']] > 0:
            #     ax.axvspan(0, time_history[0], facecolor='grey', alpha=0.3)
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

        if plot_mode == 'together':
            list_to_plot = []
            for i, name in enumerate(self.controlled_names):
                list_to_plot += [output_time_stamp, interp_funcs[i](output_time_stamp), name]
            list_to_plot += [output_time_stamp, output, out_name]
            lib.plot_to_file(*list_to_plot, *additional, xlabel='Time, s', ylabel='?',
                             fileName=file_name, xlim=time_segment, plotMoreFunction=plot_more_function2)
        else:
            list_to_plot = []
            for i, name in enumerate(self.controlled_names):
                list_to_plot += [output_time_stamp, interp_funcs[i](output_time_stamp), name]
            lib.save_to_file(*list_to_plot,
                             output_time_stamp, output, out_name,
                             *additional,
                             fileName=file_name[:file_name.rfind('.')] + '_all_data.csv',)
            lib.plot_to_file(*list_to_plot, xlabel='Time', ylabel='?', save_csv=False,
                             fileName=file_name[:file_name.rfind('.')] + '_in.png',
                             xlim=time_segment, ylim=[0., None],
                             plotMoreFunction=plot_more_function2)
            lib.plot_to_file(output_time_stamp, output, out_name, xlabel='Time', ylabel='?', save_csv=False,
                             fileName=file_name[:file_name.rfind('.')] + '_out.png',
                             xlim=time_segment, ylim=[1.03 * min(np.min(output), 0.), None],
                             plotMoreFunction=plot_more_function2)
            if additional_plot:
                lib.plot_to_file(*additional, xlabel='Time', ylabel='?', save_csv=False,
                                 fileName=file_name[:file_name.rfind('.')] + '_add.png',
                                 xlim=time_segment, ylim=[0., None],
                                 plotMoreFunction=plot_more_function2)
        if time_segment is not None:
            i = (time_segment[0] <= output_time_stamp) & (output_time_stamp <= time_segment[1])
            print('CO output integral =', lib.integral(output_time_stamp[i], output[i]))

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
        self.output_history_dt = np.full_like(self.output_history, -1.)

        self.controlling_signals_history[0] = self.controlled

        self.additional_graph = {name: np.zeros(len(self.output_history)) for name in self.process_to_control.plot}

        self.process_to_control.reset()


def custom_experiment():
    PC = ProcessController(TestModel())
    for i in range(5):
        PC.set_controlled([i + 0.2 * i1 for i1 in range(1, 6)])
        PC.time_forward(30)
    PC.get_process_output()
    for c in '12345678':
        PC.plot(f'PC_plots/example{c}.png', out_name=c, plot_mode='separately')


if __name__ == '__main__':

    custom_experiment()

    pass
