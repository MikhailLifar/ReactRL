import copy
# import warnings
from types import MethodType

import matplotlib
import numpy as np
from typing import List, Dict

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from tensorforce.environments import Environment

from ProcessController import ProcessController

from lib import simpleSmooth

from rewrad_variants import *

from test_models import *


class RL2207_Environment(Environment):
    def __init__(self, PC: ProcessController,
                 names_to_state: list = None,
                 discrete_actions: [Dict[str, List[float]], bool] = None,
                 continuous_actions: [Dict[str, List[float]], bool] = None,
                 reward_spec: [str, callable] = None,
                 episode_time=500, time_step=10,
                 state_spec: dict = None,
                 reset_mode='bottom_state',
                 initial_values: dict = None,
                 preprocess_time=0,
                 log_scaling_dict=None,
                 **kwargs):

        """

        :param PC:
        :param model_type:
        :param reward_spec:
        :param episode_time:
        :param time_step:
        """

        Environment.__init__(self)
        self.controller = PC
        self.model = self.controller.process_to_control
        self.model_type = 'continuous'
        if (hasattr(self.model, 'model_type')) and (self.model.model_type == 'discrete'):
            raise NotImplementedError('Models with discrete inputs are not supported for now')

        if discrete_actions is not None:
            self.actions_type = 'discrete'
        else:
            self.actions_type = 'continuous'
            if (hasattr(self.model, 'model_type')) and (self.model.model_type == 'discrete'):
                raise ValueError(f'Error: discrete model cannot hold continuous actions')

        self.time_step = time_step
        # TODO I don't like this statement
        if self.time_step <= self.controller.analyser_dt:
            self.controller.analyser_dt = self.time_step / 2
        self.episode_time = episode_time

        self.input_names = self.controller.controlled_names
        if initial_values is None:
            self.initial_values = {name: self.model.bottom['input'][name] for name in self.input_names}
        else:
            assert isinstance(initial_values, dict)
            self.initial_values = copy.deepcopy(initial_values)
        if preprocess_time > 0:
            self.PC_preprocess = {'in_flows': self.initial_values,
                                  'process_time': preprocess_time}
            self.controller.const_preprocess(**self.PC_preprocess)
        else:
            self.PC_preprocess = None

        self.cumm_episode_target = 0.
        self.best_episode_target = -np.inf

        self.end_episode = False
        self.success = False
        self.count_episodes = 0

        self.stored_integral_data = dict()
        self.stored_integral_data['integral'] = np.full(1000, -1.)
        self.stored_integral_data['smooth_50_step'] = None
        self.stored_integral_data['smooth_1000_step'] = None

        if names_to_state is None:
            names_to_state = self.model.names['output']
        assert names_to_state
        for name in names_to_state:
            assert name in self.model.names['output'], f'The model does not return the name: {name}'
        self.inds_to_state = [i for i, name in enumerate(self.model.names['output']) if name in names_to_state]
        self.names_to_state = copy.deepcopy(names_to_state)

        one_state_row_len = len(names_to_state)
        # one_state_row_len = len(self.model.limits['input']) + len(self.model.limits['output'])
        if state_spec is None:
            state_spec = dict()
            state_spec['rows'] = 2
        if 'use_differences' not in state_spec:
            state_spec['use_differences'] = False
        state_spec['shape'] = (state_spec['rows'], one_state_row_len)
        self.state_spec = copy.deepcopy(state_spec)

        self.state_memory = np.zeros((2 * self.state_spec['rows'] + 2, one_state_row_len))

        self.discrete_actions = copy.deepcopy(discrete_actions)
        self.continuous_actions = copy.deepcopy(continuous_actions)
        self.action_vector = np.empty(1)
        self.names_of_action = None
        self.idxs_of_action = np.empty(1)

        self.reset_mode = reset_mode

        # self.reward_type = 'each_step'
        self.reward_name = ''
        self.reward = None
        self.assign_reward(reward_spec)

        self.target_type = 'one_row'
        if kwargs['target_type'] == 'episode':
            self.target_type = 'episode'

        # TODO try to generalize normalize_coef evaluation
        if 'normalize_coef' in kwargs:
            self.normalize_coef = kwargs['normalize_coef']
        else:
            self.normalize_coef = normalize_coef(self)
        assert self.normalize_coef >= 0

        # Log preprocessing parameters.
        # Log preprocessing may be used when output values of the process
        # are distributed not uniformly between min and max bounds,
        # exactly when these values are mostly much closer to the min bound.
        # I've tried log scaling with L2001 model, but results were the same as without scaling
        self.if_use_log_scale = log_scaling_dict is not None
        self.to_log_inds = None
        self.to_log_scales = None
        self.norm_to_log = None
        if self.if_use_log_scale:
            to_log_inds_scales = np.array([[self.controller.output_ind[name],
                                                 log_scaling_dict[name]]
                                           for name in self.controller.output_names if name in log_scaling_dict])
            self.to_log_inds = to_log_inds_scales[:, 0]
            self.to_log_scales = to_log_inds_scales[:, 1]
            self.norm_to_log = np.vstack(((self.model.get_bounds('max', 'output') -
                                           self.model.get_bounds('min', 'output'))[self.to_log_inds],
                                          self.model.get_bounds('min', 'output')[self.to_log_inds]))

        # self.save_policy = False
        # self.policy_df = pd.DataFrame(columns=[*self.in_gas_names, 'time_steps'])

        # info
        self.env_info = f'model_type: {self.model_type}\n' \
                        f'actions_type: {self.actions_type}\n' \
                        f'names to state: {self.names_to_state}\n' \
                        f'reward: {self.reward_name}\n' \
                        f'state_spec: shape {self.state_spec["shape"]}, ' \
                        f'use_differences {self.state_spec["use_differences"]}\n' \
                        f'length of episode: {self.episode_time}\n' \
                        f'step length: {self.time_step}\n'

        self.names_to_plot = None
        if 'names_to_plot' in kwargs:
            self.names_to_plot = kwargs['names_to_plot']
        else:
            self.names_to_plot = ['target']

    def assign_reward(self, reward_spec: [str, callable]):
        if reward_spec is None:
            raise ValueError('You should assign reward function!')

        elif isinstance(reward_spec, str):
            if reward_spec in ('full_ep_mean', 'full_ep_median', 'full_ep_max'):
                subtype = reward_spec.split('_')[2]
                self.reward = MethodType(get_reward_func(
                    params={'name': 'full_ep_2', 'subtype': f'{subtype}_mode', 'depth': 25}
                ), self)
            elif reward_spec in ('each_step_base', 'full_ep_base'):
                # TODO here is crutch with bias
                bias = 0.02
                if isinstance(self.model, TestModel):
                    bias = -0.3
                self.reward = MethodType(get_reward_func(
                    params={'name': reward_spec, 'bias': bias}
                ), self)
            else:
                self.reward = MethodType(get_reward_func(params={'name': reward_spec}), self)

            # elif reward_spec == 'hybrid':
            #     self.reward = MethodType(get_reward_func(params={
            #         'name': reward_spec,
            #         'subtype': 'mean_mode',
            #         'depth': 25,
            #         'part': 0.9,
            #     }), self)
            #     self.reward_type = 'hybrid'

            self.reward_name = reward_spec

        elif callable(reward_spec):
            self.reward = MethodType(reward_spec, self)
            self.reward_name = 'callable'

    def log_preprocess(self, measurement: np.ndarray):
        part_to_preprocess = (measurement[self.to_log_inds] -
                                          self.norm_to_log[1]) / self.norm_to_log[0]
        measurement[self.to_log_inds] = np.log(1 + self.to_log_scales * part_to_preprocess)

    def states(self):
        lower = self.model.get_bounds('min', 'output')[self.inds_to_state]
        upper = self.model.get_bounds('max', 'output')[self.inds_to_state]
        if self.if_use_log_scale:
            lower[self.to_log_inds] = 0.
            upper[self.to_log_inds] = np.log(1. + self.to_log_scales) + 1e-5
        # print(self.model.get_bounds('min', 'input'))
        # print(self.model.get_bounds('min', 'output'))
        states_shape = self.state_spec['shape']
        assert states_shape[1] == lower.size
        min_values = np.repeat(lower.reshape(1, -1), states_shape[0], axis=0)
        max_values = np.repeat(upper.reshape(1, -1), states_shape[0], axis=0)
        if self.state_spec['use_differences']:
            min_values[1::2, :] = lower - upper
            max_values[1::2, :] = -min_values[1::2]
        return dict(type='float', shape=states_shape,
                    min_value=min_values, max_value=max_values)

    def actions(self):
        # WARNING: only the models with continuous inputs are supported for now
        min_bounds = self.model.get_bounds('min', 'input')
        max_bounds = self.model.get_bounds('max', 'input')
        inputs_shape = min_bounds.shape
        action_vector = np.full(inputs_shape, -1)
        idxs = []
        if self.actions_type == 'continuous':
            if isinstance(self.continuous_actions, bool):
                self.idxs_of_action = self.action_vector = action_vector
                return dict(type='float', shape=inputs_shape, min_value=min_bounds, max_value=max_bounds)
            lower = np.empty(inputs_shape)
            upper = np.empty(inputs_shape)
            for i, name in enumerate(self.model.names['input']):
                if name in self.continuous_actions:
                    if isinstance(self.continuous_actions[name], (int, float)):
                        action_vector[i] = self.continuous_actions[name]
                    elif isinstance(self.continuous_actions[name], (list, tuple)):
                        lower[i] = self.continuous_actions[name][0]
                        upper[i] = self.continuous_actions[name][1]
                        for v in (lower[i], upper[i]):
                            assert (min_bounds[i] <= v) and (v <= max_bounds[i])
                        assert lower[i] < upper[i]
                        idxs.append(i)
                    elif self.continuous_actions[name] == 'enable':
                        lower[i] = min_bounds[i]
                        upper[i] = max_bounds[i]
                        idxs.append(i)
                    else:
                        raise ValueError(f'Error: failed to determine what should being done with input name: {name}')
                else:
                    raise ValueError(f'Error: All input names should be specified, but at least one was not: {name}')
            self.action_vector = action_vector
            self.idxs_of_action = np.array(idxs)
            return dict(type='float', shape=(len(idxs)), min_value=lower[idxs], max_value=upper[idxs])

        elif self.actions_type == 'discrete':
            if isinstance(self.discrete_actions, bool):
                # return dict(type='int', shape=inputs_shape, num_values=21)
                raise NotImplementedError
            names_of_action = []
            number_of_actions = []
            for i, name in enumerate(self.model.names['input']):
                if isinstance(self.discrete_actions[name], (int, float)):
                    action_vector[i] = self.discrete_actions[name]
                elif isinstance(self.discrete_actions[name], (list, tuple)):
                    l = len(self.discrete_actions[name])
                    assert l > 1
                    number_of_actions.append(l)
                    idxs.append(i)
                    names_of_action.append(name)
            self.names_of_action = np.array(names_of_action)
            self.action_vector = action_vector
            self.idxs_of_action = np.array(idxs)
            return dict(type='int', shape=(len(self.idxs_of_action)), num_values=max(number_of_actions))

        raise ValueError('Unexpected actions type error')

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def terminal(self):
        if self.controller.get_current_time() >= self.episode_time:
            self.end_episode = True
            self.success = False
            if self.target_type == 'one_row':
                self.cumm_episode_target = self.controller.integrate_along_history([0, self.episode_time],
                                                                                   target_mode=True)
            elif self.target_type == 'episode':
                self.cumm_episode_target = self.controller.get_long_term_target()
            cumm_target = self.cumm_episode_target
            int_arr_size = self.stored_integral_data['integral'].size
            if self.count_episodes >= int_arr_size:
                new_integral_arr = np.full(int_arr_size + 1000, -1.)
                new_integral_arr[:self.count_episodes] = self.stored_integral_data['integral']
                self.stored_integral_data['integral'] = new_integral_arr
            self.stored_integral_data['integral'][self.count_episodes] = cumm_target
            if cumm_target > self.best_episode_target:
                self.best_episode_target = cumm_target
                self.success = True
                print('ATTENTION!')
                print(f'new record: {cumm_target:.2f}')
            self.count_episodes += 1
            # if self.save_policy:
            #     self.policy_df.to_excel(f'policy_store/policy{self.count_episodes}.xlsx')
            #     self.policy_df = pd.DataFrame(columns=self.policy_df.columns)
        return self.end_episode

    def update_env(self, act):
        if self.actions_type == 'continuous':
            if self.idxs_of_action.size < self.action_vector.size:
                model_inputs = self.action_vector
                model_inputs[self.idxs_of_action] = act
            else:
                model_inputs = act
        else:
            # model_inputs = act / 20.
            # if isinstance(self.discrete_actions, bool):
            #     raise NotImplementedError
            # else:
            model_inputs = self.action_vector
            for i, idx in enumerate(self.idxs_of_action):
                model_inputs[idx] = self.discrete_actions[self.names_of_action[i]][act[i]]

        self.controller.set_controlled(model_inputs)
        self.controller.time_forward(dt=self.time_step)
        current_measurement = self.controller.get_process_output()[1][-1][self.inds_to_state]
        if self.if_use_log_scale:
            current_measurement = copy.deepcopy(current_measurement)
            self.log_preprocess(current_measurement)

        # if self.save_policy:
        #     ind = self.policy_df.shape[0]
        #     if self.model.names['input']:
        #         for i, action_name in enumerate(self.model.names['input']):
        #             self.policy_df.loc[ind, action_name] = model_inputs[i]
        #     self.policy_df.loc[ind, 'time_steps'] = self.delta_t

        self.state_memory[1:] = self.state_memory[:-1]
        self.state_memory[0] = np.array([*current_measurement])
        rows_num = self.state_spec['rows']
        if self.state_spec['use_differences']:
            out = np.zeros(self.state_spec['shape'])
            out[::2] = self.state_memory[:rows_num//2 + 1]
            out[1::2] = self.state_memory[:rows_num//2] - self.state_memory[1:rows_num//2 + 1]
            return out
        else:
            return self.state_memory[:rows_num]

    def execute(self, actions):
        next_state = self.update_env(actions)
        terminal = self.terminal()
        reward = self.reward()
        return next_state, terminal, reward

    def reset(self, num_parallel=None):
        # TODO: fix all crutches and make class more abstract
        self.end_episode = False
        self.success = False

        self.controller.reset()

        in_values = None
        current_measurement = None

        if self.reset_mode == 'normal':
            in_values = np.array([self.initial_values[name] for name in self.input_names])
            if self.PC_preprocess is not None:
                self.controller.const_preprocess(**self.PC_preprocess)
        elif self.reset_mode == 'random':
            in_values = np.random.random(len(self.input_names)) * \
                        (self.model.get_bounds('max', 'input') - self.model.get_bounds('min', 'input')) + \
                        self.model.get_bounds('min', 'input')  # unnorm
            to_process_flows = {self.input_names[i]: in_values[i] for i in range(len(self.input_names))}
            self.controller.const_preprocess(to_process_flows, process_time=np.random.randint(1, 11)*self.episode_time)
        elif self.reset_mode == 'bottom_state':
            current_measurement = np.zeros(len(self.names_to_state))
            i0 = 0
            for i, name in enumerate(self.model.names['output']):
                if name in self.names_to_state:
                    current_measurement[i0] = self.model.limits['output'][i][0]
                    i0 += 1
        else:
            raise ValueError

        if self.reset_mode != 'bottom_state':
            self.controller.set_controlled(in_values)
            self.controller.time_forward(dt=self.time_step)
            current_measurement = self.controller.get_process_output()[1][-1][self.inds_to_state]

        self.state_memory[0] = np.array([*current_measurement])
        self.state_memory[1:] = self.state_memory[0]
        rows_num = self.state_spec['rows']
        if self.state_spec['use_differences']:
            assert (rows_num % 2 == 1) and (rows_num > 1), 'If use differences, shape must be (k, 3)' \
                                                           ' where k is odd and k > 1'
            out = np.zeros(self.state_spec['shape'])
            out[::2] = self.state_memory[:rows_num//2 + 1]
            out[1::2] = self.state_memory[:rows_num//2] - self.state_memory[1:rows_num//2 + 1]
            return out
        else:
            return self.state_memory[:rows_num]

    # def create_graphs(self, nomer, folder=''):
    #     time_ax = self.out_data['time']
    #     fig, ax = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [8, 2]})
    #     ax[0].set_title(f'{self.integral:.2f}')
    #     ax[0].plot(time_ax, self.out_data['CO2'])
    #     ax[1].set_title('policy')
    #     ax[1].plot(time_ax, self.out_data['actions'])
    #     plt.savefig(f'{folder}conversion{nomer}.png')
    #     plt.close(fig)

    def describe_to_file(self, filename):
        with open(filename, 'a') as fout:
            fout.write('\n-----Environment-----\n')
            fout.write(self.env_info + '\n')
            fout.write('-----ProcessController-----\n')
            fout.write(self.controller.get_info() + '\n')
            fout.write('-----Model-----\n')
            fout.write(self.model.add_info + '\n')

    def summary_graphs(self, folder=''):
        fig, ax = plt.subplots(1, figsize=(15, 8))
        normalized_integral_arr = self.stored_integral_data['integral'][:self.count_episodes] / self.episode_time
        x_vector = np.arange(normalized_integral_arr.size)
        self.stored_integral_data['smooth_50_step'] = simpleSmooth(x_vector[::20],
                                                                   normalized_integral_arr[::20],
                                                                   50, kernel='Gauss',
                                                                   expandParams={'stableReflMeanCount': 10})
        self.stored_integral_data['smooth_1000_step'] = simpleSmooth(x_vector[::20],
                                                                     normalized_integral_arr[::20],
                                                                     1000, kernel='Gauss',
                                                                     expandParams={'stableReflMeanCount': 10})

        df = pd.DataFrame(columns=('n_integral', 'smooth_50_step', 'smooth_1000_step'), index=x_vector)
        df['n_integral'] = normalized_integral_arr
        df.loc[:self.stored_integral_data['smooth_50_step'].size - 1, 'smooth_50_step'] = self.stored_integral_data['smooth_50_step']
        df.loc[:self.stored_integral_data['smooth_1000_step'].size - 1, 'smooth_1000_step'] = self.stored_integral_data['smooth_1000_step']
        df.to_csv(f'{folder}integral_by_step.csv', sep=';', index=False)
        ax.plot(x_vector, normalized_integral_arr, label='integral/episode_time')
        ax.plot(x_vector[::20], self.stored_integral_data['smooth_50_step'], label='short_smooth')
        ax.plot(x_vector[::20], self.stored_integral_data['smooth_1000_step'], label='long_smooth')
        ax.legend()
        plt.savefig(f'{folder}output_by_step.png')
        plt.close(fig)


def normalize_coef(env_obj):

    if isinstance(env_obj.model, LibudaModel):
        max_inputs = env_obj.model.get_bounds('max', 'input', out='dict')
        # relations = [{'O2': 0.8, 'CO': 0.4}, {'O2': 0.5, 'CO': 0.5}, {'O2': 0.4, 'CO': 0.8}]
        relations = [{'O2': 1., 'CO': value}
                     for value in [0.1 * i for i in range(1, 10)]]
        relations += [{'O2': value, 'CO': 1.}
                     for value in [0.1 * i for i in range(1, 10)]]
        koefs = []
        for d in relations:
            env_obj.controller.reset()
            env_obj.controller.set_controlled({'O2': max_inputs['O2'] * d['O2'], 'CO': max_inputs['CO'] * d['CO']})
            env_obj.controller.time_forward(env_obj.episode_time)
            if env_obj.target_type == 'one_row':
                cumm_target = env_obj.controller.integrate_along_history(target_mode=True)
            else:
                cumm_target = env_obj.controller.get_long_term_target()
            # assert cumm_target > 0, 'cumm. target should be positive'
            if cumm_target > 0:
                koefs.append(1 / cumm_target)
        if len(koefs):
            return np.min(koefs)
        else:
            warnings.warn('Failed to compute coefficient. Default one will be used.')
            return 10.

    elif isinstance(env_obj.model, TestModel):
        target_step_estim = 1 / 3 * np.linalg.norm(env_obj.model.get_bounds('max', 'output')
                                      - env_obj.model.get_bounds('min', 'output'))
        return 1 / target_step_estim / env_obj.episode_time
