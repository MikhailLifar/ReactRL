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

import lib

from rewrad_variants import *

from test_models import *


# class State:
#     def __init__(self, PC, names_to_state, state_spec):
#         pass
#
#     def update(self):
#         pass
#
#     def get(self):
#         pass
#
#     def describe(self):
#         pass


class RL2207_Environment(Environment):
    def __init__(self, PC: ProcessController,
                 names_to_state: list = None,
                 state_spec: dict = None,
                 action_spec: [Dict, str] = 'continuous',
                 reward_spec: [str, callable] = None,
                 episode_time=None, time_step=None,
                 reset_mode='bottom_state',
                 log_scaling_dict=None,
                 init_callback=None,
                 dynamic_normalization=None,
                 **kwargs):

        """

        :param PC:
        :param model_type:
        :param action_spec: {'type', 'transform', 'shape', 'info'}
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

        if isinstance(action_spec, str):
            action_spec = {'type': action_spec}
        if not action_spec.get('transform_action', False):
            action_spec['info'] = 'default - variate all inputs within predefined ranges'
        self.action_spec = copy.deepcopy(action_spec)
        self.transform_action = None
        if self.action_spec['type'] == 'continuous':
            if getattr(self.model, 'model_type', 'continuous') == 'discrete':
                raise ValueError(f'Error: discrete model cannot hold continuous actions')
        elif self.action_spec['type'] != 'discrete':
            raise ValueError(f'Wrong action type: {self.action_spec["type"]}')
        self.time_input_dependence = kwargs.get('time_input_dependence', lambda x, t: x)
        self.input_dt = kwargs.get('input_dt', time_step)

        assert episode_time is not None

        if isinstance(time_step, (float, int)):
            self.time_step = lambda action: time_step
            if time_step <= self.controller.analyser_dt:
                self.controller.analyser_dt = time_step / 10  # TODO potential bugs with analyser dt when time step length is variable
            if time_step < self.input_dt:
                self.input_dt = self.time_step
        elif callable(time_step):
            self.time_step = time_step
        else:
            raise ValueError(f'Wrong value for parameter time_step: {time_step}')
        self.last_actual_time_step = None  # TODO this is crutch

        self.episode_time = episode_time

        self.input_names = self.controller.controlled_names

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

        self.dynamic_normalization = dynamic_normalization is not None
        self.dyn_norm_idx = None
        self.dyn_norm_bounds = None
        self.dyn_norm_alpha = None
        if self.dynamic_normalization:
            self.dyn_norm_idx = [i for i, name in enumerate(self.model.names['output'])
                                 if name in dynamic_normalization['names']]
            self.dyn_norm_bounds = np.zeros((2, len(self.dyn_norm_idx)))
            self.dyn_norm_bounds[1] += 1.e-5
            self.dyn_norm_alpha = dynamic_normalization['alpha']

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

        self.reset_mode = copy.deepcopy(reset_mode)

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

        if init_callback is not None:
            init_callback(self)

        # info
        self.env_info = f'model_type: {self.model_type}\n' \
                        f'names to state: {self.names_to_state}\n' \
                        f'state_spec: shape {self.state_spec["shape"]}, ' \
                        f'use_differences {self.state_spec["use_differences"]}\n' \
                        f'action_type: {self.action_spec["type"]}\n' \
                        f'action_info: {self.action_spec["info"]}\n' \
                        f'reward: {self.reward_name}\n' \
                        f'episode_time: {self.episode_time}\n' \
                        f'time_step: {time_step}\n'

        assert self.normalize_coef >= 0

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

    def dyn_norm_proc(self, full_env_response):
        renorm_part = full_env_response[self.dyn_norm_idx]
        # self.dyn_norm_bounds[0] = np.min(np.vstack(self.dyn_norm_bounds[0], renorm_part * (1. - self.dyn_norm_alpha * np.sign(renorm_part))))

        update_idx = renorm_part < self.dyn_norm_bounds[0]
        if np.any(update_idx):
            self.dyn_norm_bounds[0, update_idx] = renorm_part[update_idx] *\
                                                  (1. - self.dyn_norm_alpha * np.sign(renorm_part[update_idx]))

        update_idx = renorm_part > self.dyn_norm_bounds[1]
        if np.any(update_idx):
            self.dyn_norm_bounds[1, update_idx] = renorm_part[update_idx] *\
                                                  (1. + self.dyn_norm_alpha * np.sign(renorm_part[update_idx]))

        full_env_response[self.dyn_norm_idx] = (renorm_part - self.dyn_norm_bounds[0]) / \
                                               (self.dyn_norm_bounds[1] - self.dyn_norm_bounds[0])

    def states(self):
        lower = self.model.get_bounds('min', 'output')[self.inds_to_state]
        upper = self.model.get_bounds('max', 'output')[self.inds_to_state]

        lower[self.dyn_norm_idx] = 0.
        upper[self.dyn_norm_idx] = 1.

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

        if self.action_spec['type'] == 'continuous':
            if set(self.action_spec.keys()) <= {'type', 'info'}:

                def default_transform(x):
                    return x * (max_bounds - min_bounds) + min_bounds

                self.action_spec['transform_action'] = default_transform
                self.action_spec['shape'] = (len(min_bounds), )

            self.transform_action = self.action_spec['transform_action']
            return dict(type='float', shape=self.action_spec['shape'], min_value=0., max_value=1.)

        else:
            raise NotImplementedError('Discrete actions are yet to be implemented')

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
        time_step = self.time_step(act)
        time_step = min(self.episode_time, self.controller.time + time_step)
        self.last_actual_time_step = time_step
        model_inputs = self.transform_action(act)
        temp = 0.
        while temp < time_step - self.input_dt + 1.e-9:
            model_inputs = self.time_input_dependence(model_inputs, self.controller.time)
            self.controller.set_controlled(model_inputs)
            self.controller.time_forward(dt=self.input_dt)
            temp += self.input_dt
        if temp < time_step:
            self.controller.time_forward(time_step - temp)

        full_env_response = self.controller.get_process_output()[1][-1].copy()
        if self.dynamic_normalization:
            self.dyn_norm_proc(full_env_response)

        current_measurement = full_env_response[self.inds_to_state]
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

        if isinstance(self.reset_mode, dict):
            if self.reset_mode['kind'] == 'predefined':
                preprocess = {k: v for k, v in self.reset_mode if k != 'kind'}
                current_measurement = self.controller.const_preprocess(**preprocess)
            elif self.reset_mode['kind'] == 'random':

                bottom = self.reset_mode.get('bottom', None)
                if bottom is None:
                    bottom = self.model.get_bounds('min', 'input')

                top = self.reset_mode.get('top', None)
                if top is None:
                    top = self.model.get_bounds('max', 'input')

                in_values = np.random.random(len(self.input_names)) * (top - bottom) + bottom

                to_process_flows = {self.input_names[i]: in_values[i] for i in range(len(self.input_names))}
                current_measurement = self.controller.const_preprocess(to_process_flows,
                                                                       time=self.reset_mode.get('time', np.random.randint(1, 4) * self.reset_mode['time_step']),
                                                                       dt=self.reset_mode.get('dt', self.controller.analyser_dt),
                                                                       )
            elif self.reset_mode['kind'] == 'predefined_step':
                self.controller.set_controlled(self.reset_mode['step'])
                self.controller.time_forward(self.reset_mode['time_step'])
                current_measurement = self.controller.get_process_output()[1][-1]
            else:
                raise ValueError
        elif self.reset_mode == 'bottom_state':
            current_measurement = self.model.get_bounds('min', 'output')
        else:
            raise ValueError

        current_measurement = current_measurement[self.inds_to_state]
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

    def describe_to_file(self, filename):
        with open(filename, 'a') as fout:
            fout.write('\n-----Environment-----\n')
            fout.write(self.env_info + '\n')
            fout.write('-----ProcessController-----\n')
            fout.write(self.controller.get_info() + '\n')
            fout.write('-----Model-----\n')
            fout.write(self.model.get_model_info() + '\n')

    def plot_learning_curve(self, folder):
        normalized_integral_arr = self.stored_integral_data['integral'][:self.count_episodes] / self.episode_time
        x_vector = np.arange(normalized_integral_arr.size)
        self.stored_integral_data['smooth_50_step'] = lib.simpleSmooth(x_vector[::20],
                                                                       normalized_integral_arr[::20],
                                                                       50, kernel='Gauss',
                                                                       expandParams={'stableReflMeanCount': 10})
        self.stored_integral_data['smooth_1000_step'] = lib.simpleSmooth(x_vector[::20],
                                                                         normalized_integral_arr[::20],
                                                                         1000, kernel='Gauss',
                                                                         expandParams={'stableReflMeanCount': 10})

        df = pd.DataFrame(columns=('n_integral', 'smooth_50_step', 'smooth_1000_step'), index=x_vector)
        df['n_integral'] = normalized_integral_arr
        df.loc[:self.stored_integral_data['smooth_50_step'].size - 1, 'smooth_50_step'] = self.stored_integral_data['smooth_50_step']
        df.loc[:self.stored_integral_data['smooth_1000_step'].size - 1, 'smooth_1000_step'] = self.stored_integral_data['smooth_1000_step']
        df.to_csv(f'{folder}/integral_by_step.csv', sep=';', index=False)

        lib.plot_to_file(x_vector, normalized_integral_arr, 'integral/episode_time',
                         x_vector[::20], self.stored_integral_data['smooth_50_step'], 'short_smooth',
                         x_vector[::20], self.stored_integral_data['smooth_1000_step'], 'long_smooth',
                         ylim=[0., None], fileName=f'{folder}/output_by_step.png', save_csv=False)


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
