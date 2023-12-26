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


class State:
    """
    State(PC) -> state
    PC: ProcessController
    state: agent network input

    Features:
    All the scalar's in the state have or fixed bounds (default),
      or dynamically adjusted
    """
    def __init__(self, PC: ProcessController,
                 shape,
                 history_shift_t=None,
                 transform=None, inner_transform=None, inner=None,
                 info='',
                 dynamic_normalization=None):
        self.PC = PC
        self.model = PC.process_to_control
        self.shape = shape
        self.measurement_len = len(self.model.names['output'])

        if history_shift_t is None:
            history_shift_t = np.zeros(1, dtype=np.float32)
        assert history_shift_t[-1] == 0
        self.history_shift_t = history_shift_t
        self.history_shift_idx = (self.history_shift_t // PC.analyser_dt).astype('int')

        if transform is None:
            transform = lambda s0, i: s0
        self.transform = transform
        self.inner = inner
        if inner_transform is None:
            inner_transform = lambda s0, s1, i: None
        self.inner_transform = inner_transform

        self.dynamic_normalization = dynamic_normalization is not None
        self.dyn_norm_idx = None
        self.dyn_norm_bounds = None
        self.dyn_norm_alpha = None
        if self.dynamic_normalization:
            self.dyn_norm_idx = [i for i, name in enumerate(self.model.names['output'])
                                 if name in dynamic_normalization['names']]
            self.dyn_norm_idx = np.array(self.dyn_norm_idx)
            self.dyn_norm_bounds = np.zeros((2, len(self.dyn_norm_idx)))
            self.dyn_norm_bounds[1] += 1.e-5
            self.dyn_norm_alpha = dynamic_normalization.get('alpha', 0.2)

        # TODO functionality may be extended to support more narrow bounds
        lower = self.model.get_bounds('min', 'output')
        upper = self.model.get_bounds('max', 'output')
        self.bounds = np.vstack((lower, upper))
        if self.dynamic_normalization:
            self.bounds[0, self.dyn_norm_idx] = 0.
            self.bounds[1, self.dyn_norm_idx] = 1.

        self.info = info

    def normalize(self, s0):
        # dynamic normalization update
        renorm_part = s0[self.dyn_norm_idx]
        # self.dyn_norm_bounds[0] = np.min(np.vstack(self.dyn_norm_bounds[0], renorm_part * (1. - self.dyn_norm_alpha * np.sign(renorm_part))))

        if np.any(renorm_part < self.dyn_norm_bounds[0]):
            update_idx = renorm_part < self.dyn_norm_bounds[0]
            self.dyn_norm_bounds[0, update_idx] = renorm_part[update_idx] *\
                                                  (1. - self.dyn_norm_alpha * np.sign(renorm_part[update_idx]))

        if np.any(renorm_part > self.dyn_norm_bounds[1]):
            update_idx = renorm_part > self.dyn_norm_bounds[1]
            self.dyn_norm_bounds[1, update_idx] = renorm_part[update_idx] *\
                                                  (1. + self.dyn_norm_alpha * np.sign(renorm_part[update_idx]))

        s0 = s0.copy()
        # dynamic normalization
        s0[self.dyn_norm_idx] = (renorm_part - self.dyn_norm_bounds[0]) / \
                                (self.dyn_norm_bounds[1] - self.dyn_norm_bounds[0])
        # fixed normalization
        s0 = (s0 - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

        return s0

    def get(self, s0=None):
        if s0 is None:
            PC = self.PC
            last_ind = np.where(PC.output_history_dt == -1)[0][0] - 1
            s0 = self.PC.output_history[last_ind - self.history_shift_idx, :]

        for i, row in enumerate(s0):
            s0[i] = self.normalize(row)

        s1 = self.transform(s0, self.inner)
        self.inner = self.inner_transform(s0, s1, self.inner)
        return s1

    def get_info(self):
        string = self.info + f'shape: {self.shape}\n'
        return string


def get_state(string, PC: ProcessController, **kwargs):
    model = PC.process_to_control

    shape = (len(model.names['output']), )
    times = None
    transform = None
    inner_transform = None
    inner = None
    info = kwargs.get('info', 'vanilla')

    if string == 'LG:CO2&O2&CO':
        transform = lambda s0, i: np.squeeze(s0[0:3])
        info = f'{string}\n'

    elif string == 'LG:(CO2&O2&CO)x(points)':
        n = kwargs['points']
        times = np.arange(n)[::-1] * kwargs.get('step', 3.)
        transform = lambda s0, i: s0[:, 0:3]
        info = f'{string}\n'
        shape = (n, 3)

    return State(PC, shape, times,
                 transform=transform, inner_transform=inner_transform, inner=inner,
                 info=info,
                 dynamic_normalization=kwargs.get('dynamic_normalization', None))


def get_state_sequential():
    raise NotImplementedError


class RL2207_Environment(Environment):
    def __init__(self, PC: ProcessController,
                 state_string,
                 state_args: dict = None,
                 action_spec: [Dict, str] = 'continuous',
                 reward_spec: [str, callable] = None,
                 episode_time=None, time_step=None,
                 reset_mode='bottom_state',
                 init_callback=None,
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

        if state_args is None:
            state_args = dict()
        self.state_obj = get_state(state_string, self.controller, **state_args)

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

        # self.save_policy = False
        # self.policy_df = pd.DataFrame(columns=[*self.in_gas_names, 'time_steps'])

        if init_callback is not None:
            init_callback(self)

        # info
        self.env_info = f'model_type: {self.model_type}\n' \
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
                self.reward = MethodType(get_reward_func(
                    params={'name': reward_spec}
                ), self)
            else:
                self.reward = MethodType(get_reward_func(params={'name': reward_spec}), self)

            self.reward_name = reward_spec

        elif callable(reward_spec):
            self.reward = MethodType(reward_spec, self)
            self.reward_name = 'callable'

    def states(self):
        return dict(type='float', shape=self.state_obj.shape, min_value=0., max_value=1.)

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
        time_step = min(self.episode_time - self.controller.time, time_step)
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
        self.controller.get_process_output()
        return self.state_obj.get()

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

        out = current_measurement
        if len(self.state_obj.shape) > 1:
            out = self.state_obj.get(np.tile(out, (self.state_obj.shape[0], 1)))
        return out

    def describe_to_file(self, filename):
        with open(filename, 'a') as fout:
            fout.write('\n-----State-----\n')
            fout.write(self.state_obj.get_info())
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
