import \
    copy
from types import \
    MethodType

import matplotlib
# import \
#     numpy as np

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
                 model_type: str = None,
                 reward_spec: [str, callable] = None,
                 episode_time=500, time_step=10,
                 state_spec: dict = None,
                 initial_values: dict = None,
                 preprocess_time=0):

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

        if model_type is None:
            self.model_type = 'continuous'
        else:
            self.model_type = model_type
        try:
            if self.model.model_type != self.model_type:
                raise ValueError(f'Error: mismatch of model type: only {self.model.model_type} is allowed,\n'
                                 f'but {self.model_type} has been received')
        except AttributeError:
            pass

        self.time_step = time_step
        # TODO I don't like this statement
        if self.time_step <= self.controller.analyser_dt:
            self.controller.analyser_dt = self.time_step / 2
        self.episode_time = episode_time

        self.input_names = self.controller.controlled_names
        if initial_values is None:
            self.initial_values = {name: 0. for name in self.input_names}
        else:
            assert isinstance(initial_values, dict)
            self.initial_values = copy.deepcopy(initial_values)
        if preprocess_time > 0:
            self.PC_preprocess = {'in_flows': self.initial_values,
                                  'process_time': preprocess_time}
            self.controller.const_preprocess(**self.PC_preprocess)
        else:
            self.PC_preprocess = None

        self.integral = 0.
        self.best_integral = -np.inf

        self.end_episode = False
        self.success = False
        self.count_episodes = 0

        self.stored_integral_data = dict()
        self.stored_integral_data['integral'] = np.full(1000, -1.)
        self.stored_integral_data['smooth_50_step'] = None
        self.stored_integral_data['smooth_1000_step'] = None

        one_state_row_len = len(self.model.limits['input']) + len(self.model.limits['output'])
        if state_spec is None:
            state_spec = dict()
            state_spec['rows'] = 2
        if 'use_differences' not in state_spec:
            state_spec['use_differences'] = False
        state_spec['shape'] = (state_spec['rows'], len(self.model.limits['input']) + len(self.model.limits['output']))
        self.state_spec = copy.deepcopy(state_spec)

        self.state_memory = np.zeros((10, one_state_row_len))

        self.reset_mode = 'normal'  # crutch

        # self.reward_type = 'each_step'
        self.reward_name = ''
        self.reward = None
        self.assign_reward(reward_spec)

        # TODO try to generalize normalize_coef evaluation
        self.normalize_coef = normalize_coef(self)
        assert self.normalize_coef >= 0

        # self.save_policy = False
        # self.policy_df = pd.DataFrame(columns=[*self.in_gas_names, 'time_steps'])

        # add info
        self.additional_info = f'model_type: {self.model_type}\n' \
                               f'reward: {self.reward_name}\n' \
                               f'state_spec: shape {self.state_spec["shape"]}, ' \
                               f'use_differences {self.state_spec["use_differences"]}\n' \
                               f'length of episode: {self.episode_time}\n' \
                               f'step length: {self.time_step}\n'

    def assign_reward(self, reward_spec: [str, callable]):
        if reward_spec is None:
            raise ValueError('You should assign reward function!')

        elif isinstance(reward_spec, str):
            if reward_spec in ('full_ep_mean', 'full_ep_median', 'full_ep_max'):
                subtype = reward_spec.split('_')[2]
                self.reward = MethodType(get_reward_func(
                    params={'name': 'full_ep_2', 'subtype': f'{subtype}_mode', 'depth': 25}
                ), self)
                return
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

    def states(self):
        lower = np.hstack((self.model.get_bounds('min', 'input'),
                           self.model.get_bounds('min', 'output')))
        upper = np.hstack((self.model.get_bounds('max', 'input'),
                           self.model.get_bounds('max', 'output')))
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
        lower = self.model.get_bounds('min', 'input')
        upper = self.model.get_bounds('max', 'input')
        if self.model_type == 'continuous':
            actions_shape = lower.shape
            return dict(type='float',
                        shape=actions_shape,
                        min_value=lower,
                        max_value=upper)
        elif self.model_type == 'discrete':
            return dict(type='int', shape=lower.shape, num_values=21)
        raise ValueError('Unexpected actions type error')

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def terminal(self):
        if self.controller.get_current_time() >= self.episode_time:
            self.end_episode = True
            self.success = False
            self.integral = self.controller.integrate_along_history([0, self.episode_time],
                                                                    target_mode=True)
            integral = self.integral
            int_arr_size = self.stored_integral_data['integral'].size
            if self.count_episodes >= int_arr_size:
                new_integral_arr = np.full(int_arr_size + 1000, -1.)
                new_integral_arr[:self.count_episodes] = self.stored_integral_data['integral']
                self.stored_integral_data['integral'] = new_integral_arr
            self.stored_integral_data['integral'][self.count_episodes] = integral
            if integral > self.best_integral:
                self.best_integral = integral
                self.success = True
                print('ATTENTION!')
                print(f'new record: {integral:.2f}')
            self.count_episodes += 1
            # if self.save_policy:
            #     self.policy_df.to_excel(f'policy_store/policy{self.count_episodes}.xlsx')
            #     self.policy_df = pd.DataFrame(columns=self.policy_df.columns)
        return self.end_episode

    def update_env(self, act):
        if self.model_type == 'continuous':
            model_inputs = act
        else:
            # model_inputs = act / 20.
            raise NotImplementedError

        self.controller.set_controlled(model_inputs)
        self.controller.time_forward(dt=self.time_step)
        current_measurement = self.controller.get_process_output()[1][-1]

        # if self.save_policy:
        #     ind = self.policy_df.shape[0]
        #     if self.model.names['input']:
        #         for i, action_name in enumerate(self.model.names['input']):
        #             self.policy_df.loc[ind, action_name] = model_inputs[i]
        #     self.policy_df.loc[ind, 'time_steps'] = self.delta_t

        self.state_memory[1:] = self.state_memory[:-1]
        self.state_memory[0] = np.array([*model_inputs, *current_measurement])
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

        if self.reset_mode == 'normal':
            in_values = np.array([self.initial_values[name] for name in self.input_names])
            if self.PC_preprocess is not None:
                self.controller.const_preprocess(**self.PC_preprocess)
        elif self.reset_mode == 'random':
            in_values = np.random.random(len(self.input_names)) * \
                        (self.model.get_bounds('max', 'input') - self.model.get_bounds('min', 'input')) + \
                        self.model.get_bounds('min', 'input')  # unnorm
            to_process_flows = {self.input_names[i]: in_values[i] for i in range(len(self.input_names))}
            self.controller.const_preprocess(to_process_flows, process_time=10)
        else:
            raise ValueError

        self.controller.set_controlled(in_values)
        self.controller.time_forward(dt=self.time_step)
        current_measurement = self.controller.get_process_output()[1][-1]

        self.state_memory[0] = np.array([*in_values, *current_measurement])
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
        with open(filename, 'w') as fout:
            fout.write('--Environment information--\n')
            fout.write(self.additional_info + '\n')
            fout.write('--Model information--\n')
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
            integral = env_obj.controller.integrate_along_history()
            # assert integral > 0, 'integral should be positive'
            if integral > 0:
                koefs.append(1 / integral)
        return np.min(koefs)

    elif isinstance(env_obj.model, TestModel):
        target_step_estim = 1 / 3 * np.linalg.norm(env_obj.model.get_bounds('max', 'output')
                                      - env_obj.model.get_bounds('min', 'output'))
        return 1 / target_step_estim / env_obj.episode_time
