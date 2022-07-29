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


class RL2207_Environment(Environment):
    def __init__(self, PC: ProcessController,
                 model_type: str = None,
                 reward_spec: [str, callable] = None,
                 episode_time=500, time_step=10,
                 state_spec: dict = None,
                 initial_flows: dict = None,
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
        self.episode_time = episode_time

        self.integral = 0.
        self.best_integral = 0.

        self.end_episode = False
        self.success = False
        self.count_episodes = 0

        self.integral_plot = dict()
        self.integral_plot['integral'] = np.full(1000, -1.)
        self.integral_plot['smooth_50_step'] = None
        self.integral_plot['smooth_1000_step'] = None

        if state_spec is None:
            state_spec = dict()
            state_spec['shape'] = (2, 3)
        if 'use_differences' not in state_spec:
            state_spec['use_differences'] = False
        self.state_spec = copy.deepcopy(state_spec)

        self.state_memory = np.zeros((10, 3))

        self.reset_mode = 'normal'  # crutch

        self.reward_type = 'each_step'
        self.reward_name = ''
        self.reward = None
        self.assign_reward(reward_spec)

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
            self.reward = MethodType(get_reward_func(params={'name': reward_spec}), self)
            # elif reward_spec == 'full_ep_mean':
            #     self.reward = MethodType(get_reward_func(params={
            #         'name': 'full_ep_2',
            #         'subtype': 'mean_mode',
            #         'depth': 25,
            #     }), self)
            #     self.reward_type = 'full_ep'
            #
            # elif reward_spec == 'full_ep_median':
            #     self.reward = MethodType(get_reward_func(params={
            #         'name': 'full_ep_2',
            #         'subtype': 'median_mode',
            #         'depth': 25,
            #     }), self)
            #     self.reward_type = 'full_ep'

            # elif reward_spec == 'hybrid':
            #     self.reward = MethodType(get_reward_func(params={
            #         'name': reward_spec,
            #         'subtype': 'mean_mode',
            #         'depth': 25,
            #         'part': 0.9,
            #     }), self)
            #     self.reward_type = 'hybrid'
            # else:
            #     raise ValueError(f'Invalid assignment for the reward function: {reward_spec}')
            self.reward_name = reward_spec

        elif callable(reward_spec):
            self.reward = MethodType(reward_spec, self)
            self.reward_name = 'callable'

    def states(self):
        lower = np.hstack((self.model.get_bounds('min', 'input'),
                           self.model.get_bounds('min', 'output')))
        upper = np.hstack((self.model.get_bounds('max', 'input'),
                           self.model.get_bounds('max', 'output')))
        states_shape = self.state_spec['shape']
        assert states_shape[1] == lower.size
        min_values = np.repeat(lower, states_shape[0])
        max_values = np.repeat(upper, states_shape[0])
        if self.state_spec['use_differences']:
            min_values[1::2] = lower - upper
            max_values[1::2] = -min_values[1::2]
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
            int_arr_size = self.integral_plot['integral'].size
            if self.count_episodes >= int_arr_size:
                new_integral_arr = np.full(int_arr_size + 1000, -1.)
                new_integral_arr[:self.count_episodes] = self.integral_plot['integral']
                self.integral_plot['integral'] = new_integral_arr
            self.integral_plot['integral'][self.count_episodes] = integral
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

        self.controller.set_controlled(model_inputs)  # normalization implemented
        self.controller.time_forward(dt=self.time_step)
        current_CO2 = self.controller.get_process_output()[1][-1]

        # if self.save_policy:
        #     ind = self.policy_df.shape[0]
        #     if self.model.names['input']:
        #         for i, action_name in enumerate(self.model.names['input']):
        #             self.policy_df.loc[ind, action_name] = model_inputs[i]
        #     self.policy_df.loc[ind, 'time_steps'] = self.delta_t

        self.state_memory[1:] = self.state_memory[:-1]
        self.state_memory[0] = np.array([*model_inputs, current_CO2])
        rows_num = self.state_spec['shape'][0]
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
            in_flows = np.array([self.initial_flows[name] for name in self.in_gas_names])  # unnorm
            if self.hc_preprocess is not None:
                self.controller.const_preprocess(**self.hc_preprocess)
        elif self.reset_mode == 'random':
            in_flows = np.random.random(len(self.in_gas_names))  # normalized
            unnorm_in_flows = in_flows * self.normalize_in_koefs[::2] + self.normalize_in_koefs[1::2]
            to_process_flows = {self.in_gas_names[i]: unnorm_in_flows[i] for i in range(len(self.in_gas_names))}
            self.controller.const_preprocess(to_process_flows, process_time=10)
        else:
            raise ValueError

        self.controller.set_controlled(in_flows * self.normalize_in_koefs[::2] + self.normalize_in_koefs[1::2])  # denormalization
        self.controller.time_forward(dt=self.time_step)
        current_CO2 = self.controller.get_process_output()[1][-1]
        current_CO2 = (current_CO2 - self.normalize_out_koefs[1]) / self.normalize_out_koefs[0]  # normalization implemented

        self.state_memory[0] = np.array([*in_flows, current_CO2])
        self.state_memory[1:] = self.state_memory[0]
        rows_num = self.state_spec['shape'][0]
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
        normalized_integral_arr = self.integral_plot['integral'][:self.count_episodes] / self.episode_time
        x_vector = np.arange(normalized_integral_arr.size)
        self.integral_plot['smooth_50_step'] = simpleSmooth(x_vector[::20],
                                                            normalized_integral_arr[::20],
                                                            50, kernel='Gauss',
                                                            expandParams={'stableReflMeanCount': 10})
        self.integral_plot['smooth_1000_step'] = simpleSmooth(x_vector[::20],
                                                              normalized_integral_arr[::20],
                                                              1000, kernel='Gauss',
                                                              expandParams={'stableReflMeanCount': 10})

        df = pd.DataFrame(columns=('n_integral', 'smooth_50_step', 'smooth_1000_step'), index=x_vector)
        df['n_integral'] = normalized_integral_arr
        df.loc[:self.integral_plot['smooth_50_step'].size - 1, 'smooth_50_step'] = self.integral_plot['smooth_50_step']
        df.loc[:self.integral_plot['smooth_1000_step'].size - 1, 'smooth_1000_step'] = self.integral_plot['smooth_1000_step']
        df.to_csv(f'{folder}integral_by_step.csv', sep=';', index=False)
        ax.plot(x_vector, normalized_integral_arr, label='integral/episode_time')
        ax.plot(x_vector[::20], self.integral_plot['smooth_50_step'], label='short_smooth')
        ax.plot(x_vector[::20], self.integral_plot['smooth_1000_step'], label='long_smooth')
        ax.legend()
        plt.savefig(f'{folder}output_by_step.png')
        plt.close(fig)


# def normalize_koef(env_obj: MyEnvironment):
#     max_values = env_obj.model.max_values(out='array')
#     # relations = [{'O2': 0.8, 'CO': 0.4}, {'O2': 0.5, 'CO': 0.5}, {'O2': 0.4, 'CO': 0.8}]
#     relations = [{'O2': 1., 'CO': value}
#                  for value in [0.1 * i for i in range(1, 10)]]
#     relations += [{'O2': value, 'CO': 1.}
#                  for value in [0.1 * i for i in range(1, 10)]]
#     koefs = []
#     for d in relations:
#         env_obj.hc.reset()
#         env_obj.hc.set_controlled({'O2': max_values[0] * d['O2'], 'CO': max_values[1] * d['CO']})
#         env_obj.hc.time_forward(env_obj.episode_time)
#         integral = env_obj.hc.integrate_along_history()
#         # assert integral > 0, 'integral should be positive'
#         if integral > 0:
#             koefs.append(1 / integral)
#     return np.min(koefs)
