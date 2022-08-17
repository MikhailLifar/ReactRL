import os
# import datetime
# import gc

import \
    itertools

# import tensorflow as tflow
import numpy as np
# import \
#     tensorforce
from tensorforce.agents import Agent

# from test_models import BaseModel
from test_models import TestModel
from ProcessController import ProcessController

from RL2207_Environment import RL2207_Environment, Environment
import lib

from usable_functions import make_subdir_return_path, make_unique_filename

ADD_PLOTS = ['theta_CO', 'theta_O']


def create_tforce_agent(environment: RL2207_Environment, agent_name, params: dict = None):
    if params is None:
        params = dict()
    if 'memory' not in params:
        if agent_name == 'dqn':
            params['memory'] = 100
        # elif agent_name == 'vpg':
        elif agent_name == 'dpg':
            params['memory'] = 10000
        # elif agent_name == 'ppo':
    if 'batch_size' not in params:
        params['batch_size'] = 16
    agent = Agent.create(agent=agent_name, environment=environment, **params)
    return agent


def run_episode(environment: RL2207_Environment, agent, independent: bool = False, deterministic: bool = False):
    # Initialize episode
    state = environment.reset()
    # print(state)
    terminal = False
    while not terminal:
        # Run episode
        actions = agent.act(states=state, independent=independent, deterministic=deterministic)
        state, terminal, reward = environment.execute(actions=actions)
        print(reward)
        # print(state)
        # print(environment.hc.gas_flow)
        if not independent:
            agent.observe(terminal=terminal, reward=reward)
    print('Episode end')


def debug_run(environment: RL2207_Environment, agent, out_folder=None, n_episodes=10):
    subdir_path = make_subdir_return_path(out_folder)
    environment.reset_mode = 'normal'
    for i in range(n_episodes):
        run_episode(environment, agent)
        if i == n_episodes - 2:
            agent.save(directory=subdir_path + '/agent', format='numpy')
        elif i == n_episodes - 1:
            environment.controller.plot(f'{subdir_path}/{environment.integral:.2f}conversion{i}.png', additional_plot=ADD_PLOTS, plot_mode='separately',
                                        out_name='target')
    agent = Agent.load(directory=f'{subdir_path}/agent', format='numpy', environment=environment)
    run_episode(environment, agent)
    environment.controller.plot(f'{subdir_path}/{environment.integral:.2f}conversion_test.png', additional_plot=ADD_PLOTS, plot_mode='separately',
                                out_name='target')


def run(environment: RL2207_Environment, agent, out_folder='run_RL_out', n_episodes=None, test=False,
        create_unique_folder=True):
    if test:
        if create_unique_folder:
            dir_path = make_subdir_return_path(out_folder, postfix='_test')
        else:
            dir_path = out_folder
        test_run(environment, agent, n_episodes=n_episodes, out_folder=dir_path,
                 deterministic=True)
        return
    if n_episodes is None:
        n_episodes = 30000
    if create_unique_folder:
        dir_path = make_subdir_return_path(out_folder)
    else:
        dir_path = out_folder
    freq = n_episodes // 30  # previously 200
    # Loop over episodes
    environment.describe_to_file(f'{dir_path}/_info.txt')
    environment.reset_mode = 'normal'

    prev_graph_ind = 0
    prev_max_integral = 1e-9

    for i in range(n_episodes):
        # Initialize episode
        run_episode(environment, agent)
        if not (i % freq) or (i > n_episodes - 5):
            # env.create_graphs(i, 'run_RL_out/')
            # --DEBUG--
            # conversion = environment.hc.get_conversion()[1]
            # print(conversion)
            # fig, ax = plt.subplots(1, figsize=(15, 8))
            # ax.plot(conversion)
            # fig.savefig(f'{dir_path}/{environment.integral:.2f}another_conversion{i}.png')
            # plt.close(fig)
            # --DEBUG--
            environment.controller.plot(f'{dir_path}/{environment.integral:.2f}conversion{i}.png',
                                        additional_plot=ADD_PLOTS, plot_mode='separately', out_name='target')
            environment.summary_graphs(f'{dir_path}/')
            prev_graph_ind = i

        if environment.success:
            if (i - prev_graph_ind > 100) or\
                    ((environment.integral - prev_max_integral) / prev_max_integral > 0.07):
                environment.controller.plot(f'{dir_path}/{environment.integral:.2f}conversion{i}.png',
                                            additional_plot=ADD_PLOTS, plot_mode='separately', out_name='target')
                environment.summary_graphs(f'{dir_path}/')
                prev_graph_ind = i

            prev_max_integral = environment.integral
            agent.save(directory=dir_path + '/agent', format='numpy')
    # # folder renaming
    # new_path = make_subdir_return_path(out_folder, prefix=f'{environment.best_integral / environment.episode_time * 100:.2}_')
    # os.rename(dir_path, new_path)
    # dir_path = new_path
    # testing
    environment.episode_time = 500
    test_agent = Agent.load(directory=f'{dir_path}/agent', format='numpy', environment=environment)
    os.makedirs(f'{dir_path}/testing', exist_ok=False)
    os.makedirs(f'{dir_path}/testing_deterministic', exist_ok=False)
    test_run(environment, test_agent, out_folder=f'{dir_path}/testing')
    test_run(environment, test_agent, out_folder=f'{dir_path}/testing_deterministic', deterministic=True)


def test_run(environment: RL2207_Environment, agent, out_folder, n_episodes=None, deterministic=False,
             reset_mode='normal'):
    environment.describe_to_file(f'{out_folder}/_info.txt')
    environment.reset_mode = reset_mode
    # environment.reset_mode = 'normal'
    if n_episodes is None:
        n_episodes = 10
    for i in range(n_episodes):
        run_episode(environment, agent, independent=True, deterministic=deterministic)
        # env.create_graphs(i, 'run_RL_out/')
        environment.controller.plot(f'{out_folder}/{environment.integral:.2f}conversion{i}.png',
                                    additional_plot=ADD_PLOTS, plot_mode='separately', out_name='target')
    # environment.summary_graphs(f'{out_folder}/')
    agent.save(directory=out_folder + '/agent', format='numpy')


def fill_variable_dict(values: list, values_names: tuple):
    variable_dict = dict()
    variable_dict['env'] = dict()
    variable_dict['agent'] = dict()
    variable_dict['model'] = dict()
    for i1, name in enumerate(values_names):
        find = False
        for subset_name in ('model', 'env', 'agent'):
            if f'{subset_name}:' in name:
                find = True
                sub_name = name[name.find(':') + 1:]
                # special cases
                if sub_name == 'state_spec':
                    variable_dict[subset_name][sub_name] = {
                        'rows': values[i1][0],
                        'use_differences': values[i1][1]
                    }
                # general case
                else:
                    variable_dict[subset_name][sub_name] = values[i1]
        # assert find, f'Error! Invalid name: {name}'
        if not find:
            variable_dict[name] = values[i1]

    return variable_dict


def train_list(names: tuple,
               params_variants,
               out_path: str = None,
               const_params: dict = None,
               controller: ProcessController = None,
               unique_folder=True,
               n_episodes=12000):
    # raise NotImplementedError('Need to redefine the method')
    assert len(names) == len(params_variants[0]), 'Error: lengths mismatch'
    # initialize inner dictionaries in const params
    # for further correct behavior
    for subd_name in ('env', 'model', 'agent'):
        if subd_name not in const_params:
            const_params[subd_name] = dict()
    plot_list = []
    if unique_folder:
        out_path = make_subdir_return_path(out_path, with_date=True, unique=True)
    model_obj = controller.process_to_control
    for set_values in params_variants:
        variable_params = fill_variable_dict(set_values, names)
        # model parametrization
        model_obj.reset()
        if len(variable_params['model']) or len(const_params['model']):
            model_obj.assign_and_eval_values(**(variable_params['model']),
                                             **(const_params['model']))
        # environment parametrization
        env_obj = Environment.create(
            environment=RL2207_Environment(controller, **(const_params['env']), **(variable_params['env'])),
            max_episode_timesteps=6000)
        # agent parametrization
        if 'agent_name' in const_params:
            agent_name = const_params['agent_name']
        else:
            agent_name = variable_params['agent_name']
        agent_rl = create_tforce_agent(env_obj, agent_name,
                                       params={**(const_params['agent']), **(variable_params['agent'])})
        # run training
        the_folder = make_subdir_return_path(out_path, name='_', with_date=False, unique=True)
        # describe the agent to file
        with open(f'{the_folder}/_info.txt', 'a') as f:
            f.write(f'----Agent----\n')
            f.write(f'agent: {agent_name}\n')
            agent_params = {**const_params['agent'], **variable_params['agent']}
            for p in agent_params:
                f.write(f'{p}: {agent_params[p]}\n')
        run(env_obj, agent_rl,
            out_folder=the_folder,
            n_episodes=n_episodes, create_unique_folder=False)
        # collect training results
        x_vector = np.arange(env_obj.integral_plot['integral'][:env_obj.count_episodes].size)[::20]
        this_train_ress = env_obj.integral_plot['smooth_1000_step']
        if x_vector.size > this_train_ress.size:
            x_vector = x_vector[:this_train_ress.size]
        label = ''
        for name in variable_params:
            label += f'{variable_params[name]}, '
        plot_list += [x_vector, this_train_ress, {'label': label}]
    filename = make_unique_filename('training_results.png')
    lib.plot_to_file(*plot_list, fileName=f'{out_path}/{filename}.png', title='training results\nfor different parameters',
                     xlabel='step',
                     ylabel='integral/episode_time', )


def train_grid(*value_sets,
               names: tuple,
               **train_list_args):

    assert len(names) == len(value_sets), 'Error: lengths mismatch'
    params_variants = list(itertools.product(*value_sets))
    # if tuple names contains subtuple of names
    contains_tuple = False
    for it in names:
        if isinstance(it, tuple):
            contains_tuple = True
            break
    # if contains...
    if contains_tuple:
        # realisation of grid not for the single parameter,
        # but for the sets of parameters,
        # i. e. creation grid of the form
        # [
        #  [a11, a12, a13..., b11, b12..., ...], [a11, a12, a13..., b21, b22..., ...], [a11, a12, a13..., b31, b32.., ...],
        #  [a21, a22, a23..., b11, b12..., ...], [a21, a22, a23..., b21, b22..., ...], [a21, a22, a23..., b31, b32.., ...],
        #  ]
        for i, _ in enumerate(params_variants):
            new_params_set = []
            for j, it in enumerate(names):
                if isinstance(it, tuple):
                    for k, _ in enumerate(it):
                        new_params_set.append(params_variants[i][j][k])
                else:
                    new_params_set.append(params_variants[i][j])
            params_variants[i] = new_params_set
        new_names = []
        for it in names:
            if isinstance(it, tuple):
                for name in it:
                    new_names.append(name)
            else:
                new_names.append(it)
        names = tuple(new_names)
    train_list(names, params_variants, **train_list_args)


if __name__ == '__main__':

    np.random.seed(100)

    def target(x):
        target_v = np.array([2., 1., 3.])
        return -np.linalg.norm(x - target_v)

    my_env = RL2207_Environment(ProcessController(TestModel(), target_func_to_maximize=target,
                                                  supposed_step_count=100, supposed_exp_time=1000),
                                state_spec={'rows': 1, 'use_differences': False},
                                reward_spec='each_step_base')
    env = Environment.create(environment=my_env,
                             max_episode_timesteps=100000)
    rl_agent = create_tforce_agent(env, 'vpg')
    run(env, rl_agent, test=False, out_folder='run_RL_out/current_training',
        n_episodes=10000)
