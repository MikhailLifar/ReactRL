import os
# import datetime
# import gc

import itertools

import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tflow
# import tensorforce
from tensorforce.agents import Agent

# from test_models import BaseModel
from test_models import *
from ProcessController import ProcessController

from RL2207_Environment import RL2207_Environment, Environment
import lib

from usable_functions import make_subdir_return_path, make_unique_filename

import PC_setup
import PC_run
import predefined_policies as policies


def create_tforce_agent(environment: RL2207_Environment, agent_name, **params):
    if 'memory' not in params:
        if agent_name == 'dqn':
            params['memory'] = 100
        # elif agent_name == 'vpg':
        elif (agent_name == 'dpg') or (agent_name == 'ddpg'):
            params['memory'] = 10_000
        # elif agent_name == 'ppo':
    if 'batch_size' not in params:
        params['batch_size'] = 16
    agent = Agent.create(agent=agent_name, environment=environment, **params)
    return agent


def run_episode(environment: RL2207_Environment, agent, deterministic: bool = False, reset_callback=None):
    # Initialize episode
    state = environment.reset()

    if reset_callback is not None:
        reset_callback(environment)
    
    # print(state)
    terminal = False
    independent = deterministic
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

    return environment.cumm_episode_target


def debug_run(environment: RL2207_Environment, agent, out_folder=None, n_episodes=10):
    subdir_path = make_subdir_return_path(out_folder)
    environment.reset_mode = 'normal'
    for i in range(n_episodes):
        run_episode(environment, agent)
        if i == n_episodes - 2:
            agent.save(directory=subdir_path + '/agent', format='numpy')
        elif i == n_episodes - 1:
            environment.controller.plot(f'{subdir_path}/{environment.cumm_episode_target:.2f}conversion{i}.png')
    agent = Agent.load(directory=f'{subdir_path}/agent', format='numpy', environment=environment)
    run_episode(environment, agent)
    environment.controller.plot(f'{subdir_path}/{environment.cumm_episode_target:.2f}conversion_test.png')


def run(environment: RL2207_Environment, agent, out_folder='run_RL_out', n_episodes=None, test=False,
        create_unique_folder=True, reset_callback=None, test_callback=None, reset_on_test=False,
        eval_agent=None, eval_period=None, ):

    reset_on_test = reset_callback if reset_on_test else None
    if eval_agent is None:
        eval_agent = lambda agent, env: run_episode(env, agent, deterministic=True, reset_callback=reset_on_test)

    if test:
        if create_unique_folder:
            dir_path = make_subdir_return_path(out_folder, postfix='_test')
        else:
            dir_path = out_folder
        test_run(environment, agent, n_episodes=n_episodes, out_folder=dir_path,
                 deterministic=True, reset_callback=reset_on_test, test_callback=test_callback)
        return
    if n_episodes is None:
        n_episodes = 30000
    if create_unique_folder:
        dir_path = make_subdir_return_path(out_folder)
    else:
        dir_path = out_folder
    # plot_period = 2  # temporary! For debug purposes
    plot_period = n_episodes // 30  # previously 200
    if plot_period < 3:  # TODO: crutch here
        plot_period = 3
    environment.describe_to_file(f'{dir_path}/info.txt')

    prev_graph_ind = 0
    prev_max_integral = -np.inf

    if eval_period is None:
        eval_period = max(3, n_episodes // 10)
    max_agent_metric = -np.inf
    agent_metric_data = []

    # Loop over episodes
    for i in range(n_episodes):
        run_episode(environment, agent, reset_callback=reset_callback)

        if not (i % eval_period) or environment.success:
            agent_metric_data.append([i, eval_agent(agent, environment)])
            if agent_metric_data[-1][1] > max_agent_metric:
                agent.save(directory=dir_path + '/best_agent', format='numpy')
                environment.state_obj.saveDynNorm(f'{dir_path}/best_agent_dyn_norm.npy')
                max_agent_metric = agent_metric_data[-1][1]

        if not (i % plot_period) or (i > n_episodes - 5):
            # env.create_graphs(i, 'run_RL_out/')
            # --DEBUG--
            # conversion = environment.hc.get_conversion()[1]
            # print(conversion)
            # fig, ax = plt.subplots(1, figsize=(15, 8))
            # ax.plot(conversion)
            # fig.savefig(f'{dir_path}/{environment.integral:.2f}another_conversion{i}.png')
            # plt.close(fig)
            # --DEBUG--
            environment.controller.plot(f'{dir_path}/{environment.cumm_episode_target:.2f}conversion{i}.png')
            environment.plot_learning_curve(f'{dir_path}')
            if agent_metric_data:
                arr = np.array(agent_metric_data)
                lib.plot_to_file(arr[:, 0], arr[:, 1],
                                 {'label': 'agent metric', 'c': 'b'},
                                 xlabel='Episode number', ylabel='Agent metric',
                                 ylim=[-1.e-2, None],
                                 fileName=f'{dir_path}/agent_metric.png')
            prev_graph_ind = i

        if environment.success:
            if (i - prev_graph_ind > 100) or\
                    ((environment.cumm_episode_target - prev_max_integral) / prev_max_integral > 0.07):
                environment.controller.plot(f'{dir_path}/{environment.cumm_episode_target:.2f}conversion{i}.png')
                prev_graph_ind = i

            prev_max_integral = environment.cumm_episode_target

    # agent.save(directory=dir_path + '/last_agent', format='numpy')

    # # folder renaming
    # new_path = make_subdir_return_path(out_folder, prefix=f'{environment.best_integral / environment.episode_time * 100:.2}_')
    # os.rename(dir_path, new_path)
    # dir_path = new_path

    # testing
    # NOTE: The value 500 seconds for episode time was used for Libuda like tests.
    # It is not suitable for KMC model
    # environment.episode_time = 500

    ret = dict()
    test_agent = Agent.load(directory=f'{dir_path}/best_agent', format='numpy', environment=environment)
    environment.state_obj.loadDynNorm(f'{dir_path}/best_agent_dyn_norm.npy')
    os.makedirs(f'{dir_path}/testing', exist_ok=False)
    os.makedirs(f'{dir_path}/testing_deterministic', exist_ok=False)
    ret['stochastic_test'] = test_run(environment, test_agent, out_folder=f'{dir_path}/testing',
                                      reset_callback=reset_on_test, test_callback=test_callback)
    ret['deterministic_test'] = test_run(environment, test_agent, out_folder=f'{dir_path}/testing_deterministic', deterministic=True,
                                         reset_callback=reset_on_test, test_callback=test_callback)
    return ret


def test_run(environment: RL2207_Environment, agent, out_folder, n_episodes=None, deterministic=False,
             reset_callback=None, test_callback=None):
    environment.describe_to_file(f'{out_folder}/info.txt')

    reset_mode = environment.reset_mode
    if isinstance(reset_mode, dict) and (reset_mode['kind'] == 'random'):
        environment.reset_mode = 'bottom_state'  # TODO: crutch here

    test_it = 0
    if test_callback is not None:
        cumm_rewards = []
        while test_callback(environment, test_it):
            cumm_rewards.append(run_episode(environment, agent, deterministic=deterministic, reset_callback=reset_callback))
            environment.controller.plot(f'{out_folder}/{environment.cumm_episode_target:.2f}conversion{test_it}.png')
            test_it += 1
        cumm_rewards = np.array(cumm_rewards)

    else:
        if n_episodes is None:
            n_episodes = 10
        cumm_rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            cumm_rewards[i] = run_episode(environment, agent, deterministic=deterministic, reset_callback=reset_callback)
            environment.controller.plot(f'{out_folder}/{environment.cumm_episode_target:.2f}conversion{i}.png')

    agent.save(directory=out_folder + '/agent', format='numpy')

    return {'mean_on_test': np.mean(cumm_rewards), 'max_on_test': np.max(cumm_rewards)}


def run_agents_test(PC: ProcessController, agents_paths, env_params, model_params,
                    **test_run_ops):
    model = PC.process_to_control
    if isinstance(env_params, dict):
        env_params = [env_params] * len(agents_paths)
    if isinstance(model_params, dict):
        model_params = [model_params] * len(agents_paths)
    for path, env_p, model_p in zip(agents_paths, env_params, model_params):
        PC.reset()
        model.assign_and_eval_values(**model_p)
        env = Environment.create(
            environment=RL2207_Environment(PC, **env_p),
            max_episode_timesteps=6000)
        env.actions()
        agent = Agent.load(path, format='numpy', environment=env)
        parent_fold, agent_name = os.path.split(path)
        agent_name = agent_name[agent_name.find('_') + 1:]  # assuming agent folders are like 'agent_/id/'
        test_fold = f'{parent_fold}/test_{agent_name}'
        # os.makedirs(test_fold, exist_ok=False)
        os.makedirs(test_fold, exist_ok=True)
        test_run(env, agent, test_fold, **test_run_ops)


def get_eval_test_agent_on_fixed_t_dependencies(t_dependencies):
    def eval_agent(agent, env):
        original_dependence = env.time_input_dependence
        original_reset = env.reset_mode
        if isinstance(original_reset, dict) and (original_reset['kind'] == 'random'):
            env.reset_mode = 'bottom_state'

        rews = np.empty(len(t_dependencies))
        for i, dependence in enumerate(t_dependencies):
            env.time_input_dependence = dependence
            rews[i] = run_episode(env, agent, deterministic=True, reset_callback=None)
            # env.controller.plot(f'./DEBUG/curve_{i}.png')  # DEBUG only !!!

        env.time_input_dependence = original_dependence
        env.reset_mode = original_reset
        return np.mean(rews)

    test_cache = []

    def test_callback(env, iteration):
        if iteration >= len(t_dependencies):
            env.time_input_dependence = test_cache.pop()
            return False
        if not test_cache:
            test_cache.append(env.time_input_dependence)
        env.time_input_dependence = t_dependencies[iteration]
        return True

    return eval_agent, test_callback


def examine_agent_vs_different_curves(environment, agent, curves, outfolder, **kwargs):
    """
    Note: only for an agent trained to deal with arbitrary CO

    :param agent:
    :param environment:
    :param curves:
    :param outfolder:
    :param kwargs:
    :return:
    """

    cache = []

    def callback_(env, iteration):
        if iteration >= len(curves):
            env.time_input_dependence = cache.pop()
            return False
        if not cache:
            cache.append(env.time_input_dependence)
        env.time_input_dependence = lambda act, t: np.array([act[0], curves[iteration](t)])
        return True

    test_run(environment, agent, out_folder=outfolder, test_callback=callback_, **kwargs)


def agent_test_x_co_control_problem(agent_path, outfolder):
    vary_x_co = {'type': 'continuous',
                 'transform_action': lambda x: np.array([x[0], 1. - x[0]]),
                 'shape': 1,
                 'info': 'control x_co'}
    PC_obj = PC_setup.general_PC_setup('LibudaG')
    my_env = RL2207_Environment(
        PC_obj,
        state_spec={'rows': 1, 'use_differences': False},
        action_spec=vary_x_co,
        names_to_state=['B', 'A', 'outputC'],
        reward_spec='each_step_base',
        target_type='one_row',
        episode_time=50.,
        time_step=5.,
        normalize_coef=1.,
        input_dt=0.1,
    )
    my_env = Environment.create(environment=my_env, max_episode_timesteps=6000)
    my_env.actions()
    agent = Agent.load(agent_path, format='numpy', environment=my_env)
    test_run(my_env, agent, out_folder=outfolder, n_episodes=5, deterministic=True)


def agent_test_both_control_problem(agent_path, outfolder):
    PC_obj = PC_setup.general_PC_setup('LibudaG')
    my_env = RL2207_Environment(
        PC_obj,
        state_spec={'rows': 1, 'use_differences': False},
        names_to_state=['B', 'A', 'outputC'],
        reward_spec='each_step_base',
        target_type='one_row',
        episode_time=50.,
        time_step=5.,
        normalize_coef=1.,
        input_dt=0.1,
    )
    my_env = Environment.create(environment=my_env, max_episode_timesteps=6000)
    my_env.actions()
    agent = Agent.load(agent_path, format='numpy', environment=my_env)
    test_run(my_env, agent, out_folder=outfolder, n_episodes=5, deterministic=True)


def agent_test_both_control_unseen_rates(agent_path, outfolder, ):
    PC_obj = PC_setup.general_PC_setup('LibudaG')
    model = PC_obj.process_to_control

    rates = [
        {'rate_ads_A': 10., 'rate_ads_B': 1., },
        {'rate_ads_A': 2., 'rate_ads_B': 1., },
        {'rate_ads_A': 1., 'rate_ads_B': 1., },
        {'rate_ads_A': 0.5, 'rate_ads_B': 1., },
        {'rate_ads_A': 0.1, 'rate_ads_B': 1., },
        {'rate_ads_A': 1., 'rate_ads_B': 10., },
        {'rate_ads_A': 1., 'rate_ads_B': 2., },
        {'rate_ads_A': 1., 'rate_ads_B': 1., },
        {'rate_ads_A': 1., 'rate_ads_B': 0.5, },
        {'rate_ads_A': 1., 'rate_ads_B': 0.1, },
    ]

    cache = {f'rate_{suff}': model[f'rate_{suff}'] for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')}
    for i, params in enumerate(rates):
        model.set_params(params)
        my_env = RL2207_Environment(
            PC_obj,
            state_spec={'rows': 1, 'use_differences': False},
            names_to_state=['B', 'A', 'outputC'],
            reward_spec='each_step_base',
            target_type='one_row',
            episode_time=50.,
            time_step=5.,
            input_dt=0.1,
            init_callback=PC_run.get_estimate_rate_callback(),
        )
        my_env = Environment.create(environment=my_env, max_episode_timesteps=6000)
        my_env.actions()
        agent = Agent.load(agent_path, format='numpy', environment=my_env, )
        run_episode(my_env, agent, deterministic=True)
        my_env.controller.plot(f'{outfolder}/{my_env.cumm_episode_target:.2f}_try{i}.png')

    model.set_params(cache)


def agent_test_arbitrary_co_problem(agent_path, outfolder):
    # curves = [
    #     policies.ConstantPolicy({'value': 0.75}),
    #     policies.ConstantPolicy({'value': 0.5}),
    #     policies.ConstantPolicy({'value': 0.25}),
    #     policies.TwoStepPolicy({'1': 0., '2': 1., 't1': 5., 't2': 5., }),
    #     policies.TwoStepPolicy({'1': 0., '2': 1., 't1': 10., 't2': 10., }),
    #     policies.TwoStepPolicy({'1': 0.1, '2': 0.6, 't1': 5., 't2': 5., }),
    #     policies.TwoStepPolicy({'1': 0.1, '2': 0.6, 't1': 10., 't2': 10., }),
    #     policies.TwoStepPolicy({'1': 0.6, '2': 1., 't1': 5., 't2': 5., }),
    #     policies.TwoStepPolicy({'1': 0.6, '2': 1., 't1': 10., 't2': 10., }),
    #
    #     # special, test only curves
    #     # TRIANGLE
    #     policies.TrianglePolicy({'1': 0., '2': 1., 't1': 5., 't2': 5., }),
    #     policies.TrianglePolicy({'1': 0., '2': 1., 't1': 10., 't2': 10., }),
    #     policies.TrianglePolicy({'1': 0.2, '2': 0.8, 't1': 5., 't2': 5., }),
    #     policies.TrianglePolicy({'1': 0.2, '2': 0.8, 't1': 10., 't2': 10., }),
    #
    #     # SAW
    #     policies.TrianglePolicy({'1': 1., '2': 0., 't1': 0., 't2': 10., }),
    #     policies.TrianglePolicy({'1': 0., '2': 1., 't1': 0., 't2': 10., }),
    #     policies.TrianglePolicy({'1': 1., '2': 0., 't1': 0., 't2': 5., }),
    #     policies.TrianglePolicy({'1': 0., '2': 1., 't1': 0., 't2': 5., }),
    #
    #     # SIN
    #     policies.SinPolicy({'A': 0.5, 'T': 5., 'alpha': 0., 'bias': 0.5, }),
    #     policies.SinPolicy({'A': 0.5, 'T': 10., 'alpha': 0., 'bias': 0.5, }),
    #     policies.SinPolicy({'A': 0.3, 'T': 5., 'alpha': 0., 'bias': 0.7, }),
    #     policies.SinPolicy({'A': 0.3, 'T': 10., 'alpha': 0., 'bias': 0.7, }),
    #     policies.SinPolicy({'A': 0.3, 'T': 5., 'alpha': 0., 'bias': 0.3, }),
    #     policies.SinPolicy({'A': 0.3, 'T': 10., 'alpha': 0., 'bias': 0.3, }),
    # ]

    curves = [
        policies.AnyStepPolicy({'nsteps': 4,
                                '1': 0.75, '2': 0.45, '3': 1., '4': 0.2,
                                't1': 10., 't2': 10., 't3': 20., 't4': 10.})
    ]

    for co_curve in curves:
        co_curve.update_policy({'t_init': -0.2})

    PC_obj = PC_setup.general_PC_setup('LibudaG')
    PC_obj.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
                                          'thetaA_init': 0., 'thetaB_init': 0.,
                                          'thetaA_max': 0.5, 'thetaB_max': 0.5,
                                          'rate_ads_A': 0.14895, 'rate_ads_B': 0.06594 * 4})
    vary_o2_co_is_arbitrary = {'type': 'continuous',
                               'transform_action': lambda x: x,
                               'shape': 1,
                               'info': 'control O2, CO(t) is random'}
    my_env = RL2207_Environment(
        PC_obj,
        state_spec={'rows': 1, 'use_differences': False},
        names_to_state=['B', 'A', 'outputC'],
        action_spec=vary_o2_co_is_arbitrary,
        reward_spec='each_step_base',
        target_type='one_row',
        episode_time=100.,
        time_step=1.,
        normalize_coef=1.,
        input_dt=0.1,
    )
    my_env = Environment.create(environment=my_env, max_episode_timesteps=6000)
    agent = Agent.load(agent_path, format='numpy', environment=my_env)
    my_env.actions()
    examine_agent_vs_different_curves(my_env, agent, curves, outfolder=outfolder,
                                      deterministic=True)


def test_if_runs(PC_obj, outfolder):
    my_env = RL2207_Environment(
        PC_obj,
        state_string='LG:(CO2&O2&CO)x(points)',
        state_args={
            'dynamic_normalization': {
                'names': 'outputC'
            },
            'points': 3,
        },
        reward_spec='each_step_base',
        target_type='one_row',
        episode_time=500,
        time_step=10,
        rate_estimate=0.1,
    )
    my_env = Environment.create(environment=my_env, max_episode_timesteps=6000)
    agent = create_tforce_agent(my_env, 'vpg')
    test_run(my_env, agent, out_folder=outfolder, n_episodes=5, deterministic=False)


def main():
    # from targets_metrics import get_target_func
    import PC_setup

    np.random.seed(100)

    # TEST RUN

    # def target(x):
    #     target_v = np.array([2., 1., 3.])
    #     return -np.linalg.norm(x - target_v)

    # my_env = RL2207_Environment(ProcessController(TestModel(), target_func_to_maximize=target,
    #                                               supposed_step_count=100, supposed_exp_time=1000),
    #                             state_spec={'rows': 1, 'use_differences': False},
    #                             reward_spec='each_step_base')
    # env = Environment.create(environment=my_env,
    #                          max_episode_timesteps=100000)
    # rl_agent = create_tforce_agent(env, 'vpg')
    # run(env, rl_agent, test=False, out_folder='run_RL_out/current_training',
    #     n_episodes=10000)

    # PRETRAINED AGENT ARCHITECTURE

    # # PC_obj = PC_setup.general_PC_setup('Libuda2001',
    # #                                    ('to_model_constructor', {'Ts': 433.}),
    # #                                    ('to_PC_constructor', {'target_func_to_maximize': lambda x: x[4]}),
    # #                                    )
    # PC_obj = PC_setup.general_PC_setup('LibudaD',
    #                                    ('to_model_constructor', {'Ts': 433.}),
    #                                    ('to_PC_constructor', {'target_func_to_maximize': lambda x: x[4]}),
    #                                    )
    # PC_obj = PC_setup.general_PC_setup('LibudaG')

    # test_if_runs(PC_obj, f'./DEBUG/test_state_class')

    # my_env = RL2207_Environment(
    #     PC_obj,
    #     state_spec={'rows': 1, 'use_differences': False},
    #     names_to_state=['CO2', 'O2(Pa)', 'CO(Pa)'],
    #     reward_spec='full_ep_mean',
    #     target_type='one_row',
    #     episode_time=500,
    #     time_step=10,
    # )

    # my_env = Environment.create(environment=my_env, max_episode_timesteps=6000)
    # # rl_agent = create_tforce_agent(my_env, 'ac',
    # #                                network=dict(type='layered',
    # #                                             layers=[dict(type='flatten'),
    # #                                                     dict(type='dense', size=16, activation='relu')]),
    # #                                critic=dict(type='layered',
    # #                                             layers=[dict(type='flatten'),
    # #                                                     dict(type='dense', size=16, activation='relu')]))
    # rl_agent = create_tforce_agent(my_env, 'vpg')
    # # rl_agent = Agent.load('run_RL_out/agents/220804_LMT_0_agent', format='numpy', environment=my_env)
    # # rl_agent = Agent.load('temp/for_english/stationary_agent', format='numpy', environment=my_env)
    # # rl_agent = Agent.load('temp/for_english/periodic_agent_2', format='numpy', environment=my_env)
    # # test_run(my_env, rl_agent, 'temp/for_english/L2001_test_run', 5, deterministic=True)
    # # test_run(my_env, rl_agent, 'temp/for_english/LD_test_run', 5, deterministic=True)

    # my_env = Environment.create(environment=RL2207_Environment(
    #     PC_obj,
    #     state_spec={'rows': 2, 'use_differences': False},
    #     names_to_state=['B', 'A', 'outputC'],
    #     reward_spec='each_step_base',
    #     target_type='one_row',
    #     episode_time=50.,
    #     time_step=5.,
    #     normalize_coef=1.,
    #     input_dt=0.1,
    # ), max_episode_timesteps=6000)
    # my_env.actions()
    # rl_agent = Agent.load('./231002_sudden_discovery/agent', format='numpy', environment=my_env)
    # print(rl_agent.get_specification())
    # print(rl_agent.get_architecture())

    # agent_test_both_control_problem('ARTICLE/best_stationary_agent/agent', 'ARTICLE/best_stationary_agent/agent_test')
    # agent_test_both_control_unseen_rates('run_RL_out/agent_test/best_stationary_agent/agent_rows1',
    #                                      'run_RL_out/agent_test/best_stationary_agent/agent_test_unseen_rates')
    # agent_test_both_control_unseen_rates('run_RL_out/agent_test/best_stationary_agent/agent_rows3',
    #                                      'run_RL_out/agent_test/best_stationary_agent/agent_test_unseen_rates')
    # agent_test_both_control_unseen_rates('run_RL_out/agent_test/best_both_control_random_start/agent_rows1',
    #                                      'run_RL_out/agent_test/best_both_control_random_start/test_unseen_rates_rows1')
    # agent_test_both_control_unseen_rates('run_RL_out/agent_test/best_both_control_random_start/agent_rows3',
    #                                      'run_RL_out/agent_test/best_both_control_random_start/test_unseen_rates_rows3')
    # agent_test_x_co_control_problem('ARTICLE/best_x_co_agent/agent', 'ARTICLE/best_x_co_agent/agent_test')

    # agent_test_arbitrary_co_problem('./ARTICLE/agents/best_CORTP_no_sampling_agent/230717_CORTPS_it2',
    #                                 './ARTICLE/agents/best_CORTP_no_sampling_agent/agent_test')

    # run_agents_test(PC_setup.get_dyn_adv_LibudaG(),
    #                 agents_paths=[
    #                     'ARTICLE/agents/diff_rates_both_control/agent_dyn_adv',
    #                 ],
    #                 env_params={
    #                     'episode_time': 1000.,
    #                     'time_step': 9.98,
    #                     'names_to_state': ['B', 'A', 'outputC'],
    #                     'state_string': 'LG:(CO2&O2&CO)x(points)',
    #                     'state_args': {'points': 2,
    #                                    'step': 10.,
    #                                    'dynamic_normalization': None,  # TODO don't forget that old agent is incompatible with a new code
    #                                    },
    #                     'reward_spec': 'each_step_base',
    #                     'input_dt': 0.1,
    #                     'target_type': 'one_row',
    #                 },
    #                 model_params=[
    #                     {}
    #                 ],
    #                 n_episodes=1, deterministic=True,
    #                 )

    run_agents_test(PC_setup.get_dyn_adv_LibudaG(),
                    agents_paths=[
                        # 'ARTICLE/agents/diff_rates_both_control/agent_it_20',
                        # 'ARTICLE/agents/diff_rates_both_control/agent_it_40',
                        # 'ARTICLE/agents/diff_rates_both_control/agent_it_94',
                        # 'ARTICLE/agents/diff_rates_both_control/agent_it_65',
                        # 'ARTICLE/agents/diff_rates_both_control/agent_libuda',
                        # 'ARTICLE/agents/diff_rates_both_control/agent_max_dyn_adv',
                        'ARTICLE/agents/diff_rates_both_control/agent_dyn_adv_new',
                    ],
                    env_params={
                        'episode_time': 1000.,
                        'time_step': 10.,
                        'names_to_state': ['B', 'A', 'outputC'],
                        'state_string': 'LG:(CO2&O2&CO)x(points)',
                        'state_args': {'points': 2,
                                       'step': 10.,
                                       'dynamic_normalization': {
                                           'names': ['outputC'],
                                           'alpha': 0.2
                                           },
                                       # 'dynamic_normalization': None,
                                       },
                        'reward_spec': 'each_step_base',
                        'input_dt': 0.1,
                        'target_type': 'one_row',
                        # 'reset_mode': {'kind': 'predefined_step', 'step': np.zeros(2)},
                    },
                    model_params=[
                        # {'rate_ads_A': 0.1, 'rate_ads_B': 1.0, 'rate_des_A': 0.07162, 'rate_react': 5.98734, },
                        # {'rate_ads_A': 10., 'rate_ads_B': 0.1, 'rate_des_A': 0.07162, 'rate_react': 5.98734, },
                        # {'rate_ads_A': 0.14895, 'rate_ads_B': 0.06594, 'rate_des_A': 10.0, 'rate_react': 10.0, },
                        # {'rate_ads_A': 0.14895, 'rate_ads_B': 0.06594, 'rate_des_A': 0.1, 'rate_react': 0.1, 'thetaB_init': 0.},
                        # {'rate_des_A': 0.01, 'rate_react': 0.01},
                        {'rate_des_A': 0.1, 'rate_react': 0.1},
                    ],
                    n_episodes=5, deterministic=True,
                    )


if __name__ == '__main__':
    main()

