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


def run_episode(environment: RL2207_Environment, agent, independent: bool = False, deterministic: bool = False,
                reset_callback=None):
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

    if reset_callback is not None:
        reset_callback(environment)

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
        eval_agent = lambda agent, env: env.cumm_episode_target

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
    prev_max_integral = 1e-9

    if eval_period is None:
        eval_period = max(3, n_episodes // 10)
    max_agent_metric = -np.inf
    agent_metric_data = []

    # Loop over episodes
    for i in range(n_episodes):
        run_episode(environment, agent, reset_callback=reset_callback)
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
            environment.summary_graphs(f'{dir_path}/')
            prev_graph_ind = i

        if environment.success:
            if (i - prev_graph_ind > 100) or\
                    ((environment.cumm_episode_target - prev_max_integral) / prev_max_integral > 0.07):
                environment.controller.plot(f'{dir_path}/{environment.cumm_episode_target:.2f}conversion{i}.png')
                environment.summary_graphs(f'{dir_path}/')
                prev_graph_ind = i

            prev_max_integral = environment.cumm_episode_target

        if not (i % eval_period) or environment.success:
            agent_metric_data.append([i, eval_agent(agent, environment)])
            if agent_metric_data[-1][1] > max_agent_metric:
                agent.save(directory=dir_path + '/best_agent', format='numpy')
                max_agent_metric = agent_metric_data[-1][1]

    # agent.save(directory=dir_path + '/last_agent', format='numpy')
    
    agent_metric_data = np.array(agent_metric_data)
    lib.plot_to_file(agent_metric_data[:, 0], agent_metric_data[:, 1],
                     {'label': 'agent metric', 'c': 'b'},
                     xlabel='Episode number', ylabel='Agent metric',
                     ylim=[-1.e-2, None],
                     fileName=f'{dir_path}/agent_metric.png')

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
    os.makedirs(f'{dir_path}/testing', exist_ok=False)
    os.makedirs(f'{dir_path}/testing_deterministic', exist_ok=False)
    ret['stochastic_test'] = test_run(environment, test_agent, out_folder=f'{dir_path}/testing',
                                      reset_callback=reset_on_test, test_callback=test_callback)
    ret['deterministic_test'] = test_run(environment, test_agent, out_folder=f'{dir_path}/testing_deterministic', deterministic=True,
                                         reset_callback=reset_on_test, test_callback=test_callback)
    return ret


def test_run(environment: RL2207_Environment, agent, out_folder, n_episodes=None, deterministic=False,
             reset_mode='bottom_state', reset_callback=None, test_callback=None):
    environment.describe_to_file(f'{out_folder}/info.txt')
    environment.reset_mode = reset_mode

    test_it = 0
    if test_callback is not None:
        cumm_rewards = []
        while test_callback(environment, test_it):
            cumm_rewards.append(run_episode(environment, agent, independent=True, deterministic=deterministic, reset_callback=reset_callback))
            environment.controller.plot(f'{out_folder}/{environment.cumm_episode_target:.2f}conversion{test_it}.png')
            test_it += 1
        cumm_rewards = np.array(cumm_rewards)

    else:
        if n_episodes is None:
            n_episodes = 10
        cumm_rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            cumm_rewards[i] = run_episode(environment, agent, independent=True, deterministic=deterministic, reset_callback=reset_callback)
            environment.controller.plot(f'{out_folder}/{environment.cumm_episode_target:.2f}conversion{i}.png')

    agent.save(directory=out_folder + '/agent', format='numpy')

    return {'mean_on_test': np.mean(cumm_rewards), 'max_on_test': np.max(cumm_rewards)}


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

    # # PC_obj = ProcessController(
    # #             LibudaModelWithDegradation(
    # #                 init_cond={'thetaCO': 0., 'thetaO': 0.},
    # #                 Ts=273+160,  # 273+160
    # #                 v_d=0.01,
    # #                 v_r=0.1,
    # #                 border=4.),
    # #             target_func_to_maximize=get_target_func('CO2_value'),
    # #             supposed_step_count=100, supposed_exp_time=1000)
    #
    # # PC_obj = PC_setup.general_PC_setup('Libuda2001',
    # #                                    ('to_model_constructor', {'Ts': 433.}),
    # #                                    ('to_PC_constructor', {'target_func_to_maximize': lambda x: x[4]}),
    # #                                    )
    # PC_obj = PC_setup.general_PC_setup('LibudaD',
    #                                    ('to_model_constructor', {'Ts': 433.}),
    #                                    ('to_PC_constructor', {'target_func_to_maximize': lambda x: x[4]}),
    #                                    )
    #
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
    # print(rl_agent.get_architecture())

    # EXAMINE agent against curves
    import predefined_policies as policies

    curves = [
        policies.ConstantPolicy({'value': 0.75}),
        policies.ConstantPolicy({'value': 0.5}),
        policies.ConstantPolicy({'value': 0.25}),
        policies.TwoStepPolicy({'1': 0., '2': 1., 't1': 5., 't2': 5., }),
        policies.TwoStepPolicy({'1': 0., '2': 1., 't1': 10., 't2': 10., }),
        policies.TwoStepPolicy({'1': 0.1, '2': 0.6, 't1': 5., 't2': 5., }),
        policies.TwoStepPolicy({'1': 0.1, '2': 0.6, 't1': 10., 't2': 10., }),
        policies.TwoStepPolicy({'1': 0.6, '2': 1., 't1': 5., 't2': 5., }),
        policies.TwoStepPolicy({'1': 0.6, '2': 1., 't1': 10., 't2': 10., }),

        # special, test only curves
        # TRIANGLE
        policies.TrianglePolicy({'1': 0., '2': 1., 't1': 5., 't2': 5., }),
        policies.TrianglePolicy({'1': 0., '2': 1., 't1': 10., 't2': 10., }),
        policies.TrianglePolicy({'1': 0.2, '2': 0.8, 't1': 5., 't2': 5., }),
        policies.TrianglePolicy({'1': 0.2, '2': 0.8, 't1': 10., 't2': 10., }),

        # SAW
        policies.TrianglePolicy({'1': 1., '2': 0., 't1': 0., 't2': 10., }),
        policies.TrianglePolicy({'1': 0., '2': 1., 't1': 0., 't2': 10., }),
        policies.TrianglePolicy({'1': 1., '2': 0., 't1': 0., 't2': 5., }),
        policies.TrianglePolicy({'1': 0., '2': 1., 't1': 0., 't2': 5., }),

        # SIN
        policies.SinPolicy({'A': 0.5, 'T': 5., 'alpha': 0., 'bias': 0.5, }),
        policies.SinPolicy({'A': 0.5, 'T': 10., 'alpha': 0., 'bias': 0.5, }),
        policies.SinPolicy({'A': 0.3, 'T': 5., 'alpha': 0., 'bias': 0.7, }),
        policies.SinPolicy({'A': 0.3, 'T': 10., 'alpha': 0., 'bias': 0.7, }),
        policies.SinPolicy({'A': 0.3, 'T': 5., 'alpha': 0., 'bias': 0.3, }),
        policies.SinPolicy({'A': 0.3, 'T': 10., 'alpha': 0., 'bias': 0.3, }),
    ]

    for co_curve in curves:
        co_curve.update_policy({'t_init': -0.2})

    PC_obj = PC_setup.general_PC_setup('LibudaG', ('to_model_constructor', {'params': {}}))
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
        episode_time=30.,
        time_step=1.,
        normalize_coef=1.,
        input_dt=0.1,
    )
    my_env = Environment.create(environment=my_env, max_episode_timesteps=6000)
    agent = Agent.load('ARTICLE/bestCORTPagent/agent', format='numpy', environment=my_env)
    my_env.actions()
    examine_agent_vs_different_curves(my_env, agent, curves, outfolder='ARTICLE/bestCORTPagent/agent_test',
                                      deterministic=True)

    pass


if __name__ == '__main__':
    main()

