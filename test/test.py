import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
# import itertools
import os

import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from test_models import TestModel, generate_max_rank_matr

from ProcessController import ProcessController
import test_models as models
from targets_metrics import *


def run_test_model():
    model_obj = TestModel()
    vector_to_apply = np.random.randint(-5, 5, 5)
    print(f'input-vector: {vector_to_apply}')
    for _ in range(20):
        print(f'out vector: {model_obj.update(vector_to_apply, 1.)}')


def benchmark_RL_agents():
    import tensorforce
    import tensorforce.environments
    # import tensorforce.agents
    import run_RL
    # import tensorforce.execution as tf_exec

    agent_name = 'vpg'
    env_name = 'CartPole'

    # env = tensorforce.environments.Environment.create(
    #     environment='gym', level=env_name, max_episode_timesteps=500,
    # )
    env = tensorforce.environments.OpenAIGym(f'{env_name}-v0', visualize=False)
    agent = run_RL.create_tforce_agent(env, agent_name)

    # runner = tf_exec.Runner(agent=agent, environment=env, max_episode_timesteps=500)
    #
    # runner.run(num_episodes=300)
    # runner.run(num_episodes=20, evaluation=True)
    # runner.close()

    folder = f'./benchmark_RL/{agent_name}_{env_name}'
    os.makedirs(folder, exist_ok=False)

    num_episodes = 2000
    cum_rewards = np.zeros(num_episodes)
    for i in range(num_episodes):
        states = env.reset()
        terminal = False
        sum_reward = 0.
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            sum_reward += reward
            if terminal:
                cum_rewards[i] = sum_reward

    def plot_rews(fname: str):
        fig, ax = plt.subplots()
        ax.plot(np.arange(num_episodes) + 1, cum_rewards, 'b-d')
        ax.set_xlabel('Episode number')
        ax.set_ylabel('Cumulative reward')
        fig.savefig(f'{folder}/{fname}')
        plt.close(fig)

    plot_rews('training_rews.png')

    env = tensorforce.environments.OpenAIGym(f'{env_name}-v0', visualize=True)
    num_episodes = 30
    cum_rewards = np.zeros(num_episodes)
    for i in range(num_episodes):
        states = env.reset()
        terminal = False
        sum_reward = 0.
        while not terminal:
            actions = agent.act(states=states, independent=True, deterministic=True)
            states, terminal, reward = env.execute(actions=actions)
            # agent.observe(terminal=terminal, reward=reward)
            sum_reward += reward
            if terminal:
                cum_rewards[i] = sum_reward

    plot_rews('eval_rews.png')


def working_with_csv_test():
    import lib

    csv_path = './PC_plots/Ziff/Ziff_2023_3_10__2/Ziff_summarize_CO2.csv'
    ops, df = lib.read_plottof_csv(csv_path, ret_ops=True)
    lib.plot_from_file({'label': 'Average CO2 prod. rate', 'linestyle': 'solid',
                        'marker': 'h', 'c': 'purple',
                        'twin': True, },
                       {'label': 'Average O2 coverage', 'linestyle': (0, (1, 1)),
                        'marker': 'x', 'c': 'blue'},
                       {'label': 'Average CO coverage', 'linestyle': (0, (5, 5)),
                        'marker': '+', 'c': 'red'},
                       csvFileName=csv_path,
                       fileName='./test.png',
                       xlabel=f'O2, {1.e4:.4g} Pa', ylabel='coverages', title='Summarize across benchmark',
                       twin_params={'ylabel': 'CO2 prod. rate'},
                       )


def test_policy():
    import predefined_policies as pp
    plt.switch_backend('TkAgg')

    # any step policy
    # nsteps = 5
    # policy = pp.AnyStepPolicy(nsteps, dict())
    #
    # params = {str(i): 10 + i for i in range(1, nsteps + 1)}
    # tparams = {f't{i}': i for i in range(1, nsteps + 1)}
    # policy.set_policy({**params, **tparams})
    # policy(np.linspace(0, 50, 51))

    # fourier series policy
    policy = pp.FourierSeriesPolicy(5, {'a_s': np.array([1., 1., 1., 1., 1.]), 'length': 10})
    t = np.linspace(0, 60, 300)
    p_of_t = policy(t)
    p_of_t_2 = np.array([policy(t0) for t0 in t])
    fig, ax = plt.subplots()
    ax.plot(t, p_of_t_2, c='b', linewidth=5)
    ax.plot(t, p_of_t, c='r', linewidth=3)
    plt.show()


def test_jobs_functions():
    import multiple_jobs_functions as jobfuncs

    size = [10, 10]
    PC_obj = ProcessController(models.KMC_CO_O2_Pt_Model((*size, 1), log_on=False,
                                                          O2_top=1.1e5, CO_top=1.1e5,
                                                          CO2_rate_top=1.4e6, CO2_count_top=1.e4,
                                                          T=373.),
                               analyser_dt=0.25e-7,
                               target_func_to_maximize=get_target_func('CO2_count'),
                               target_func_name='CO2_count',
                               target_int_or_sum='sum',
                               RESOLUTION=1,  # always should be 1 if we use KMC, otherwise we will get wrong results!
                               supposed_step_count=100,  # memory controlling parameters
                               supposed_exp_time=1.e-5)
    PC_obj.set_plot_params(input_lims=[-1e-5, None], input_ax_name='Pressure, Pa',
                           output_lims=[-1e-2, None],
                           additional_lims=[-1e-2, 1. + 1.e-2],
                           # output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$',
                           output_ax_name='CO x O events count')
    PC_obj.set_metrics(
                       # ('CO2', CO2_integral),
                       ('CO2 count', CO2_count),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
                       )

    jobfuncs.run_jobs_list(
        jobfuncs.one_turn_search_iteration,
        **(jobfuncs.jobs_list_from_grid(
            (0.25, 0.2, 0.1),
            (3.e-8, 5.e-8, 1.e-7, 2.e-7),
            names=('x1', 't0')
        )),
        names_groups=(),
        const_params={'x0': 0.3, 't1': 2.e-7},
        sort_iterations_by='Total_CO2_Count',
        PC=PC_obj,
        python_interpreter='../RL_10_21/venv/bin/python',
        out_fold_path='PC_plots/220324_one_turn_search',
        separate_folds=False,
    )


def Libuda_coefs_estimation():
    import test_models
    model = test_models.LibudaModel()
    model.update([1.e-4, 1.e-4], 1, True)


if __name__ == '__main__':
    # working_with_csv_test()

    test_policy()

    # test_jobs_functions()

    # Libuda_coefs_estimation()

    # benchmark_RL_agents()

    # model testing
    # run_test_model()

    # M = generate_max_rank_matr(5, 5)
    # print(M)

    # d = {'CO_A': 0.0001, 'CO_bias_f': 0.0, 'CO_bias_t': 0.21801443183436786, 'CO_k': 0.6283185307179586, 'O2_A': 0.0001, 'O2_bias_f': 8.623641039884324e-05, 'O2_bias_t': 0.26438923328940805, 'O2_k': 0.3141592653589793}

    # A = 0.0001
    # k = 0.6283185307179586
    # b_t = 0.21801443183436786
    # b_f = 0.0

    # A = 2.e-5
    # k = 0.1 * np.pi
    # b_t = 0.0
    # b_f = 3.e-5
    #
    # def f(t):
    #     res = A * np.sin(k * t + b_t) + b_f
    #     res[res < 0.] = 0.
    #     res[res > 1.e-4] = 1.e-4
    #     return res
    #
    # x = np.linspace(0., 500., 5000)
    # y = f(x)
    # plt.plot(x, y)
    # plt.show()

    # M = generate_max_rank_matr(3, 3)
    # np.save('./M_3x3.npy', M, allow_pickle=False)

    pass
