import numpy as np

from parallel_trainRL_funcs import *
from targets_metrics import *
from predefined_policies import *

from multiple_jobs_functions import *
import PC_setup
import PC_run


def main():
    # train_grid_parallel(['full_ep_mean', 'full_ep_base', 'each_step_base'],
    #                      [(1, False), (2, False), (3, False),
    #                       (4, False), (5, False)],
    #                      [0.01, 0.01, 0.01],
    #                      names=('env:reward_spec', 'env:state_spec', 'model:v_d'),
    #                      const_params={
    #                          'env': {'model_type': 'continuous',
    #                                  'episode_time': episode_time,
    #                                  'time_step': time_step,
    #                                  },
    #                          'agent_name': 'vpg',
    #                          'model': {
    #                              'v_r': 0.1,
    #                              'border': 4.
    #                              }
    #                      },
    #                      out_path='run_RL_out/current_training/220804_test',
    #                      python_interpreter='../RL_21/venv/bin/python',
    #                      on_cluster=True,
    #                      controller=ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0.},
    #                                                        Ts=273+160), target_func_to_maximize=CO2_value,
    #                                                   supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                                   supposed_exp_time=2 * episode_time),
    #                      n_episodes=30000,
    #                      unique_folder=False,
    #                      at_same_time=45)

    # LibudaDegradPC = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0.},
    #                                                               Ts=273+160),
    #                                    target_func_to_maximize=CO2_value,
    #                                    supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                    supposed_exp_time=2 * episode_time)
    # LibudaDegradPC.set_plot_params(input_ax_name='Pressure', input_lims=None,
    #                                output_lims=[0., 0.06], output_ax_name='CO2_formation_rate')

    # Pt2210_PC = ProcessController(PtModel(init_cond={'thetaO': 0., 'thetaCO': 0.}),
    #                                   target_func_to_maximize=CO2_value,
    #                                   supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                   supposed_exp_time=2 * episode_time)
    # Pt2210_PC.set_plot_params(input_ax_name='Pressure', input_lims=None,
    #                             output_ax_name='CO2 form. rate', output_lims=None)

    # PC_LReturnK3K1 = ProcessController(LibudaModelReturnK3K1AndPressures(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=273 + 160),
    #                                    RESOLUTION=10,
    #                                    # target_func_to_maximize=CO2xConversion,
    #                                    long_term_target_to_maximize=get_target_func('CO2xConversion_I', eps=1.),
    #                                    supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                    supposed_exp_time=2 * episode_time)
    # PC_LReturnK3K1.set_plot_params(input_ax_name='Pressure', input_lims=[0., None],
    #                                output_ax_name='???', output_lims=[0., None])
    # PC_PtReturnK3K1 = ProcessController(PtReturnK1K3Model(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=440),
    #                                    # target_func_to_maximize=CO2xConversion,
    #                                    long_term_target_to_maximize=get_target_func('CO2_sub_outs_I', alpha=0.1),
    #                                    supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                    supposed_exp_time=2 * episode_time)

    # PC_LReturnK3K1 = ProcessController(LibudaModelReturnK3K1AndPressures(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=273 + 160),
    #                                    RESOLUTION=10,
    #                                    # target_func_to_maximize=CO2xConversion,
    #                                    long_term_target_to_maximize=get_target_func('CO2xConversion_I', eps=1.),
    #                                    supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                    supposed_exp_time=2 * episode_time)
    # episode_time = 2.e-5
    # time_step = 2.e-6
    # size = (5, 5)
    # PC_KMC = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=False,
    #                                               O2_bottom=1.1e2, CO_bottom=1.1e2,
    #                                               O2_top=1.1e4, CO_top=1.1e4,
    #                                               CO2_rate_top=1.4e6, CO2_count_top=3.e2,
    #                                               T=373.),
    #                            analyser_dt=0.5e-6,
    #                            target_func_to_maximize=get_target_func('CO2_count'), RESOLUTION=1,
    #                            supposed_step_count=2 * int(episode_time / time_step + 1),  # memory controlling parameters
    #                            supposed_exp_time=2 * episode_time)

    # PC_obj = PC_setup.general_PC_setup('ZGBTwo')
    PC_obj = PC_setup.general_PC_setup('LibudaG')
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithT')
    # PC_obj = PC_setup.default_PC_setup('Ziff')
    # PC_obj = PC_setup.general_PC_setup('Ziff', ('to_model_constructor', 'CO2_count_top', 2.e+3))
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithT',
    #                                    ('to_model_constructor', {'params': {'reaction_rate_top': pow(3., 10)}})
    #                                    )
    # PC_obj = PC_setup.general_PC_setup('Libuda2001',
    #                                    ('to_model_constructor', {'params': {'reaction_rate_top': pow(3., 10)}})
    #                                    )

    PC_obj.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
                                          'thetaA_max': 0.5, 'thetaB_max': 0.5,
                                          'rate_ads_A': 0.14895, 'rate_ads_B': 0.26376,  'rate_des_B': 0.,
                                          'rate_des_A': 0.07162, 'rate_react': 5.98734,
                                          })
    # PC_obj.process_to_control.set_params({'rate_ads_A': 0.14895, 'rate_ads_B': 0.06594,  'rate_des_B': 0.,
    #                                       'rate_des_A': 0.1, 'rate_react': 0.1, })
    PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0., })

    # Gauss_x_Conv_x_Conv = get_target_func('(Gauss)x(Conv)x(Conv)_I', default=1e-4, sigma=1e-5 * episode_time, eps=1e-4)
    # name1 = '(Gauss)x(Conv)x(Conv)'
    # Gauss_x__Conv_Plus_Conv_ = get_target_func('(Gauss)x(Conv+Conv)_I', default=1e-4, sigma=1e-5 * episode_time, eps=1e-4)
    # name2 = '(Gauss)x(Conv+Conv)'

    ## LibudaG
    # episode_time = 100
    # episode_time = 30.
    # time_step = 10
    # time_step = 5.

    # def get_fourier_time_dependence(terms_number, length):
    #     random_coefs = np.random.random(2 * terms_number) - 0.5
    #     random_coefs /= np.sum(np.abs(random_coefs))
    #     fourier_policy = FourierSeriesPolicy(terms_number, {'a_sin': random_coefs[:terms_number],
    #                                                         'a_cos': random_coefs[terms_number:],
    #                                                         'length': length})
    #
    #     def time_dependence_(act, t):
    #         return np.array([act[0], fourier_policy(t) / 2 + 0.5])
    #
    #     def callback_(env):
    #         # this callback considers .time_input_dependence to be of FourierSeriesPolicy type
    #         random_coefs_0 = np.random.random(2 * terms_number) - 0.5
    #         random_coefs_0 /= np.sum(np.abs(random_coefs_0))
    #         fourier_policy.update_policy({'a_sin': random_coefs_0[:terms_number],
    #                                    'a_cos': random_coefs_0[terms_number:],
    #                                    'length': length})
    #
    #     return time_dependence_, callback_

    # t_init = -0.2
    #
    # test_co_curves = [
    #     ConstantPolicy({'value': 0.75}),
    #     ConstantPolicy({'value': 0.5}),
    #     ConstantPolicy({'value': 0.25}),
    #     TwoStepPolicy({'1': 0., '2': 1., 't1': 5., 't2': 5., }),
    #     TwoStepPolicy({'1': 0., '2': 1., 't1': 10., 't2': 10., }),
    #     TwoStepPolicy({'1': 0.1, '2': 0.6, 't1': 5., 't2': 5., }),
    #     TwoStepPolicy({'1': 0.1, '2': 0.6, 't1': 10., 't2': 10., }),
    #     TwoStepPolicy({'1': 0.6, '2': 1., 't1': 5., 't2': 5., }),
    #     TwoStepPolicy({'1': 0.6, '2': 1., 't1': 10., 't2': 10., }),
    # ]
    #
    # train_co_curves = [
    #     ConstantPolicy({'value': 0.75}),
    #     ConstantPolicy({'value': 0.2}),
    #     ConstantPolicy({'value': 0.1}),
    #     TwoStepPolicy({'1': 0., '2': 1., 't1': 10., 't2': 10., }),
    # ]
    #
    # for co_curve in test_co_curves + train_co_curves:
    #     co_curve.update_policy({'t_init': t_init})
    #
    # def get_random_turns_time_dependence(period, bounds):
    #     rtp = RandomTurnsPolicy({'period': period, 'bounds': bounds, 't_init': t_init})
    #
    #     def time_dependence_(act, t):
    #         return np.array([act[0], rtp(t)])
    #
    #     def callback_(env):
    #         rtp.reset()
    #
    #     return time_dependence_, callback_
    #
    # # def get_random_turns_plus_sampling_dependence(period, bounds):
    # #     rtp = RandomTurnsPolicy({'period': period, 'bounds': bounds, 't_init': t_init})
    # #     store_current = [rtp]  # crutch
    # #
    # #     def time_dependence_(act, t):
    # #         return np.array([act[0], store_current[0](t)])
    # #
    # #     def callback_(env):
    # #         if np.random.uniform() < 0.25:
    # #             store_current[0] = np.random.choice(train_co_curves)  # p=[0.5 / 3] * 3 + [0.5]
    # #         else:
    # #             rtp.reset()
    # #             store_current[0] = rtp
    # #
    # #     return time_dependence_, callback_
    #
    # def eval_agent(agent, env):
    #     original_dependence = env.time_input_dependence
    #
    #     rews = np.empty(len(test_co_curves))
    #     for i, curve_ in enumerate(test_co_curves):
    #         env.time_input_dependence = lambda act, t: np.array([act[0], curve_(t)])
    #         rews[i] = run_episode(env, agent, deterministic=True, reset_callback=None)
    #         # env.controller.plot(f'./DEBUG/curve_{i}.png')  # DEBUG only !!!
    #
    #     env.time_input_dependence = original_dependence
    #     return np.mean(rews)
    #
    # test_cache = []  # TODO such a painful crutch
    #
    # def test_callback(env, iteration):
    #     if iteration >= len(test_co_curves):
    #         env.time_input_dependence = test_cache.pop()
    #         return False
    #     if not test_cache:
    #         test_cache.append(env.time_input_dependence)
    #     env.time_input_dependence = lambda act, t: np.array([act[0], test_co_curves[iteration](t)])
    #     return True
    #
    # # # fourier_dependence, fourier_callback = get_fourier_time_dependence(5, episode_time)
    # rtp_dependence, rtp_callback = get_random_turns_time_dependence(10., (0., 1.))
    # # rtp_with_sampling_dependence, rtp_with_sampling_callback = get_random_turns_plus_sampling_dependence(10., (0., 1.))
    # vary_o2_co_is_arbitrary = {'type': 'continuous',
    #                            'transform_action': lambda x: x,
    #                            'shape': 1,
    #                            'info': 'control O2, CO(t) is random'}
    #
    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   [(rtp_dependence, rtp_callback, test_callback)],
    #                   # [(rtp_with_sampling_dependence, rtp_with_sampling_callback, test_callback)],
    #                   # [(0.14895, 0.06594 * 4, 0.1), ],
    #                   # ['vpg', 'ppo'],
    #                   ['vpg', ],
    #                   ['auto',
    #                    dict(type='layered', layers=[dict(type='flatten'),
    #                                                 dict(type='dense', size=64, activation='relu'),
    #                                                 dict(type='dense', size=64, activation='relu')]),
    #                    ],
    #                   ['each_step_base'],
    #                   [{'rows': 1, 'use_differences': False},
    #                    {'rows': 2, 'use_differences': False},
    #                    {'rows': 3, 'use_differences': False},
    #                    ],
    #                   [1., 2.5, 5.],
    #                   names=(('env:time_input_dependence', 'reset_callback', 'test_callback'),
    #                          # ('model:rate_ads_A', 'model:rate_ads_B'),
    #                          'agent_name',
    #                          'agent:network',
    #                          'env:reward_spec', 'env:state_spec', 'env:time_step',
    #                          )
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'eval_agent': eval_agent,
    #                   'env': {
    #                           'episode_time': episode_time,
    #                           # 'time_step': time_step,
    #                           'names_to_state': ['B', 'A', 'outputC'],
    #                           'action_spec': vary_o2_co_is_arbitrary,
    #                           'input_dt': 0.1,
    #                           'target_type': 'one_row',
    #                           'init_callback': PC_run.get_estimate_rate_callback(top_rate_estim=0.01),
    #                          },
    #                   # 'agent': {'network': dict(type='layered', layers=[dict(type='flatten'),
    #                   #                                                   dict(type='dense', size=16, activation='relu'),
    #                   #                                                   dict(type='dense', size=5, activation='relu'),
    #                   #                                                   dict(type='dense', size=16, activation='relu')]),
    #                   #           'baseline': dict(type='layered', layers=[dict(type='flatten')]),
    #                   #           'baseline_optimizer': dict(optimizer='adam', learning_rate=1e-3),
    #                   #           'l2_regularization': 1.e-3},
    #                   'model': {},
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::mean_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/LibudaG/231120_CORTP_NoSampling_dynrates',
    #               at_same_time=110,
    #               )

    # vary rates
    variants = [pow(10., x) for x in np.linspace(-2., 1., 4)]

    episode_time = 240.
    # time step range, let's say, from 1. to 100.
    # my time_step action scheme: min_value + a1 * pow(10, 1) + a2 * pow(10, 2)

    def time_step(act):
        a1, a2 = act[-2:]
        return 1. + max(0., (a1 - 0.2) / 0.8) * 10. + max(0., (a2 - 0.2) / 0.8) * 100.

    variable_time_step_len = {'type': 'continuous',
                              'transform_action': lambda x: x[:-2],
                              'shape': 4,
                              'info': 'control O2, CO and chose time step itself'}

    run_jobs_list(**get_for_RL_iterations(),
                  **merge_job_lists(
                      jobs_list_from_grid(variants, variants, names=('model:rate_ads_A', 'model:rate_ads_B')),
                      jobs_list_from_grid(variants, variants, names=('model:rate_des_A', 'model:rate_react')),
                      ),
                  const_params={
                      'n_episodes': 40,
                      'agent_name': 'vpg',
                      'env': {
                              'episode_time': episode_time,
                              'time_step': time_step,
                              'names_to_state': ['B', 'A', 'outputC'],
                              'state_spec': {'rows': 3, 'use_differences': False},
                              'action_spec': variable_time_step_len,
                              'reward_spec': 'each_step_base',
                              'input_dt': 0.1,
                              'target_type': 'one_row',
                              'dynamic_normalization': {'names': ['outputC'], 'alpha': 0.2},
                              'init_callback': PC_run.get_estimate_rate_callback(),
                             },
                      'agent': {},
                      'model': {},
                  },
                  PC=PC_obj,
                  repeat=3,
                  sort_iterations_by='deterministic_test::max_on_test',
                  cluster_command_ops=False,
                  python_interpreter='../RL_10_21/venv/bin/python',
                  out_fold_path='./run_RL_out/LibudaG/231205_var_time_step_dyn_norm',
                  at_same_time=110,
                  )

    # Reference == vanilla Libuda optimization
    # vary_x_co = {'type': 'continuous',
    #              'transform_action': lambda x: np.array([x[0], 1. - x[0]]),
    #              'shape': 1,
    #              'info': 'control x_co'}
    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   ['vpg'],
    #                   ['each_step_base'],
    #                   [{'rows': 1, 'use_differences': False},
    #                    {'rows': 2, 'use_differences': False},
    #                    {'rows': 3, 'use_differences': False},
    #                    ],
    #                   # [(0.01, 1.), (0.1, 10.), (1., 100.)],
    #                   names=('agent_name', 'env:reward_spec', 'env:state_spec',
    #                          # ('env:time_step', 'env:episode_time'),
    #                          )
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'env': {
    #                           'episode_time': episode_time,
    #                           'time_step': time_step,
    #                           'names_to_state': ['B', 'A', 'outputC'],
    #                           # 'action_spec': vary_x_co,
    #                           'input_dt': 0.1,
    #                           'target_type': 'one_row',
    #                           'reset_mode': {'kind': 'random', 'time': 10., 'dt': 0.1},  # 'predefined', 'bottom_state', 'random'
    #                           # 'preprocess': {'in_values': {'inputB': 1., 'inputA': 0.451059329508262},
    #                           #                'time': 20., 'dt': 1.},
    #                           'init_callback': PC_run.estimate_rate_callback,
    #                          },
    #                   'agent': {},
    #                   'model': {},
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::max_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/LibudaG/230727_debug',
    #               at_same_time=110,
    #               )

    # SWITCH RATES, AGNOSTIC AGENT
    # model = PC_obj.process_to_control
    # k2, k4 = model['rate_des_A'], model['rate_react']
    # test_pool = [
    #     ConstantPolicy({'value': k2}),
    #     ConstantPolicy({'value': k2 * 10}),
    #     ConstantPolicy({'value': k2 * 100}),
    #     TwoStepPolicy({'1': k2, '2': k2 * 100, 't1': 5., 't2': 5., }),
    #     TwoStepPolicy({'1': k2, '2': k2 * 100, 't1': 10., 't2': 10., }),
    #     TwoStepPolicy({'1': k2, '2': k2 * 10, 't1': 5., 't2': 5., }),
    #     TwoStepPolicy({'1': k2, '2': k2 * 10, 't1': 10., 't2': 10., }),
    #     TwoStepPolicy({'1': k2 * 10, '2': k2 * 100, 't1': 5., 't2': 5., }),
    #     TwoStepPolicy({'1': k2 * 10, '2': k2 * 100, 't1': 10., 't2': 10., }),
    # ]
    #
    # for curve in test_pool:
    #     curve.update_policy({'t_init': t_init})
    #
    # def get_rate_switch_dependence(PC):
    #     rtp_k2 = RandomTurnsPolicy({'bounds': (k2, k2 * 100.), 'period': 10., 't_init': t_init})
    #     # rtp_k4 = RandomTurnsPolicy({'bounds': (k4 / 10, k4), 'period': 10., 't_init': t_init})
    #
    #     def time_dependence_(act, t):
    #         model.set_params({'rate_des_A': rtp_k2(t),
    #                           })
    #         return act
    #
    #     def callback_(env):
    #         rtp_k2.reset()
    #
    #     return time_dependence_, callback_
    #
    # # def get_random_turns_plus_sampling_dependence(period, bounds):
    # #     rtp = RandomTurnsPolicy({'period': period, 'bounds': bounds, 't_init': t_init})
    # #     store_current = [rtp]  # crutch
    # #
    # #     def time_dependence_(act, t):
    # #         return np.array([act[0], store_current[0](t)])
    # #
    # #     def callback_(env):
    # #         if np.random.uniform() < 0.25:
    # #             store_current[0] = np.random.choice(train_co_curves)  # p=[0.5 / 3] * 3 + [0.5]
    # #         else:
    # #             rtp.reset()
    # #             store_current[0] = rtp
    # #
    # #     return time_dependence_, callback_
    #
    # def eval_agent(agent, env):
    #     original_dependence = env.time_input_dependence
    #
    #     rews = np.empty(len(test_pool))
    #     for i, curve_ in enumerate(test_pool):
    #
    #         def dependence_(act, t):
    #             model.set_params({'rate_des_A': curve_(t)})
    #             return act
    #
    #         env.time_input_dependence = dependence_
    #         rews[i] = run_episode(env, agent, deterministic=True, reset_callback=None)
    #         # env.controller.plot(f'./DEBUG/curve_{i}.png')  # DEBUG only !!!
    #
    #     env.time_input_dependence = original_dependence
    #     return np.mean(rews)
    #
    # test_cache = []  # TODO such a painful crutch
    #
    # def test_callback(env, iteration):
    #     if iteration >= len(test_pool):
    #         env.time_input_dependence = test_cache.pop()
    #         return False
    #     if not test_cache:
    #         test_cache.append(env.time_input_dependence)
    #
    #     def dependence_(act, t):
    #         model.set_params({'rate_des_A': test_pool[iteration](t)})
    #         return act
    #
    #     env.time_input_dependence = dependence_
    #     return True
    #
    # rate_switch_dependence, rate_switch_callback = get_rate_switch_dependence(PC_obj)
    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   [(rate_switch_dependence, rate_switch_callback, test_callback), ],
    #                   # ['vpg', 'ppo'],
    #                   ['vpg', ],
    #                   ['each_step_base'],
    #                   [{'rows': 1, 'use_differences': False},
    #                    {'rows': 2, 'use_differences': False},
    #                    {'rows': 3, 'use_differences': False},
    #                    ],
    #                   [1., 2.5, 5.],
    #                   names=(('env:time_input_dependence', 'reset_callback', 'test_callback'),
    #                          'agent_name',
    #                          'env:reward_spec', 'env:state_spec', 'env:time_step',
    #                          )
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'eval_agent': eval_agent,
    #                   'env': {
    #                           'episode_time': episode_time,
    #                           # 'time_step': time_step,
    #                           'names_to_state': ['outputC', 'B', 'A', ],
    #                           'input_dt': 0.1,
    #                           'target_type': 'one_row',
    #                           'reset_mode': {'kind': 'random', 'time': 20., 'dt': 1.,
    #                                          'bottom': np.array([0., 0.]), 'top': np.array([1., 1.]), },
    #                           'normalize_coef': 10.,
    #                          },
    #                   'agent': {'network': dict(type='layered', layers=[dict(type='flatten'),
    #                                                                     dict(type='dense', size=16, activation='relu'),
    #                                                                     dict(type='dense', size=5, activation='relu'),
    #                                                                     dict(type='dense', size=16, activation='relu')]),
    #                             'baseline': dict(type='layered', layers=[dict(type='flatten')]),
    #                             'baseline_optimizer': dict(optimizer='adam', learning_rate=1e-3),
    #                             'l2_regularization': 1.e-3},
    #                   'model': {
    #                       'reaction_rate_top': 10.,
    #                   },
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::max_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/LibudaG/230729_vary_rates_agent_ignorant',
    #               at_same_time=110,
    #               )

    # SEARCH FOR BEST AGENT
    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   # [(rtp_dependence, rtp_callback, test_callback)],
    #                   [(rtp_with_sampling_dependence, rtp_with_sampling_callback, test_callback)],
    #                   [(0.14895, 0.06594 * 4, 0.1), ],
    #                   [dict(type='layered', layers=[dict(type='flatten')]),
    #                    dict(type='layered', layers=[dict(type='flatten'), dict(type='dense', size=16, activation='relu')]),
    #                    dict(type='layered', layers=[dict(type='flatten'),
    #                                                 dict(type='dense', size=16, activation='relu'),
    #                                                 dict(type='dense', size=5, activation='relu'),
    #                                                 dict(type='dense', size=16, activation='relu')]),
    #                    ],
    #                   [(None, None), (dict(type='layered', layers=[dict(type='flatten')]), dict(optimizer='adam', learning_rate=1e-3))],
    #                   [1.e-3, 1.],
    #                   # ['vpg', 'ppo'],
    #                   ['vpg', ],
    #                   ['each_step_base'],
    #                   [{'rows': 1, 'use_differences': False},
    #                    {'rows': 2, 'use_differences': False},
    #                    ],
    #                   names=(('env:time_input_dependence', 'reset_callback', 'test_callback'),
    #                          ('model:rate_ads_A', 'model:rate_ads_B', 'model:reaction_rate_top'),
    #                          'agent:network', ('agent:baseline', 'agent:baseline_optimizer'), 'agent:l2_regularization',
    #                          'agent_name',
    #                          'env:reward_spec', 'env:state_spec',
    #                          )
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'eval_agent': eval_agent,
    #                   'env': {
    #                           'episode_time': episode_time,
    #                           'time_step': time_step,
    #                           'names_to_state': ['B', 'A', 'outputC'],
    #                           'action_spec': vary_o2_co_is_arbitrary,
    #                           'input_dt': 0.1,
    #                           'target_type': 'one_row',
    #                           'normalize_coef': 1.,
    #                          },
    #                   'agent': {},
    #                   'model': {},
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::max_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/LibudaG/230715_CO(t)_RTP_vpg_optimization',
    #               at_same_time=110,
    #               )

    # LibudaGWithT

    # SWITCH T DEPENDENCE,
    # episode_time = 220.
    # episode_time = 120.
    # t_init = -0.5
    #
    # t_half = episode_time / 2
    # t_third = episode_time / 3
    # T1, T2 = 440., 560.
    # min_step = episode_time / 30.
    # T_policies = [
    #         # TwoStepPolicy({'1': 440., '2': 460., 't1': 27., 't2': 3., 't_init': t_init}),
    #         # TwoStepPolicy({'1': 460., '2': 440., 't1': 3., 't2': 27., 't_init': t_init}),
    #         TwoStepPolicy({'1': T1, '2': T2, 't1': t_half, 't2': t_half, 't_init': t_init}),
    #         TwoStepPolicy({'1': T2, '2': T1, 't1': t_half, 't2': t_half, 't_init': t_init}),
    #         TwoStepPolicy({'1': T1, '2': T2, 't1': t_third, 't2': 2 * t_third, 't_init': t_init}),
    #         TwoStepPolicy({'1': T2, '2': T1, 't1': 2 * t_third, 't2': t_third, 't_init': t_init}),
    #                   ]
    #
    # def get_T_switch_dependence():
    #     store_current = [T_policies[0]]  # crutch
    #
    #     def time_dependence_(act, t):
    #         return np.array([act[0], act[1], store_current[0](t)])
    #
    #     def callback_(env):
    #         if np.random.uniform() < 0.25:
    #             store_current[0] = np.random.choice(T_policies)
    #
    #     return time_dependence_, callback_
    #
    # T_switch_dependence, T_switch_callback = get_T_switch_dependence()
    #
    # def func_maker(T_curve):
    #     return lambda act, t: [act[0], act[1], T_curve(t)]
    # eval_agent, test_callback = get_eval_test_agent_on_fixed_t_dependencies([
    #     func_maker(T_curve) for T_curve in T_policies])
    #
    # vary_both_T_is_arbitrary = {'type': 'continuous',
    #                             'transform_action': lambda x: x,
    #                             'shape': 2,
    #                             'info': 'control O2, CO; T is changing without agent knowing'}
    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   [(T_switch_dependence, T_switch_callback, test_callback), ],
    #                   # ['vpg', 'ppo'],
    #                   ['vpg', ],
    #                   ['each_step_base'],
    #                   [{'rows': 1, 'use_differences': False},
    #                    {'rows': 2, 'use_differences': False},
    #                    {'rows': 3, 'use_differences': False},
    #                    {'rows': 4, 'use_differences': False},
    #                    {'rows': 5, 'use_differences': False},
    #                    ],
    #                   # [1., 2.5, 5.],
    #                   [min_step, 2.5*min_step, 5.*min_step],
    #                   names=(('env:time_input_dependence', 'reset_callback', 'test_callback'),
    #                          'agent_name',
    #                          'env:reward_spec', 'env:state_spec', 'env:time_step',
    #                          )
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'eval_agent': eval_agent,
    #                   'env': {
    #                           'episode_time': episode_time,
    #                           # 'time_step': time_step,
    #                           'names_to_state': ['B', 'A', 'outputC'],
    #                           'action_spec': vary_both_T_is_arbitrary,
    #                           'input_dt': 0.1,
    #                           'target_type': 'one_row',
    #                           'reset_mode': {'kind': 'random', 'time': 20., 'dt': 1.,
    #                                          'bottom': np.array([0., 0., 440.]), 'top': np.array([1., 1., 440.]), },
    #                           'init_callback': PC_run.get_estimate_rate_callback(top_rate_estim=0.1),
    #                          },
    #                   'agent': {'network': dict(type='layered', layers=[dict(type='flatten'),
    #                                                                     dict(type='dense', size=16, activation='relu'),
    #                                                                     dict(type='dense', size=5, activation='relu'),
    #                                                                     dict(type='dense', size=16, activation='relu')]),
    #                             'baseline': dict(type='layered', layers=[dict(type='flatten')]),
    #                             'baseline_optimizer': dict(optimizer='adam', learning_rate=1e-3),
    #                             'l2_regularization': 1.e-3},
    #                   'model': {},
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::max_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/LibudaG/230925_T1(440)_T2(560)_ignorant_agent',
    #               at_same_time=110,
    #               )

    # vary_x_co_action_spec = {'type': 'continuous',
    #                          'transform_action': lambda x: [1 - x[0], x[0], 300 * x[1] + 400],
    #                          'shape': 2,
    #                          'info': 'control x_co = CO / (CO + O2) and T'}
    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   ['vpg', 'ppo'],
    #                   ['each_step_base'],
    #                   [{'rows': 1, 'use_differences': False},
    #                    {'rows': 3, 'use_differences': False},
    #                    {'rows': 5, 'use_differences': False},
    #                    ],
    #                   [(20., 1., 0.5), (2., 0.1, 5.)],
    #                   names=('agent_name', 'env:reward_spec', 'env:state_spec',
    #                          ('env:episode_time', 'env:time_step', 'env:normalize_coef'), )
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'env': {
    #                           # 'episode_time': episode_time,
    #                           # 'time_step': time_step,
    #                           'names_to_state': ['B', 'A', 'outputC', 'T'],
    #                           'action_spec': vary_x_co_action_spec,
    #                           'target_type': 'one_row',
    #                           # 'normalize_coef': 2.e-5,
    #                          },
    #                   'agent': {},
    #                   'model': {},
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::max_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/LibudaGWithT/230519_action_spec_debug_2',
    #               at_same_time=110,
    #               )

    # ZGB
    # episode_time = 2.e+5
    # time_step = 2.e+3
    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   ['vpg', 'ppo'],
    #                   ['each_step_base'],
    #                   [{'rows': 1, 'use_differences': False},
    #                    {'rows': 3, 'use_differences': False},
    #                    {'rows': 5, 'use_differences': False},
    #                    {'rows': 13, 'use_differences': False},
    #                    {'rows': 13, 'use_differences': True}],
    #                   names=('agent_name', 'env:reward_spec', 'env:state_spec')
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'env': {'episode_time': episode_time,
    #                           'time_step': time_step,
    #                           'names_to_state': ['CO2_count', 'x'],
    #                           'continuous_actions': True,
    #                           # 'continuous_actions': {'CO': [0., 1.e+4], 'O2': 1.e+4},
    #                           # 'discrete_actions': True,
    #                           # 'discrete_actions': {'CO': [7e-5, 0.], 'O2': 7.e-5},
    #                           'target_type': 'one_row',
    #                           'normalize_coef': 2.e-5,
    #                          },
    #                   'agent': {},
    #                   'model': {},
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::max_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/230418_Ziff_debug',
    #               at_same_time=30,
    #               )

    # run_jobs_list(**get_for_RL_iterations(),
    #               **jobs_list_from_grid(
    #                   ['full_ep_mean'],
    #                   [{'rows': 3, 'use_differences': False},
    #                    {'rows': 5, 'use_differences': False},
    #                    {'rows': 13, 'use_differences': False},
    #                    {'rows': 13, 'use_differences': True}],
    #                   names=('env:reward_spec', 'env:state_spec')
    #               ),
    #               const_params={
    #                   'n_episodes': 40,
    #                   'env': {'episode_time': episode_time,
    #                           'time_step': time_step,
    #                           'names_to_state': ['outputC', 'B', 'A'],
    #                           'continuous_actions': True,
    #                           # 'continuous_actions': {'CO': [0., 1.e+4], 'O2': 1.e+4},
    #                           # 'discrete_actions': True,
    #                           # 'discrete_actions': {'CO': [7e-5, 0.], 'O2': 7.e-5},
    #                           'target_type': 'one_row',
    #                           'normalize_coef': 1.e-3,
    #                          },
    #                   'agent_name': 'ppo',
    #                   'agent': {},
    #                   'model': {},
    #               },
    #               PC=PC_obj,
    #               repeat=3,
    #               sort_iterations_by='deterministic_test::max_on_test',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./run_RL_out/230418_LibudaG_debug',
    #               at_same_time=30,
    #               )

    # MAIN TEST FOR LIBUDA
    # train_grid_parallel(['full_ep_mean', 'full_ep_base', 'each_step_base'],
    #                      [(1, False), (2, False), (3, False),
    #                       (4, False), (5, False)],
    #                      [0.01, 0.01, 0.01],
    #                      names=('env:reward_spec', 'env:state_spec', 'model:v_d'),
    #                      const_params={
    #                          'env': {'model_type': 'continuous',
    #                                  'episode_time': episode_time,
    #                                  'time_step': time_step,
    #                                  },
    #                          'agent_name': 'vpg',
    #                          'model': {
    #                              'v_r': 0.1,
    #                              'border': 4.
    #                              }
    #                      },
    #                      out_path='run_RL_out/current_training/???',
    #                      file_to_execute_path='code/parallel_trainRL.py',
    #                      python_interpreter='../RL_21/venv/bin/python',
    #                      on_cluster=True,
    #                      controller=ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0.},
    #                                                        Ts=273+160), target_func_to_maximize=target,
    #                                                   supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                                   supposed_exp_time=2 * episode_time),
    #                      n_episodes=30_000,
    #                      unique_folder=False,
    #                      at_same_time=45)

    # train_grid_parallel([(Gauss_x_Conv_x_Conv, name1), (Gauss_x__Conv_Plus_Conv_, name2)],
    #                     [(1.e-5, 1.e-4), (1.e-4, 1.e-4), (1.e-3, 1.e-3), (1.e-2, 1.e-2), ],
    #                     ['full_ep_mean', 'full_ep_base'],
    #                     [(1, False), (3, False)],
    #                     names=(('long_term_target', 'target_func_name'), ('model:O2_top', 'model:CO_top'), 'env:reward_spec', 'env:state_spec'),
    #                     const_params={
    #                          'env': {'model_type': 'continuous',
    #                                  'names_to_state': ['CO2', 'O2(Pa)', 'CO(Pa)'],
    #                                  'episode_time': episode_time,
    #                                  'time_step': time_step,
    #                                  'log_scaling_dict': None,
    #                                  'names_to_plot': ['CO2', 'long_term_target'],
    #                                  'target_type': 'episode',
    #                                  },
    #                          'agent_name': 'vpg',
    #                          'model': {
    #                              }
    #                      },
    #                     out_path='run_RL_out/221130_variate_new_targets',
    #                     file_to_execute_path='repos/parallel_trainRL.py',
    #                     python_interpreter='../RL_10_21/venv/bin/python',
    #                     on_cluster=False,
    #                     controller=PC_obj,
    #                     n_episodes=30_000,
    #                     unique_folder=False,
    #                     at_same_time=45)

    # agent, actor, critic, critic_opt, baseline, baseline_opt
    # train_grid_parallel([('ac', 'auto', 'auto', '#exclude', '#exclude', '#exclude',),
    #                      ('ac', 'auto', 'auto', dict(optimizer='adam', learning_rate=1e-3), '#exclude', '#exclude',),
    #                      ('ppo', 'auto', '#exclude', '#exclude', '#exclude', '#exclude',),
    #                      ('ppo',
    #                       dict(type='layered', layers=[dict(type='flatten'),
    #                                                    dict(type='dense', size=32, activation='relu')]),
    #                       '#exclude', '#exclude',
    #                       dict(type='layered', layers=[dict(type='flatten'),
    #                                                    dict(type='dense', size=32, activation='relu')]),
    #                       0.3),
    #                      ('ppo',
    #                       dict(type='layered', layers=[dict(type='flatten'),
    #                                                    dict(type='dense', size=32, activation='relu')]),
    #                       '#exclude', '#exclude',
    #                       dict(type='layered', layers=[dict(type='flatten'),
    #                                                    dict(type='dense', size=32, activation='relu')]),
    #                       dict(optimizer='adam', learning_rate=1e-3)),
    #                      ],
    #                     ['full_ep_mean', 'each_step_base',
    #                      ],
    #                     names=(('agent_name',
    #                             'agent:network',
    #                             'agent:critic',
    #                             'agent:critic_optimizer',
    #                             'agent:baseline',
    #                             'agent:baseline_optimizer',
    #                             ),
    #                            'env:reward_spec'),
    #                     const_params={
    #                          'env': {'model_type': 'continuous',
    #                                  'episode_time': episode_time,
    #                                  'time_step': time_step,
    #                                  'state_spec': {'rows': 3, 'use_differences': False},
    #                                  'log_scaling_dict': None,
    #                                  },
    #                          'model': {
    #                              'O2_top': 10.e-5,
    #                              'CO_top': 100.e-5,
    #                              'v_r': 0.1,
    #                              'v_d': 0.01,
    #                              'border': 4.,
    #                              },
    #                          'agent': {
    #                              'batch_size': 32,
    #                              'update_frequency': 32,
    #                          }
    #                      },
    #                     repeat=5,
    #                     out_path='run_RL_out/current_training/220909_new_agents_CO_top_100',
    #                     file_to_execute_path='repos/parallel_trainRL.py',
    #                     python_interpreter='../RL_10_21/venv/bin/python',
    #                     on_cluster=False,
    #                     controller=LibudaDegradPC,
    #                     n_episodes=30_000,
    #                     unique_folder=False,
    #                     at_same_time=60
    #                     )


if __name__ == '__main__':
    main()
