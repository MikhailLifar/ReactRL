from parallel_trainRL_funcs import *
from test_models import *


def target(x):
    return x[0]


# train_grid_parallel(['full_ep_mean', 'full_ep_mean', 'each_step_base', 'each_step_base'],
#                      names=('env:reward_spec', ),
#                      const_params={
#                          'env': {'model_type': 'continuous',
#                                  'state_spec': {
#                                      'rows': 1,
#                                      'use_differences': False,
#                                      },
#                                  'episode_time': 500},
#                          'agent_name': 'vpg',
#                      },
#                      out_path='run_RL_out/current_training/diff_rewards',
#                      python_interpreter='../RL_10_21/venv/bin/python',
#                      on_cluster=False,
#                      controller=ProcessController(TestModel(), target_func_to_maximize=target,
#                                                   supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
#                                                   supposed_exp_time=2 * episode_time),
#                      n_episodes=10_000,
#                      unique_folder=False,)

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
#                                                        Ts=273+160), target_func_to_maximize=target,
#                                                   supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
#                                                   supposed_exp_time=2 * episode_time),
#                      n_episodes=30000,
#                      unique_folder=False,
#                      at_same_time=45)

episode_time = 500
time_step = 10
LibudaDegradPC = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0.},
                                                              Ts=273+160),
                                   target_func_to_maximize=target,
                                   supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
                                   supposed_exp_time=2 * episode_time)
LibudaDegradPC.set_plot_params(input_ax_name='Pressure', input_lims=None,
                               output_lims=[0., 0.06], output_ax_name='CO2_formation_rate')

train_grid_parallel(['vpg', 'ppo'],
                    [(dict(type='layered', layers=[dict(type='flatten'),
                                                   dict(type='dense', size=32, activation='relu')]),
                      dict(optimizer='adam', learning_rate=1e-3)),
                     (dict(type='layered', layers=[dict(type='flatten'),
                                                   dict(type='dense', size=32, activation='relu')]),
                      0.3),
                     (dict(type='layered', layers=[dict(type='flatten'),
                                                   dict(type='dense', size=32, activation='relu')]),
                      1.)],
                    ['full_ep_mean', 'full_ep_base', 'each_step_base'],
                    names=('agent_name',
                           ('agent:baseline',
                            'agent:baseline_optimizer',),
                           'env:reward_spec'),
                    const_params={
                         'env': {'model_type': 'continuous',
                                 'episode_time': episode_time,
                                 'time_step': time_step,
                                 'state_spec': {'rows': 3, 'use_differences': False},
                                 'log_scaling_dict': None,
                                 },
                         'model': {
                             'O2_top': 10.e-5,
                             'CO_top': 10.e-5,
                             'v_r': 0.1,
                             'v_d': 0.01,
                             'border': 4.,
                             },
                         'agent': {
                             'batch_size': 32,
                             'update_frequency': 32,
                             'network': dict(type='layered', layers=[dict(type='flatten'),
                                                                     dict(type='dense', size=32, activation='relu')])
                         }
                     },
                    repeat=3,
                    out_path='run_RL_out/current_training/220826_vpg_ppo_baseline',
                    file_to_execute_path='repos/parallel_trainRL.py',
                    python_interpreter='../RL_10_21/venv/bin/python',
                    on_cluster=False,
                    controller=LibudaDegradPC,
                    n_episodes=30_000,
                    unique_folder=False,
                    at_same_time=60)

# train_grid_parallel([('vpg', '#exclude'),
#                      ('ac', dict(type='layered', layers=[dict(type='flatten'), dict(type='dense', size=16, activation='relu')])),
#                      ('ac', dict(type='layered', layers=[dict(type='flatten'), dict(type='dense', size=16, activation='relu'),
#                                                           dict(type='dense', size=16, activation='relu')])), ],
#                     [dict(type='layered', layers=[dict(type='flatten'), dict(type='dense', size=16, activation='relu')]),
#                      dict(type='layered', layers=[dict(type='flatten'), dict(type='dense', size=16, activation='relu'),
#                                                   dict(type='dense', size=16, activation='relu')])],
#                     ['full_ep_mean', 'full_ep_base', 'each_step_base'],
#                     names=(('agent_name', 'agent:critic'), 'agent:network', 'env:reward_spec'),
#                     const_params={
#                          'env': {'model_type': 'continuous',
#                                  'episode_time': episode_time,
#                                  'time_step': time_step,
#                                  'state_spec': {'rows': 3, 'use_differences': False},
#                                  'log_scaling_dict': None,
#                                  },
#                          'model': {
#                              'O2_top': 10.e-5,
#                              'CO_top': 10.e-5,
#                              'v_r': 0.1,
#                              'v_d': 0.01,
#                              'border': 4.,
#                              }
#                      },
#                     repeat=3,
#                     out_path='run_RL_out/current_training/220822_vpg_ac_custom_nets',
#                     file_to_execute_path='repos/parallel_trainRL.py',
#                     python_interpreter='../RL_10_21/venv/bin/python',
#                     on_cluster=False,
#                     controller=LibudaDegradPC,
#                     n_episodes=30_000,
#                     unique_folder=False,
#                     at_same_time=60)

# train_grid_parallel([(100.e-5, (1, False), 1e-4, 0.7), (10.e-5, (2, False), 1e-3, 0.7), (1.e-5, (3, False), 1e-4, 0.7), ],
#                     ['full_ep_mean', 'each_step_base'],
#                     names=(('model:CO_top', 'env:state_spec',
#                             'agent:learning_rate', 'agent:entropy_regularization'),
#                            'env:reward_spec'),
#                     const_params={
#                          'env': {'model_type': 'continuous',
#                                  'episode_time': episode_time,
#                                  'time_step': time_step,
#                                  'log_scaling_dict': None,
#                                  },
#                          'agent_name': 'vpg',
#                          'model': {
#                              'O2_top': 10.e-5,
#                              'v_r': 0.1,
#                              'v_d': 0.01,
#                              'border': 4.,
#                              }
#                      },
#                     repeat=3,
#                     out_path='run_RL_out/current_training/220810_repeat_old',
#                     file_to_execute_path='repos/parallel_trainRL.py',
#                     python_interpreter='../RL_10_21/venv/bin/python',
#                     on_cluster=False,
#                     controller=LibudaDegradPC,
#                     n_episodes=1_000,
#                     unique_folder=False,
#                     at_same_time=45)
