from parallel_trainRL_funcs import *
from test_models import *


def target(x):
    return x[0]


#train_greed_parallel(['full_ep_mean', 'full_ep_mean', 'each_step_base', 'each_step_base'],
                     #names=('env:reward_spec', ),
                     #const_params={
                         #'env': {'model_type': 'continuous',
                                 #'state_spec': {
                                     #'rows': 1,
                                     #'use_differences': False,
                                     #},
                                 #'episode_time': 500},
                         #'agent_name': 'vpg',
                     #},
                     #out_path='run_RL_out/current_training/diff_rewards',
                     #python_interpreter='../RL_10_21/venv/bin/python',
                     #on_cluster=False,
                     #controller=ProcessController(TestModel(), target_func_to_maximize=target,
                                                  #supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
                                                  #supposed_exp_time=2 * episode_time),
                     #n_episodes=10_000,
                     #unique_folder=False,)

episode_time = 500
time_step = 10
# train_greed_parallel(['full_ep_mean', 'full_ep_base', 'each_step_base'],
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

PC = ProcessController(LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0.}, Ts=273+160),
                       target_func_to_maximize=target,
                       supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
                       supposed_exp_time=2 * episode_time)
PC.set_plot_params(input_ax_name='Pressure', output_lims=[0., 0.05], output_ax_name='CO2_formation_rate')
train_list_parallel([[1.e-5, 10.e-5, (1, False)],
                     [1.e-5, 10.e-5, (2, False)],
                     [1.e-5, 10.e-5, (3, False)],
                     [10.e-5, 100.e-5, (1, False)],
                     [10.e-5, 100.e-5, (2, False)],
                     [10.e-5, 100.e-5, (3, False)],
                     [100.e-5, 1.e-2, (1, False)],
                     [100.e-5, 1.e-2, (2, False)],
                     [100.e-5, 1.e-2, (3, False)]],
                    names=('model:O2_top', 'model:CO_top', 'env:state_spec'),
                    const_params={
                         'env': {'model_type': 'continuous',
                                 'time_step': time_step,
                                 'reward_spec': 'full_ep_mean',
                                 'log_scaling_dict': {'CO2': 10},
                                 },
                         'agent_name': 'vpg',
                     },
                    repeat=3,
                    out_path='run_RL_out/current_training/220808_diff_limitations',
                    file_to_execute_path='code/parallel_trainRL.py',
                    python_interpreter='../RL_10_21/venv/bin/python',
                    on_cluster=False,
                    controller=PC,
                    n_episodes=30_000,
                    unique_folder=False,
                    at_same_time=45)
