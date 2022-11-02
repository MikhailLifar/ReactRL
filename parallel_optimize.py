from optimize_funcs import iter_optimize_cluster
# from parse_and_run import *
from test_models import *
from ProcessController import *
from predefined_policies import *
from targets_metrics import *


def main_cluster_function():
    # # EXPERIMENTAL DATA READING
    # dfs = []
    # exp_df1, T_exp, _ = read_exp_data('220422_O2_cutoff', 0)  # read raw data
    # dfs.append(exp_df1)
    # exp_df2, _, _ = read_exp_data('220422_CO_cutoff', 1)  # read raw data
    # dfs.append(exp_df2)
    # exp_df3, _, _ = read_exp_data('220422_switch', 2)  # read raw data
    # dfs.append(exp_df3)
    #
    # for i in range(len(dfs)):
    #     dfs[i] = cut_period_from_exp(dfs[i], 375, 915)  # slice a piece of data we need
    #     get_input_Pa_if_not_exist(dfs[i])
    #
    # # MODEL CONSTRUCTION
    # model_obj = models.LibudaModel_Parametrize(
    #                   params={'p1': 1.18e-4, 'p2': 20, 'p3': 1e-07, 'p4': 4.68, },
    #                   init_cond={'thetaCO': 0., 'thetaO': 0.})
    # model_id = 'libuda2001_p'
    #
    # if model_obj is None:
    #     model_obj = get_model(model_id)
    # model_bounds = get_model_bounds(model_id)
    # model_obj.change_temperature(T_exp)

    # MAIN PART

    # optimize_different_methods(create_func_to_approximate(exp_df, model_obj=model_obj,
    #                                                     label_name='CO2_Pa_out',
    #                                                     in_cols=['O2_Pa_in', 'CO_Pa_in'],
    #                                                     rename_dict={'O2_Pa_in': 'O2', 'CO_Pa_in': 'CO'},
    #                                                     conv_params={'RESOLUTION': 20}),
    #                            try_num=15,
    #                            optimize_bounds=model_bounds,
    #                            out_folder='parse_and_run_out/optimize',
    #                            debug_params={'DEBUG': True, 'folder': 'auto'},
    #                            cut_ends=(False, True))

    # iter_optimize_cluster(create_to_approximate_many_frames(dfs, model_obj=model_obj,
    #                                                     label_name='CO2_Pa_out',
    #                                                     in_cols=['O2_Pa_in', 'CO_Pa_in'],
    #                                                     rename_dict={'O2_Pa_in': 'O2', 'CO_Pa_in': 'CO'},
    #                                                     koefs=np.array([20, 10, 5]),
    #                                                     conv_params={'RESOLUTION': 20}),
    #                                                 method='Nelder-Mead',
    #                                                 try_num=10,
    #                                                 optimize_bounds=model_bounds,
    #                                                 out_folder='parse_and_run_out/optimize',
    #                                                 debug_params={'DEBUG': True, 'folder': 'auto', 'ind_picture': None},
    #                                                 cut_left=False, cut_right=False,)

    model_discret_step = 1.
    episode_time = 500

    # PC_LDegrad = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160,
    #                                                                                 v_d=0.01, v_r=0.1, border=4.),
    #                                          target_func_to_maximize=CO2_value)
    # PC_LDegrad.set_plot_params(output_lims=[0., 0.06], output_ax_name='CO2_formation_rate',
    #                            input_ax_name='Pressure, Pa')

    # Pt2210_PC = ProcessController(test_models.PtModel(init_cond={'thetaO': 0., 'thetaCO': 0.}),
    #                               target_func_to_maximize=CO2_value,
    #                               supposed_step_count=2 * round(episode_time / model_discret_step),  # memory controlling parameters
    #                               supposed_exp_time=2 * episode_time)
    # Pt2210_PC.set_plot_params(input_ax_name='Pressure, x10^5', input_lims=None,
    #                         output_ax_name='CO2 form. rate', output_lims=None)

    PC_LReturnK3K1 = ProcessController(LibudaModelReturnK3K1(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=273+160),
                                       target_func_to_maximize=CO2xConversion)
    # PC_PtReturnK3K1 = ProcessController(PtReturnK1K3Model(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=440),
    #                                     target_func_to_maximize=CO2xConversion,
    #                                     supposed_step_count=2 * (int(episode_time / model_discret_step) + 1),  # memory controlling parameters
    #                                     supposed_exp_time=2 * episode_time)

    PC_obj = PC_LReturnK3K1

    PC_obj.set_plot_params(output_lims=[0., None], output_ax_name='CO2_formation_rate',
                           input_ax_name='Pressure, Pa')
    # PC_obj.process_to_control.assign_and_eval_values(O2_top=2., CO_top=10.)

    iter_optimize_cluster(func_to_optimize_policy(PC_obj, ConstantPolicy(dict()), episode_time, model_discret_step),
                          optimize_bounds={
                              'O2_value': [0., 10e-5], 'CO_value': [0., 10e-5],
                              },
                          cut_left=False, cut_right=False,
                          method='Nelder-Mead',
                          try_num=30,
                          on_cluster=False,
                          python_interpreter='../RL_10_21/venv/bin/python',
                          file_to_execute_path='repos/parallel_optimize.py',
                          unique_folder=False,
                          out_path='optimize_out/221027_new_target1_optimize',
                          debug_params={'DEBUG': True, 'folder': 'auto'},
                          )

    # Pt TESTS
    # iter_optimize_cluster(func_to_optimize_policy(PC_LDegrad, SinPolicy(dict()), episode_time, 1.),
    #                       optimize_bounds={
    #                           'O2_A': [0., 10.e-5], 'O2_omega': [0.02 * np.pi, 0.2 * np.pi], 'O2_alpha': [0., 2 * np.pi], 'O2_bias': [0., 10.e-5],
    #                           'CO_A': [0., 10.e-5], 'CO_omega': [0.02 * np.pi, 0.2 * np.pi], 'CO_alpha': [0., 2 * np.pi], 'CO_bias': [0., 10.e-5],
    #                           },
    #                       cut_left=False, cut_right=False,
    #                       method='Nelder-Mead',
    #                       try_num=30,
    #                       on_cluster=False,
    #                       python_interpreter='../RL_10_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       unique_folder=False,
    #                       out_path='optimize_out/221020_test_new_func_sin',
    #                       debug_params={'DEBUG': True, 'folder': 'auto'},
    #                       )

    # iter_optimize_cluster(func_to_optimize_sin_sol(PC_LDegrad, episode_len=500,
    #                                                dt=50),
    #                       optimize_bounds={
    #                           'O2_A': [0., 10e-5], 'O2_k': [0., 0.1 * np.pi], 'O2_bias_t': [0., 2 * np.pi], 'O2_bias_f': [0., 10.e-5],
    #                           'CO_A': [0., 10e-5], 'CO_k': [0., 0.2 * np.pi], 'CO_bias_t': [0., 2 * np.pi], 'CO_bias_f': [0., 10.e-5],
    #                           },
    #                       cut_left=False, cut_right=False,
    #                       method='Nelder-Mead',
    #                       try_num=20,
    #                       on_cluster=False,
    #                       python_interpreter='../RL_10_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       unique_folder=False,
    #                       out_path='optimize_out/220816_LDegrad_sin_sol',
    #                       debug_params={'DEBUG': True, 'folder': 'auto'},
    #                       )

    # iter_optimize_cluster(func_to_optimize_two_step_sol(PC_LDegrad, episode_len=500),
    #                       optimize_bounds={
    #                           'O2_1': [0., 10e-5], 'CO_1': [0., 10e-5],
    #                           'O2_2': [0., 10e-5], 'CO_2': [0., 10e-5],
    #                           'time_1': [5., 500.], 'time_2': [5., 500.],
    #                           },
    #                       cut_left=False, cut_right=False,
    #                       method='Nelder-Mead',
    #                       try_num=20,
    #                       on_cluster=False,
    #                       python_interpreter='../RL_10_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       unique_folder=False,
    #                       out_path='optimize_out/220816_LDegrad_2_step',
    #                       debug_params={'DEBUG': True, 'folder': 'auto'},
    #                       )

    # iter_optimize_cluster(func_to_optimize_stationary_sol(PC_LDegrad, episode_len=500),
    #                       optimize_bounds={'O2': [0., 10e-5], 'CO': [0., 10e-5]}, cut_left=False, cut_right=False,
    #                       method='Nelder-Mead',
    #                       try_num=20,
    #                       on_cluster=False,
    #                       python_interpreter='../RL_10_21/venv/bin/python',
    #                       file_to_execute_path='parallel_optimize.py',
    #                       unique_folder=False,
    #                       out_path='optimize_out/220815_stationary',
    #                       debug_params={'DEBUG': True, 'folder': 'auto'},
    #                       )

    # run_df_and_plot_with_exp(exp_df, model_obj=model_obj, out_folder='parse_and_run_out')

    pass


if __name__ == '__main__':
    main_cluster_function()
