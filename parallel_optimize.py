from optimize_funcs import *
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
    
    #PC_L2001_low_T = ProcessController(test_models.LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+25),
                                             #target_func_to_maximize=target)
    
    #PC_L2001_old = ProcessController(test_models.LibudaModel(init_cond={'thetaCO': 0.5, 'thetaO': 0.25, }, Ts=440.),
                                             #target_func_to_maximize=target)
    
    #PC_LDegrad_old = ProcessController(test_models.LibudaModelWithDegradation(init_cond={'thetaCO': 0.5, 'thetaO': 0.25, }, Ts=440.,
                                                                                    #v_d=0.01, v_r=1.5, border=4.),
                                             #target_func_to_maximize=target)
    
    #PC_L2001 = ProcessController(test_models.LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160),
                                             #target_func_to_maximize=target)
    
    #PC_LDegrad = ProcessController(test_models.LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160,
                                                                                    #v_d=0.01, v_r=0.1, border=4.),
                                             #target_func_to_maximize=target)
    
    #Pt2210_PC = ProcessController(test_models.PtModel(init_cond={'thetaO': 0., 'thetaCO': 0.}),
                                  #target_func_to_maximize=target,
                                  #supposed_step_count=2 * round(episode_time / time_step),  # memory controlling parameters
                                  #supposed_exp_time=2 * episode_time)
    
    PC_LReturnK3K1 = ProcessController(LibudaModelReturnK3K1(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=273+160),
                                       long_term_target_to_maximize=get_target_func('CO2xConversion_I', eps=1.),
                                       #target_func_to_maximize=CO2_sub_outs,
                                   )
    #PC_PtReturnK3K1 = ProcessController(PtReturnK1K3Model(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=440),
                                   #target_func_to_maximize=target_1,  # CO2_value, CO2xConversion, CO2_sub_outs
                                   #supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
                                   #supposed_exp_time=2 * episode_time)

    PC_obj = PC_LReturnK3K1
    # O2_top = 10.e-4
    # CO_top = 10.e-4
    # max_top = max(O2_top, CO_top)

    PC_obj.set_metrics(('integral CO2', CO2_integral),
                   ('O2 conversion', overall_O2_conversion),
                   ('CO conversion', overall_CO_conversion))
    # PC_obj.process_to_control.assign_and_eval_values(O2_top=O2_top, CO_top=CO_top)
    PC_obj.set_plot_params(output_lims=None, output_ax_name='?',
                           input_ax_name='Pressure, Pa')

    optimize_list_cluster([(10.e-5, 10.e-5, 'Nelder-Mead'),
                           (10.e-4, 10.e-5, 'Powell'),
                           (10.e-5, 10.e-4, 'Nelder-Mead'),
                           (10.e-4, 10.e-4, 'Powell')],
                          ('model:O2_top', 'model:CO_top', 'iter_optimize:method'),
                          TwoStepPolicy,
                          {
                              'O2_1': 'model_lims', 'O2_2': 'model_lims',
                              'O2_t1': [5., 100.], 'O2_t2': [5., 100.],
                              'CO_1': 'model_lims', 'CO_2': 'model_lims',
                              'CO_t1': [5., 100.], 'CO_t2': [5., 100.],
                          },
                          './optimize_out/221109_opt_list_1st_try',
                          PC_obj,
                          const_params={
                              'model': {

                              },
                              'iter_optimize': {
                                  'try_num': 30,
                                  'debug_params': {'folder': 'auto', 'DEBUG': True, 'ind_picture': True},
                                  'cut_left': False,
                                  'cut_right': False,
                              },
                              'episode_len': 500.,
                              'time_step': 1.,
                              'to_plot': {'out_names': ['CO2', 'long_term_target'],
                                          'additional_plot': ['thetaCO', 'thetaO']}
                          },
                          python_interpreter='../RL_21/venv/bin/python',
                          file_to_execute_path='repos/parallel_optimize.py',
                          on_cluster=False)
    
    # iter_optimize_cluster(
    #                       # func_to_optimize_two_step_sol(PC_obj, 500, 1.,
    #                                                     # to_plot={'out_names': ['CO2', 'long_term_target'], 'additional_plot': ['theta_CO', 'theta_O']}),
    #                       func_to_optimize_policy(PC_obj, TwoStepPolicy(dict()), 500, 1.,
    #                                               to_plot={'out_names': ['CO2', 'long_term_target'], 'additional_plot': ['thetaCO', 'thetaO']}),
    #                       optimize_bounds={
    #                           'O2_1': [0., O2_top], 'O2_2': [0., O2_top],
    #                           'O2_t1': [5., 100.], 'O2_t2': [5., 100.],
    #                           'CO_1': [0., CO_top], 'CO_2': [0., CO_top],
    #                           'CO_t1': [5., 100.], 'CO_t2': [5., 100.],
    #                           # 'time_1': [5., 100.], 'time_2': [5., 100.],
    #                           },
    #                       cut_left=False, cut_right=False,
    #                       method='Nelder-Mead',
    #                       try_num=30,
    #                       on_cluster=False,
    #                       python_interpreter='../RL_10_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       unique_folder=False,
    #                       out_path='optimize_out/221109_fig_debug_common_func',
    #                       debug_params={'DEBUG': True, 'folder': 'auto'},
    #                       )
    
    #iter_optimize_cluster(func_to_optimize_policy(PC_obj, ConstantPolicy(dict()), 500, 1.),
                          #optimize_bounds={
                              #'O2_value': [0., 10e-5], 'CO_value': [0., 10e-5],
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=30,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/221028_Pd_fixed_CO2_subs_outs',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)
    
    #iter_optimize_cluster(func_to_optimize_policy(PC_obj, TwoStepPolicy(dict()), 500, 1.),
                          #optimize_bounds={
                              #'O2_1': [0., 10e-5], 'CO_1': [0., 10e-5],
                              #'O2_2': [0., 10e-5], 'CO_2': [0., 10e-5],
                              #'time_1': [10., 50.], 'time_2': [10., 50.],
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=30,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/221021_old_func_two_step',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)
    
    # Pt TESTS
    #Pt2210_PC.process_to_control.assign_and_eval_values(O2_top=2., CO_top=10.)
    #iter_optimize_cluster(func_to_optimize_sin_sol(Pt2210_PC, episode_time, 1.),
                          #optimize_bounds={
                              #'O2_A': [0., 2.], 'O2_k': [0.02 * np.pi, 0.2 * np.pi], 'O2_bias_t': [0., 2 * np.pi], 'O2_bias_f': [0., 2.],
                              #'CO_A': [0., 10.], 'CO_k': [0.02 * np.pi, 0.2 * np.pi], 'CO_bias_t': [0., 2 * np.pi], 'CO_bias_f': [0., 10.],
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=30,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/221012_Pt_2210_sin_less_O2',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)
    
    #PC_Pt2210.process_to_control.assign_and_eval_values(O2_top=2., CO_top=10.)    # assign limitations
    #iter_optimize_cluster(func_to_optimize_two_step_sol(PC_LDegrad, 500, 1.),
                          #optimize_bounds={
                              #'O2_1': [0., 10e-5], 'CO_1': [0., 10e-5],
                              #'O2_2': [0., 10e-5], 'CO_2': [0., 10e-5],
                              #'time_1': [10., 50.], 'time_2': [10., 50.],
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=30,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/221021_old_func_two_step',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)
    
    # new generic func to optimize any policy
    #iter_optimize_cluster(func_to_optimize_policy(PC_LDegrad, SinPolicy(dict()), episode_time, time_step),
                          #optimize_bounds={
                              #'O2_A': [0., 10.e-5], 'O2_omega': [0., 0.1 * np.pi], 'O2_alpha': [0., 2 * np.pi], 'O2_bias': [0., 10.e-5],
                              #'CO_A': [0., 10.e-5], 'CO_omega': [0., 0.1 * np.pi], 'CO_alpha': [0., 2 * np.pi], 'CO_bias': [0., 10.e-5],
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=30,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/221020_old_func_old_bounds_sin',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)
    
    #iter_optimize_cluster(func_to_optimize_stationary_sol(PC_L2001_low_T, 500),
                          #optimize_bounds={
                              #'O2': [0., 10e-5], 'CO': [0., 40e-5]
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=20,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/220929_L2001_T25_O210_CO40',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)
    
    #iter_optimize_cluster(func_to_optimize_sin_sol(PC_L2001, 500, 1.),
                          #optimize_bounds={
                              #'O2_A': [0., 10.e-5], 'O2_k': [0., 0.1 * np.pi], 'O2_bias_t': [0., 2 * np.pi], 'O2_bias_f': [0., 10.e-5],
                              #'CO_A': [0., 10.e-5], 'CO_k': [0., 0.1 * np.pi], 'CO_bias_t': [0., 2 * np.pi], 'CO_bias_f': [0., 10.e-5],
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=30,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/220817_L2001_sin_sol_Nelder_Mead',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)
    
    #iter_optimize_cluster(func_to_optimize_two_step_sol(PC_L2001, 500, 10.),
                          #optimize_bounds={
                              #'O2_1': [0., 10e-5], 'CO_1': [0., 10e-5],
                              #'O2_2': [0., 10e-5], 'CO_2': [0., 10e-5],
                              #'time_1': [0., 500.], 'time_2': [0., 500.],
                              #},
                          #cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=30,
                          #on_cluster=True,
                          #python_interpreter='../RL_21/venv/bin/python',
                          #file_to_execute_path='code/parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/220817_L2001_2_step_Nelder_Mead',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)

    #iter_optimize_cluster(func_to_optimize_stationary_sol(PC_LDegrad, episode_len=500),
                          #optimize_bounds={'O2': [0., 10e-5], 'CO': [0., 10e-5]}, cut_left=False, cut_right=False,
                          #method='Nelder-Mead',
                          #try_num=20,
                          #on_cluster=False,
                          #python_interpreter='../RL_10_21/venv/bin/python',
                          #file_to_execute_path='parallel_optimize.py',
                          #unique_folder=False,
                          #out_path='optimize_out/220815_stationary',
                          #debug_params={'DEBUG': True, 'folder': 'auto'},
                          #)

    # run_df_and_plot_with_exp(exp_df, model_obj=model_obj, out_folder='parse_and_run_out')

    pass


if __name__ == '__main__':
    main_cluster_function()
