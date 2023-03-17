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

    episode_time = 500
    
    # PC_L2001_low_T = ProcessController(test_models.LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+25),
    #                                          target_func_to_maximize=target)
    #
    # PC_L2001_old = ProcessController(test_models.LibudaModel(init_cond={'thetaCO': 0.5, 'thetaO': 0.25, }, Ts=440.),
    #                                          target_func_to_maximize=target)
    #
    # PC_LDegrad_old = ProcessController(test_models.LibudaModelWithDegradation(init_cond={'thetaCO': 0.5, 'thetaO': 0.25, }, Ts=440.,
    #                                                                                 v_d=0.01, v_r=1.5, border=4.),
    #                                          target_func_to_maximize=target)
    #
    # PC_L2001 = ProcessController(test_models.LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160),
    #                                          target_func_to_maximize=target)
    #
    # PC_LDegrad = ProcessController(test_models.LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160,
    #                                                                                 v_d=0.01, v_r=0.1, border=4.),
    #                                          target_func_to_maximize=target)
    #
    # Pt2210_PC = ProcessController(test_models.PtModel(init_cond={'thetaO': 0., 'thetaCO': 0.}),
    #                               target_func_to_maximize=target,
    #                               supposed_step_count=2 * round(episode_time / time_step),  # memory controlling parameters
    #                               supposed_exp_time=2 * episode_time)

    # get_target_func('CO2xConversion_I', eps=1.)
    # PC_LExtendedReturn = ProcessController(LibudaModelReturnK3K1AndPressures(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=273 + 160),
    #                                        long_term_target_to_maximize=get_target_func('Gauss(CO_sub_default)xConvSum_I', default=1e-4,
    #                                                                                     sigma=1e-5 * episode_time, eps=1.),
    #                                        # target_func_to_maximize=CO2_sub_outs,
    #                                        )
    # PC_PtReturnK3K1 = ProcessController(PtReturnK1K3Model(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=440),
    #                                target_func_to_maximize=target_1,  # CO2_value, CO2xConversion, CO2_sub_outs
    #                                supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                supposed_exp_time=2 * episode_time)

    size = [10, 10]
    PC_KMC = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=True,
                                                  O2_top=1.1e5, CO_top=1.1e5,
                                                  CO2_rate_top=2.e5, CO2_count_top=2.e3,
                                                  T=373.),
                               analyser_dt=1.e-7,
                               target_func_to_maximize=get_target_func('CO2_count'),
                               target_func_name='CO2_count',
                               target_int_or_sum='sum',
                               RESOLUTION=1,  # ATTENTION! Always should be 1 if we use KMC, otherwise we will get wrong results!
                               supposed_step_count=100,  # memory controlling parameters
                               supposed_exp_time=1.e-5)

    PC_obj = PC_KMC
    # O2_top = 10.e-4
    # CO_top = 10.e-4
    # max_top = max(O2_top, CO_top)

    PC_obj.set_metrics(
                       # ('integral CO2', CO2_integral),
                       ('CO2 count', CO2_count),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
    )
    # PC_obj.process_to_control.assign_and_eval_values(O2_top=O2_top, CO_top=CO_top)
    # PC_obj.set_plot_params(output_lims=None, output_ax_name='?',
    #                        input_ax_name='Pressure, Pa')
    PC_KMC.set_plot_params(input_lims=[-1e-5, None], input_ax_name='Pressure, Pa',
                           output_lims=[-1e-2, None],
                           additional_lims=[-1e-2, 1. + 1.e-2],
                           # output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$',
                           output_ax_name='CO x O events count')

    # gauss_target_1 = get_target_func('Gauss(CO_sub_default)xConv_I', default=1e-4, sigma=1e-5 * episode_time, eps=1e-4)
    # gauss_target_2 = get_target_func('Gauss(CO_sub_default)xConvSum_I', default=1e-4, sigma=1e-5 * episode_time, eps=1e-4)
    # gauss_target_3 = get_target_func('Gauss(CO_sub_default)x(Conv+alpha)_I', default=1e-4, sigma=1e-5 * episode_time,
    #                                  alpha=0.2, eps=1e-4)
    # gauss_target_4 = get_target_func('Gauss(CO_sub_default)xConv_I', default=1e-3, sigma=1e-4 * 500, eps=1e-4)
    # gauss_target_5 = get_target_func('Gauss(CO_sub_default)xConvSum_I', default=1e-3, sigma=1e-4 * 500, eps=1e-4)
    # gauss_target_6 = get_target_func('Gauss(CO_sub_default)x(Conv+alpha)_I', default=1e-3, sigma=1e-4 * 500,
    #                                  alpha=0.2, eps=1e-4)
    # name1 = '(Gauss)x(Conv)x(Conv)'
    # name2 = '(Gauss)x(Conv + Conv)'
    # name3 = '(Gauss)x(Conv + alpha)x(Conv + alpha)'

    # EXTENSIONS

    # def same_period_ext(d):
    #     for name in ('t1', 't2'):
    #         d[f'O2_{name}'] = d[f'CO_{name}'] = d[name]
    #         del d[name]

    # def get_CO_fixed_ext(CO_value):
    #     def CO_fixed_ext(d):
    #         d['CO_2'] = d['CO_1'] = CO_value
    #         d['CO_t2'] = d['CO_t1'] = 10.
    #     return CO_fixed_ext

    # def get_fixed_sum_ext(sum_value):
    #     def fixed_sum_ext(d): d['CO_value'] = sum_value - d['O2_value']
    #     return fixed_sum_ext

    # def get_equal_periods_ext(n, t_value):
    #     def fixed_sum_ext(d):
    #         for i in range(n):
    #             d[f't{i}'] = t_value
    #     return fixed_sum_ext

    # def get_complicated_ext(sum_press, ncycle, t_value):
    #     def complicated_ext(d):
    #         for i in range(1, ncycle + 1):
    #             d[f'O2_t{i}'] = d[f'CO_t{i}'] = t_value
    #             d[f'CO_{i}'] = d[f'x{i}'] * sum_press
    #             d[f'O2_{i}'] = sum_press - d[f'CO_{i}']
    #             # del d[f'x{i}']
    #     return complicated_ext

    def get_discrete_turns_ext(sum_press, ncycle, t_value, min_bound=0.):
        def complicated_ext(d):
            O2_level = d['O2']
            CO_level = sum_press - d['O2']
            for i in range(1, ncycle + 1):
                d[f'O2_t{i}'] = d[f'CO_t{i}'] = t_value

                if d[f'alpha{i}'] < 1./3:
                    d[f'O2_{i}'] = min_bound
                    d[f'CO_{i}'] = CO_level
                elif d[f'alpha{i}'] < 2./3:
                    d[f'O2_{i}'] = O2_level
                    d[f'CO_{i}'] = min_bound
                else:
                    d[f'O2_{i}'] = O2_level
                    d[f'CO_{i}'] = CO_level

        return complicated_ext

    # OPTIMIZER CALL
    episode_time = 1.e-6
    sum_of_pressures = 1.e+5
    cyclesteps = 2
    nsteps = 4
    # ext = get_fixed_sum_ext(sum_of_pressures)
    # ext = get_equal_periods_ext(4, 2.e-7)
    # ext = get_complicated_ext(sum_of_pressures, cyclesteps, episode_time / nsteps)
    ext = get_discrete_turns_ext(sum_of_pressures, cyclesteps, episode_time / nsteps)
    iter_optimize_cluster(func_to_optimize_policy(
                                PC_obj, AnyStepPolicy(cyclesteps, dict()), episode_time, episode_time / 50,
                                expand_description=ext,
                                to_plot={'out_names': ['CO2_count'], 'additional_plot': ['thetaCO', 'thetaO']}),
                          optimize_bounds={
                              # f'x{i}': (0., 1.) for i in range(1, cyclesteps + 1)
                              'O2': (0., 1.e+5),
                              **{f'alpha{i}': (0., 1.) for i in range(1, cyclesteps + 1)},
                              },
                          cut_left=False, cut_right=False,
                          method='Nelder-Mead',
                          try_num=30,
                          on_cluster=False,
                          python_interpreter='../RL_10_21/venv/bin/python',
                          file_to_execute_path='repos/parallel_optimize.py',
                          unique_folder=False,
                          out_path='optimize_out/220317_discrete_turns_debug',
                          debug_params={'DEBUG': True, 'folder': 'auto'},
                          )

    # optimize_grid_cluster([(gauss_target_1, name1), (gauss_target_2, name2), (gauss_target_3, name3)],
    #                       [(1.e-5, 1.e-4), (1.e-4, 1.e-4), (1.e-3, 1.e-3), (1.e-2, 1.e-2)],
    #                       names=(('long_term_target', 'target_func_name'), ('model:O2_top', 'model:CO_top')),
    #                       policy_type=TwoStepPolicy,
    #                       optimize_bounds={
    #                           'O2_1': 'model_lims', 'O2_2': 'model_lims',
    #                           # 'O2_t1': [5., 100.], 'O2_t2': [5., 100.],
    #                           'CO_1': 'model_lims', 'CO_2': 'model_lims',
    #                           # 'CO_t1': [5., 100.], 'CO_t2': [5., 100.],
    #                           't1': [5., 100.], 't2': [5., 100.],
    #                       },
    #                       out_path='./optimize_out/221124_many_targets_same_time',
    #                       PC_obj=PC_obj,
    #                       const_params={
    #                           'model': {
    #
    #                           },
    #                           'iter_optimize': {
    #                               'method': 'Nelder-Mead',
    #                               'try_num': 30,
    #                               'debug_params': {'folder': 'auto', 'DEBUG': True, 'ind_picture': True},
    #                               'cut_left': False,
    #                               'cut_right': False,
    #                           },
    #                           'expand_description': same_period_ext,
    #                           'episode_time': 500.,
    #                           'time_step': 1.,
    #                           'to_plot': {'out_names': ['CO2', 'long_term_target'],
    #                                       'additional_plot': ['thetaCO', 'thetaO']}
    #                       },
    #                       python_interpreter='../RL_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       on_cluster=False)

    # optimize_list_cluster([(1.e-5, 1.e-4, same_period_ext, gauss_target_1, name1),  # target 1
    #                        (1.e-4, 1.e-4, same_period_ext, gauss_target_1, name1),
    #                        (1.e-3, 1.e-3, same_period_ext, gauss_target_1, name1),
    #                        (1.e-2, 1.e-2, same_period_ext, gauss_target_1, name1),
    #                        (1.e-5, 1.e-4, same_period_ext, gauss_target_2, name2),  # target 2
    #                        (1.e-4, 1.e-4, same_period_ext, gauss_target_2, name2),
    #                        (1.e-3, 1.e-3, same_period_ext, gauss_target_2, name2),
    #                        (1.e-2, 1.e-2, same_period_ext, gauss_target_2, name2),
    #                        (1.e-5, 1.e-4, same_period_ext, gauss_target_3, name3),  # target 3
    #                        (1.e-4, 1.e-4, same_period_ext, gauss_target_3, name3),
    #                        (1.e-3, 1.e-3, same_period_ext, gauss_target_3, name3),
    #                        (1.e-2, 1.e-2, same_period_ext, gauss_target_3, name3),
    #                        ],
    #                       ('model:O2_top', 'model:CO_top', 'expand_description', 'long_term_target', 'target_func_name'),
    #                       TwoStepPolicy,
    #                       {
    #                           'O2_1': 'model_lims', 'O2_2': 'model_lims',
    #                           # 'O2_t1': [5., 100.], 'O2_t2': [5., 100.],
    #                           'CO_1': 'model_lims', 'CO_2': 'model_lims',
    #                           # 'CO_t1': [5., 100.], 'CO_t2': [5., 100.],
    #                           't1': [5., 100.], 't2': [5., 100.],
    #                       },
    #                       out_path='./optimize_out/221124_many_targets_same_time',
    #                       PC_obj=PC_obj,
    #                       const_params={
    #                           'model': {
    #
    #                           },
    #                           'iter_optimize': {
    #                               'method': 'Nelder-Mead',
    #                               'try_num': 30,
    #                               'debug_params': {'folder': 'auto', 'DEBUG': True, 'ind_picture': True},
    #                               'cut_left': False,
    #                               'cut_right': False,
    #                           },
    #                           'episode_time': 500.,
    #                           'time_step': 1.,
    #                           'to_plot': {'out_names': ['CO2', 'long_term_target'],
    #                                       'additional_plot': ['thetaCO', 'thetaO']}
    #                       },
    #                       python_interpreter='../RL_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       on_cluster=False)

    # CO2_fixed_00001 = get_CO_fixed_ext(1.e-4)
    # CO2_fixed_0001 = get_CO_fixed_ext(1.e-3)
    # optimize_list_cluster([(10.e-5, 1.e-3, CO2_fixed_00001),
    #                        (10.e-4, 1.e-3, CO2_fixed_00001),
    #                        (10.e-3, 1.e-3, CO2_fixed_00001),
    #                        (10.e-2, 1.e-3, CO2_fixed_00001),
    #                        (10.e-4, 1.e-2, CO2_fixed_0001),
    #                        (10.e-3, 1.e-2, CO2_fixed_0001),
    #                        (10.e-2, 1.e-2, CO2_fixed_0001),
    #                        (10.e-1, 1.e-2, CO2_fixed_0001),
    #                        ],
    #                       ('model:O2_top', 'model:CO_top', 'expand_description'),
    #                       TwoStepPolicy,
    #                       {
    #                           'O2_1': 'model_lims', 'O2_2': 'model_lims',
    #                           'O2_t1': [5., 100.], 'O2_t2': [5., 100.],
    #                           #'CO_1': 'model_lims', 'CO_2': 'model_lims',
    #                           #'CO_t1': [5., 100.], 'CO_t2': [5., 100.],
    #                       },
    #                       out_path='./optimize_out/221110_opt_list_CO2xConv',
    #                       PC_obj=PC_obj,
    #                       const_params={
    #                           'model': {
    #
    #                           },
    #                           'iter_optimize': {
    #                               'method': 'Nelder-Mead',
    #                               'try_num': 30,
    #                               'debug_params': {'folder': 'auto', 'DEBUG': True, 'ind_picture': True},
    #                               'cut_left': False,
    #                               'cut_right': False,
    #                           },
    #                           'episode_time': 500.,
    #                           'time_step': 1.,
    #                           'to_plot': {'out_names': ['CO2', 'long_term_target'],
    #                                       'additional_plot': ['thetaCO', 'thetaO']}
    #                       },
    #                       python_interpreter='../RL_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       on_cluster=False)
    
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

    #iter_optimize_cluster(func_to_optimize_stationary_sol(PC_LDegrad, episode_time=500),
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
