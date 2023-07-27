from optimize_funcs import *
# from parse_and_run import *
from test_models import *
from ProcessController import *
from predefined_policies import *
from targets_metrics import *
import PC_setup

from multiple_jobs_functions import run_jobs_list

from PC_run import get_to_optimize_SBP_const_ratio


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

    # episode_time = 500

    # DEBUG
    # def test_func(func_description: dict, DEBUG=False, folder=None, ind_picture=None):
    #     x = func_description['x']
    #     return (x - 2) ** 4 + 1
    
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

    # # O2_top = 10.e-4
    # # CO_top = 10.e-4
    # # max_top = max(O2_top, CO_top)
    #
    # # PC_obj.process_to_control.assign_and_eval_values(O2_top=O2_top, CO_top=CO_top)
    # # PC_obj.set_plot_params(output_lims=None, output_ax_name='?',
    # #                        input_ax_name='Pressure, Pa')

    # PC_obj = PC_setup.default_PC_setup('Pd_monte_coffee')
    # PC_obj = PC_setup.default_PC_setup('Ziff')
    # PC_obj = PC_setup.general_PC_setup('Ziff', ('to_PC_constructor', 'analyser_dt', 5.e+1))
    # PC_obj = PC_setup.general_PC_setup('Libuda2001', ('to_PC_constructor',
    #                                                   {'target_func_to_maximize': None,
    #                                                    'long_term_target_to_maximize': get_target_func('CO2_plus_CO_conv_I', eps=1.e-5, alpha=1.),
    #                                                    'target_func_name': 'CO2_plus_CO_conv_I', }))
    PC_obj = PC_setup.general_PC_setup('LibudaG',
                                       ('to_PC_constructor', {'long_term_target_to_maximize': get_target_func('CO2_plus_CO_conv_I', eps=1.e-3, alpha=1., beta=1/0.02),
                                                              'target_func_to_maximize': None
                                                              }),
                                       )

    # PC_obj.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
    #                                       'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'thetaA_max': 0.5, 'thetaB_max': 0.5, })
    # PC_obj.process_to_control.set_Libuda()
    PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0., })

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
    #     def equal_periods_ext(d):
    #         for i in range(n):
    #             d[f't{i}'] = t_value
    #     return equal_periods_ext

    # def get_complicated_ext(sum_press, ncycle, t_value):
    #     def complicated_ext(d):
    #         for i in range(1, ncycle + 1):
    #             d[f'O2_t{i}'] = d[f'CO_t{i}'] = t_value
    #             d[f'CO_{i}'] = d[f'x{i}'] * sum_press
    #             d[f'O2_{i}'] = sum_press - d[f'CO_{i}']
    #             # del d[f'x{i}']
    #     return complicated_ext

    # def get_discrete_turns_ext(sum_press, ncycle, t_value, min_bound=0.):
    #     def complicated_ext(d):
    #         O2_level = d['O2']
    #         CO_level = sum_press - d['O2']
    #         for i in range(1, ncycle + 1):
    #             d[f'O2_t{i}'] = d[f'CO_t{i}'] = t_value
    #
    #             if d[f'alpha{i}'] < 1./3:
    #                 d[f'O2_{i}'] = min_bound
    #                 d[f'CO_{i}'] = CO_level
    #             elif d[f'alpha{i}'] < 2./3:
    #                 d[f'O2_{i}'] = O2_level
    #                 d[f'CO_{i}'] = min_bound
    #             else:
    #                 d[f'O2_{i}'] = O2_level
    #                 d[f'CO_{i}'] = CO_level
    #
    #     return complicated_ext

    # def get_switch_between_2_ext(regimes_dict, steps):
    #
    #     def switch_between_pure(d):
    #         if ('total' in d) and ('first_part' in d):
    #             d['t1'], d['t2'] = d['first_part'] * d['total'], (1 - d['first_part']) * d['total']
    #
    #         for i in range(1, steps + 1):
    #             d[f'O2_t{i}'] = d[f'CO_t{i}'] = d[f't{i}']
    #
    #         d.update(regimes_dict)
    #
    #     return switch_between_pure

    # def get_SBP_ext_for_Ziff(first_turned):
    #
    #     def switch_between_pure(d):
    #         d['x_1'] = 1. - (first_turned == 'O2') * 1.
    #         d['x_2'] = 1. - (first_turned == 'CO') * 1.
    #
    #         if ('total' in d) and ('first_part' in d):
    #             d['t1'], d['t2'] = int(d['first_part'] * d['total']), int((1 - d['first_part']) * d['total'])
    #
    #         d['x_t1'], d['x_t2'] = d['t1'], d['t2']
    #
    #     return switch_between_pure

    # def get_SBPOverlap_ext_Ziff():
    #     # for 4 step policy
    #     # output x1..x4, x_t1..x_t4
    #
    #     def complicated_ext(d):
    #         d['x_1'], d['x_3'] = 0., 1.
    #         d['x_2'] = d['x_4'] = 0.5
    #
    #     return complicated_ext

    # def get_O2_CO_from_x_co_ext(pressures_sum):
    #     def new_ext(d):
    #         d['O2_value'] = pressures_sum * (1 - d['x_value'])
    #         d['CO_value'] = pressures_sum * d['x_value']
    #
    #     return new_ext

    # OPTIMIZER CALL

    # LibudaG
    # rates optimization
    def log10_space_constrains(d):
        for k, v in d.items():
            d[k] = pow(10, v)

    run_jobs_list(**get_for_repeated_opt_iterations(get_to_optimize_SBP_const_ratio(
                                                        PC_obj,
                                                        np.array([1., 0.]), np.array([0., 1.]), np.array([2., 200.]),
                                                        resolutions=[3, 3, 3]),
                                                    optimize_bounds={f'rate_{suff}': (-2., 1.) for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')},
                                                    constrains=log10_space_constrains,
                                                    cut_left=False, cut_right=False,
                                                    method='Nelder-Mead', try_num=50,
                                                    call_after_opt_params={'plot_both_best': True, 'folder': 'auto'},
                                                    optimize_options={'maxiter': 3}),

                  const_params={},
                  PC=PC_obj,
                  repeat=1,
                  sort_iterations_by='fvalue',
                  cluster_command_ops=False,
                  python_interpreter='../RL_10_21/venv/bin/python',
                  out_fold_path='./optimize_out/LibudaG/230726_ratio_if_conversion_alpha1',
                  separate_folds=False,
                  at_same_time=30,
                  )

    # vanilla x_co control
    # def x_co_constrain(d):
    #     d['inputB_value'] = 1 - d['x_A']
    #     d['inputA_value'] = d['x_A']
    #
    # episode_time = 50
    # run_jobs_list(**get_for_repeated_opt_iterations(func_to_optimize_policy(
    #                                                     PC_obj,
    #                                                     # AnyStepPolicy(cyclesteps, dict()),
    #                                                     ConstantPolicy(),
    #                                                     episode_time, episode_time / 1000,
    #                                                     t_start_count_from=30.),
    #                                                 # optimize_bounds={'inputB_value': (0., 1.), 'inputA_value': (0., 1.)},
    #                                                 optimize_bounds={'x_A': (0., 1.)},
    #                                                 constrains=x_co_constrain,
    #                                                 cut_left=False, cut_right=False,
    #                                                 method='Nelder-Mead', try_num=10,
    #                                                 debug_params={'DEBUG': True, 'folder': 'auto'},
    #                                                 optimize_options={'maxiter': 500}),
    #
    #               const_params={},
    #               PC=PC_obj,
    #               repeat=1,
    #               sort_iterations_by='fvalue',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./optimize_out/LibudaG/230721_libuda_x_co_opt',
    #               separate_folds=False,
    #               at_same_time=30,
    #               )

    # L2001
    # CO2_plus_CO_conv_reward
    # cyclesteps = 2
    # episode_time = 500
    # run_jobs_list(**get_for_repeated_opt_iterations(func_to_optimize_policy(
    #                                                     PC_obj,
    #                                                     AnyStepPolicy(cyclesteps, dict()),
    #                                                     episode_time, episode_time / 1000,
    #                                                     expand_description=get_switch_between_2_ext({'O2_1': 0., 'CO_1': 1.e-4, 'O2_2': 1.e-4, 'CO_2': 0.}, 2),),
    #                                                 optimize_bounds={'t1': (2., 100.), 't2': (2., 100.)},
    #                                                 cut_left=False, cut_right=False,
    #                                                 method='Nelder-Mead', try_num=30,
    #                                                 debug_params={'DEBUG': True, 'folder': 'auto'},
    #                                                 optimize_options={'maxiter': 3}),
    #
    #               const_params={},
    #               PC=PC_obj,
    #               repeat=1,
    #               sort_iterations_by='fvalue',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./optimize_out/230706_L2001_CO2_plus_CO_conv_I',
    #               separate_folds=False,
    #               at_same_time=30,
    #               )

    # SBP

    # # PdMCoffee
    # episode_time = 2.e-6
    # # sum_of_pressures = 1.e+5
    # cyclesteps = 2
    # # nsteps = 4
    # # ext = get_discrete_turns_ext(sum_of_pressures, cyclesteps, episode_time / nsteps)
    # ext = get_switch_between_pure_ext({'O2': 1.e+5, 'CO': 1.e+5, }, first_turned='O2')

    # Ziff
    # cyclesteps = 2
    # cyclesteps = 4
    # episode_time = 2.e+5
    # # ext = get_SBP_ext_for_Ziff(first_turned='O2')
    # ext = get_SBPOverlap_ext_Ziff()
    #
    # run_jobs_list(**get_for_repeated_opt_iterations(func_to_optimize_policy(
    #                                                     PC_obj,
    #                                                     AnyStepPolicy(cyclesteps, dict()),
    #                                                     episode_time, episode_time / 1000,
    #                                                     expand_description=ext,
    #                                                     to_plot={'out_names': ['CO2_count'], 'additional_plot': ['thetaCO', 'thetaO']}),
    #                                                 # optimize_bounds={'t1': (1.e-8, 1.5e-8), 't2': (1.5e-7, 2.e-7)},
    #                                                 # optimize_bounds={'total': (0.5e-7, 1.e-7), 'first_part': (0.6, 0.8)},  # first good PdMCoffee SBP point
    #                                                 # optimize_bounds={'total': (1e+3, 3.e+3), 'first_part': (0.4, 0.6)},  # first good Ziff SBP point
    #                                                 optimize_bounds={'x_t1': (800, 1200), 'x_t3': (800, 1200), 'x_t2': (0, 1200), 'x_t4': (0, 1200), },  # SBP with overlap, near good SBP point
    #                                                 cut_left=False, cut_right=False,
    #                                                 method='Nelder-Mead', try_num=30,
    #                                                 debug_params={'DEBUG': True, 'folder': 'auto'},
    #                                                 optimize_options={'maxiter': 3}),
    #
    #               const_params={},
    #               PC=PC_obj,
    #               repeat=1,
    #               sort_iterations_by='fvalue',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./optimize_out/230420_Ziff_SBPOverlap_2000_0.5_good_point',
    #               separate_folds=False,
    #               at_same_time=30,
    #               )

    # Ziff

    # CONSTANT
    # episode_time = 2.e+5  # Ziff model

    # episode_time = 2.e-6  # Pd MonteCoffee
    # sum_of_pressures = 1.e+5
    #
    # run_jobs_list(**get_for_repeated_opt_iterations(func_to_optimize_policy(
    #                                                     PC_obj, ConstantPolicy(dict()), episode_time, episode_time / 200,
    #                                                     expand_description=get_O2_CO_from_x_co_ext(sum_of_pressures),
    #                                                     to_plot={'out_names': ['CO2_count'], 'additional_plot': ['thetaCO', 'thetaO']}),
    #                                                 optimize_bounds={'x_value': (0.2, 0.4)},  # 0.4, 0.55 for Ziff
    #                                                 cut_left=False, cut_right=False,
    #                                                 method='Nelder-Mead', try_num=20,
    #                                                 debug_params={'DEBUG': True, 'folder': 'auto'},
    #                                                 optimize_options={'maxiter': 30}),
    #               PC=PC_obj,
    #               out_fold_path='./optimize_out/DEBUG/230415_PdMCoffee_debug',
    #               separate_folds=False,
    #               repeat=1,
    #               const_params={},
    #               sort_iterations_by='fvalue',
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               at_same_time=30,
    #               )

    # iter_optimize_cluster(func_to_optimize_policy(
    #                             PC_obj, AnyStepPolicy(cyclesteps, dict()), episode_time, episode_time / 1000,
    #                             expand_description=ext,
    #                             to_plot={'out_names': ['CO2_count'], 'additional_plot': ['thetaCO', 'thetaO']}),
    #                       optimize_bounds={
    #                           # f'x{i}': (0., 1.) for i in range(1, cyclesteps + 1)
    #                           # 'O2': (0., 1.e+5),
    #                           # **{f'alpha{i}': (0., 1.) for i in range(1, cyclesteps + 1)},
    #                           't1': (1.e-8, 2.e-7), 't2': (1.e-8, 2.e-7),
    #                           },
    #                       cut_left=False, cut_right=False,
    #                       method='Nelder-Mead',
    #                       try_num=30,
    #                       on_cluster=False,
    #                       python_interpreter='../RL_10_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       unique_folder=False,
    #                       out_path='optimize_out/230405_switch_between_pure',
    #                       debug_params={'DEBUG': True, 'folder': 'auto'},
    #                       )

    # iter_optimize_cluster(func_to_optimize_policy(
    #                             PC_obj, AnyStepPolicy(cyclesteps, dict()), episode_time, episode_time / 50,
    #                             expand_description=ext,
    #                             to_plot={'out_names': ['CO2_count'], 'additional_plot': ['thetaCO', 'thetaO']}),
    #                       optimize_bounds={
    #                           # f'x{i}': (0., 1.) for i in range(1, cyclesteps + 1)
    #                           'O2': (0., 1.e+5),
    #                           **{f'alpha{i}': (0., 1.) for i in range(1, cyclesteps + 1)},
    #                           },
    #                       cut_left=False, cut_right=False,
    #                       method='Nelder-Mead',
    #                       try_num=30,
    #                       on_cluster=False,
    #                       python_interpreter='../RL_10_21/venv/bin/python',
    #                       file_to_execute_path='repos/parallel_optimize.py',
    #                       unique_folder=False,
    #                       out_path='optimize_out/220317_discrete_turns_debug',
    #                       debug_params={'DEBUG': True, 'folder': 'auto'},
    #                       )

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

    pass


if __name__ == '__main__':
    main_cluster_function()
