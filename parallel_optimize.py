import numpy as np

from optimize_funcs import *
from test_models import *
from ProcessController import *
from predefined_policies import *
from targets_metrics import *
import PC_setup

from multiple_jobs_functions import *

from PC_run import get_to_optimize_SBP_const_ratio, get_to_fit_L2001_by_LG


def main():
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
                                       # ('to_PC_constructor', {'long_term_target_to_maximize': get_target_func('CO2_plus_CO_conv_I', eps=1.e-3, alpha=1., beta=1/0.02),
                                       #                        'target_func_to_maximize': None
                                       #                        }),
                                       )

    # PC_obj.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
    #                                       'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'thetaA_max': 0.5, 'thetaB_max': 0.5, })
    # PC_obj.process_to_control.set_Libuda()
    # PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'rate_des_A': 0.1, 'rate_react': 0.1,
    #                                       })
    PC_obj.process_to_control.set_params({'rate_ads_A': 0.14895, 'rate_ads_B': 0.06594,  'rate_des_B': 0.,
                                          'rate_des_A': 0.1, 'rate_react': 0.1, })
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

    # rate constants fitting
    # run_jobs_list(**get_for_repeated_opt_iterations(get_to_fit_L2001_by_LG(),
    #                                                 # optimize_bounds={f'rate_{suff}': (-2., 1.) for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')},
    #                                                 optimize_bounds={'rate_ads_A': (0.1, 0.2), 'rate_ads_B': (0.05, 0.1), 'rate_des_A': (0.05, 0.1), 'rate_react': (2., 10.)},
    #                                                 cut_left=False, cut_right=False,
    #                                                 method='Nelder-Mead', try_num=20,
    #                                                 optimize_options={'maxiter': 3}),
    #
    #               const_params={},
    #               PC=PC_obj,
    #               repeat=1,
    #               sort_iterations_by='fvalue',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./optimize_out/LibudaG/231021_L2001_fit',
    #               separate_folds=False,
    #               at_same_time=110,
    #               )

    # rates optimization
    # def log10_space_constrains(d):
    #     for k, v in d.items():
    #         d[k] = pow(10, v)
    #
    # run_jobs_list(**get_for_repeated_opt_iterations(get_to_optimize_SBP_const_ratio(
    #                                                     PC_obj,
    #                                                     np.array([1., 0.]), np.array([0., 1.]), np.array([2., 200.]),
    #                                                     resolutions=[3, 3, 3]),
    #                                                 optimize_bounds={f'rate_{suff}': (-2., 1.) for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')},
    #                                                 constrains=log10_space_constrains,
    #                                                 cut_left=False, cut_right=False,
    #                                                 method='Nelder-Mead', try_num=50,
    #                                                 call_after_opt_params={'plot_both_best': True, 'folder': 'auto'},
    #                                                 optimize_options={'maxiter': 3}),
    #
    #               const_params={},
    #               PC=PC_obj,
    #               repeat=1,
    #               sort_iterations_by='fvalue',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./optimize_out/LibudaG/230726_ratio_if_conversion_alpha1',
    #               separate_folds=False,
    #               at_same_time=30,
    #               )

    # vanilla x_co/pressures control

    # def x_co_constrain(d):
    #     d['inputB_value'] = 1 - d['x_A']
    #     d['inputA_value'] = d['x_A']

    # def get_same_t_constrains(steps_in_cycle):
    #
    #     def constrains_(d):
    #         for i in range(1, steps_in_cycle + 1):
    #             d[f'inputB_t{i}'] = d[f'inputA_t{i}'] = d[f't{i}']
    #
    #     return constrains_

    # def sin_constrains(d):
    #     d['alpha'] = 0.
    #     for name in ('T', 'alpha'):
    #         d[f'inputB_{name}'] = d[f'inputA_{name}'] = d[name]
    #
    # episode_time = 250.
    # cyclesteps = 2
    #
    # run_jobs_list(**get_for_repeated_opt_iterations(func_to_optimize_policy(
    #                                                     PC_obj,
    #                                                     # AnyStepPolicy(cyclesteps),
    #                                                     # TrianglePolicy(),
    #                                                     # ConstantPolicy(),
    #                                                     SinPolicy(),
    #                                                     episode_time, episode_time / 1000,
    #                                                     t_start_count_from=30.),
    #                                                 # optimize_bounds={'inputB_value': (0., 1.), 'inputA_value': (0., 1.)},
    #                                                 # optimize_bounds={'inputB_1': (0., 1.), 'inputA_1': (0., 1.),
    #                                                 #                  'inputB_2': (0., 1.), 'inputA_2': (0., 1.),
    #                                                 #                  't1': (5., 25.), 't2': (5., 25.),
    #                                                 #                  },
    #                                                 optimize_bounds={'inputB_A': (0., 1.), 'inputA_A': (0., 1.),
    #                                                                  'inputB_bias': (0., 1.), 'inputA_bias': (0., 1.),
    #                                                                  'T': (2., 25.),
    #                                                                  },
    #                                                 # optimize_bounds={'x_A': (0., 1.)},
    #                                                 # constrains=get_same_t_constrains(cyclesteps),
    #                                                 constrains=sin_constrains,
    #                                                 # constrains=x_co_constrain,
    #                                                 cut_left=False, cut_right=False,
    #                                                 method='Nelder-Mead', try_num=30,
    #                                                 call_after_opt_params={'DEBUG': True, 'folder': 'auto'},
    #                                                 optimize_options={'maxiter': 10}),
    #
    #               const_params={},
    #               PC=PC_obj,
    #               repeat=1,
    #               sort_iterations_by='fvalue',
    #               cluster_command_ops=False,
    #               python_interpreter='../RL_10_21/venv/bin/python',
    #               out_fold_path='./optimize_out/LibudaG/230729_both_control_sin',
    #               separate_folds=False,
    #               at_same_time=30,
    #               )

    # REFERENCE, NM FOR DIFFERENT RATES, STEADY-STATE
    episode_time = 1000.
    # variants = [pow(10., x) for x in np.linspace(-2., 1., 4)]
    variants = [1.e-2, 2.e-2, 1.e-1, 2.e-1, 1., 2., 10.]

    run_jobs_list(**get_for_param_opt_iterations(func_to_optimize_policy,
                                                 optimize_bounds={'inputB_value': (0., 1.), 'inputA_value': (0., 1.)},
                                                 ),
                  # **merge_job_lists(
                  #     jobs_list_from_grid(variants, variants, names=('model:rate_ads_A', 'model:rate_ads_B')),
                  #     jobs_list_from_grid(variants, variants, names=('model:rate_des_A', 'model:rate_react')),
                  #     ),
                  **jobs_list_from_grid(variants, variants, names=('model:rate_des_A', 'model:rate_react')),
                  const_params={
                      'to_func_to_optimize': {
                          'PC_obj': PC_obj,
                          'policy_obj': ConstantPolicy(),
                          'episode_len': episode_time,
                          'time_step': episode_time / 1000,
                          't_start_count_from': 760.
                      },
                      'to_iter_optimize': {
                          'method': 'Nelder-Mead', 'try_num': 5,
                          'call_after_opt_params': {'DEBUG': True, 'folder': 'auto', 'ind_picture': True},
                          'optimize_options': {'maxiter': 10},
                          'cut_left': False, 'cut_right': False,
                      },
                  },
                  sort_iterations_by='min_fun',
                  PC=PC_obj,
                  repeat=1,
                  cluster_command_ops=False,
                  python_interpreter='../RL_10_21/venv/bin/python',
                  out_fold_path='./optimize_out/LibudaG/240402_stationary_diff_rates',
                  at_same_time=110,
                  )

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
    
    pass


if __name__ == '__main__':
    main()
