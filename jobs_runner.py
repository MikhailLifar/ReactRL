import numpy as np

# from ProcessController import ProcessController
# from test_models import *
# from targets_metrics import *
from multiple_jobs_functions import *
from iterations import *

# from optimize_funcs import get_for_param_opt_iterations
from predefined_policies import *

import PC_setup


def main():

    # PC_obj = PC_setup.general_PC_setup('ZGB')
    # PC_obj = PC_setup.general_PC_setup('ZGBk', ('to_PC_constructor', 'supposed_exp_time', 1.e+8),
    #                                            ('to_PC_constructor', 'supposed_step_count', int(1.e+5))
    #                                    )
    #PC_obj = PC_setup.general_PC_setup('Lynch',
                                       #('to_PC_constructor', 'analyser_dt', 0.1),
                                       #)
    #PC_obj = PC_setup.general_PC_setup('ZGBk',
                                       #('to_PC_constructor', {
                                           #'analyser_dt': 1.e+3,
                                           ## 'supposed_exp_time': 1.6e+2,
                                       #}),
                                       #)
    #PC_obj = PC_setup.general_PC_setup('LibudaGWithT',
                                       #('to_model_constructor', {'params': {}}),
                                       #)
    # PC_obj = PC_setup.general_PC_setup('LibudaG')
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithT')

    # PC_obj.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
    #                                       'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'thetaA_max': 0.5, 'thetaB_max': 0.5,
    #
    #                                       'rate_ads_A': 1., 'rate_ads_B': 1.,  # adjusted to be ZGB like
    #                                       'rate_des_A': 0., 'rate_des_B': 0.,
    #                                       'rate_react': 0.3,
    #
    #                                       })
    # PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0., })
    # PC_obj.process_to_control.set_params({'thetaA_init': 0.1254, 'thetaB_init': 0.0517, })  # optimal steady-state
    # PC_obj.process_to_control.set_params({'C_B_inhibit_A': 1., 'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'thetaA_max': 0.5, 'thetaB_max': 0.5,
    #                                       'rate_ads_A': 0.14895, 'rate_ads_B': 0.06594 * 4,
    #                                       })  # Libuda eqns CO RTP
    # c_b_a = 0.01  # (0.01, 0.1, 1.)
    # rate_react = 0.1  # (0.001, 0.01, 0.1, 1.)
    # PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'rate_ads_A': 0.1, 'rate_ads_B': 0.1,
    #                                       'rate_des_A': 0.1, 'rate_react': rate_react,  # 0.1
    #                                       'C_B_inhibit_A': c_b_a,
    #                                       })  # low des, react rates

    # MCKMC
    # DYN ADV POLICY AND OPT STEADY STATE POLICY
    run_jobs_list(
        MCKMC_policy_iteration,
        **jobs_list_from_grid(
            [
                './231002_sudden_discovery/rl_agent_sol.csv',
                './231002_sudden_discovery/nelder_mead_sol.csv',
            ],
            [0.3, 0.9],
            [0., 0.1, 1.],
            names=('plottoffile', 'covOLimit', 'diffusion_level')
        ),
        names_groups=(),
        PC=PC_setup.general_PC_setup('LibudaG'),
        const_params={
            'surfShape': (5, 5, 1),
            'snapshotPeriod': 0.1,
            't_end': 10.,
            'analyser_dt': 1.,
        },
        unique_folder=False,
        separate_folds=False,
        out_fold_path='./PC_plots/MCKMC/240403_RL_NM_opt_sols',
        python_interpreter='/opt/anaconda_py38_1/bin/python',
        cluster_command_ops=False,
        at_same_time=300,
    )

    # STEADY-STATE
    # run_jobs_list(
    #     MCKMC_policy_iteration,
    #     **jobs_list_from_grid(
    #         np.linspace(0., 1., 26),
    #         [0.3, 0.9],
    #         [0., 0.1, 1.],
    #         names=('xCO', 'covOLimit', 'diffusion_level')
    #     ),
    #     names_groups=(),
    #     PC=PC_setup.general_PC_setup('LibudaG'),
    #     const_params={
    #         'surfShape': (5, 5, 1),
    #         'snapshotPeriod': 0.1,
    #         't_end': 10.,
    #         'analyser_dt': 1.,
    #     },
    #     unique_folder=False,
    #     separate_folds=False,
    #     out_fold_path='./PC_plots/MCKMC/240401_steady_state',
    #     python_interpreter='/opt/anaconda_py38_1/bin/python',
    #     cluster_command_ops=False,
    #     at_same_time=300,
    # )
    
    # LibudaGWithT
    # diff temperatures steady-state
    #variants = np.linspace(0., 1., 40).reshape(-1, 1)
    #variants = np.hstack((variants, -1 * variants + 1)).tolist()

    ##T = 100 + 273
    ##episode_time = 40000

    ##T = 300 + 273
    ##episode_time = 2.
    
    #T = 500 + 273
    #episode_time = 0.2

    #run_jobs_list(
        #**(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': ConstantPolicy()},
                                     #'inputA_value', {'name': 'mean_reaction_rate', 'column': 0},
                                     #additional_names=('thetaB', 'thetaA'),
                                     #take_from_the_end=0.33,
                                     #)),
        #params_variants=variants,
        #names=('inputA_value', 'inputB_value'),
        #names_groups=(),
        #const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
                      #'T_value': T,
                      #},
        #sort_iterations_by='mean_reaction_rate',
        #PC=PC_obj,
        #repeat=1,
        #out_fold_path=f'PC_plots/LibudaGWithT/230518_steady_state_T{T}',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=100,
    #)
    
    ## diff temperatures, vary gases and temperature
    #variants = np.linspace(0., 1., 40).reshape(-1, 1)
    
    #T = 100 + 273
    #episode_time = 40000
    #period_time = 2000
    
    ##T = 300 + 273
    ##episode_time = 2.
    ##period_time = 0.1
    
    ##T = 500 + 273
    ##episode_time = 0.2
    ##period_time = 0.01
    
    #def transform_2_from_x_co(d):
        #d.update({'inputA_value': d['x_co'], 'inputB_value': (1. - d['x_co'])})
    
    ## vary temperature
    #run_jobs_list(
        #**(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': TwoStepPolicy()},
                                     #'x_co', {'name': 'mean_reaction_rate', 'column': 0},
                                     #transform_params=transform_2_from_x_co,
                                     #additional_names=('thetaB', 'thetaA'),
                                     #take_from_the_end=0.75,
                                     #kwargs_to_sum_plot={'ylim': (0., 0.6), 'twin_ylim': (0., None), },
                                     #)),
        #params_variants=variants,
        #names=('x_co', ),
        #names_groups=(),
        #const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
                      #'T_1': T - 100., 'T_2': T + 100.,
                      #'T_t1': period_time / 2, 'T_t2': period_time / 2,
                      #},
        #sort_iterations_by='mean_reaction_rate',
        #PC=PC_obj,
        #repeat=1,
        #out_fold_path=f'PC_plots/LibudaGWithT/230520_vary_T_T{T}',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=100,
    #)

    ## vary gases
    #def transform_1_from_x_co(d):
        #d.update({'inputA_1': max(d['x_co'] - 0.1, 0.), 'inputA_2': min(d['x_co'] + 0.1, 1.),
                  #'inputB_1': min((1 - d['x_co']) + 0.1, 1.), 'inputB_2': max((1 - d['x_co']) - 0.1, 0.)})
    
    #run_jobs_list(
        #**(get_for_common_variations({'inputA': TwoStepPolicy(), 'inputB': TwoStepPolicy(), 'T': ConstantPolicy()},
                                     #'x_co', {'name': 'mean_reaction_rate', 'column': 0},
                                     #transform_params=transform_1_from_x_co,
                                     #additional_names=('thetaB', 'thetaA'),
                                     #take_from_the_end=0.75,
                                     #kwargs_to_sum_plot={'ylim': (0., 0.6), 'twin_ylim': (0., None), },
                                     #)),
        #params_variants=variants,
        #names=('x_co', ),
        #names_groups=(),
        #const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
                      #'T_value': T,
                      #'inputB_t1': period_time / 2, 'inputB_t2': period_time / 2,
                      #'inputA_t1': period_time / 2, 'inputA_t2': period_time / 2,
                      #},
        #sort_iterations_by='mean_reaction_rate',
        #PC=PC_obj,
        #repeat=1,
        #out_fold_path=f'PC_plots/LibudaGWithT/230520_vary_gases_T{T}',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=100,
    #)
    
    # diff temperatures steady-state
    #variants = np.linspace(0., 1., 40).reshape(-1, 1)
    #variants = np.hstack((variants, -1 * variants + 1)).tolist()
    
    ##T = 100 + 273
    ##episode_time = 40000
    
    ##T = 300 + 273
    ##episode_time = 2.
    
    #T = 500 + 273
    #episode_time = 0.2
    
    #run_jobs_list(
        #**(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': ConstantPolicy()},
                                     #'inputA_value', {'name': 'mean_reaction_rate', 'column': 0},
                                     #additional_names=('thetaB', 'thetaA'),
                                     #take_from_the_end=0.33,
                                     #kwargs_to_sum_plot={'ylim': (0., 0.6), 'twin_ylim': (0., None), },
                                     #)),
        #params_variants=variants,
        #names=('inputA_value', 'inputB_value'),
        #names_groups=(),
        #const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
                      #'T_value': T,
                      #},
        #sort_iterations_by='mean_reaction_rate',
        #PC=PC_obj,
        #repeat=1,
        #out_fold_path=f'PC_plots/LibudaGWithT/230520_steady_state_T{T}',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=100,
    #)
    
    # frequency
    #points_num = 49 # 25
    #variants = np.linspace(5., -7., points_num).reshape(-1, 1)
    ## variants = np.hstack((np.linspace(0., optimal_ratio, int(points_num * optimal_ratio)),
    ##                       np.linspace(optimal_ratio, 1., points_num - int(points_num * optimal_ratio)))).reshape(-1, 1)

    ## times = {400: 3000, 500: 500, 600: 10., 700: 0.5, 800: 0.05}
    
    #T_1, T_2 = 700, 500
    ##T_1, T_2 = 600, 400

    #def transform_from_log_omega(d):
        #T = 2 ** (-d['log_omega'])
        #d.update({'T_t1': T / 2, 'T_t2': T / 2,
                  #'episode_time': 100 * T})

    #run_jobs_list(
        #**(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': TwoStepPolicy()},
                                     #'log_omega', {'name': 'mean_reaction_rate', 'column': 0},
                                     #transform_params=transform_from_log_omega,
                                     #additional_names=('thetaB', 'thetaA'),
                                     #take_from_the_end=0.5,
                                     #kwargs_to_sum_plot={'ylim': (0., 0.6), 'twin_ylim': (0., 15.), },
                                     #)),
        #params_variants=variants.tolist(),
        #names=('log_omega', ),
        #names_groups=(),
        #const_params={'calc_dt': lambda x: x / 10000,    # x / 1000
                      #'inputB_value': 1., 'inputA_value': 1.,
                      #'T_1': T_1, 'T_2': T_2,
                      #},
        #sort_iterations_by='mean_reaction_rate',
        #PC=PC_obj,
        #repeat=1,
        #out_fold_path=f'PC_plots/LibudaGWithT/230520_rate_vs_frequency',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=100,
    #)
    
    # # steady-state
    #points_num = 40
    ## points_num = 10  # DEBUG
    #variants = np.linspace(400., 780., points_num).reshape(-1, 1)
    ##variants = np.linspace(400., 600., points_num).reshape(-1, 1)  # changed rates
    ## variants = np.hstack((np.linspace(0., optimal_ratio, int(points_num * optimal_ratio)),
    ##                       np.linspace(optimal_ratio, 1., points_num - int(points_num * optimal_ratio)))).reshape(-1, 1)
    
    #times = {400: 3000, 500: 500, 600: 10., 700: 0.5, 800: 0.05}
    
    #def transform_from_T_steady_state(d):
        #d.update({'T_value': d['T'], 'episode_time': times[next(filter(lambda T: (d['T'] // 100) - T // 100 < 1, times.keys()
                                                                       #))]})
    
    #run_jobs_list(
        #**(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': ConstantPolicy()},
                                     #'T', {'name': 'mean_reaction_rate', 'column': 0},
                                     #transform_params=transform_from_T_steady_state,
                                     #additional_names=('thetaB', 'thetaA'),
                                     #take_from_the_end=0.5,
                                     #kwargs_to_sum_plot={'ylim': (0., 0.6), 'twin_ylim': None, },
                                     #)),
        #params_variants=variants.tolist(),
        #names=('T', ),
        #names_groups=(),
        #const_params={'calc_dt': lambda x: x / 1000,
                      #'inputB_value': 1., 'inputA_value': 1.,
                      #},
        #sort_iterations_by='mean_reaction_rate',
        #PC=PC_obj,
        #repeat=1,
        #out_fold_path=f'PC_plots/LibudaGWithT/230520_diff_T_steady_state',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=100,
    #)

    # def for_one_x0_several_x1(x0):
    #     variants_of_x1 = filter(lambda x: (x < 9.e-5) and (x > 1.e-5),
    #                             (x0 + 0.2, x0 - 0.2, x0 + 0.4, x0 - 0.4,
    #                              x0 + 0.4, x0 + 0.6,))
    #     return tuple(map(lambda x: (x0, x), variants_of_x1))
    #
    # variants = list(map(for_one_x0_several_x1, np.arange(0.1, 0.35, 0.05)))
    # new_params = []
    # for p in variants:
    #     new_params += [*p]
    # variants = new_params
    
    # LibudaG
    # return, ratio on rates dependence
    # points_num = 40  # 30, 40
    # variants = np.linspace(0.01, 10., points_num)
    # run_jobs_list(
    #     **(get_for_opt_policy_search('rate_ads_A', 'rate_ads_B', variants,
    #                                  map_grid_resolutions=(300, 300),
    #                                  inputs_start=np.array([1., 0.]), inputs_end=np.array([0., 1.]),
    #                                  period_bounds=np.array([0.2, 1000.]), resolutions=[200, 20, 10])),  # [200, 20, 10]
    #     params_variants=variants.reshape(-1, 1).tolist(),
    #     names=('rate_ads_A', ),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by=None,
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/LibudaG/230722_draw_maps',
    #     cluster_command_ops={'n': 1, 'm': 3000},
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # LIBUDA, LYNCH
    
    #episode_time = 200  # Libuda: 500
    #max_flow = 1.  # Libuda: 1.e-4

    # SBP
    #run_jobs_list(
        #**(get_for_SBP_iteration(episode_time, 'O2', ziff_model=False,
                                 #out_name_to_observe='CO2')),
        #**(jobs_list_from_grid(
            ## (i * 5. for i in range(1, 4)),
            ## (i * 5. for i in range(4, 8)),
            ##(i * 5. for i in range(8, 12)),
            #(10, 20, 40),
            #map(lambda x: 0.1 * x, range(1, 10)),
            #names=('total', 'first_part'),
        #)),
        #names_groups=(),
        #const_params={'O2_max': max_flow, 'CO_max': max_flow},
        #sort_iterations_by='CO2',
        #PC=PC_obj,
        #repeat=1,
        #out_fold_path='PC_plots/230428_Lynch_SBP',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=110,
    #)

    # steady state
    # episode_time = 1000
    # variants = np.linspace(0., 1., 100).reshape(-1, 1)
    #
    # def transform(d):
    #     d['inputB_value'] = 1 - d['x']
    #     d['inputA_value'] = d['x']
    #
    # run_jobs_list(
    #     **(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy()},
    #                                  'x', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  transform_params=transform,
    #                                  additional_names=('thetaB', 'thetaA'),
    #                                  take_from_the_end=0.33,
    #                                  )),
    #     params_variants=variants,
    #     names=('x', ),
    #     names_groups=(),
    #     const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/LibudaG/diff_react_rate/231030_c_b_a({c_b_a:.3g})_steady_state',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=110,
    # )

    # # # Sergey's approach / rate vs frequency, CO part is equal to O2 part
    # points_num = 100  # 49, 25
    # variants = np.linspace(10., -10., points_num).reshape(-1, 1)
    # # variants = np.hstack((np.linspace(0., optimal_ratio, int(points_num * optimal_ratio)),
    # #                       np.linspace(optimal_ratio, 1., points_num - int(points_num * optimal_ratio)))).reshape(-1, 1)
    #
    # def transform_from_log_omega(d):
    #     T = 2 ** (-d['log_omega'])
    #     d.update({'inputB_t1': T / 2, 'inputB_t2': T / 2, 'inputA_t1': T / 2, 'inputA_t2': T / 2, 'episode_time': 100 * T})
    #
    # run_jobs_list(
    #     **(get_for_common_variations({'inputA': TwoStepPolicy({'1': 0., '2': 1.}),
    #                                   'inputB': TwoStepPolicy({'1': 1., '2': 0.})},  # low rates
    #                                  # {'inputA': TwoStepPolicy({'1': 0.47, '2': 1., 't1': 1., 't2': 1.}),
    #                                  #  'inputB': ConstantPolicy()},  # Sergey's
    #                                  'log_omega', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  transform_params=transform_from_log_omega,
    #                                  additional_names=('thetaB', 'thetaA'),
    #                                  take_from_the_end=0.5,
    #                                  kwargs_to_sum_plot={'ylim': (0., 0.6), 'twin_ylim': (0., 0.015 * rate_react / 0.1), },
    #                                  )),
    #     params_variants=variants.tolist(),
    #     names=('log_omega', ),
    #     names_groups=(),
    #     const_params={'calc_dt': lambda x: x / 1000,    # 1000, 10_000
    #                   'preprocess': {'in_values': {'inputB': 0.5, 'inputA': 0.5, },
    #                                  'time': 1000.,
    #                                  'dt': 1.},
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/LibudaG/diff_react_rate/231030_c_b_a({c_b_a:.3g})_rate_vs_freq',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # ZGB k
    ## benchmark
    #pressure_unit = 1.e+4
    #m, n = PC_obj.process_to_control['surface_size']
    #time_unit = m * n

    ## run for y2 value from the article
    #iteration_func = get_for_Ziff_iterations(pressure_unit, 50 * time_unit, take_from_the_end=0.1, CO2_output_column=0,
                                             #out_names_to_plot=('CO2_prod_rate', ))['iteration_function']
    #iteration_func(PC_obj, {'x': 0.531}, './PC_plots/230426_ZGBk_true_optimal/', 666)

    #points_num = 24
    #variants = (np.linspace(0., 0.35, points_num // 8),
                #np.linspace(0.35, 0.55, 6 * points_num//8),
                #np.linspace(0.55, 1., points_num//8))
    #variants = np.hstack(variants)
    
    # search for y2 value from the article
    #variants = np.linspace(0.53, 0.535, 10)
    
    ## run for y2 value from the article
    ## variants = np.array([0.531] * 3)
    
    #episode_time = 100 * (150 + 10) * time_unit
    ##episode_time = 10 * time_unit  # DEBUG
    
    #run_jobs_list(
        #**(get_for_steady_state_variations(episode_time, 'x', {'name': 'CO2_prod_rate', 'column': 0}, take_from_the_end=0.1,
                                           #names_to_plot={'input': None, 'output': ('CO2_prod_rate', )})),
        #params_variants=variants.reshape(-1, 1).tolist(),
        #names=('x', ),
        #names_groups=(),
        #const_params={},
        #sort_iterations_by='CO2_prod_rate',
        #PC=PC_obj,
        #repeat=5,  # 3
        #out_fold_path='PC_plots/ZGBk/230513_ZGBk_search_for_y2',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=110,
    #)
    
    # dynamic advantage run
    #MC_time_step = PC_obj.process_to_control.m * PC_obj.process_to_control.n
    ##MC_time_step //= 1000
    #t_p, t_d = 150 * MC_time_step, 10 * MC_time_step,  # original values
    #x_p, x_d = 0.535, 0.5
    #run_jobs_list(
        #repeat_periods_calc_rate,
        #params_variants=[({'x': x_p}, t_p, {'x': x_d}, t_d)],
        #names=('part1', 't1', 'part2', 't2'),
        #names_groups=(),
        #const_params={
            #'periods_number': 100
        #},
        #sort_iterations_by='rate_mean',
        #PC=PC_obj,
        #repeat=50,
        #out_fold_path='PC_plots/230510_ZGBk_dynamic_advantage_test',
        #separate_folds=False,
        #cluster_command_ops={'n': 1, 'm': 3000},
        #python_interpreter='../RL_10_21/venv/bin/python',
        #at_same_time=110,
    #)

    # ZGB
    # episode_time = 150_000
    # variants = np.linspace(0., 1., 100).reshape(-1, 1)
    #
    # def transform(d):
    #     d['x_value'] = d['x']
    #
    # run_jobs_list(
    #     **(get_for_common_variations({'x': ConstantPolicy()}, 'x', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  transform_params=transform,
    #                                  additional_names=('thetaO', 'thetaCO'),
    #                                  take_from_the_end=0.33,
    #                                  )),
    #     params_variants=variants,
    #     names=('x', ),
    #     names_groups=(),
    #     const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/ZGB/230923_steady_state',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )


if __name__ == '__main__':
    main()
