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
    # size = [20, 20]
    # PC_KMC = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=True,
    #                                               O2_top=1.1e5, CO_top=1.1e5,
    #                                               CO2_rate_top=3.e5, CO2_count_top=1.e4,
    #                                               T=373.),
    #                            analyser_dt=1,
    #                            target_func_to_maximize=get_target_func('CO2_count'),
    #                            target_func_name='CO2_count',
    #                            target_int_or_sum='sum',
    #                            RESOLUTION=1,  # ATTENTION! Always should be 1 if we use KMC, otherwise we will get wrong results!
    #                            supposed_step_count=1000,  # memory controlling parameters
    #                            supposed_exp_time=1.e-5)

    # PC_obj = PC_setup.default_PC_setup('ZGB')
    # PC_obj = PC_setup.general_PC_setup('Libuda2001', ('to_model_constructor', {'Ts': 433.}))
    # PC_obj = PC_setup.general_PC_setup('LibudaD', ('to_model_constructor', {'Ts': 433.}))
    # PC_obj = PC_setup.general_PC_setup('ZGBk')
    # PC_obj = PC_setup.general_PC_setup('ZGBk',
    #                                    ('to_PC_constructor', {
    #                                        'analyser_dt': 1.e+3,
    #                                        # 'supposed_exp_time': 1.6e+2,
    #                                    }),
    #                                    )
    # PC_obj = PC_setup.general_PC_setup('Lynch')
    # PC_obj = PC_setup.general_PC_setup('VanNeer')
    PC_obj = PC_setup.general_PC_setup('LibudaGWithT',
                                       ('to_model_constructor', {'params': {'reaction_rate_top': pow(3., 10)}}),
                                       )

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

    # LibudaGWithT

    # experiment with artificially chosen k
    # noinspection PyTupleAssignmentBalance,PyTypeChecker
    def find_constants():

        import scipy.optimize as sp_opt

        kB = 0.008314463

        def F(p, roots, Ts):
            v_0, E = p
            return (v_0 * np.exp(-E / kB / Ts[0]) - roots[0],
                    v_0 * np.exp(-E / kB / Ts[1]) - roots[1])

        # var 1, default
        v1, E1 = sp_opt.fsolve(F, (0, 0), args=((0.5, 0.8), (400, 600)))
        # v1_des, E1_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 10.), (400, 600)), maxfev=10000)
        v2, E2 = sp_opt.fsolve(F, (0, 0), args=((0.5, 0.8), (400, 600)))
        # v2_des, E2_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 10.), (400, 600)), maxfev=10000)
        # v3, E3 = sp_opt.fsolve(F, (0, 0), args=((0.05, 10.), (400, 600)))

        # var 2
        # v1_des, E1_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 100.), (400, 600)), maxfev=10000)  # 10 -> 100
        # v2_des, E2_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 100.), (400, 600)), maxfev=10000)  # 10 -> 100

        # var 3
        # v3, E3 = sp_opt.fsolve(F, (0, 0), args=((0.005, 10.), (400, 600)), maxfev=10000)  # 0.05 -> 0.005

        # var 4
        # v3, E3 = sp_opt.fsolve(F, (0, 0), args=((0.005, 30.), (400, 600)), maxfev=10000)  # 0.05 -> 0.005, 10. -> 30.

        # var 5
        # v1_des, E1_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 50.), (400, 600)), maxfev=10000)  # 10 -> 50
        # v2_des, E2_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 50.), (400, 600)), maxfev=10000)  # 10 -> 50
        # v3, E3 = sp_opt.fsolve(F, (0, 0), args=((0.005, 100.), (400, 600)), maxfev=10000)  # 0.05 -> 0.005, 10. -> 100.

        # var 6
        # v1_des, E1_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 100.), (400, 600)), maxfev=10000)  # 10 -> 100
        # v2_des, E2_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 100.), (400, 600)), maxfev=10000)  # 10 -> 100
        # v3, E3 = sp_opt.fsolve(F, (0, 0), args=((0.005, 100.), (400, 600)), maxfev=10000)  # 0.05 -> 0.005, 10. -> 100.

        # var 7
        # v1_des, E1_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 100.), (400, 600)), maxfev=10000)  # 10 -> 100
        # v2_des, E2_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 100.), (400, 600)), maxfev=10000)  # 10 -> 100
        # v3, E3 = sp_opt.fsolve(F, (0, 0), args=((0.005, 200.), (400, 600)), maxfev=100000)  # 0.05 -> 0.005, 10. -> 200.

        # var 8
        v1_des, E1_des = sp_opt.fsolve(F, (0, 0), args=((0.4, 100.), (400, 600)), maxfev=10000)  # 0.1 -> 0.4, 10 -> 100
        v2_des, E2_des = sp_opt.fsolve(F, (0, 0), args=((0.1, 100.), (400, 600)), maxfev=10000)  # 10 -> 100
        v3, E3 = sp_opt.fsolve(F, (0, 0), args=((0.005, 100.), (400, 600)), maxfev=100000)  # 0.05 -> 0.005, 10. -> 100.

        return {
            'E_ads_A': E1, 'E_ads_B': E2,
            'E_des_A': E1_des, 'E_des_B': E2_des,
            'E_react': E3,
            'rate_ads_A_0': v1,
            'rate_des_A_0': v1_des,
            'rate_ads_B_0': v2,
            'rate_des_B_0': v2_des,
            'rate_react_0': v3,
        }
    PC_obj.process_to_control.set_params(find_constants())

    # diff temperatures, vary gases and temperature
    # variants = np.linspace(0., 1., 40).reshape(-1, 1)
    #
    # # T = 100 + 273
    # # episode_time = 40000
    # # period_time = 2000
    #
    # # T = 300 + 273
    # # episode_time = 2.
    # # period_time = 0.1
    #
    # T = 500 + 273
    # episode_time = 0.2
    # period_time = 0.01
    #
    # def transform_2_from_x_co(d):
    #     d.update({'inputA_value': d['x_co'], 'inputB_value': (1. - d['x_co'])})
    #
    # # vary temperature
    # run_jobs_list(
    #     **(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': TwoStepPolicy()},
    #                                  'x_co', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  transform_params=transform_2_from_x_co,
    #                                  additional_names=('thetaB', 'thetaA'),
    #                                  take_from_the_end=0.75,
    #                                  )),
    #     params_variants=variants,
    #     names=('x_co', ),
    #     names_groups=(),
    #     const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
    #                   'T_1': T - 100., 'T_2': T + 100.,
    #                   'T_t1': period_time / 2, 'T_t2': period_time / 2,
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/LibudaGWithT/230517_vary_T_T{T}',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # vary gases
    # def transform_1_from_x_co(d):
    #     d.update({'inputA_1': max(d['x_co'] - 0.1, 0.), 'inputA_2': min(d['x_co'] + 0.1, 1.),
    #               'inputB_1': min((1 - d['x_co']) + 0.1, 1.), 'inputB_2': max((1 - d['x_co']) - 0.1, 0.)})
    #
    # run_jobs_list(
    #     **(get_for_common_variations({'inputA': TwoStepPolicy(), 'inputB': TwoStepPolicy(), 'T': ConstantPolicy()},
    #                                  'x_co', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  transform_params=transform_1_from_x_co,
    #                                  additional_names=('thetaB', 'thetaA'),
    #                                  take_from_the_end=0.75,
    #                                  )),
    #     params_variants=variants,
    #     names=('x_co', ),
    #     names_groups=(),
    #     const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
    #                   'T_value': T,
    #                   'inputB_t1': period_time / 2, 'inputB_t2': period_time / 2,
    #                   'inputA_t1': period_time / 2, 'inputA_t2': period_time / 2,
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/LibudaGWithT/230517_vary_gases_T{T}',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # diff temperatures steady-state
    # variants = np.linspace(0., 1., 40).reshape(-1, 1)
    # variants = np.hstack((variants, -1 * variants + 1)).tolist()
    #
    # T = 100 + 273
    # episode_time = 40000
    #
    # # T = 300 + 273
    # # episode_time = 2.
    # #
    # # T = 500 + 273
    # # episode_time = 0.2
    #
    # run_jobs_list(
    #     **(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': ConstantPolicy()},
    #                                  'inputA_value', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  additional_names=('thetaB', 'thetaA'),
    #                                  take_from_the_end=0.33,
    #                                  )),
    #     params_variants=variants,
    #     names=('inputA_value', 'inputB_value'),
    #     names_groups=(),
    #     const_params={'episode_time': episode_time, 'calc_dt': lambda x: x / 1000,
    #                   'T_value': T,
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/LibudaGWithT/230518_steady_state_T{T}',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # test common variations
    # episode_time = 200
    # run_jobs_list(
    #     **(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': TwoStepPolicy(), 'T': SinPolicy()},
    #                                  'inputA_value', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  names_to_plot={'input': ('T', ), 'output': ('outputC', )},
    #                                  additional_names=('thetaB', 'thetaA'),
    #                                  take_from_the_end=0.33,
    #                                  )),
    #     **(jobs_list_from_grid(
    #         (10., 20., 50.),
    #         (0.1, 0.5, 0.7),
    #         (10., 40.),
    #         names=('T_T', 'inputA_value', 'inputB_t1'),
    #     )),
    #     names_groups=(),
    #     const_params={'episode_time': episode_time, 'policy_step': 0.1,
    #                   'T_A': 50, 'T_bias': 500, 'T_alpha': 0.,
    #                   'inputB_1': 0.7, 'inputB_2': 0.2, 'inputB_t2': 10.,
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path='PC_plots/LibudaGWithT/230516_test_common_variations',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # switch between end values
    # episode_time = 200
    # periods_number = 100
    # run_jobs_list(
    #     **(get_for_SB2_iteration_flexible(periods_number, {'name': 'mean_reaction_rate', 'column': 0},
    #                                       calc_analyser_dt=lambda t: t / 10,
    #                                       names_to_plot={'input': ('T',), 'output': ('outputC',)},
    #                                       additional_names=('thetaB', 'thetaA', 'error'),
    #                                       )),
    #     **(jobs_list_from_grid(
    #         # (i * 5. for i in range(1, 4)),
    #         # (i * 5. for i in range(4, 8)),
    #         # (i * 5. for i in range(8, 12)),
    #         (2 ** (i - 4) for i in range(12)),
    #         # map(lambda x: 0.01 * x, range(1, 100)),
    #         [0.5],
    #         names=('total', 'first_part'),
    #     )),
    #     names_groups=(),
    #     const_params={'part1': {'inputB': 1., 'inputA': 1., 'T': 400},
    #                   'part2': {'inputB': 1., 'inputA': 1., 'T': 500},
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path='PC_plots/LibudaGWithT/230512_SBE',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # frequency
    points_num = 25
    variants = np.linspace(5., -7., points_num).reshape(-1, 1)
    # variants = np.hstack((np.linspace(0., optimal_ratio, int(points_num * optimal_ratio)),
    #                       np.linspace(optimal_ratio, 1., points_num - int(points_num * optimal_ratio)))).reshape(-1, 1)

    # times = {400: 3000, 500: 500, 600: 10., 700: 0.5, 800: 0.05}

    # T_1, T_2 = 700, 500
    T_1, T_2 = 600, 400

    def transform_from_log_omega(d):
        T = 2 ** (-d['log_omega'])
        d.update({'T_t1': T / 2, 'T_t2': T / 2,
                  'episode_time': 100 * T})

    run_jobs_list(
        **(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': TwoStepPolicy()},
                                     'log_omega', {'name': 'mean_reaction_rate', 'column': 0},
                                     transform_params=transform_from_log_omega,
                                     additional_names=('thetaB', 'thetaA'),
                                     take_from_the_end=0.5,
                                     )),
        params_variants=variants.tolist(),
        names=('log_omega', ),
        names_groups=(),
        const_params={'calc_dt': lambda x: x / 10000,
                      'inputB_value': 1., 'inputA_value': 1.,
                      'T_1': T_1, 'T_2': T_2,
                      },
        sort_iterations_by='mean_reaction_rate',
        PC=PC_obj,
        repeat=1,
        out_fold_path=f'PC_plots/LibudaGWithT/change_k_low_T_adsorption_high_T_desorption/230522_rate_vs_frequency',
        separate_folds=False,
        cluster_command_ops=False,
        python_interpreter='../RL_10_21/venv/bin/python',
        at_same_time=100,
    )

    # steady-state
    # points_num = 40
    # # points_num = 10  # DEBUG
    # # variants = np.linspace(470., 780., points_num).reshape(-1, 1)
    # variants = np.linspace(400., 600., points_num).reshape(-1, 1)
    # # variants = np.linspace(470., 780., points_num).reshape(-1, 1)  # DEBUG
    # # variants = np.hstack((np.linspace(0., optimal_ratio, int(points_num * optimal_ratio)),
    # #                       np.linspace(optimal_ratio, 1., points_num - int(points_num * optimal_ratio)))).reshape(-1, 1)
    #
    # times = {400: 3000, 500: 500, 600: 10., 700: 0.5, 800: 0.05}
    #
    # def transform_from_T_steady_state(d):
    #     d.update({'T_value': d['T'], 'episode_time': times[next(filter(lambda T: (d['T'] // 100) - T // 100 < 1, times.keys()
    #                                                                    ))]})
    #
    # run_jobs_list(
    #     **(get_for_common_variations({'inputA': ConstantPolicy(), 'inputB': ConstantPolicy(), 'T': ConstantPolicy()},
    #                                  'T', {'name': 'mean_reaction_rate', 'column': 0},
    #                                  transform_params=transform_from_T_steady_state,
    #                                  additional_names=('thetaB', 'thetaA'),
    #                                  take_from_the_end=0.5,
    #                                  )),
    #     params_variants=variants.tolist(),
    #     names=('T', ),
    #     names_groups=(),
    #     const_params={'calc_dt': lambda x: x / 1000,
    #                   'inputB_value': 1., 'inputA_value': 1.,
    #                   },
    #     sort_iterations_by='mean_reaction_rate',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path=f'PC_plots/LibudaGWithT/change_k_low_T_adsorption_high_T_desorption/230522_diff_T_steady_state',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # ZGB k
    # benchmark

    # pressure_unit = 1.e+4
    # m, n = PC_obj.process_to_control['surface_size']
    # time_unit = m * n

    # # run for y2 value from the article
    # iteration_func = get_for_Ziff_iterations(pressure_unit, 50 * time_unit, take_from_the_end=0.1, CO2_output_column=0,
    #                                          out_names_to_plot=('CO2_prod_rate', ))['iteration_function']
    # iteration_func(PC_obj, {'x': 0.531}, './PC_plots/230426_ZGBk_true_optimal/', 666)

    # points_num = 24
    # variants = (np.linspace(0., 0.35, points_num // 8),
    #             np.linspace(0.35, 0.55, 6 * points_num//8),
    #             np.linspace(0.55, 1., points_num//8))
    # variants = np.hstack(variants)

    # search for y2 value from the article
    # variants = np.linspace(0.53, 0.535, 10)

    # # run for y2 value from the article
    # # variants = np.array([0.531] * 3)

    # episode_time = 100 * (150 + 10) * time_unit
    # episode_time = 10 * time_unit  # DEBUG
    #
    # run_jobs_list(
    #     **(get_for_steady_state_variations(episode_time, 'x', {'name': 'CO2_prod_rate', 'column': 0}, take_from_the_end=0.1,
    #                                        names_to_plot={'input': None, 'output': ('CO2_prod_rate', )})),
    #     params_variants=variants.reshape(-1, 1).tolist(),
    #     names=('x', ),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by='CO2_prod_rate',
    #     PC=PC_obj,
    #     repeat=1,  # 3
    #     out_fold_path='PC_plots/ZGBk/230513_ZGBk_search_for_y2',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=110,
    # )

    # dynamic advantage run
    # MC_time_step = PC_obj.process_to_control.m * PC_obj.process_to_control.n
    # MC_time_step //= 1000  # DEBUG
    # t_p, t_d = 150 * MC_time_step, 10 * MC_time_step,  # original values
    # x_p, x_d = 0.535, 0.5
    # run_jobs_list(
    #     repeat_periods_calc_rate,
    #     params_variants=[({'x': x_p}, t_p, {'x': x_d}, t_d),
    #                      ({'x': x_p - 0.01}, t_p, {'x': x_d}, t_d),
    #                      ({'x': x_p - 0.02}, t_p, {'x': x_d}, t_d),
    #                      ({'x': x_p - 0.05}, t_p, {'x': x_d}, t_d),
    #                      ({'x': x_p}, t_p, {'x': x_d - 0.01}, t_d),
    #                      ({'x': x_p}, t_p, {'x': x_d - 0.02}, t_d),
    #                      ({'x': x_p}, t_p, {'x': x_d - 0.05}, t_d),
    #                      ],
    #     names=('part1', 't1', 'part2', 't2'),
    #     names_groups=(),
    #     const_params={
    #         'periods_number': 100
    #     },
    #     sort_iterations_by='rate_mean',
    #     PC=PC_obj,
    #     repeat=5,
    #     out_fold_path='PC_plots/ZGBk/230513_ZGBk_avoid_poisoning',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=110,
    # )

    # ZGB

    # benchmark
    # pressure_unit = 1.e+4
    # points_num = 26
    # run_jobs_list(
    #     **(get_for_Ziff_iterations(pressure_unit, 24.e+4, take_from_the_end=0.1, CO2_output_column=0)),
    #     params_variants=np.linspace(0., 1., points_num).reshape(-1, 1).tolist(),
    #     names=('x', ),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=3,
    #     out_fold_path='PC_plots/230413_DEBUG_Ziff_model',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # run_jobs_list(
    #     **(get_for_SBP_iteration(2.e-6, 'O2')),
    #     **(jobs_list_from_grid(
    #         (0.25e-7, 0.5e-7, 0.75e-7),
    #         map(lambda x: 0.1 * x, range(1, 10)),
    #         names=('total', 'first_part'),
    #     )),
    #     names_groups=(),
    #     const_params={'O2_max': 1.e+5, 'CO_max': 1.e+5},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=3,
    #     out_fold_path='PC_plots/230410_SwitchBetweenPure_20x20',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # pressure_unit = 1.e+4
    # # points_num = 26
    # points_num = 3
    # run_jobs_list(
    #     **(get_for_Ziff_iterations(pressure_unit, 2.e-6)),
    #     params_variants=[((1 - x) * 10 * pressure_unit, x * 10 * pressure_unit)
    #                      # for x in np.linspace(0., 1., points_num)],
    #                      for x in np.linspace(0.5, 1., points_num)],
    #     names=('O2', 'CO'),
    #     ...
    # )

    # # the best stationary obtained by optimization
    # one_turn_search_iteration(PC_obj, {'x0': 0.295, 'x1': 0.295,
    #                                    't0': 1.e-6, 't1': 1.e-6, },
    #                           'PC_plots/2303_KMC_const_best', 0)


if __name__ == '__main__':
    main()
