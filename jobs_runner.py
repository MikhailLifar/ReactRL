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
    PC_obj = PC_setup.general_PC_setup('LibudaGWithT')

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
    points_num = 36
    variants = np.linspace(400., 600., points_num).reshape(-1, 1)
    # variants = np.hstack((np.linspace(0., optimal_ratio, int(points_num * optimal_ratio)),
    #                       np.linspace(optimal_ratio, 1., points_num - int(points_num * optimal_ratio)))).reshape(-1, 1)
    episode_time = 1000
    # episode_time = 1000  # Libuda: 500
    run_jobs_list(
        **(get_for_steady_state_variations(episode_time, 'T', {'name': 'CO2', 'column': 0},
                                           out_names_to_plot=('outputC', ), take_from_the_end=0.33,
                                           additional_names=('thetaB', 'thetaA'))),
        params_variants=variants.tolist(),
        names=('T', ),
        names_groups=(),
        const_params={'inputB': 1., 'inputA': 1.},
        sort_iterations_by='CO2',
        PC=PC_obj,
        repeat=1,
        out_fold_path='PC_plots/LibudaGWithT/230512_steady_state_diff_T',
        separate_folds=False,
        cluster_command_ops=False,
        python_interpreter='../RL_10_21/venv/bin/python',
        at_same_time=100,
    )

    # VanNeer
    # benchmark
    # points_num = 26
    # variants = np.linspace(-3., 2., points_num).reshape(-1, 1)
    # episode_time = 1
    # run_jobs_list(
    #     **(get_for_VanNeer_iterations(episode_time, take_from_the_end=0.5, CO2_output_column=0,
    #                                   out_names_to_plot='CO2')),
    #     params_variants=variants.tolist(),
    #     names=('log_omega', ),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path='PC_plots/230504_VanNeer_benchmark',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # LIBUDA, LYNCH

    # SBP
    # run_jobs_list(
    #     **(get_for_SBP_iteration(500, 'O2', ziff_model=False,
    #                              out_name_to_observe='CO2')),
    #     **(jobs_list_from_grid(
    #         # (i * 5. for i in range(1, 4)),
    #         # (i * 5. for i in range(4, 8)),
    #         (i * 5. for i in range(8, 12)),
    #         map(lambda x: 0.1 * x, range(1, 10)),
    #         names=('total', 'first_part'),
    #     )),
    #     names_groups=(),
    #     const_params={'O2_max': 1.e-4, 'CO_max': 1.e-4},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path='PC_plots/230419_Libuda_SBP_part3',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=110,
    # )

    # benchmark
    # pressure_unit = 1.e-5 * 1.388
    # # pressure_unit = 0.1  # Libuda: 1.e-5
    # points_num = 36
    # # variants = np.linspace(0., 1., points_num).reshape(-1, 1)
    # optimal_ratio = 0.388 / (1 + 0.388)  # L2001
    # # optimal_ratio = 0.25 / (1 + 0.25)  # LD
    # variants = np.hstack((np.linspace(0., optimal_ratio, int(points_num * optimal_ratio)),
    #                       np.linspace(optimal_ratio, 1., points_num - int(points_num * optimal_ratio)))).reshape(-1, 1)
    # # variants = np.hstack((-1 * variants + 1, variants)) * 10 * pressure_unit
    # episode_time = 500
    # # episode_time = 1000  # Libuda: 500
    # run_jobs_list(
    #     **(get_for_Ziff_iterations(pressure_unit, episode_time, take_from_the_end=0.5, CO2_output_column=0,
    #                                out_names_to_plot='CO2')),
    #     params_variants=variants.tolist(),
    #     # names=('O2', 'CO'),
    #     names=('x_co', ),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path='PC_plots/Libuda/230512_test_get_for_variations',
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
    # iteration_func = get_for_Ziff_iterations(pressure_unit, 5000 * time_unit, take_from_the_end=0.1, CO2_output_column=0,
    #                                          out_names_to_plot=('CO2_prod_rate', ))['iteration_function']
    # iteration_func(PC_obj, {'x': 0.531}, './PC_plots/ZGBk_true_optimal/', 666)

    # points_num = 24
    # variants = (np.linspace(0., 0.35, points_num // 8),
    #             np.linspace(0.35, 0.55, 6 * points_num//8),
    #             np.linspace(0.55, 1., points_num//8))
    # variants = np.hstack(variants)
    # run_jobs_list(
    #     **(get_for_Ziff_iterations(pressure_unit, time_unit, take_from_the_end=0.1, CO2_output_column=0,
    #                                out_names_to_plot=('CO2_prod_rate', ))),
    #     params_variants=variants.reshape(-1, 1).tolist(),
    #     names=('x', ),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=3,
    #     out_fold_path='PC_plots/230425_ZGBk_profile',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=110,
    # )

    # dynamic advantage run
    # MC_time_step = PC_obj.process_to_control.m * PC_obj.process_to_control.n
    # # MC_time_step //= 1000
    # t_p, t_d = 150 * MC_time_step, 10 * MC_time_step,  # original values
    # x_p, x_d = 0.535, 0.5
    # run_jobs_list(
    #     repeat_periods_calc_rate,
    #     params_variants=[({'x': x_p}, t_p, {'x': x_d}, t_d)],
    #     names=('part1', 't1', 'part2', 't2'),
    #     names_groups=(),
    #     const_params={
    #         'periods_number': 100
    #     },
    #     sort_iterations_by='rate_mean',
    #     PC=PC_obj,
    #     repeat=50,
    #     out_fold_path='PC_plots/230510_ZGBk_dynamic_advantage_test',
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

    # Ziff_iteration = get_for_Ziff_iterations(1.e+4, 2.e-6)['iteration_function']
    # Ziff_iteration(PC_obj, {'O2': 7.1e+4, 'CO': 2.9e+4}, './PC_plots/KMC', 0)


if __name__ == '__main__':
    main()
