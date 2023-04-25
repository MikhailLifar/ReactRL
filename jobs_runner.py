import numpy as np

from ProcessController import ProcessController
from test_models import *
from targets_metrics import *
from multiple_jobs_functions import *

from optimize_funcs import get_for_param_opt_iterations
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

    # PC_obj = PC_setup.default_PC_setup('Ziff')
    PC_obj = PC_setup.general_PC_setup('Libuda2001')

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

    # LIBUDA

    # SBP
    run_jobs_list(
        **(get_for_SBP_iteration(500, 'O2', ziff_model=False,
                                 out_name_to_observe='CO2')),
        **(jobs_list_from_grid(
            # (i * 5. for i in range(1, 4)),
            # (i * 5. for i in range(4, 8)),
            (i * 5. for i in range(8, 12)),
            map(lambda x: 0.1 * x, range(1, 10)),
            names=('total', 'first_part'),
        )),
        names_groups=(),
        const_params={'O2_max': 1.e-4, 'CO_max': 1.e-4},
        sort_iterations_by='CO2',
        PC=PC_obj,
        repeat=1,
        out_fold_path='PC_plots/230419_Libuda_SBP_part3',
        separate_folds=False,
        cluster_command_ops=False,
        python_interpreter='../RL_10_21/venv/bin/python',
        at_same_time=110,
    )

    # benchmark
    # pressure_unit = 1.e-5
    # pairs_num = 26
    # variants = np.linspace(0., 1., pairs_num).reshape(-1, 1)
    # variants = np.hstack((variants, -1 * variants + 1)) * 10 * pressure_unit
    # # pairs_num = 3
    # run_jobs_list(
    #     **(get_for_Ziff_iterations(pressure_unit, 500, take_from_the_end=0.1, CO2_output_column=0,
    #                                out_names_to_plot='CO2')),
    #     params_variants=variants.tolist(),
    #     names=('O2', 'CO'),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=1,
    #     out_fold_path='PC_plots/230419_Libuda_benchmark',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # Ziff Switch Between Pure
    # run_jobs_list(
    #     **(get_for_SBP_iteration(2.e+5, 'O2', ziff_model=True)),
    #     **(jobs_list_from_grid(
    #         (i * 1.e+3 for i in range(1, 4)),
    #         map(lambda x: 0.1 * x, range(1, 10)),
    #         names=('total', 'first_part'),
    #     )),
    #     names_groups=(),
    #     const_params={'O2_max': 1.e+5, 'CO_max': 1.e+5},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=3,
    #     out_fold_path='PC_plots/230415_Ziff_80x25_SBP',
    #     separate_folds=False,
    #     cluster_command_ops=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=100,
    # )

    # Ziff benchmark
    # pressure_unit = 1.e+4
    # pairs_num = 26
    # run_jobs_list(
    #     **(get_for_Ziff_iterations(pressure_unit, 24.e+4, take_from_the_end=0.1, CO2_output_column=0)),
    #     params_variants=np.linspace(0., 1., pairs_num).reshape(-1, 1).tolist(),
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

    # run_jobs_list(
    #     **(get_for_SBP_iteration(2.e-6)),
    #     **(jobs_list_from_grid(
    #         (2.e-8, 5.e-8, 1.e-7, 2.e-7),
    #         (2.e-8, 5.e-8, 1.e-7, 2.e-7),
    #         names=('t0', 't1'),
    #     )),
    #     names_groups=(),
    #     const_params={'O2_max': 1.e+5, 'CO_max': 1.e+5},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     repeat=3,
    #     out_fold_path='PC_plots/230405_SwitchBetweenPure_20x20',
    #     separate_folds=False,
    #     on_cluster=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     at_same_time=80,
    # )

    # pressure_unit = 1.e+4
    # # pairs_num = 26
    # pairs_num = 3
    # run_jobs_list(
    #     **(get_for_Ziff_iterations(pressure_unit, 2.e-6)),
    #     params_variants=[((1 - x) * 10 * pressure_unit, x * 10 * pressure_unit)
    #                      # for x in np.linspace(0., 1., pairs_num)],
    #                      for x in np.linspace(0.5, 1., pairs_num)],
    #     names=('O2', 'CO'),
    #     names_groups=(),
    #     const_params={},
    #     sort_iterations_by='CO2',
    #     PC=PC_obj,
    #     # repeat=3,
    #     repeat=2,
    #     on_cluster=False,
    #     python_interpreter='../RL_10_21/venv/bin/python',
    #     out_fold_path='PC_plots/230404_debug',
    #     separate_folds=False,
    #     at_same_time=80,
    # )

    # # the best stationary obtained by optimization
    # one_turn_search_iteration(PC_obj, {'x0': 0.295, 'x1': 0.295,
    #                                    't0': 1.e-6, 't1': 1.e-6, },
    #                           'PC_plots/2303_KMC_const_best', 0)

    # Ziff_iteration = get_for_Ziff_iterations(1.e+4, 2.e-6)['iteration_function']
    # Ziff_iteration(PC_obj, {'O2': 7.1e+4, 'CO': 2.9e+4}, './PC_plots/KMC', 0)


if __name__ == '__main__':
    main()
