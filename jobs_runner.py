import numpy as np

from ProcessController import ProcessController
from test_models import *
from targets_metrics import *
from multiple_jobs_functions import *


def main():
    size = [40, 40]
    PC_KMC = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=True,
                                                  O2_top=1.1e5, CO_top=1.1e5,
                                                  CO2_rate_top=3.e5, CO2_count_top=1.e4,
                                                  T=373.),
                               analyser_dt=0.25e-7,
                               target_func_to_maximize=get_target_func('CO2_count'),
                               target_func_name='CO2_count',
                               target_int_or_sum='sum',
                               RESOLUTION=1,  # ATTENTION! Always should be 1 if we use KMC, otherwise we will get wrong results!
                               supposed_step_count=100,  # memory controlling parameters
                               supposed_exp_time=1.e-5)
    PC_obj = PC_KMC
    PC_obj.set_metrics(
                       # ('integral CO2', CO2_integral),
                       ('CO2 count', CO2_count),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
    )

    PC_KMC.set_plot_params(input_lims=[-1e-5, None], input_ax_name='Pressure, Pa',
                           output_lims=[-1e-2, None],
                           additional_lims=[-1e-2, 1. + 1.e-2],
                           # output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$',
                           output_ax_name='CO x O events count')

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

    # **(jobs_list_from_grid(
    #     (0.3, 0.35),
    #     (0.25, 0.2, 0.1),
    #     (3.e-8, 5.e-8, 1.e-7, 2.e-7),
    #     names=('x0', 'x1', 't0')
    # )),

    pressure_unit = 1.e+4
    # pairs_num = 26
    pairs_num = 3
    run_jobs_list(
        **(get_for_Ziff_iterations(pressure_unit, 2.e-6)),
        params_variants=[((1 - x) * 10 * pressure_unit, x * 10 * pressure_unit)
                         # for x in np.linspace(0., 1., pairs_num)],
                         for x in np.linspace(0.5, 1., pairs_num)],
        names=('O2', 'CO'),
        names_groups=(),
        const_params={},
        sort_iterations_by='CO2',
        PC=PC_obj,
        # repeat=3,
        repeat=2,
        on_cluster=False,
        python_interpreter='../RL_10_21/venv/bin/python',
        out_fold_path='PC_plots/230404_debug',
        separate_folds=False,
        at_same_time=80,
    )

    # # the best stationary obtained by optimization
    # one_turn_search_iteration(PC_obj, {'x0': 0.295, 'x1': 0.295,
    #                                    't0': 1.e-6, 't1': 1.e-6, },
    #                           'PC_plots/2303_KMC_const_best', 0)


if __name__ == '__main__':
    main()
