from ProcessController import ProcessController
from test_models import *
from targets_metrics import *
from multiple_jobs_functions import *


def main():
    size = [10, 10]
    PC_KMC = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=True,
                                                  O2_top=1.1e5, CO_top=1.1e5,
                                                  CO2_rate_top=2.e5, CO2_count_top=2.e3,
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

    run_jobs_list(
        one_turn_search_iteration,
        **(jobs_list_from_grid(
            (0.3, 0.35),
            (0.25, 0.2, 0.1),
            (3.e-8, 5.e-8, 1.e-7, 2.e-7),
            names=('x0', 'x1', 't0')
        )),
        names_groups=(),
        const_params={'t1': 2.e-7},
        sort_iterations_by='Total_CO2_Count',
        PC=PC_obj,
        repeat=3,
        on_cluster=False,
        python_interpreter='../RL_10_21/venv/bin/python',
        out_fold_path='PC_plots/2220324_one_turn_search',
        separate_folds=False,
    )


if __name__ == '__main__':
    main()
