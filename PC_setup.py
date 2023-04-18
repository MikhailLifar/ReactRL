from ProcessController import ProcessController
from test_models import *
from targets_metrics import *


def default_PC_setup(PCid: str):
    if PCid == 'LibudaG':
        PC_LibudaG = ProcessController(GeneralizedLibudaModel(params={
            'thetaA_max': 0.5, 'thetaB_max': 0.25,
            'thetaA_init': 0., 'thetaB_init': 0.25,
            'rate_ads_A': 0.14895,
            'rate_des_A': 0.07162,
            'rate_ads_B': 0.06594,
            'rate_react': 5.98734,
            'C_B_inhibit_A': 0.3,
            'C_A_inhibit_B': 1.,
        }),
                                       analyser_dt=1,
                                       target_func_to_maximize=lambda x: x[0],
                                       target_func_name='C production rate',
                                       target_int_or_sum='int',
                                       RESOLUTION=20,
                                       supposed_step_count=2000,  # memory controlling parameters
                                       supposed_exp_time=2e+4)
        PC_obj = PC_LibudaG

        PC_obj.set_plot_params(input_lims=[-1e-5, 1.1], input_ax_name='?',
                               output_lims=[-1e-2, None],
                               additional_lims=[-1e-2, 1. + 1.e-2],
                               output_ax_name='C prod rate')

        PC_obj.set_metrics(
                       ('integral C', lambda time_arr, arr: integral(time_arr, arr[:, 0])),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
        )

    elif PCid == 'Ziff':
        # size = [128, 256]
        # size = [80, 80]
        size = [80, 25]
        PC_Ziff = ProcessController(KMC_Ziff_model(*size,
                                                   # log_on=True,
                                                   # O2_top=1.1e5, CO_top=1.1e5,
                                                   # CO2_rate_top=3.e5,
                                                   CO2_count_top=1.e4,
                                                   # T=373.,
                                                   ),
                                    analyser_dt=2e+2,
                                    target_func_to_maximize=get_target_func('CO2_value'),
                                    target_func_name='CO2_count',
                                    target_int_or_sum='sum',
                                    RESOLUTION=1,  # ATTENTION! Always should be 1 if we use KMC, otherwise we will get wrong results!
                                    supposed_step_count=2000,  # memory controlling parameters
                                    supposed_exp_time=1e+6)
        PC_obj = PC_Ziff

        PC_obj.set_plot_params(input_lims=[-1e-5, None], input_ax_name='Pressure, Pa',
                               output_lims=[-1e-2, None],
                               additional_lims=[-1e-2, 1. + 1.e-2],
                               # output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$',
                               output_ax_name='CO x O events count')

        PC_obj.set_metrics(
                       # ('integral CO2', CO2_integral),
                       ('CO2 count', lambda time_arr, arr: np.sum(arr[:, 0])),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
        )

    elif PCid == 'Pd_monte_coffee':
        size = [20, 20]
        PC_KMC = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=False,
                                                      O2_top=1.1e5, CO_top=1.1e5,
                                                      CO2_rate_top=5.e5, CO2_count_top=1.e4,
                                                      T=373.),
                                   analyser_dt=0.125e-8,
                                   target_func_to_maximize=get_target_func('CO2_count'),
                                   target_func_name='CO2_count',
                                   target_int_or_sum='sum',
                                   RESOLUTION=1,  # ATTENTION! Always should be 1 if we use KMC, otherwise we will get wrong results!
                                   supposed_step_count=1000,  # memory controlling parameters
                                   supposed_exp_time=1.e-5)

        PC_obj = PC_KMC

        PC_obj.set_metrics(
                       # ('integral CO2', CO2_integral),
                       ('CO2 count', CO2_count),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
        )

        PC_obj.set_plot_params(input_lims=[-1e-5, None], input_ax_name='Pressure, Pa',
                               output_lims=[-1e-2, None],
                               additional_lims=[-1e-2, 1. + 1.e-2],
                               # output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$',
                               output_ax_name='CO x O events count')

    else:
        raise ValueError

    return PC_obj

