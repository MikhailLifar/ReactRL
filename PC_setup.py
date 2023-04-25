import copy

from ProcessController import ProcessController
from test_models import *
from targets_metrics import *


DEFAULT_PARAMS = {
        'Libuda2001': {
            'model_class': LibudaModelReturnK3K1AndPressures,
            'to_model_constructor': {},
            'to_PC_constructor': {
                'analyser_dt': 1,
                'target_func_to_maximize': get_target_func('CO2_value'),
                'target_func_name': 'CO2 form. rate',
                'target_int_or_sum': 'int',
                'RESOLUTION': 10,
                'supposed_step_count': 10000,
                'supposed_exp_time': 10000,
            },
            'to_set_plot_params': {
                'input_lims': [-1e-5, None],
                'input_ax_name': 'Pressure, Pa',
                'output_lims': [-1e-2, None],
                'output_ax_name': 'CO2 formation rate, $(Pt atom * sec)^{-1}$',
                'additional_lims': [-1e-2, 1. + 1.e-2],
            },
            'metrics': (('integral CO2', CO2_integral),
                        # ('O2 conversion', overall_O2_conversion),
                        # ('CO conversion', overall_CO_conversion)
                        )
        },
        'LibudaG': None,
        'ZGB': {
            'model_class': ZGBModel,
            # 'size': [80, 25],  # [128, 256]
            'to_model_constructor': {
                'm': 80, 'n': 25,
                # log_on=True,
                # O2_top=1.1e5, CO_top=1.1e5,
                # CO2_rate_top=3.e5,
                'CO2_count_top': 1.e4,
                # T=373.,
            },
            'to_PC_constructor': {
                'analyser_dt': 2e+2,
                # 'target_func_to_maximize': get_target_func('CO2_value'),
                # 'target_func_name': 'CO2_count',
                'target_func_to_maximize': get_target_func('CO2_value'),
                'target_func_name': 'CO2_prod_rate',
                # 'target_int_or_sum': 'sum',
                'target_int_or_sum': 'int',
                'RESOLUTION': 1,
                'supposed_step_count': 100000,
                'supposed_exp_time': 1e+6,
            },
            'to_set_plot_params': {
                'input_lims': [-1e-5, None],
                'input_ax_name': 'Pressure, Pa',
                'output_lims': [-1e-2, None],
                # 'output_ax_name': 'CO2 formation rate, $(Pt atom * sec)^{-1}$',
                'additional_lims': [-1e-2, 1. + 1.e-2],
                # 'output_ax_name': 'CO x O events count'
                'output_ax_name': 'CO2_prod_rate'
            },
            'metrics': (
                        # ('CO2 count', lambda time_arr, arr: np.sum(arr[:, 0]), ),
                        ('integral CO2', CO2_integral),
                        # ('O2 conversion', overall_O2_conversion),
                        # ('CO conversion', overall_CO_conversion)
                        )
        }
    }

# ZGBk setup
DEFAULT_PARAMS['ZGBk'] = copy.deepcopy(DEFAULT_PARAMS['ZGB'])
DEFAULT_PARAMS['ZGBk']['model_class'] = ZGBkModel
DEFAULT_PARAMS['ZGBk']['to_model_constructor'].update({'m': 100, 'n': 100, 'k': 0.02})
DEFAULT_PARAMS['ZGBk']['to_PC_constructor'].update({'analyser_dt': 1.e+3,
                                                    'supposed_step_count': int(1e+5),
                                                    'supposed_exp_time': 1e+7,
                                                    })


def default_PC_setup(model_id: str):

    if model_id == 'LibudaG':
        PC_LibudaG = ProcessController(GeneralizedLibudaModel(
                                            # params={
                                            # 'thetaA_max': 0.5, 'thetaB_max': 0.25,
                                            # 'thetaA_init': 0., 'thetaB_init': 0.25,
                                            # 'rate_ads_A': 0.14895,
                                            # 'rate_des_A': 0.07162,
                                            # 'rate_ads_B': 0.06594,
                                            # 'rate_react': 5.98734,
                                            # 'C_B_inhibit_A': 0.3,
                                            # 'C_A_inhibit_B': 1.,
                                            # },
                                            resample_when_reset=True,
                                        ),
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

    elif model_id == 'Ziff':
        parameters = DEFAULT_PARAMS['Ziff']
        model = parameters['model_class'](**parameters['to_model_constructor'])
        PC_obj = ProcessController(model, **(parameters['to_PC_constructor']))
        PC_obj.set_plot_params(**(parameters['to_set_plot_params']))
        PC_obj.set_metrics(*(parameters['metrics']))

    elif model_id == 'Pd_monte_coffee':
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


def general_PC_setup(model_id, *change_parameters):
    # change_parameters: tuples of (k1, k2 in params[k1], k3 in params[k2]... v)
    parameters = copy.deepcopy(DEFAULT_PARAMS[model_id])
    for tup in change_parameters:
        d = parameters
        for k in tup[:-2]:
            d = d[k]
        d[tup[-2]] = tup[-1]

    model = parameters['model_class'](**parameters['to_model_constructor'])
    PC_obj = ProcessController(model, **(parameters['to_PC_constructor']))
    PC_obj.set_plot_params(**(parameters['to_set_plot_params']))
    PC_obj.set_metrics(*(parameters['metrics']))

    return PC_obj

