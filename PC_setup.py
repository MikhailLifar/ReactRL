import copy

from ProcessController import ProcessController
from test_models import *
from targets_metrics import *


DEFAULT_PARAMS = {
        'Libuda2001': {
            'model_class': LibudaModelReturnK3K1AndPressures,
            'to_model_constructor': {},
            'to_PC_constructor': {
                'target_func_to_maximize': get_target_func('coordinate'),
                'target_func_name': 'CO2_form_rate',
                'RESOLUTION': 10,
            },
            'metrics': (
                        ('integral CO2', CO2_integral),
                        # ('O2 conversion', overall_O2_conversion),
                        ('CO conversion', overall_CO_conversion)
                        )
        },
        'LibudaG': {
            'model_class': GeneralizedLibudaModel,
            'to_model_constructor': {
                # Libuda original params
                'set_Libuda': True,
                'resample_when_reset': False,
            },
            'to_PC_constructor': {
                'analyser_dt': 1.e-1,
                'target_func_to_maximize': lambda x: x[0],
                'target_func_name': 'C production rate',
                'target_int_or_sum': 'int',
                'RESOLUTION': 1,
            },
            'metrics': (
                ('integral C', lambda time_arr, arr: integral(time_arr, arr[:, 0])),
                # ('O2 conversion', overall_O2_conversion),
                # ('CO conversion', overall_CO_conversion)
            )
        },
        'Lynch': {
            'model_class': LynchModel,
            'to_model_constructor': {},
            'to_PC_constructor': {
                'analyser_dt': 1,
                'target_func_to_maximize': get_target_func('coordinate'),
                'target_func_name': 'CO2_prod_rate',
                'RESOLUTION': 97,
            },
            'metrics': (
                        # ('CO2 count', lambda time_arr, arr: np.sum(arr[:, 0]), ),
                        ('integral CO2', CO2_integral),
                        # ('O2 conversion', overall_O2_conversion),
                        # ('CO conversion', overall_CO_conversion)
                        )
        },
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
                'target_func_to_maximize': get_target_func('coordinate'),
                'target_func_name': 'CO2_prod_rate',
                # 'target_int_or_sum': 'sum',
                'target_int_or_sum': 'int',
                'RESOLUTION': 1,
            },
            'metrics': (
                        # ('CO2 count', lambda time_arr, arr: np.sum(arr[:, 0]), ),
                        ('integral CO2', CO2_integral),
                        # ('O2 conversion', overall_O2_conversion),
                        # ('CO conversion', overall_CO_conversion)
                        )
        },
        'Bassett': {
            'model_class': BassettModel,
            'to_model_constructor': {},
            'to_PC_constructor': {
                'analyser_dt': 1.e-1,
                'target_func_to_maximize': get_target_func('coordinate'),
                'target_func_name': 'CO2_prod_rate',
                'target_int_or_sum': 'int',
                'RESOLUTION': 1,
            },
            'metrics': (
                        # ('CO2 count', lambda time_arr, arr: np.sum(arr[:, 0]), ),
                        ('integral CO2', lambda time_arr, arr: integral(time_arr, arr[:, 0])),
                        # ('O2 conversion', overall_O2_conversion),
                        # ('CO conversion', overall_CO_conversion)
                        )
        },
        'MCKMC': {
            'model_class': MCKMCModel,
            'to_model_constructor': dict(
                surf_shape=(5, 5, 1),
                log_on=True,
                O2_top=1.e-4, CO_top=1.e-4,
                CO2_rate_top=1.4e2, CO2_count_top=1.e2,
                T=440.,
            ),
            'to_PC_constructor': {
                'analyser_dt': 1.e-1,
                'target_func_to_maximize': get_target_func('coordinate'),
                'target_func_name': 'CO2_prod_rate',
                'target_int_or_sum': 'int',
                'RESOLUTION': 1,
            },
            'metrics': (
                        # ('CO2 count', lambda time_arr, arr: np.sum(arr[:, 0]), ),
                        ('integral CO2', lambda time_arr, arr: integral(time_arr, arr[:, 0])),
                        # ('O2 conversion', overall_O2_conversion),
                        # ('CO conversion', overall_CO_conversion)
                        )
        }
    }


DEFAULT_PLOT_SPECS = {
    'Libuda2001': [
        # input
        {'names': ('O2', 'CO'), 'groups': 'input', 'styles': (None, None),
         'to_fname': '_input',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-6, 1.e-4 + 5.e-6],
             'save_csv': True,
         }},
        # add
        {'names': ('thetaO', 'thetaCO'), 'groups': ('add', 'add'), 'styles': None,
         'to_fname': '_add',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # CO2 rate
        {'names': ('CO2', ), 'groups': ('output', ), 'styles': None,
         'to_fname': '_output_CO2',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'reaction rate',
             'ylim': [-5.e-3, None],
             'save_csv': True,
                     }},
        # # target
        # {'names': ('reaction rate', ), 'groups': ('target', ), 'styles': None,
        #  'to_fname': '_target',
        #  'to_plot_to_f': {
        #      'xlabel': 'Time, s', 'ylabel': 'reaction rate',
        #      'ylim': [-5.e-3, None],
        #      'save_csv': True,
        #              }},
    ],
    'LibudaG': [
        # input
        {'names': ('inputB', 'inputA'), 'groups': 'input', 'styles': (None, None),
         'to_fname': '_input',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # add
        {'names': ('thetaB', 'thetaA'), 'groups': ('add', 'add'), 'styles': None,
         'to_fname': '_add',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # output
        {'names': ('outputC', ), 'groups': ('output', ), 'styles': None,
         'to_fname': '_output',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'reaction rate',
             'ylim': [-5.e-3, None],
             'save_csv': True,
                     }},
        # target
        # {'names': ('reaction rate', ), 'groups': ('target', ), 'styles': None,
        #  'to_fname': '_target',
        #  'to_plot_to_f': {
        #      'xlabel': 'Time, s', 'ylabel': 'reaction rate',
        #      'ylim': [-5.e-3, None],
        #      'save_csv': True,
        #              }},
    ],
    'ZGB': [
        # input
        {'names': ('x', ), 'groups': 'input', 'styles': (None, ),
         'to_fname': '_input',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'Pressure',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # add
        {'names': ('thetaO', 'thetaCO'), 'groups': ('add', 'add'), 'styles': None,
         'to_fname': '_add',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # output
        {'names': ('CO2_prod_rate', ), 'groups': ('output', ), 'styles': None,
         'to_fname': '_output',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'reaction rate',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
                     }},
    ],
    'ZGBTwo': [
        # input
        {'names': ('O2', 'CO'), 'groups': 'input', 'styles': (None, None),
         'to_fname': '_input',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'Pressure',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # add
        {'names': ('thetaO', 'thetaCO'), 'groups': ('add', 'add'), 'styles': None,
         'to_fname': '_add',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # output
        {'names': ('CO2_prod_rate', ), 'groups': ('output', ), 'styles': None,
         'to_fname': '_output',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'reaction rate',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
                     }},
    ],
    'Bassett': [
        # input
        {'names': ('O2', 'CO'), 'groups': 'input', 'styles': (None, None),
         'to_fname': '_input',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # add
        {'names': ('thetaO', 'thetaCO', 'thetaOsub'), 'groups': 'add', 'styles': None,
         'to_fname': '_add',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': ' ',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # output
        {'names': ('CO2', ), 'groups': ('output', ), 'styles': None,
         'to_fname': '_output',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'reaction rate',
             'ylim': [-5.e-3, None],
             'save_csv': True,
                     }},
    ],
    'MCKMC': [
        # input
        {'names': ('O2', 'CO'), 'groups': 'input', 'styles': (None, ),
         'to_fname': '_input',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'Pressure',
             'ylim': [-5.e-6, 1.e-4 + 5.e-6],
             'save_csv': True,
         }},
        # add
        {'names': ('thetaO', 'thetaCO'), 'groups': ('add', 'add'), 'styles': None,
         'to_fname': '_add',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': '?',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
         }},
        # output
        {'names': ('CO2_rate', ), 'groups': ('output', ), 'styles': None,
         'to_fname': '_output',
         'to_plot_to_f': {
             'xlabel': 'Time, s', 'ylabel': 'reaction rate',
             'ylim': [-5.e-2, 1. + 5.e-2],
             'save_csv': True,
                     }},
    ],
}

# Libuda2001 plot specs

# LibudaD setup
DEFAULT_PARAMS['LibudaD'] = copy.deepcopy(DEFAULT_PARAMS['Libuda2001'])
DEFAULT_PARAMS['LibudaD']['model_class'] = LibudaModelWithDegradation
DEFAULT_PARAMS['LibudaD']['to_model_constructor'].update({'v_d': 0.01, 'v_r': 0.1, 'border': 4.})
DEFAULT_PLOT_SPECS['LibudaD'] = DEFAULT_PLOT_SPECS['Libuda2001']

# LibudaGWithT setup
DEFAULT_PARAMS['LibudaGWithT'] = copy.deepcopy(DEFAULT_PARAMS['LibudaG'])
DEFAULT_PARAMS['LibudaGWithT']['model_class'] = LibudaGWithTemperature
# DEFAULT_PARAMS['LibudaGWithT']['to_set_plot_params'].update({
#     'input_lims': [380, 620],
#     'input_ax_name': 'Temperature, K',
# })
DEFAULT_PLOT_SPECS['LibudaGWithT'] = copy.deepcopy(DEFAULT_PLOT_SPECS['LibudaG'])
DEFAULT_PLOT_SPECS['LibudaGWithT'][0] = {'names': ('inputB', 'inputA', 'T'), 'groups': 'input', 'styles': (None, None, {'twin': True}),
                                         'to_fname': '_input',
                                         'to_plot_to_f': {
                                             'xlabel': 'Time, s', 'ylabel': '?',
                                             'ylim': [-5.e-2, 1. + 5.e-2],
                                             'twin_params': {
                                                 'ylabel': 'T, K',
                                                 'ylim': [250, 900]
                                             },
                                             'save_csv': True,
                                         }}

# LibudaGWithTEnergies setup
DEFAULT_PARAMS['LibudaGWithTEs'] = copy.deepcopy(DEFAULT_PARAMS['LibudaGWithT'])
DEFAULT_PARAMS['LibudaGWithTEs']['model_class'] = LibudaGWithTEnergies
DEFAULT_PLOT_SPECS['LibudaGWithTEs'] = copy.deepcopy(DEFAULT_PLOT_SPECS['LibudaGWithT'])

# ZGBk setup
DEFAULT_PARAMS['ZGBk'] = copy.deepcopy(DEFAULT_PARAMS['ZGB'])
DEFAULT_PARAMS['ZGBk']['model_class'] = ZGBkModel
DEFAULT_PARAMS['ZGBk']['to_model_constructor'].update({'m': 100, 'n': 100, 'k': 0.02})
DEFAULT_PARAMS['ZGBk']['to_PC_constructor'].update({'analyser_dt': 1.e+3,
                                                    })
# ZGBTwoInputs setup
DEFAULT_PARAMS['ZGBTwo'] = copy.deepcopy(DEFAULT_PARAMS['ZGB'])
DEFAULT_PARAMS['ZGBTwo']['model_class'] = ZGBTwoInputsModel


def general_PC_setup(model_id, *change_parameters):
    # change_parameters: tuples of (k1, k2 in params[k1], k3 in params[k2]... v)
    parameters = copy.deepcopy(DEFAULT_PARAMS[model_id])
    for tup in change_parameters:
        d = parameters
        for k in tup[:-1]:
            d = d[k]
        d.update(tup[-1])

    model = parameters['model_class'](**parameters['to_model_constructor'])
    PC_obj = ProcessController(model, **(parameters['to_PC_constructor']))
    # PC_obj.set_plot_params(**(parameters['to_set_plot_params']))
    PC_obj.set_metrics(*(parameters['metrics']))

    PC_obj.set_plot_specs(*(DEFAULT_PLOT_SPECS[model_id]))

    return PC_obj


