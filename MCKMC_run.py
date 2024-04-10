import os
import numpy as np

import lib
import PC_setup


def KMC_compatibility_test():
    PC_obj = PC_setup.general_PC_setup('MCKMC', ('to_model_constructor',
                                                 {
                                                     'surf_shape': (20, 20, 1),
                                                     'logDir': './PC_plots/MCKMC/compatibility_test/20x20',
                                                 }),
                                                ('to_PC_constructor', {'analyser_dt': 100.}))
    PC_obj.reset()
    PC_obj.set_controlled((1., 1.))
    PC_obj.time_forward(1000.)
    PC_obj.get_process_output()


def MCKMC_DE_compatibility_test():
    PC_obj = PC_setup.general_PC_setup('MCKMC', ('to_model_constructor', {'surf_shape': (25, 25, 1),
                                                                          'log_on': True,
                                                                          'snapshotDir': './PC_plots/MCKMC/de_compatibility',
                                                                          }))
    PC_DE = PC_setup.general_PC_setup('LibudaG')
    PC_DE.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 0.3,
                                          'thetaA_max': 0.5, 'thetaB_max': 0.25,
                                          'rate_des_A': 0.1, 'rate_react': 0.1,
                                          })
    PC_DE.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0., })

    # O2 then CO
    PC_obj.reset()
    PC_DE.reset()

    PC_obj.set_controlled((1., 0.))
    PC_obj.time_forward(10.)
    PC_obj.set_controlled((0., 1.))
    PC_obj.time_forward(10.)
    PC_obj.get_and_plot('./PC_plots/MCKMC/de_compatibility/MCKMC_O2_then_CO.png')

    PC_DE.set_controlled((1., 0.))
    PC_DE.time_forward(10.)
    PC_DE.set_controlled((0., 1.))
    PC_DE.time_forward(10.)
    PC_DE.get_and_plot('./PC_plots/MCKMC/de_compatibility/DE_O2_then_CO.png')


def MCKMC_one_turn_steady_state_tries():
    diffusion_level = 0.
    shape = (5, 5, 1)

    res_dir = './PC_plots/MCKMC/diffusion_influence'
    O2_CO_dir = f'{res_dir}/O2_then_CO_d({diffusion_level:.2f})'
    CO_O2_dir = f'{res_dir}/CO_then_O2_d({diffusion_level:.2f})'
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(O2_CO_dir, exist_ok=True)
    os.makedirs(CO_O2_dir, exist_ok=True)

    # O2, then CO
    PC_obj = PC_setup.general_PC_setup('MCKMC', ('to_model_constructor', {'surf_shape': shape,
                                                                          'logDir': O2_CO_dir,
                                                                          'snapshotDir': O2_CO_dir,
                                                                          'diffusion_level': diffusion_level,
                                                                          }))

    PC_obj.reset()
    PC_obj.set_controlled((1., 0.))
    PC_obj.time_forward(10.)
    PC_obj.set_controlled((0., 1.))
    PC_obj.time_forward(10.)
    PC_obj.get_and_plot(f'{O2_CO_dir}/O2_then_CO.png')

    # CO then O2
    # PC_obj = PC_setup.general_PC_setup('MCKMC', ('to_model_constructor', {'surf_shape': (25, 25, 1),
    #                                                                       'log_on': True,
    #                                                                       'snapshotDir': './PC_plots/MCKMC/de_compatibility',
    #                                                                       'init_covs': (0.05, 0.95, 0.),
    #                                                                       }))
    PC_obj = PC_setup.general_PC_setup('MCKMC', ('to_model_constructor', {'surf_shape': shape,
                                                                          'logDir': CO_O2_dir,
                                                                          'snapshotDir': CO_O2_dir,
                                                                          'diffusion_level': diffusion_level,
                                                                          }))
    PC_obj.reset()
    PC_obj.set_controlled((0., 1.))
    PC_obj.time_forward(10.)
    PC_obj.set_controlled((1., 0.))
    PC_obj.time_forward(10.)
    PC_obj.get_and_plot(f'{CO_O2_dir}/CO_then_O2.png')

    # # stationary
    # PC_obj = PC_setup.general_PC_setup('MCKMC', ('to_model_constructor', {'surf_shape': (25, 25, 1),
    #                                                                       'log_on': True,
    #                                                                       'snapshotDir': './PC_plots/MCKMC/de_compatibility',
    #                                                                       }))
    # PC_obj.reset()


def MCKMC_run_policy(logDir,
                     snapshotPeriod,
                     plottoffile, t_start=None, t_end=None,
                     surfShape=(5, 5, 1),
                     diffusion_level=0.,
                     OORepLim = 0.3,
                     OCORepLim = 1.1,
                     analyser_dt=1.,
                     int_segment=None,
                     ):

    os.makedirs(logDir, exist_ok=True)

    PC_obj = PC_setup.general_PC_setup('MCKMC',
                                       ('to_model_constructor', {
                                           'surf_shape': surfShape,
                                           'diffusion_level': diffusion_level,
                                           'OORepLim': OORepLim,
                                           'OCORepLim': OCORepLim,
                                           'logDir': logDir,
                                           'saveLog': logDir is not None,
                                           'snapshotPeriod': snapshotPeriod,
                                       }),
                                       ('to_model_constructor', {
                                           'analyser_dt': analyser_dt
                                       }))
    PC_obj.reset()

    options, _ = lib.read_plottof_csv(plottoffile, ret_ops=True)
    Oidx = options[2::3].index('inputB') * 3 + 2
    COidx = options[2::3].index('inputA') * 3 + 2

    times = options[Oidx - 2]
    O_control = options[Oidx - 1]
    CO_control = options[COidx - 1]

    # limit times
    if t_start is None:
        t_start = times[0]
    if t_end is None:
        t_end = times[-1]
    idx = (times >= t_start) & (times <= t_end)
    times = times[idx] - t_start
    O_control = O_control[idx]
    CO_control = CO_control[idx]

    Ostart = O_control[0]
    COstart = CO_control[0]

    PC_obj.set_controlled((Ostart, COstart))
    turning_points = np.where(np.abs(O_control[1:] - O_control[:-1]) > 1.e-5)[0]
    if len(turning_points):
        turning_points = turning_points + 1
        step_end_times = times[turning_points - 1]

        PC_obj.time_forward(step_end_times[0])
        step_end_times = np.array(step_end_times[1:].tolist() + [times[-1]])
        for O_val, CO_val, t in zip(O_control[turning_points], CO_control[turning_points], step_end_times):
            PC_obj.set_controlled((O_val, CO_val))
            PC_obj.time_forward(t - PC_obj.time)
    else:
        PC_obj.time_forward(times[-1])

    ret = {}
    output_dt, output = PC_obj.get_process_output()
    ret['mean_CO2_rate'] = output[:, 0]
    for name in ('thetaO', 'thetaCO'):
        ret[name] = PC_obj.additional_graph[name][:ret['mean_CO2_rate'].size]

    if int_segment is not None:
        idx = (int_segment[0] <= output_dt) & (output_dt <= int_segment[1])
    else:
        idx = np.full_like(output[:, 0], True, dtype='bool')
    for k, v in ret.items():
        ret[k] = np.mean(v[idx])

    PC_obj.plot(f'{logDir}/MCKMC.png')
    if PC_obj.target_func is not None:
        ret['return'] = PC_obj.integrate_along_history(target_mode=True,
                                           time_segment=int_segment)
    elif PC_obj.long_term_target is not None:
        ret['return'] = PC_obj.get_long_term_target(time_segment=int_segment)

    return ret


if __name__ == '__main__':
    # MCKMC_one_turn_steady_state_tries()
    # MCKMC_run_policy('./DEBUG/MCKMCDebug/runPolicy/try0',
    #                  snapshotPeriod=0.2,
    #                  plottoffile='./PC_plots/LibudaG/230923_rates_original_steady_state/'
    #                              'common_variations_x(0.40)_40_all_data.csv',
    #                  t_end=30.,
    #                  )

    KMC_compatibility_test()
