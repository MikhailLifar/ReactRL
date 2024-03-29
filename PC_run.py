import os
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib
from ProcessController import *
from predefined_policies import *
from test_models import *
from targets_metrics import *

from usable_functions import make_subdir_return_path
from lib import plot_to_file

import PC_setup


def estimate_max_rate_libuda_like(PC: ProcessController, state1=(1, 0), state2=(0, 1), turn_time=100.,
                                  return_col=0, plotpath=None):
    input_bounds = [
        PC.process_to_control.get_bounds('min', kind='input', out='array'),
        PC.process_to_control.get_bounds('max', kind='input', out='array')
    ]

    inputs1 = input_bounds[0] + np.array(state1) * (input_bounds[1] - input_bounds[0])
    inputs2 = input_bounds[0] + np.array(state2) * (input_bounds[1] - input_bounds[0])

    PC.reset()
    PC.set_controlled(inputs1)
    PC.time_forward(turn_time)
    PC.set_controlled(inputs2)
    PC.time_forward(turn_time)
    PC.set_controlled(inputs1)
    PC.time_forward(turn_time)

    _, out = PC.get_process_output()

    # plotpath = './DEBUG/estimate_for_LibudaG.png'  # DEBUG only!!!
    if plotpath is not None:
        PC.plot(plotpath)

    return np.max(out[:, return_col])


# def estimate_max_rate_hard_way():
#     raise NotImplementedError


def get_estimate_rate_callback(**kwargs):

    def callback(env):
        if 'top_rate_estim' in kwargs:
            top_rate_estim = kwargs['top_rate_estim']
        else:
            top_rate_estim = estimate_max_rate_libuda_like(env.controller, **kwargs)
        env.rate_estimate = 2. / top_rate_estim
        env.model.assign_and_eval_values(reaction_rate_top=1.6*top_rate_estim)

    return callback


def run_constant_policies_bunch(PC: ProcessController,
                                episode_time, time_step,
                                out_foldpath: str,
                                input_limits=None, variants=None,):
    if input_limits is None:
        input_limits = PC.process_to_control.limits['input']
    else:
        assert len(input_limits) == len(PC.process_to_control.limits['input'])
    policies = []
    for _ in input_limits:
        policies.append(ConstantPolicy(dict()))
    if variants is None:
        assert len(input_limits) == 2
        variants = np.array([(i * 0.1, (10 - i) * 0.1) for i in range(1, 10)])
    for v in variants:
        PC.reset()
        for i, p in enumerate(policies):
            p.update_policy({'value': v[i] * input_limits[i][1]})
        PC.process_by_policy_objs(policies, episode_time, time_step)
        run_id = '_'.join(map(lambda pair: f'{pair[0]}:{pair[1]:.2f}', zip(PC.controlled_names, v)))
        PC.get_and_plot(f'{out_foldpath}/{run_id}.png')


def custom_experiment():
    # # 1st try
    # PC = ProcessController(TestModel())
    # for i in range(5):
    #     PC.set_controlled([i + 0.2 * i1 for i1 in range(1, 6)])
    #     PC.time_forward(30)
    # PC.get_process_output()
    # for c in '12345678':
    #     PC.plot(f'PC_plots/example{c}.png', out_name=c, plot_mode='separately')

    # def CO2_value(x):
    #     # target_v = np.array([2., 1., 3., -1., 0.,
    #     #                      1., -1., 3., -2., 3.])
    #     target_v = np.array([2., 1., 3.])
    #     return -np.linalg.norm(x - target_v)
    #
    # PC = ProcessController(TestModel(), target_func_to_maximize=CO2_value)
    # for i in range(5):
    #     PC.set_controlled([i + 0.2 * i1 for i1 in range(1, 6)])
    #     PC.time_forward(30)
    # print(PC.integrate_along_history(target_mode=True))
    # for c in [f'out{i}' for i in range(1, 4)]:
    #     PC.plot(f'PC_plots/example_{c}.png', out_name=c, plot_mode='separately')
    # PC.plot(f'PC_plots/example_target.png', out_name='CO2_value', plot_mode='separately')

    pass


def test_PC_with_Libuda():

    # PC = ProcessController(LibudaModelWithDegradation(init_cond={'thetaO': 0.25, 'thetaCO': 0.5}, Ts=440),
    #                        target_func_to_maximize=CO2_value)
    # # for i in range(5):
    # #     PC.set_controlled([(i + 0.2 * i1) * 1.e-5 for i1 in range(1, 3)])
    # #     PC.time_forward(30)
    # PC.set_controlled({'O2': 10.e-5, 'CO': 4.2e-5})
    # PC.time_forward(500)
    # print(PC.integrate_along_history(target_mode=True))
    # # PC.plot(f'PC_plots/example_RL_21_10_task.png', out_name='CO2_value', plot_mode='separately')

    # TEST if more CO is better under the old conditions of the tests for the russian article
    # PC_L2001_old = ProcessController(LibudaModelWithDegradation(init_cond={'thetaO': 0.25, 'thetaCO': 0.5}, Ts=440,
    #                                                             v_d=0.01, v_r=1.5, border=4.),
    #                                  target_func_to_maximize=CO2_value)

    # episode_len = 500
    # for i in range(2, 11, 2):
    #     PC_L2001_old.reset()
    #     O2 = i * 1e-5 / 4
    #     CO = i * 1e-5
    #     PC_L2001_old.set_controlled({'O2': O2, 'CO': i * 1e-5})
    #     PC_L2001_old.time_forward(episode_len)
    #     R = PC_L2001_old.integrate_along_history(time_segment=[0., episode_len])
    #     print(f'CO: {CO}, O2: {O2}, R: {R}')

    PC = ProcessController(LibudaModelReturnK3K1AndPressures(init_cond={'thetaO': 0., 'thetaCO': 0.}, Ts=273 + 160),
                           long_term_target_to_maximize=get_target_func('CO2xConversion_I', eps=1.),
                           RESOLUTION=2
                           )
    PC.set_plot_params(output_lims=None, output_ax_name='?',
                       input_lims=[-1.e-3, 1.e-2], input_ax_name='Pressure, Pa')
    # TwoStepPolicy({'1': 3.7e-3, '2': 1.e-3, 't1': 60., 't2': 100., })
    PC.process_by_policy_objs([TwoStepPolicy({'1': 3.828125e-4, '2': 3.828125e-4, 't1': 5.09375, 't2': 5., }),
                               ConstantPolicy({'value': 1e-3})],
                              500.,
                              10.)
    # PC.get_and_plot('PC_plots/debug/debug1.png',
    #                 plot_params={
    #                  'plot_mode': 'separately',
    #                  'out_names': ['CO2', 'long_term_target'],
    #                  'additional_plot': ['thetaCO', 'thetaO']
    #                  },
    #                 get_params={'RESOLUTION': 10})
    PC.get_process_output()
    PC.plot('PC_plots/debug/debug2.png', plot_mode='separately',
            out_names=['CO2', 'long_term_target'],
            additional_plot=['thetaCO', 'thetaO'])

    # # find optimal log_scale
    # average_rate = PC.integrate_along_history(target_mode=True) / PC.get_current_time()
    # max_rate, = PC.process_to_control.get_bounds('max', 'output')
    # log_scale = 5_000
    # print(np.log(1 + log_scale * average_rate / max_rate))
    # print(np.log(1 + log_scale))


def count_conversion_given_exp(csv_path, L_model: LibudaModel):
    df = lib.read_plottof_csv(csv_path, False)
    new_df = pd.DataFrame(columns=['CO_conversion', 'O2_conversion'])

    CO2_name = 'CO2' if 'CO2 x' in df.columns else 'target'

    F_CO2 = L_model.CO2_rate_to_F_value(df[f'{CO2_name} y'].to_numpy())
    F_CO = L_model.pressure_to_F_value(df['CO y'].to_numpy(), 'CO')
    F_O2 = L_model.pressure_to_F_value(df['O2 y'].to_numpy(), 'O2')
    for F in (F_CO, F_O2):
        F[abs(F) < 1e-9] = 1e-9

    new_df['CO_conversion'] = F_CO2 / F_CO
    new_df['O2_conversion'] = F_CO2 / F_O2
    new_df['Time'] = df[f'{CO2_name} x']

    filedir, filename = os.path.split(csv_path)
    filename, _ = os.path.splitext(filename)
    new_path = f'{filedir}/{filename}_conv.csv'
    new_df.to_csv(new_path, index=False)


def Libuda2001_CO_cutoff_policy(PC_obj: ProcessController, program, dest_dir='./PC_plots/Libuda_orginal',
                                p_total=1e-4,
                                transform_x_co_on=None,
                                transform_x_co_off=None,
                                plot=True, ret_idxs=None):

    # default transforms (as in Libuda article)
    def p_co(x_co):
        return p_total * x_co * 7 / np.sqrt(8) / (np.sqrt(7) + 7 * x_co / np.sqrt(8) - np.sqrt(7) * x_co)

    def default_transform_on(x_co):
        CO_p = p_co(x_co)
        return {'O2': p_total - CO_p, 'CO': CO_p}

    def default_transform_off(x_co):
        return {'O2': p_total - p_co(x_co), 'CO': 0.}

    if transform_x_co_on is None:
        transform_x_co_on = default_transform_on
        assert transform_x_co_off is None
        transform_x_co_off = default_transform_off

    def _one_co_on(x_co):
        # CO_p = p_total * x_co  # simple formula
        PC_obj.set_controlled(transform_x_co_on(x_co))
        PC_obj.time_forward(100)
        PC_obj.set_controlled(transform_x_co_off(x_co))
        PC_obj.time_forward(75)

    def run_full_cutoff_series():
        x_co = 0.05
        while x_co < 0.96:
            _one_co_on(x_co)
            x_co += 0.05

    if ret_idxs is not None:
        assert len(program) == 1, 'Can return data only for a one step program'

    ret = None
    for arg in program:
        PC_obj.reset()
        fname = None
        if isinstance(arg, tuple):
            for v in arg:
                _one_co_on(v)
            fname = 'series.png'
        elif isinstance(arg, float):
            _one_co_on(arg)
            fname = f'x_co_{arg:.2f}.png'
        elif isinstance(arg, str) and (arg == 'full'):
            run_full_cutoff_series()
            fname = 'full_series.png'
        output = PC_obj.get_process_output()
        if ret_idxs is not None:
            ret = output[0], output[1][:, ret_idxs]
        if plot:
            PC_obj.plot(f'{dest_dir}/{fname}')

    return ret


# def Pt_exp(path: str):
#
#     # PC_Pt2210 = ProcessController(PtModel(init_cond={'thetaO': 0., 'thetaCO': 0.}),
#     #                               target_func_to_maximize=get_target_func('CO2_value'))
#     PC_Salomons = ProcessController(PtSalomons(init_cond={'thetaO': 0., 'thetaCO': 0., 'thetaOO': 0., }),
#                                   target_func_to_maximize=get_target_func('CO2_value'))
#
#     PC_obj = PC_Salomons
#
#     # first experiment
#     low_value = 10  # 4
#     high_value = 50  # 10
#     reps = 10
#     for i in range(reps):
#         PC_obj.set_controlled({'O2': low_value, 'CO': high_value})
#         PC_obj.time_forward(20)
#         PC_obj.set_controlled({'O2': high_value, 'CO': low_value})
#         PC_obj.time_forward(30)
#     R = PC_obj.integrate_along_history(target_mode=True, time_segment=[0., 50 * reps])
#
#     PC_obj.plot(path)


def benchmark_runs(PC_obj: ProcessController, out_path: str, rate_or_count: str,
                   ):
    temperatures = (300, 373, 500)
    pressures = ((50e3, 50e3), (10e3, 90e3), (90e3, 10e3), (99e3, 1e3), )  # (O2, CO)

    run_time = 1.e-5

    folder = make_subdir_return_path(out_path, prefix='benchmark_', with_date=True, unique=True)

    for T in temperatures:
        PC_obj.process_to_control.assign_and_eval_values(T=T)
        for p_pair in pressures:
            PC_obj.reset()

            # # stationary
            # PC_obj.set_controlled(p_pair)
            # PC_obj.time_forward(run_time)

            # one turn
            PC_obj.set_controlled(p_pair)
            PC_obj.time_forward(run_time / 2)
            PC_obj.set_controlled(p_pair[::-1])
            PC_obj.time_forward(run_time / 2)

            PC_obj.get_and_plot(f'{folder}/T{T}_CO({int(p_pair[0] // 1000)})_O2({int(p_pair[1] // 1000)}).png',
                                plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                             'plot_mode': 'separately', 'out_names': [f'CO2_{rate_or_count}']})

            # plot CO2 integral as A. Guda suggested
            idxs = PC_obj.output_history_dt > -1

            if rate_or_count == 'rate':
                # rate mode
                CO2_int = np.cumsum(PC_obj.output_history[:, 0][idxs]) * PC_obj.analyser_dt
            elif rate_or_count == 'count':
                # count mode
                CO2_int = np.cumsum(PC_obj.output_history[:, 3][idxs])

            # CO2 integral over time, ever increasing
            plot_to_file(PC_obj.output_history_dt[idxs], CO2_int, 'CO2 integral',
                         fileName=f'{folder}/T{T}_O2({int(p_pair[0] // 1000)})_CO({int(p_pair[1] // 1000)})_intCO2.png',
                         title='CO2 integral over time',
                         xlabel='Time', ylabel='Integral',
                         xlim=[0., None], ylim=[-0.1, None], )


def MCKMC_simple_tests():
    # size = [20, 20]
    # PC_obj = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=True, O2_top=1.1e-4, CO_top=1.1e-4,),
    #                            target_func_to_maximize=get_target_func('CO2_value'),
    #                            RESOLUTION=1)
    # PC_obj.set_metrics(('CO2', CO2_integral),)
    # PC_obj.analyser_dt = 1
    # PC_obj.set_plot_params(input_lims=[-1e-5, 1.1e-4], input_ax_name='Pressure, Pa',
    #                        output_lims=[-1e-2, 0.06], output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$')
    # postfix = f'{size[0]}x{size[1]}'
    # for T in (300., 400., 440., 500., 600.):
    #     run_dir = f'PC_plots/KMC/Basic/{int(T)}K_{postfix}'
    #     os.mkdir(run_dir)
    #     PC_obj.process_to_control.cls_parameters['T'] = T
    #     Libuda2001_CO_cutoff_policy(PC_obj, run_dir, 0.25, 0.5, 0.75)

    # TEST IF COMPARABLE WITH ORIGINAL
    # size = [20, 20]
    # PC_obj = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=True,
    #                                               O2_top=1.e5, CO_top=1.1e5,
    #                                               CO2_rate_top=1.4e6, CO2_count_top=1.e4,
    #                                               T=373.),
    #                            analyser_dt=1.e-6,
    #                            target_func_to_maximize=get_target_func('CO2_count'),
    #                            RESOLUTION=1,  # always should be 1 if we use KMC, otherwise we will get wrong results!
    #                            )

    PC_obj = PC_setup.general_PC_setup('MCKMC', ('to_model_constructor', {'surf_shape': (25, 25, 1)}))

    # PC_obj.analyser_dt = 1.e-7
    # PC_obj.reset()
    # for i in range(4):
    #     PC_obj.set_controlled((1E3, 2E3))
    #     PC_obj.time_forward(10.e-5)
    #     PC_obj.set_controlled((10E3, 1E3))
    #     PC_obj.time_forward(10.e-5)
    # PC_obj.get_and_plot(f'PC_plots/KMC/Basic/debug_dynamic.png',
    #                     plot_params={'time_segment': [0., None], 'additional_plot': ['thetaCO', 'thetaO'],
    #                                  'plot_mode': 'separately', 'out_names': ['CO2']})

    # RUN UNDER CONDITIONS FOR RL
    # PC_obj.reset()
    # PC_obj.set_controlled((5E3, 5E3))

    # BENCHMARK
    # benchmark_runs(PC_obj, './PC_plots/model_benchmarks', 'count')

    # NEW TESTS BASED ON DYNAMIC ADV
    PC_obj.reset()
    dt = 10.

    options, _ = lib.read_plottof_csv('231002_sudden_discovery/rl_agent_sol.csv', ret_ops=True)
    Oidx = options[2::3].index('inputB') * 3 + 2
    COidx = options[2::3].index('inputA') * 3 + 2

    times = options[Oidx - 2]
    O_control = options[Oidx - 1]
    CO_control = options[COidx - 1]
    O_control *= 1.e-4
    CO_control *= 1.e-4

    Ostart = O_control[0]
    COstart = CO_control[0]

    PC_obj.set_controlled((Ostart, COstart))
    turning_points = np.where(np.abs(O_control[1:] - O_control[:-1]) > 1.e-5)[0] + 1
    step_end_times = times[turning_points - 1]

    PC_obj.time_forward(step_end_times[0])
    step_end_times = np.array(step_end_times[1:].tolist() + [times[-1]])
    for O_val, CO_val, t in zip(O_control[turning_points], CO_control[turning_points], step_end_times):
        PC_obj.set_controlled((O_val, CO_val))
        PC_obj.time_forward(t - PC_obj.time)
    PC_obj.get_and_plot('repos/MonteCoffee_modified_Pd/snapshots/PC_runned/MCKMC.png')


def Ziff_model_poisoning_speed_test():
    # size = [80, 25]
    # PC_obj = PC_setup.general_PC_setup('ZGB')
    PC_obj = PC_setup.general_PC_setup('ZGBTwo')

    time_step = 2e+3

    PC_obj.reset()
    # PC_obj.set_controlled({'x': 1.})
    PC_obj.set_controlled({'O2': 0., 'CO': 1.})
    CO_cov = 0.
    while CO_cov < 0.96:
        PC_obj.time_forward(time_step)
        time_history, _ = PC_obj.get_process_output()
        CO_cov = PC_obj.additional_graph['thetaCO'][time_history.size - 1]
    PC_obj.plot('PC_plots/ZGB/230918_dynamics_speed_test/ZGBTwo_CO_rich_speed_test.png')

    PC_obj.reset()
    # PC_obj.set_controlled({'x': 0.01})
    PC_obj.set_controlled({'O2': 0.99, 'CO': 0.01})
    O_cov = 0.
    while O_cov < 0.95:
        PC_obj.time_forward(time_step)
        time_history, _ = PC_obj.get_process_output()
        O_cov = PC_obj.additional_graph['thetaO'][time_history.size - 1]
    PC_obj.plot('PC_plots/ZGB/230918_dynamics_speed_test/ZGBTwo_O2_rich_speed_test.png')


def ZGB_snapshots():
    x = 0.57
    m, n = 80, 25
    time_unit = m * n
    episode_time = 10 * time_unit
    step = time_unit
    ZGB = PC_setup.ZGBModel(m, n, CO2_count_top=step)

    outpath = './PC_plots/ZGB/snapshots'
    time_ = 0

    def snapshot(model, filepath):
        surface = model.surface
        fig, ax = plt.subplots(figsize=(m * 16 / (m + n), n * 16 / (m + n)))

        where_co = np.where(surface == 1)
        where_o = np.where(surface == 2)

        ax.scatter(*where_co, c='r', marker='o', label='CO')
        ax.scatter(*where_o, c='b', marker='o', label='O')
        ax.set_title(f'surface state, step {int(time_)}')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.legend(loc='outside lower center', ncol=2, fancybox=True)

        fig.savefig(filepath, dpi=400, bbox_inches='tight')
        plt.close(fig)

    while time_ < episode_time:
        ZGB.update([x], step)
        time_ += step
        snapshot(ZGB, f'{outpath}/step_{int(time_)}.png')


def transition_speed_test():

    def transition_test(point1, point2, period, plot_filepath):
        PC_obj.reset()
        PC_obj.set_controlled(point1)
        PC_obj.time_forward(period)
        PC_obj.set_controlled(point2)
        PC_obj.time_forward(period)
        PC_obj.set_controlled(point1)
        PC_obj.time_forward(period)
        PC_obj.get_process_output()
        PC_obj.plot(plot_filepath)

    # # LibudaGWithT, diff temperatures
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithT')
    # folder = 'PC_plots/LibudaGWithT/230516_transition_speed'
    #
    # transition_test([1., 0., 100 + 273], [0., 1., 100 + 273], 1000, f'{folder}/373K.png')
    #
    # PC_obj.analyser_dt = 1.e-3
    # transition_test([1., 0., 300 + 273], [0., 1., 300 + 273], 0.1, f'{folder}/573K.png')
    #
    # PC_obj.analyser_dt = 1.e-5
    # transition_test([1., 0., 500 + 273], [0., 1., 500 + 273], 1.e-2, f'{folder}/773K.png')

    # LibudaG, diff rates
    PC_obj = PC_setup.general_PC_setup('LibudaG', ('to_model_constructor', {'params': {}}))
    PC_obj.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
                                          'thetaA_init': 0., 'thetaB_init': 0.,
                                          'thetaA_max': 0.5, 'thetaB_max': 0.5, })
    folder = 'PC_plots/LibudaG/230715_transition_speed'

    PC_obj.process_to_control.set_params({f'rate_{suff}': 0.05 for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')})
    transition_test([1., 0.], [0., 1.], 200, f'{folder}/all_rates_005.png')

    PC_obj.process_to_control.set_params({f'rate_{suff}': 1. for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')})
    transition_test([1., 0.], [0., 1.], 10, f'{folder}/all_rates_1.png')

    pass


def temperature_dependence_analysis():
    PC_obj = PC_setup.general_PC_setup('LibudaGWithT')
    PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0., })
    optimal_440 = {'inputB': 1., 'inputA': 0.55}
    policies = [ConstantPolicy({'value': optimal_440['inputB']}),
                ConstantPolicy({'value': optimal_440['inputA']}),
                ConstantPolicy(dict())]
    episode_time, time_step = 30., 0.1

    # for T in range(400, 440, 5):
    #     policies[-1].update_policy({'value': T})
    #     PC_obj.reset()
    #     PC_obj.process_by_policy_objs(policies, episode_time, time_step)
    #     PC_obj.get_and_plot(f'./PC_plots/LibudaGWithT/230925_temperature_effect/T{T}K.png')
    #
    # for T in range(445, 500, 5):
    #     policies[-1].update_policy({'value': T})
    #     PC_obj.reset()
    #     PC_obj.process_by_policy_objs(policies, episode_time, time_step)
    #     PC_obj.get_and_plot(f'./PC_plots/LibudaGWithT/230925_temperature_effect/T{T}K.png')

    for T in range(500, 650, 10):
        policies[-1].update_policy({'value': T})
        PC_obj.reset()
        PC_obj.process_by_policy_objs(policies, episode_time, time_step)
        PC_obj.get_and_plot(f'./PC_plots/LibudaGWithT/230925_temperature_effect/T{T}K.png')


def SBP_constant_ratio_and_max_rate(PC: ProcessController, inputs_start, inputs_end, period_bounds, resolutions: Union[int, list] = 10,
                                    DEBUG=False, **kwargs):
    original_analyzer_dt = PC.analyser_dt

    if isinstance(resolutions, int):
        resolutions = [resolutions] * 2 + [9]
    resol_const, resol_T, resol_frac_T = resolutions

    debug_folder = './DEBUG/SBPvsCONSTANT'
    t0 = t1 = None
    if DEBUG:
        t0 = time.time()

    # constant
    const_episode_time = period_bounds[1] * 3
    constant_policies = [ConstantPolicy() for _ in PC.controlled_names]
    constant_mean_rets = []
    for p in np.linspace(0., 1., resol_const):
        inputs = (1. - p) * inputs_start + p * inputs_end
        for i, obj in enumerate(constant_policies):
            obj.update_policy({'value': inputs[i]})
        PC.analyser_dt = const_episode_time / 200
        PC.reset()
        PC.process_by_policy_objs(constant_policies, const_episode_time, const_episode_time / 10)
        if DEBUG:
            PC.get_and_plot(f'{debug_folder}/constant_{p:.2f}.png')
        constant_mean_rets.append(2 * PC.get_cumulative_target([const_episode_time / 2, const_episode_time]) / const_episode_time)

    # SBP
    SBP_policies = [TwoStepPolicy() for _ in PC.controlled_names]
    SBP_mean_rets = []
    for log_T in np.linspace(np.log(period_bounds[0]), np.log(period_bounds[1]), resol_T):
        T = np.exp(log_T)
        episode_time = 30 * T
        for p in np.linspace(0.05, 0.95, resol_frac_T):
            t1, t2 = (1. - p) * T, p * T
            for i, obj in enumerate(SBP_policies):
                obj.update_policy({'1': inputs_start[i], '2': inputs_end[i], 't1': t1, 't2': t2, })
            PC.analyser_dt = episode_time / 1000
            PC.reset()
            PC.const_preprocess((1. - p) * inputs_start + p * inputs_end,
                                const_episode_time, const_episode_time / 100)
            PC.process_by_policy_objs(SBP_policies, episode_time, T / 100)
            if DEBUG:
                PC.get_and_plot(f'{debug_folder}/SBP_{T:.2f}_{p:.2f}.png')
            SBP_mean_rets.append(2 * PC.get_cumulative_target([episode_time / 2, episode_time]) / episode_time)

    # best const
    best_const_p = np.linspace(0., 1., resol_const)[np.argmax(constant_mean_rets)]

    # best SBP
    T_idx, p_idx = divmod(np.argmax(SBP_mean_rets), resol_frac_T)
    best_SBP = [np.linspace(np.log(period_bounds[0]), np.log(period_bounds[1]), resol_T)[T_idx],
                np.linspace(0.1, 0.9, resol_frac_T)[p_idx]]
    T, p = np.exp(best_SBP[0]), best_SBP[1]
    t1, t2 = p * T, (1. - p) * T

    if kwargs.get('plot_both_best', False):
        # plot const
        inputs = (1 - best_const_p) * inputs_start + best_const_p * inputs_end
        for i, obj in enumerate(constant_policies):
            obj.update_policy({'value': inputs[i]})
        PC.reset()
        PC.process_by_policy_objs(constant_policies, const_episode_time, const_episode_time / 10)
        PC.get_and_plot(f'{kwargs["folder"]}/best_constant_{kwargs["ind_picture"]}.png')
        # plot SBP
        episode_time = 30 * T
        for i, obj in enumerate(SBP_policies):
            obj.update_policy({'1': inputs_start[i], '2': inputs_end[i], 't1': t1, 't2': t2, })
        PC.analyser_dt = episode_time / 1000
        PC.reset()
        PC.const_preprocess((1. - p) * inputs_start + p * inputs_end,
                            const_episode_time, const_episode_time / 100)
        PC.process_by_policy_objs(SBP_policies, episode_time, T / 100)
        PC.get_and_plot(f'{kwargs["folder"]}/best_SBP_{kwargs["ind_picture"]}.png')

    ratio = max(SBP_mean_rets) / max(constant_mean_rets)
    max_mean_ret = max(SBP_mean_rets + constant_mean_rets)

    PC.analyser_dt = original_analyzer_dt

    if ratio <= 1.:
        maximizing_params = {'type': 'const', 'p': best_const_p}
    else:
        maximizing_params = {'type': 'SBP', 't1': t1, 't2': t2}

    if DEBUG:
        t1 = time.time()
        with open(f'{debug_folder}/results.txt', 'w') as fwrite:
            fwrite.write(f'best ratio: {ratio:.5f}\n')
            fwrite.write(f'maximum achieved with: ' + '; '.join(f'{k}: {v}' for k, v in maximizing_params.items()) + '\n')
            fwrite.write(f'max return / episode ratio: {max_mean_ret:.5f}\n')
            fwrite.write(f'elapsed: {t1 - t0:.5f}')

    return ratio, max_mean_ret, maximizing_params


def get_to_optimize_SBP_const_ratio(PC_obj, inputs_min, inputs_max, period_bounds, resolutions):

    def f_to_optimize(rates_dict, **kwargs):
        PC_obj.process_to_control.set_params(rates_dict)
        return -1 * SBP_constant_ratio_and_max_rate(PC_obj, inputs_min, inputs_max, period_bounds,
                                                    resolutions=resolutions, DEBUG=False, **kwargs)[0]

    return f_to_optimize


def reproduce_Bassett():
    PC_obj = PC_setup.general_PC_setup('Bassett')

    PC_obj.reset()
    PC_obj.process_by_policy_objs([ConstantPolicy({'value': 0.4}), ConstantPolicy({'value': 0.1})], 50, 0.1)
    PC_obj.get_and_plot('./PC_plots/Bassett/reproduce_paper/reproduce_paper.png')


def low_desorp_react_try():
    PC_obj = PC_setup.general_PC_setup('LibudaG')
    PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0.,
                                          'rate_des_A': 0.1, 'rate_react': 0.1,
                                          })

    PC_obj.reset()
    PC_obj.process_by_policy_objs([
        TwoStepPolicy({'1': 1., '2': 0., 't1': 30, 't2': 30}),
        TwoStepPolicy({'1': 0., '2': 1., 't1': 30, 't2': 30}),
    ], episode_time=3000., policy_step=5.)
    PC_obj.get_and_plot('./PC_plots/LibudaG/low_des_react_guessing/try.png')


def integral_from_csv(datapath, feature_name, xlim=None):
    _, df = lib.read_plottof_csv(datapath, ret_df=True)
    X = df[f'{feature_name} x'].to_numpy()
    Y = df[f'{feature_name} y'].to_numpy()
    if xlim is None:
        xlim = [X.min(), X.max()]
    idx = (X >= xlim[0]) & (X <= xlim[1])
    return lib.integral(X[idx], Y[idx])


def steady_state_map_data(PC_obj: ProcessController, p1_lim, p2_lim, grid_resolution,
                          savepath):
    # TODO not very general
    model = PC_obj.process_to_control
    p1 = np.linspace(*p1_lim, grid_resolution)
    p2 = np.linspace(*p2_lim, grid_resolution)
    p1, p2 = np.meshgrid(p1, p2)
    steady_state = model.steady_state_sol(p1.ravel(), p2.ravel())
    steady_state = steady_state.reshape(grid_resolution, grid_resolution)
    np.save(savepath, steady_state)
    return steady_state


def covs_reverse_map_data(PC_obj: ProcessController, covsB_lim, covsA_lim, grid_resolution, savepath):
    model = PC_obj.process_to_control
    covsB = np.linspace(*covsB_lim, grid_resolution)
    covsA = np.linspace(*covsA_lim, grid_resolution)
    covsB, covsA = np.meshgrid(covsB, covsA)
    pB, pA = model.reverse_steady_state_problem(covsB.ravel(), covsA.ravel())
    pB = pB.reshape(grid_resolution, grid_resolution)
    pA = pA.reshape(grid_resolution, grid_resolution)
    filename, ext = os.path.splitext(savepath)
    np.save(f'{filename}_pB{ext}', pB)
    np.save(f'{filename}_pA{ext}', pA)
    return pB


def check_L2001_coincidence(PC_obj, check_folder='./check', plot=True, rate_idx=None, verbose=True,
                            original_iloc=None, plot_with_check_points=False):
    original_data_path = './check/L2001_graph_data_14.csv'
    get_co_part = GeneralizedLibudaModel.co_flow_part_to_pressure_part
    if rate_idx is not None:
        rate_idx = [rate_idx]
    gen_data = Libuda2001_CO_cutoff_policy(PC_obj,
                                        ['full'],
                                        check_folder,
                                        transform_x_co_on=lambda x: {
                                                                     # 'O2': (1. - get_co_part(x)) * 1.e-4, 'CO': get_co_part(x) * 1.e-4,
                                                                     'inputB': 1. - get_co_part(x), 'inputA': get_co_part(x),
                                                                     # 'T': 440.
                                                                     # 'T': 440.
                                                                     },
                                        transform_x_co_off=lambda x: {
                                                                      # 'O2': (1. - get_co_part(x)) * 1.e-4, 'CO': 0.,
                                                                      'inputB': 1. - get_co_part(x), 'inputA': 0.,
                                                                      # 'inputB': 1., 'inputA': 0.,
                                                                      # 'inputB': 1., 'inputA': 0.,
                                                                      # 'T': 440.
                                                                      # 'T': 440.
                                                                      },
                                        plot=plot,
                                        ret_idxs=rate_idx,
                                        )

    if gen_data is None:
        assert plot
        _, df = lib.read_plottof_csv(f'{check_folder}/full_series_all_data.csv', ret_df=True)
        sim_t = df['outputC x'].to_numpy()
        sim_rate = df['outputC y'].to_numpy()
        # sim_t = df['CO2 x'].to_numpy()
        # sim_rate = df['CO2 y'].to_numpy()
    else:
        sim_t, sim_rate = gen_data[0], np.squeeze(gen_data[1])

    original_data = pd.read_csv(original_data_path, index_col=None)
    if original_iloc is not None:
        original_data = original_data.iloc[original_iloc, :]
    mape_data = np.zeros((original_data.shape[0], 2))
    mape_data[:, 0] = original_data['rate']
    sim_t_points = np.zeros(original_data.shape[0])  # to scatter plot
    for i, t in enumerate(original_data['t']):
        idx = np.argmin(np.abs(sim_t - t))
        mape_data[i, 1] = sim_rate[idx]
        sim_t_points[i] = sim_t[idx]
    mape = np.mean(np.abs((mape_data[:, 0] - mape_data[:, 1]) / mape_data[:, 0])) * 100.

    if verbose:
        res_str = f'MAPE wrt Libuda data {mape}'
        with open(f'{check_folder}/check.txt', 'w') as fwrite:
            fwrite.write(res_str)
        print(res_str)

    if plot_with_check_points:
        fig, ax = plt.subplots(figsize=(16 / 3 * 2, 9 / 3 * 2))
        ax.plot(sim_t, sim_rate)
        ax.scatter(original_data['t'], mape_data[:, 0], marker='d', color='r')
        ax.scatter(sim_t_points, mape_data[:, 1], marker='d', color='y')
        fig.savefig(f'{check_folder}/with_check_points.png')
        plt.close(fig)

    return mape


def get_to_fit_L2001_by_LG():
    PC_obj = PC_setup.general_PC_setup('LibudaG')
    model = PC_obj.process_to_control

    def f_to_opt(rates, mode='silent'):
        model.set_params(rates)
        args = {'plot': False, 'rate_idx': 0, 'verbose': False}
        if mode != 'silent':
            args.update({'plot': True, 'verbose': True})
        mape = check_L2001_coincidence(PC_obj, check_folder='./check/fit', **args)
        return mape

    return f_to_opt


def sergeys_check(PC_obj: ProcessController, omega, mean_inputs):
    assert (omega <= 100.) and (omega > 1 / 10.)
    model = PC_obj.process_to_control
    folder = './PC_plots/LibudaG/sergeys_check'

    PC_obj.analyser_dt = 0.1 / 2 / omega
    PC_obj.reset()
    PC_obj.process_by_policy_objs(
        [TwoStepPolicy({'1': 1., '2': 0., 't1': 1. / (2 * omega), 't2': 1. / (2 * omega)}),
         TwoStepPolicy({'1': 0., '2': 1., 't1': 1. / (2 * omega), 't2': 1. / (2 * omega)})],
        episode_time=100., policy_step=0.001
    )
    PC_obj.get_and_plot(f'{folder}/sergeys_check.png')

    steady_state_sol = model.steady_state_sol(*mean_inputs)[0]

    ops, _ = lib.read_plottof_csv(f'{folder}/sergeys_check_all_data.csv', ret_ops=True)
    idx_tB = ops.index('thetaB')
    idx_tA = ops.index('thetaA')
    idx_rate = ops.index('outputC')
    ops_cov = ops[idx_tB-2:idx_tB+1] + ops[idx_tA-2:idx_tA+1] + [ops[idx_tB-2], np.full_like(ops[idx_tB-2], steady_state_sol[1]), 'steady_state_tB'] + [ops[idx_tA-2], np.full_like(ops[idx_tA-2], steady_state_sol[2]), 'steady_state_tA']
    ops_rate = ops[idx_rate-2:idx_rate+1] + [ops[idx_rate-2], np.full_like(ops[0], steady_state_sol[0]), 'steady_state']
    lib.plot_to_file(*ops_cov, fileName=f'{folder}/sergeys_check_cov.png', save_csv=False)
    lib.plot_to_file(*ops_rate, fileName=f'{folder}/sergeys_check_rate.png', save_csv=False)


def sergeys_check_check():
    from test_models.GeneralizedLidudaModel import get_dfdt_Libuda, solve_ivp
    
    folder = './PC_plots/LibudaG/sergeys_check'

    ks = [0.14895,
          0.07162,
          0.06594,
          0.,
          5.98734]
    flows = [0., 1.]
    theta_max = [0.25, 0.5]
    Cs = [0.3, 1.]
    thetas = [0.25, 0.]
    
    # N, delta_t = 1000, 0.01
    T, step, substeps_in_step = 100., 0.1, 10
    N, delta_t = int(T // step), step / substeps_in_step
    curves = np.zeros((N * substeps_in_step + 1, 2))
    curves[0] = thetas
    idx = 1
    
    for _ in range(N):
        flows = flows[::-1]
        for __ in range(substeps_in_step):
            thetas = solve_ivp(get_dfdt_Libuda(flows, ks, theta_max, Cs), [0., delta_t], y0=thetas,
                               t_eval=[0, delta_t], atol=1.e-6, rtol=1.e-4, first_step=delta_t / 3)
            thetas = thetas.y[:, -1]
            curves[idx] = thetas
            idx += 1

    fig, ax = plt.subplots()
    x_arr = np.arange(substeps_in_step * N + 1) * delta_t
    ax.plot(x_arr, np.vstack([curves[:, 0], curves[:, 1], ks[4] * curves[:, 0] * curves[:, 1]]).transpose(),
            label=['thetaB', 'thetaA', 'rate'])
    ax.legend()
    fig.savefig(f'{folder}/sergeys_check_check.png')


def steady_state_plot_anltc(PC_obj: ProcessController, start_point, end_point, num_points,
                            foldpath, check_idxs=None, check_ep_time=None, **kwargs):
    if not isinstance(start_point, np.ndarray):
        start_point = np.array(start_point)
    if not isinstance(end_point, np.ndarray):
        end_point = np.array(end_point)

    model = PC_obj.process_to_control
    t = np.linspace(0., 1., num_points).reshape(-1, 1)
    t = np.tile(t, (1, len(start_point)))
    vals = start_point + (end_point - start_point) * t

    results = model.steady_state_sol(vals[:, 0], vals[:, 1])

    check_arr = None
    if check_idxs is not None:
        policies = [ConstantPolicy() for _ in start_point]
        check_arr = np.zeros((len(check_idxs), 3))
        for i, ch_idx in enumerate(check_idxs):
            ch_vals = vals[ch_idx]
            for j, p in enumerate(policies):
                p.update_policy({'value': ch_vals[j]})
            PC_obj.reset()
            PC_obj.process_by_policy_objs(policies, check_ep_time, check_ep_time / 1000.)
            _, output = PC_obj.get_and_plot(f'{foldpath}/check_{ch_idx}.png')
            check_arr[i] = [output[-1, 0],
                            PC_obj.additional_graph['thetaB'][output.shape[0] - 1],
                            PC_obj.additional_graph['thetaA'][output.shape[0] - 1]
                            ]

    t = t[:, 0]
    to_plot = [t, results[:, 0], {'label': 'rate', 'twin': True, 'c': 'purple'},
               t, results[:, 1], 'thetaB',
               t, results[:, 2], 'thetaA']

    fig, ax = plt.subplots()
    _, right_ax, _ = lib.plot_in_axis(*to_plot, ax=ax,
                                      xlabel='pCO', ylabel='coverage', title='steady_state',
                                      ylim=kwargs.get('ylim', (0, None)),
                                      twin_params={'ylabel': 'rate', 'ylim': kwargs.get('twin_ylim', (0, None))})

    # if check_idxs is not None:
    #     right_ax.scatter(t[check_idxs], check_arr[:, 0], marker='d', color='red',
    #                      label='numerical rates')

    fig.savefig(f'{foldpath}/steady_state.png')
    plt.close(fig)


def main():
    # custom_experiment()

    # test_PC_with_Libuda()

    # check_func_to_optimize()

    # count_conversion_given_exp('run_RL_out/important_results/220928_T25_diff_lims/O2_40_CO_10/8_copy.csv',
    #                            LibudaModel(Ts=273+25))
    # count_conversion_given_exp('run_RL_out/important_results/220830_REFERENCE/4_copy.csv',
    #                            LibudaModel(Ts=273+25))

    # plot_conv('run_RL_out/conversion/220928_8_conv.csv')
    # plot_conv('run_RL_out/conversion/220830_4_conv.csv')

    # Pt_exp('PC_plots/PtSalomons/1_.png')

    # test_new_k1k3_model_new_targets()

    # Libuda2001_original_simulation()

    MCKMC_simple_tests()

    # Ziff_model_poisoning_speed_test()
    # ZGB_snapshots()

    # PC_obj = PC_setup.general_PC_setup('ZGBTwo', ('to_model_constructor', {'rate_ads_O': 3., 'rate_ads_CO': 1.}))
    # run_constant_policies_bunch(PC_obj, 60_000, 2_000, 'PC_plots/ZGB/ZGBTwo_rates_work_test/O2_3_CO_1')

    # temperature_dependence_analysis()

    # reproduce_Bassett()

    # low_desorp_react_try()

    # # sudden discovery, rates measurement
    # folder = './231002_sudden_discovery'
    # int_period = 120.
    # NM_int = integral_from_csv(f'{folder}/nelder_mead_sol.csv', 'outputC', xlim=[240.-int_period, 240.])
    # NM_rate = NM_int / int_period
    # RL_int = integral_from_csv(f'{folder}/rl_agent_sol.csv', 'outputC', xlim=[240.-int_period, 240.])
    # RL_rate = RL_int / int_period
    # with open(f'{folder}/rates.txt', 'w') as fwrite:
    #     fwrite.write(f'Max stationary rate: {NM_rate}\n')
    #     fwrite.write(f'RL achieved rate: {RL_rate}\n')

    # PC_obj = PC_setup.general_PC_setup('LibudaG')
    # PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'rate_des_A': 0.1, 'rate_react': 0.1,
    #                                       })
    # steady_state_map_data(PC_setup.general_PC_setup('LibudaG'),
    #                       [0., 1.], [0., 1.], 100,
    #                       savepath='./PC_plots/LibudaG/analytic_steady_state.npy')
    # covs_reverse_map_data(PC_obj,
    #                       [0., 0.25], [0., 0.5], 100,
    #                       savepath='./PC_plots/LibudaG/press_from_covs.npy')

    # run_constant_policies_bunch(PC_setup.default_PC_setup('LibudaG'), 500, 1,
    #                             out_foldpath='PC_plots/LibudaGeneralized/DEBUG/',
    #                             plot_params={'time_segment': [0, None], 'additional_plot': ['thetaB', 'thetaA'],
    #                                          'plot_mode': 'separately', 'out_names': ['outputC']})

    # PC_obj = PC_setup.general_PC_setup('Libuda2001', ('to_PC_constructor', {'analyser_dt': 0.1}))
    # PC_obj = PC_setup.general_PC_setup('LibudaG', ('to_model_constructor', {'params': {}}))
    # rate_react = 0.1  # (0.001, 0.01, 0.1, 1.)
    # PC_obj.process_to_control.set_params({'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'rate_ads_A': 0.1, 'rate_ads_B': 0.1,
    #                                       'rate_des_A': 0.1, 'rate_react': rate_react,  # 0.1
    #                                       })  # low des, react rates
    # PC_obj.process_to_control.set_params({'C_B_inhibit_A': 1.,
    #                                       'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'thetaA_max': 0.5, 'thetaB_max': 0.5, })
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithT', ('to_model_constructor', {'T': 440., 'params': {}}))
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithTEs', ('to_model_constructor', {'T': 440., 'params': {}}))
    # get_co_part = PC_obj.process_to_control.co_flow_part_to_pressure_part
    # get_co_part = GeneralizedLibudaModel.co_flow_part_to_pressure_part
    # Libuda2001_CO_cutoff_policy(PC_obj,
    #                             ['full'],
    #                             'PC_plots/LibudaG/DEBUG/Libuda_regime',
    #                             transform_x_co_on=lambda x: {
    #                                                          # 'O2': (1. - get_co_part(x)) * 1.e-4, 'CO': get_co_part(x) * 1.e-4,
    #                                                          'inputB': 1. - get_co_part(x), 'inputA': get_co_part(x),
    #                                                          # 'T': 440.
    #                                                          # 'T': 440.
    #                                                          },
    #                             transform_x_co_off=lambda x: {
    #                                                           # 'O2': (1. - get_co_part(x)) * 1.e-4, 'CO': 0.,
    #                                                           'inputB': 1. - get_co_part(x), 'inputA': 0.,
    #                                                           # 'inputB': 1., 'inputA': 0.,
    #                                                           # 'T': 440.
    #                                                           # 'T': 440.
    #                                                           },
    #                             output_name_to_plot='outputC',
    #                             # output_name_to_plot='CO2',
    #                             add_names=('thetaB', 'thetaA', 'error'),
    #                             # add_names=('thetaO', 'thetaCO')
    #                             )
    # check_L2001_coincidence(PC_obj,
    #                         original_iloc=[i for i in range(14)],
    #                         plot_with_check_points=True)
    # sergeys_check(PC_obj, 10., (0.5, 0.5))
    # sergeys_check_check()

    # steady_state_plot_anltc(PC_obj, [1., 0.], [0., 1.], 101,
    #                         foldpath=f'./PC_plots/LibudaG/diff_react_rate/231030_r_react({rate_react:.3g})_steady_state',
    #                         check_idxs=[i for i in range(0, 100, 15)], check_ep_time=2000., twin_ylim=(0., 0.15 * rate_react),
    #                         )

    # transition_speed_test()

    # # SBP_constant_ratio
    # PC_obj = PC_setup.general_PC_setup('LibudaG')
    # PC_obj.process_to_control.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
    #                                       'thetaA_init': 0., 'thetaB_init': 0.,
    #                                       'thetaA_max': 0.5, 'thetaB_max': 0.5,
    #                                       # 'rate_ads_A': 1., 'rate_react': 1.,
    #                                       # 'rate_des_A': 0.05, 'rate_ads_B': 0.05, 'rate_des_B': 0.05,
    #                                       })
    # # PC_obj.process_to_control.set_params({f'rate_{suff}': 1. for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')})
    # SBP_constant_ratio_and_max_rate(PC_obj, np.array([1., 0.]), np.array([0., 1.]), np.array([2., 200.]), resolutions=[4, 4, 4], DEBUG=True)

    # estimate max rate
    # PC_obj = PC_setup.general_PC_setup('LibudaG')
    # # model = PC_obj.process_to_control
    # # model.set_params({'C_A_inhibit_B': 1., 'C_B_inhibit_A': 1.,
    # #                   'thetaA_init': 0., 'thetaB_init': 0.,
    # #                   'thetaA_max': 0.5, 'thetaB_max': 0.5,
    # #                   'rate_ads_A': 1., 'rate_react': 1.,
    # #                   'rate_des_A': 0.05, 'rate_ads_B': 0.05, 'rate_des_B': 0.05,
    # #                   })
    # # model.set_params({f'rate_{suff}': 10. * model[f'rate_{suff}'] for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react')})
    # max_rate = estimate_max_rate_libuda_like(PC_obj, plotpath='./PC_plots/LibudaG/DEBUG/max_rate_estim.png')
    # print(max_rate)

    # ZGB Lopez Albano
    # size = [256, 256]
    # PC_obj = PC_setup.general_PC_setup('Ziff',
    #                                    ('to_model_constructor', {'m': size[0], 'n': size[1], 'CO2_count_top': 1.e+5}),
    #                                    ('to_PC_constructor', 'supposed_step_count', 100000),
    #                                    ('to_PC_constructor', 'supposed_exp_time', 2.e+7),
    #                                    ('to_PC_constructor', 'analyser_dt', 1.e+3),
    #                                    )
    # time_unit = size[0] * size[1]
    # T = 20 * time_unit
    # # x_0 = (0.52 + 0.39) / 2
    # x_0 = 0.455
    # PC_obj.const_preprocess({'x': x_0}, 5 * time_unit)
    # PC_obj.process_by_policy_objs((SinPolicy({'A': 0.11, 'T': T, 'alpha': 0., 'bias': x_0}), ), 10 * T, 5.e+2)
    # PC_obj.get_and_plot(f'PC_plots/Ziff_Lopez_article/try3.png',
    #                     plot_params={'time_segment': [0, None], 'additional_plot': ['thetaO', 'thetaCO'],
    #                                  'plot_mode': 'separately', 'out_names': ['CO2_count']})

    # PC_obj = PC_setup.general_PC_setup('Libuda2001')
    # PC_obj.process_by_policy_objs((ConstantPolicy({'value': 10.e-5}), ConstantPolicy({'value': 3.88e-5})), 500, 1)
    # PC_obj.get_and_plot('PC_plots/L2001_optimal_const.png', plot_params={'out_names': ('CO2',)})

    # omega = 0.7
    # A = 0.4
    # T = 2 * np.pi / omega
    # PC_obj = PC_setup.general_PC_setup('Lynch')
    # PC_obj.process_by_policy_objs((
    #     SinPolicy({'A': A, 'T': T, 'alpha': np.pi, 'bias': 0.5}),
    #     SinPolicy({'A': A, 'T': T, 'alpha': 0., 'bias': 0.5}),
    # ), 100, 0.1)
    # PC_obj.get_and_plot(f'PC_plots/LynchReproduce/omega07.png',
    #                     plot_params={'time_segment': [0, None], 'additional_plot': ['thetaO', 'thetaCO'],
    #                                  'plot_mode': 'separately', 'out_names': ['CO2']})

    pass


if __name__ == '__main__':
    main()

