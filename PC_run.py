import os

import numpy as np
import pandas as pd

from ProcessController import *
from predefined_policies import *
from test_models import *
from targets_metrics import *

from usable_functions import make_subdir_return_path
from lib import plot_to_file

import PC_setup


def check_func_to_optimize():

    PC_L2001 = ProcessController(LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160),
                                 target_func_to_maximize=CO2_value,
                                 # supposed_step_count=2 * round(episode_time / time_step),  # memory controlling parameters
                                 # supposed_exp_time=2 * episode_time
                                 )
    PC_L2001.set_plot_params(output_lims=[0., None], output_ax_name='CO2_formation_rate',
                               input_ax_name='Pressure, Pa')

    # PC_LDegrad = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160,
    #                                                           v_d=0.01, v_r=0.1, border=4.),
    #                                target_func_to_maximize=CO2_value)
    # PC_LDegrad.set_plot_params(output_lims=[0., 0.06], output_ax_name='CO2_formation_rate',
    #                            input_ax_name='Pressure, Pa')

    # f = func_to_optimize_sin_sol(PC_LDegrad, 500, 1.)
    # f({'O2_A': 0., 'O2_k': 0.1 * np.pi, 'O2_bias_t': 0., 'O2_bias_f': 10.e-5,
    #    'CO_A': 2e-5, 'CO_k': 0.1 * np.pi, 'CO_bias_t': 0., 'CO_bias_f': 3.e-5},
    #   DEBUG=True, folder='PC_plots/test_sin_sol')

    # f = func_to_optimize_policy(PC_LDegrad, TwoStepPolicy(dict()), 500, 0.5)
    # f({'O2_1': 2.e-5, 'O2_2': 8.e-5, 'CO_1': 4.e-5, 'CO_2': 6.e-5,
    #    'O2_t1': 20., 'O2_t2': 40., 'CO_t1': 10., 'CO_t2': 20., },
    #   DEBUG=True, folder='PC_plots/test_func_for_any_policy/two_step_policy')

    # f = func_to_optimize_policy(PC_LDegrad, SinPolicy(dict()), 500, 1.)
    # f({'O2_A': 2.e-5, 'O2_omega': np.pi * 0.033, 'O2_alpha': np.pi, 'O2_bias': 7.e-5,
    #    'CO_A': 7.e-5, 'CO_omega': np.pi * 0.15, 'CO_alpha': np.pi / 4, 'CO_bias': 3.e-5},
    #   DEBUG=True, folder='PC_plots/test_func_for_any_policy/sin_policy')

    # f = func_to_optimize_policy(PC_LDegrad, SinOfPowerPolicy(dict()), 500, 0.5)
    # f({'O2_power': 4., 'O2_A': 4.e-5, 'O2_omega': np.pi * 0.003, 'O2_alpha': np.pi, 'O2_bias': 7.e-5,
    #    'CO_power': 1.4, 'CO_A': 4.e-5, 'CO_omega': np.pi * 0.01, 'CO_alpha': np.pi / 4, 'CO_bias': 3.e-5},
    #   DEBUG=True, folder='PC_plots/test_func_for_any_policy/sin_power_policy')

    # f = func_to_optimize_policy(PC_L2001, ConstantPolicy(dict()), 500, 1.)
    # f({'O2_value': 9.5e-5,
    #    'CO_value': 4.4e-5},
    #   DEBUG=True, folder='PC_plots/test_func_for_any_policy/constant_policy')


# def try_policy(PC_obj: ProcessController, time_seq: np.ndarray, policy_funcs, path: str) -> float:
#     def create_f_consider_bounds(func, ind_in_model):
#
#         def f(t):
#             res = func(t)
#             lower_bound = PC_obj.process_to_control.limits['input'][ind_in_model][0]
#             upper_bound = PC_obj.process_to_control.limits['input'][ind_in_model][1]
#             res[res < lower_bound] = lower_bound
#             res[res > upper_bound] = upper_bound
#             return res
#
#         return f
#
#     new_policy_funcs = [create_f_consider_bounds(f, i) for i, f in enumerate(policy_funcs)]
#
#     controlled_to_pass = np.array([func(time_seq) for func in new_policy_funcs])
#     controlled_to_pass = controlled_to_pass.transpose()
#
#     PC_obj.reset()
#     for i, dt in enumerate(time_seq):
#         PC_obj.set_controlled(controlled_to_pass[i])
#         PC_obj.time_forward(dt)
#     R = PC_obj.integrate_along_history(target_mode=True, time_segment=[0., np.sum(time_seq)])
#
#     def ax_func(ax):
#         ax.set_title(f'integral: {R:.4g}')
#
#     PC_obj.plot(path,
#                 plot_more_function=ax_func, plot_mode='separately',
#                 time_segment=[0., np.sum(time_seq)])
#
#     return R


def run_constant_policies_bunch(PC: ProcessController,
                                episode_time, resolution,
                                out_foldpath: str,
                                input_limits=None, variants=None,
                                plot_params: dict = None):
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
    assert plot_params is not None
    for v in variants:
        PC.reset()
        for i, p in enumerate(policies):
            p.set_policy({'value': v[i] * input_limits[i][1]})
        PC.process_by_policy_objs(policies, episode_time, resolution)
        run_id = '_'.join(map(lambda pair: f'{pair[0]}:{pair[1]:.2f}', zip(PC.controlled_names, v)))
        PC.get_and_plot(f'{out_foldpath}/{run_id}.png', plot_params)


def test_new_k1k3_model_new_targets():
    episode_time = 500.
    resolution = 1.

    PC_obj = ProcessController(LibudaModelReturnK3K1AndPressures(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=440),
                               supposed_step_count=2 * round(episode_time / resolution),  # memory controlling parameters
                               supposed_exp_time=2 * episode_time,
                               # target_func_to_maximize=get_target_func('CO2_sub_outs', alpha=0.1),
                               long_term_target_to_maximize=get_target_func('CO2_sub_outs_I', alpha=0.1),
                               )
    PC_obj.set_plot_params(output_lims=None, output_ax_name='CO2_formation_rate',
                           input_lims=[0., 10.e-5], input_ax_name='Pressure, Pa')
    policies = (SinPolicy({'A': 2.e-5, 'omega': 0.05 * np.pi, 'alpha': 0., 'bias': 10.e-5, }),
                TwoStepPolicy({'1': 2.e-5, '2': 5.e-5, 't1': 30, 't2': 20, }))
    # policies = (ConstantPolicy({'value': 9.e-5}),
    #             ConstantPolicy({'value': 4.4e-5}))
    PC_obj.process_by_policy_objs(policies, episode_time, resolution)
    # R = PC_obj.integrate_along_history(target_mode=True,
    #                                    time_segment=[0., episode_time])
    R = PC_obj.get_long_term_target()
    print(R)

    # save to csv
    names = ['CO2', 'O2(k3)', 'CO(k1)']
    output = PC_obj.output_history
    df = pd.DataFrame(columns=names)
    for i, name in enumerate(names):
        df[name] = output[:, i]
    df.to_csv(f'PC_plots/test_k3k1_model/CO2_O2_CO.csv', index=False)

    PC_obj.set_metrics(('CO2', CO2_integral),
                       ('O2 conversion', overall_O2_conversion),
                       ('CO conversion', overall_CO_conversion))
    PC_obj.plot(f'PC_plots/test_k3k1_model/return_{R:.3f}.png',
                plot_mode='separately',
                time_segment=[0., episode_time],
                additional_plot=['thetaCO', 'thetaO'],
                out_names=['CO2', 'long_term_target'],
                with_metrics=True)


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

    T = 273 + 160
    PC_L2001 = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=T),
                                 target_func_to_maximize=CO2_value)
    PC_L2001.set_plot_params(output_lims=[0., 0.06], output_ax_name='CO2_formation_rate',
                             input_ax_name='Pressure, Pa')
    # PC_LDegrad = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160,
    #                                                           v_d=0.01, v_r=0.1, border=4.),
    #                                target_func_to_maximize=CO2_value)
    # PC_LDegrad.set_plot_params(output_lims=[0., 0.06], output_ax_name='CO2_formation_rate',
    #                            input_ax_name='Pressure, Pa')

    time_seq = np.linspace(0., 500., 51)
    time_seq = time_seq[1:] - time_seq[:-1]
    try_policy(PC_L2001, time_seq,
               [SinOfPowerPolicy(A=5e-5, power=1, omega=1, alpha=1e-3, bias=5e-5), ConstantPolicy(value=5e-5)],
               './PC_plots/low_temperature/SinOfPow_0.png')

    # episode_len = 500
    # for pair in ((2e-5, 10e-5), (3e-5, 10e-5), (4e-5, 10e-5), ):
    #     PC_LDegrad.reset()
    #     PC_LDegrad.set_controlled({'CO': pair[0], 'O2': pair[1], })
    #     PC_LDegrad.time_forward(episode_len)
    #     R = PC_LDegrad.integrate_along_history(target_mode=True)
    #     PC_LDegrad.plot(file_name=f'PC_plots/LDegrad/O2_{int(pair[1] * 1e+5)}_CO_{int(pair[0] * 1e+5)}_R_{R:.2f}.png',
    #                     plot_mode='separately')

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


# def plot_conv(csv_path):
#     df = pd.read_csv(csv_path)
#
#     plot_file_path, _ = os.path.splitext(csv_path)
#     plot_file_path = f'{plot_file_path}.png'
#     lib.plot_to_file(df['Time'], df['CO_conversion'],  'CO conversion',
#                      df['Time'], df['O2_conversion'], 'O2 conversion',
#                      title='Conversion',
#                      ylim=None, save_csv=False, fileName=plot_file_path)


def Libuda2001_CO_cutoff_policy(PC_obj: ProcessController, program, dest_dir='./PC_plots/Libuda_orginal',
                                p_total=1e-4,
                                transform_x_co_on=None,
                                transform_x_co_off=None,
                                output_name_to_plot='CO2',
                                add_names=('thetaO', 'thetaCO')):

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
        PC_obj.time_forward(150)
        PC_obj.set_controlled(transform_x_co_off(x_co))
        PC_obj.time_forward(50)

    def plot_one_co_on(x_co):
        PC_obj.reset()
        _one_co_on(x_co)
        PC_obj.get_and_plot(f'{dest_dir}/x_co_{x_co:.2f}.png',
                            plot_params={'time_segment': [0, 200], 'additional_plot': list(add_names),
                                         'plot_mode': 'separately', 'out_names': [output_name_to_plot]})

    def run_full_cutoff_series():
        PC_obj.reset()
        x_co = 0.05
        while x_co < 0.96:
            _one_co_on(x_co)
            x_co += 0.05
        PC_obj.get_and_plot(f'{dest_dir}/full_series.png',
                            plot_params={'time_segment': [0, None], 'additional_plot': list(add_names),
                                         'plot_mode': 'separately', 'out_names': [output_name_to_plot]})

    for arg in program:
        if isinstance(arg, tuple):
            PC_obj.reset()
            for v in arg:
                _one_co_on(v)
            PC_obj.get_and_plot(f'{dest_dir}/series.png',
                                plot_params={'time_segment': [0, None], 'additional_plot': list(add_names),
                                         'plot_mode': 'separately', 'out_names': [output_name_to_plot]})
        elif isinstance(arg, float):
            plot_one_co_on(arg)
        elif isinstance(arg, str) and (arg == 'full'):
            run_full_cutoff_series()

    # plot_one_co_on(0.75)
    # plot_one_co_on(0.5)
    # plot_one_co_on(0.25)
    # run_full_cutoff_series()


def Libuda2001_original_simulation():
    PC_obj = ProcessController(LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0.25}, Ts=440.),
                               target_func_to_maximize=get_target_func('CO2_value'))
    # PC_obj.set_plot_params(output_lims=[-1.e-3, 0.05], output_ax_name='CO2 formation rate')
    Libuda2001_CO_cutoff_policy(PC_obj, ['full'])


def Pt_exp(path: str):

    # PC_Pt2210 = ProcessController(PtModel(init_cond={'thetaO': 0., 'thetaCO': 0.}),
    #                               target_func_to_maximize=get_target_func('CO2_value'))
    PC_Salomons = ProcessController(PtSalomons(init_cond={'thetaO': 0., 'thetaCO': 0., 'thetaOO': 0., }),
                                  target_func_to_maximize=get_target_func('CO2_value'))

    PC_obj = PC_Salomons
    PC_obj.set_plot_params(input_ax_name='Pressure', input_lims=None,
                           output_ax_name='CO2 form. rate', output_lims=None)

    # first experiment
    low_value = 10  # 4
    high_value = 50  # 10
    reps = 10
    for i in range(reps):
        PC_obj.set_controlled({'O2': low_value, 'CO': high_value})
        PC_obj.time_forward(20)
        PC_obj.set_controlled({'O2': high_value, 'CO': low_value})
        PC_obj.time_forward(30)
    R = PC_obj.integrate_along_history(target_mode=True, time_segment=[0., 50 * reps])

    def ax_func(ax):
        ax.set_title(f'integral: {R:.4g}')

    PC_obj.plot(path,
                plot_more_function=ax_func, plot_mode='separately',
                time_segment=[0., 50 * reps], additional_plot=['thetaCO', 'thetaO'])


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


def KMC_simple_tests():
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
    size = [20, 20]
    PC_obj = ProcessController(KMC_CO_O2_Pt_Model((*size, 1), log_on=True,
                                                  O2_top=1.1e5, CO_top=1.1e5,
                                                  CO2_rate_top=1.4e6, CO2_count_top=1.e4,
                                                  T=373.),
                               analyser_dt=1.e-6,
                               target_func_to_maximize=get_target_func('CO2_count'),
                               RESOLUTION=1,  # always should be 1 if we use KMC, otherwise we will get wrong results!
                               supposed_step_count=100,  # memory controlling parameters
                               supposed_exp_time=1.e-3)

    PC_obj.set_plot_params(input_lims=[-1e-5, None], input_ax_name='Pressure, Pa',
                           output_lims=[-1e-2, None],
                           additional_lims=[-1e-2, 1. + 1.e-2],
                           # output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$',
                           output_ax_name='CO x O events count')
    PC_obj.set_metrics(
                       # ('CO2', CO2_integral),
                       ('CO2 count', CO2_count),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
                       )

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


def Ziff_model_poisoning_speed_test():
    size = [80, 25]
    PC_Ziff = ProcessController(ZGBModel(*size,
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
                                supposed_step_count=200,  # memory controlling parameters
                                supposed_exp_time=2e+6)
    PC_obj = PC_Ziff
    PC_obj.set_metrics(
                       # ('integral CO2', CO2_integral),
                       # ('CO2 count', CO2_count),
                       ('CO2 count', lambda time_arr, arr: np.sum(arr[:, 0])),
                       # ('O2 conversion', overall_O2_conversion),
                       # ('CO conversion', overall_CO_conversion)
    )

    PC_obj.set_plot_params(input_lims=[-1e-5, None], input_ax_name='Pressure, Pa',
                           output_lims=[-1e-2, None],
                           additional_lims=[-1e-2, 1. + 1.e-2],
                           # output_ax_name='CO2 formation rate, $(Pt atom * sec)^{-1}$',
                           output_ax_name='CO x O events count')

    time_step = 2e+3

    PC_obj.reset()
    PC_obj.set_controlled({'x': 1.})
    CO_cov = 0.
    while CO_cov < 0.96:
        PC_obj.time_forward(time_step)
        time_history, _ = PC_obj.get_process_output()
        CO_cov = PC_obj.additional_graph['thetaCO'][time_history.size - 1]
    PC_obj.plot('PC_plots/Ziff_poisoning_speed_test/Ziff_CO_poisoning_speed.png',
                **{'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                   'plot_mode': 'separately', 'out_names': ['CO2_count']})

    PC_obj.reset()
    PC_obj.set_controlled({'x': 0.})
    O_cov = 0.
    while O_cov < 0.9:
        PC_obj.time_forward(time_step)
        time_history, _ = PC_obj.get_process_output()
        O_cov = PC_obj.additional_graph['thetaO'][time_history.size - 1]
    PC_obj.plot('PC_plots/Ziff_poisoning_speed_test/Ziff_O_poisoning_speed.png',
                **{'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                   'plot_mode': 'separately', 'out_names': ['CO2_count']})


def LibudaGWithT_transtion_speed_test():
    PC_obj = PC_setup.general_PC_setup('LibudaGWithT')

    folder = 'PC_plots/LibudaGWithT/230516_transition_speed'

    PC_obj.reset()
    PC_obj.set_controlled([1., 0., 100 + 273])
    PC_obj.time_forward(1000)
    PC_obj.set_controlled([0., 1., 100 + 273])
    PC_obj.time_forward(1000)
    PC_obj.set_controlled([1., 0., 100 + 273])
    PC_obj.time_forward(1000)
    PC_obj.get_and_plot(f'{folder}/373K.png',
                        plot_params={'time_segment': [0, None], 'additional_plot': ('thetaB', 'thetaA'),
                                     'plot_mode': 'separately', 'input_names': ('T', ), 'out_names': ('outputC',), })

    PC_obj.analyser_dt = 1.e-3
    PC_obj.reset()
    PC_obj.set_controlled([1., 0., 300 + 273])
    PC_obj.time_forward(0.1)
    PC_obj.set_controlled([0., 1., 300 + 273])
    PC_obj.time_forward(0.1)
    PC_obj.set_controlled([1., 0., 300 + 273])
    PC_obj.time_forward(0.1)
    PC_obj.get_and_plot(f'{folder}/573K.png',
                        plot_params={'time_segment': [0, None], 'additional_plot': ('thetaB', 'thetaA'),
                                     'plot_mode': 'separately', 'input_names': ('T', ), 'out_names': ('outputC',), })

    PC_obj.analyser_dt = 1.e-5
    PC_obj.reset()
    PC_obj.set_controlled([1., 0., 500 + 273])
    PC_obj.time_forward(1.e-2)
    PC_obj.set_controlled([0., 1., 500 + 273])
    PC_obj.time_forward(1.e-2)
    PC_obj.set_controlled([1., 0., 500 + 273])
    PC_obj.time_forward(1.e-2)
    PC_obj.get_and_plot(f'{folder}/773K.png',
                        plot_params={'time_segment': [0, None], 'additional_plot': ('thetaB', 'thetaA'),
                                     'plot_mode': 'separately', 'input_names': ('T', ), 'out_names': ('outputC',), })


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

    # KMC_simple_tests()

    # Ziff_model_poisoning_speed_test()

    # run_constant_policies_bunch(PC_setup.default_PC_setup('LibudaG'), 500, 1,
    #                             out_foldpath='PC_plots/LibudaGeneralized/DEBUG/',
    #                             plot_params={'time_segment': [0, None], 'additional_plot': ['thetaB', 'thetaA'],
    #                                          'plot_mode': 'separately', 'out_names': ['outputC']})

    PC_obj = PC_setup.general_PC_setup('Libuda2001')
    # PC_obj = PC_setup.general_PC_setup('LibudaG', ('to_model_constructor', {'params': {}}))
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithT', ('to_model_constructor', {'T': 440., 'params': {}}))
    # PC_obj = PC_setup.general_PC_setup('LibudaGWithTEs', ('to_model_constructor', {'T': 440., 'params': {}}))
    # get_co_part = PC_obj.process_to_control.co_flow_part_to_pressure_part
    get_co_part = GeneralizedLibudaModel.co_flow_part_to_pressure_part
    Libuda2001_CO_cutoff_policy(PC_obj,
                                ['full'],
                                'PC_plots/Libuda/DEBUG/Libuda_regime',
                                transform_x_co_on=lambda x: {'O2': 1. - get_co_part(x), 'CO': get_co_part(x),
                                                             # 'inputB': 1. - get_co_part(x), 'inputA': get_co_part(x),
                                                             # 'T': 440.
                                                             # 'T': 440.
                                                             },
                                transform_x_co_off=lambda x: {'O2': 1. - get_co_part(x), 'CO': 0.,
                                                              # 'inputB': 1. - get_co_part(x), 'inputA': 0.,
                                                              # 'T': 440.
                                                              # 'T': 440.
                                                              },
                                # output_name_to_plot='outputC',
                                output_name_to_plot='CO2',
                                # add_names=('thetaB', 'thetaA', 'error'),
                                add_names=('thetaO', 'thetaCO'))

    # LibudaGWithT_transtion_speed_test()

    # LGWithT transition speed
    # PC_obj.reset()
    # PC_obj.set_controlled([1., 1., 400.])
    # PC_obj.time_forward(100)
    # # PC_obj.set_controlled([1., 1., 400.])
    # # PC_obj.time_forward(100)
    # PC_obj.get_and_plot(f'PC_plots/LibudaGWithT/230512_transition/400K_long_period.png',
    #                     plot_params={'time_segment': [0, None], 'additional_plot': ['thetaB', 'thetaA'],
    #                                  'plot_mode': 'separately', 'input_names': ['T'], 'out_names': ['outputC']})

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

