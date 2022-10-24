import os

from ProcessController import *
from predefined_policies import *
from test_models import *


def check_func_to_optimize():

    def target(x):
        return x[0]

    PC_L2001 = ProcessController(LibudaModel(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160),
                                 target_func_to_maximize=target,
                                 # supposed_step_count=2 * round(episode_time / time_step),  # memory controlling parameters
                                 # supposed_exp_time=2 * episode_time
                                 )
    PC_L2001.set_plot_params(output_lims=[0., None], output_ax_name='CO2_formation_rate',
                               input_ax_name='Pressure, Pa')

    # PC_LDegrad = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160,
    #                                                           v_d=0.01, v_r=0.1, border=4.),
    #                                target_func_to_maximize=target)
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


def bunch_of_policies(PC: ProcessController):
    lims = PC.process_to_control.limits['input']
    if lims.shape[0] == 2:
        # constant policy
        first_policy = ConstantPolicy(dict())
        second_policy = ConstantPolicy(dict())
        coefs = np.array([[0.2, 1.], [0.4, 0.7], [0.5, 0.5], [0.7, 0.4], [1., 0.2], ])
        for p in coefs:
            first_policy.set_policy({'value': p[0] * lims[0][1]})
            second_policy.set_policy({'value': p[1] * lims[1][1]})
    raise NotImplementedError


def test_new_k1k3_model_new_targets():
    episode_time = 500.
    resolution = 1.

    def target_1(x):
        # formula, roughly: cost = CO2 * (1 - O2_out/O2_in - CO_out/CO_in)
        # formula was rewriting, considering O2_out = O2_in - 2 * CO2; CO_out = CO_in - CO2;
        # and using protection from division by zero
        return x[0] * (2 * x[0] / (x[1] + 1e-7) + x[0] / (x[2] + 1e-7))

    PC_obj = ProcessController(LibudaModelReturnK3K1(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=440),
                               supposed_step_count=2 * round(episode_time / resolution),  # memory controlling parameters
                               supposed_exp_time=2 * episode_time,
                               target_func_to_maximize=target_1
                               )
    PC_obj.set_plot_params(output_lims=None, output_ax_name='CO2_formation_rate',
                           input_lims=[0., 10.e-5], input_ax_name='Pressure, Pa')
    policies = (SinPolicy({'A': 2.e-5, 'omega': 0.05 * np.pi, 'alpha': 0., 'bias': 10.e-5, }),
                TwoStepPolicy({'1': 2.e-5, '2': 5.e-5, 't1': 30, 't2': 20, }))
    # policies = (ConstantPolicy({'value': 9.e-5}),
    #             ConstantPolicy({'value': 4.4e-5}))
    PC_obj.process_by_policy_objs(policies, episode_time, resolution)
    R = PC_obj.integrate_along_history(target_mode=True,
                                       time_segment=[0., episode_time])
    PC_obj.plot(f'PC_plots/new_target_1/return_{R:.3f}.png',
                plot_mode='separately',
                time_segment=[0., episode_time],
                additional_plot=['theta_CO', 'theta_O'],
                out_name='CO2')


def custom_experiment():
    # # 1st try
    # PC = ProcessController(TestModel())
    # for i in range(5):
    #     PC.set_controlled([i + 0.2 * i1 for i1 in range(1, 6)])
    #     PC.time_forward(30)
    # PC.get_process_output()
    # for c in '12345678':
    #     PC.plot(f'PC_plots/example{c}.png', out_name=c, plot_mode='separately')

    # def target(x):
    #     # target_v = np.array([2., 1., 3., -1., 0.,
    #     #                      1., -1., 3., -2., 3.])
    #     target_v = np.array([2., 1., 3.])
    #     return -np.linalg.norm(x - target_v)
    #
    # PC = ProcessController(TestModel(), target_func_to_maximize=target)
    # for i in range(5):
    #     PC.set_controlled([i + 0.2 * i1 for i1 in range(1, 6)])
    #     PC.time_forward(30)
    # print(PC.integrate_along_history(target_mode=True))
    # for c in [f'out{i}' for i in range(1, 4)]:
    #     PC.plot(f'PC_plots/example_{c}.png', out_name=c, plot_mode='separately')
    # PC.plot(f'PC_plots/example_target.png', out_name='target', plot_mode='separately')

    def target(x):
        return x[0]

    T = 273 + 160
    PC_L2001 = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=T),
                                 target_func_to_maximize=target)
    PC_L2001.set_plot_params(output_lims=[0., 0.06], output_ax_name='CO2_formation_rate',
                             input_ax_name='Pressure, Pa')
    # PC_LDegrad = ProcessController(LibudaModelWithDegradation(init_cond={'thetaCO': 0., 'thetaO': 0., }, Ts=273+160,
    #                                                           v_d=0.01, v_r=0.1, border=4.),
    #                                target_func_to_maximize=target)
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

    def target(x):
        return x[0]

    # PC = ProcessController(LibudaModelWithDegradation(init_cond={'thetaO': 0.25, 'thetaCO': 0.5}, Ts=440),
    #                        target_func_to_maximize=target)
    # # for i in range(5):
    # #     PC.set_controlled([(i + 0.2 * i1) * 1.e-5 for i1 in range(1, 3)])
    # #     PC.time_forward(30)
    # PC.set_controlled({'O2': 10.e-5, 'CO': 4.2e-5})
    # PC.time_forward(500)
    # print(PC.integrate_along_history(target_mode=True))
    # # PC.plot(f'PC_plots/example_RL_21_10_task.png', out_name='target', plot_mode='separately')

    # TEST if more CO is better under the old conditions of the tests for the russian article
    PC_L2001_old = ProcessController(LibudaModelWithDegradation(init_cond={'thetaO': 0.25, 'thetaCO': 0.5}, Ts=440,
                                                                v_d=0.01, v_r=1.5, border=4.),
                                     target_func_to_maximize=target)

    episode_len = 500
    for i in range(2, 11, 2):
        PC_L2001_old.reset()
        O2 = i * 1e-5 / 4
        CO = i * 1e-5
        PC_L2001_old.set_controlled({'O2': O2, 'CO': i * 1e-5})
        PC_L2001_old.time_forward(episode_len)
        R = PC_L2001_old.integrate_along_history(time_segment=[0., episode_len])
        print(f'CO: {CO}, O2: {O2}, R: {R}')

    # # find optimal log_scale
    # average_rate = PC.integrate_along_history(target_mode=True) / PC.get_current_time()
    # max_rate, = PC.process_to_control.get_bounds('max', 'output')
    # log_scale = 5_000
    # print(np.log(1 + log_scale * average_rate / max_rate))
    # print(np.log(1 + log_scale))


def count_conversion_given_exp(csv_path, L_model: LibudaModel):
    df = lib.read_flot_to_file_csv(csv_path, False)
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


def plot_conv(csv_path):
    df = pd.read_csv(csv_path)

    plot_file_path, _ = os.path.splitext(csv_path)
    plot_file_path = f'{plot_file_path}.png'
    lib.plot_to_file(df['Time'], df['CO_conversion'],  'CO conversion',
                     df['Time'], df['O2_conversion'], 'O2 conversion',
                     title='Conversion',
                     ylim=None, save_csv=False, fileName=plot_file_path)


def Pt_exp(path: str):

    def target(x):
        return x[0]

    PC_Pt2210 = ProcessController(PtModel(init_cond={'thetaO': 0., 'thetaCO': 0.}),
                                  target_func_to_maximize=target)
    PC_Pt2210.set_plot_params(input_ax_name='Pressure', input_lims=None,
                              output_ax_name='CO2 form. rate', output_lims=None)

    # first experiment
    low_value = 4
    high_value = 10
    reps = 10
    for i in range(reps):
        PC_Pt2210.set_controlled({'O2': low_value, 'CO': high_value})
        PC_Pt2210.time_forward(20)
        PC_Pt2210.set_controlled({'O2': high_value, 'CO': low_value})
        PC_Pt2210.time_forward(30)
    R = PC_Pt2210.integrate_along_history(target_mode=True, time_segment=[0., 50 * reps],
                                          RESOLUTION=10)

    def ax_func(ax):
        ax.set_title(f'integral: {R:.4g}')

    PC_Pt2210.plot(path,
                plot_more_function=ax_func, plot_mode='separately',
                time_segment=[0., 50 * reps], additional_plot=['theta_CO', 'theta_O'])


if __name__ == '__main__':

    # custom_experiment()

    # test_PC_with_Libuda()

    # check_func_to_optimize()

    # count_conversion_given_exp('run_RL_out/important_results/220928_T25_diff_lims/O2_40_CO_10/8_copy.csv',
    #                            LibudaModel(Ts=273+25))
    # count_conversion_given_exp('run_RL_out/important_results/220830_REFERENCE/4_copy.csv',
    #                            LibudaModel(Ts=273+25))

    # plot_conv('run_RL_out/conversion/220928_8_conv.csv')
    # plot_conv('run_RL_out/conversion/220830_4_conv.csv')

    # Pt_exp('PC_plots/Pt/1_.png')

    test_new_k1k3_model_new_targets()

    pass
