import os
import numpy as np


def get_for_SBP_iteration(episode_time, first_to_turn: str, ziff_model: bool = False,
                          out_name_to_observe='CO2_count', ):

    def switch_between_pure_iteration(PC, params: dict, foldpath, it_arg):
        O2_max, CO_max = params['O2_max'], params['CO_max']
        if ('t0' in params) and ('t1' in params):
            t0, t1 = params['t0'], params['t1']
        else:
            t0, t1 = params['first_part'] * params['total'], (1 - params['first_part']) * params['total']

        pressures_to_set = [(O2_max, 0.), (0, CO_max)]
        if first_to_turn == 'CO':
            pressures_to_set = pressures_to_set[::-1]
        if ziff_model:
            # O2, CO -> x_CO
            pressures_to_set = [(pair[1] / sum(pair), ) for pair in pressures_to_set]

        PC.reset()
        while PC.get_current_time() < episode_time:
            PC.set_controlled(pressures_to_set[0])
            PC.time_forward(t0)
            PC.set_controlled(pressures_to_set[1])
            PC.time_forward(t1)

        _, output_history = PC.get_and_plot(f'{foldpath}/SBP_{it_arg}_t0({t0:.4g})_t1({t1:.4g}).png',
                                            plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                                         'plot_mode': 'separately', 'out_names': out_name_to_observe})
        R = PC.integrate_along_history(out_name=out_name_to_observe)

        return {'CO2': R}

    return {'iteration_function': switch_between_pure_iteration}


def get_for_steady_state_variations(episode_time,
                                    name_to_variate,
                                    target: dict,
                                    additional_names=('thetaCO', 'thetaO'),
                                    transform_params=lambda x: x,
                                    out_names_to_plot=(),
                                    take_from_the_end=0.5):
    import lib
    import json

    target_name = target["name"]

    def variate_iteration(PC, params: dict, foldpath, it_arg):
        ret = [0] * (1 + len(additional_names))
        PC.reset()
        PC.set_controlled(transform_params(params))
        PC.time_forward(episode_time)
        PC.get_and_plot(f'{foldpath}/variations_{name_to_variate}({params[name_to_variate]:.2f})_{it_arg}.png',
                        plot_params={'time_segment': [0, None], 'additional_plot': additional_names,
                                     'plot_mode': 'separately', 'out_names': out_names_to_plot})
        ret[0] = PC.get_process_output()[1][:, target['column']]  # should be CO2 output column
        for i in range(1, len(ret)):
            ret[i] = PC.additional_graph[additional_names[i - 1]][:ret[0].size]

        for j, v in enumerate(ret):
            ret[j] = np.mean(v[int(v.size * (1. - take_from_the_end)):])

        ret_d = {target_name: ret[0]}
        ret_d.update({name: ret[i + 1] for i, name in enumerate(additional_names)})
        return ret_d

    def variate_summarize(foldapth):
        x_arr, avgs = [], []
        for fname in filter(lambda name: name.endswith('.json'), sorted(os.listdir(foldapth))):
            with open(f'{foldapth}/{fname}', 'r') as fread:
                d = json.load(fread)
                ret_d = d['return']
                # complicated lists expansion due to possible duplicates of x values
                x_ax_value = d['variables'][name_to_variate]
                idx = next(filter(lambda i: abs(x_ax_value - x_arr[i]) < 1.e-5, range(len(x_arr))), None)
                if idx is None:
                    avgs.append([[ret_d[target_name]] + [ret_d[name] for name in additional_names]])
                    x_arr.append(x_ax_value)
                else:
                    avgs[idx].append([ret_d[target_name]] + [ret_d[name] for name in additional_names])

        for i, matr in enumerate(avgs):
            avgs[i] = np.mean(avgs[i], axis=0)
        x_arr, avgs = np.array(x_arr), np.array(avgs)
        idxs = np.argsort(x_arr)
        x_arr, avgs = x_arr[idxs], avgs[idxs]

        to_plot = [x_arr,
                   avgs[:, 0],
                   {'label': target_name, 'linestyle': 'solid',
                                          'marker': 'h', 'c': 'purple',
                                          'twin': True,
                    }
                   ]
        if additional_names:
            to_plot += [x_arr, avgs[:, 1], {'label': additional_names[0], 'linestyle': (0, (1, 1)), 'marker': 'x', 'c': 'blue'}]
            if len(additional_names) > 1:
                to_plot += [x_arr, avgs[:, 2], {'label': additional_names[1], 'linestyle': (0, (5, 5)), 'marker': '+', 'c': 'red'}]

        lib.plot_to_file(*to_plot,
                         fileName=f'{foldapth}/variations_summarize_CO2.png',
                         xlabel=f'{name_to_variate}', ylabel='?', title='Summarize variations',
                         twin_params={'ylabel': target_name}, )

    return {'iteration_function': variate_iteration, 'summarize_function': variate_summarize}


def get_for_Ziff_iterations(pressure_unit: float, episode_time, CO2_output_column=3,
                            out_names_to_plot=('CO2_count',),
                            take_from_the_end=0.5):
    # import lib
    # import json
    #
    # def Ziff_iteration(PC, params: dict, foldpath, it_arg):
    #     if params.get('x', None) is not None:
    #         O2, CO = (1 - params['x']) * 10 * pressure_unit, params['x'] * 10 * pressure_unit
    #     else:
    #         O2, CO = params['O2'], params['CO']
    #     CO2_and_covs = [0] * 3
    #     PC.reset()
    #     PC.set_controlled(params)
    #     PC.time_forward(episode_time)
    #     PC.get_and_plot(f'{foldpath}/Ziff_O2({O2 / pressure_unit:.2f})_CO({CO / pressure_unit:.2f})_{it_arg}.png',
    #                     plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
    #                                  'plot_mode': 'separately', 'out_names': out_names_to_plot})
    #     CO2_and_covs[0] = PC.get_process_output()[1][:, CO2_output_column]  # should be CO2 output column
    #     CO2_and_covs[1] = PC.additional_graph['thetaO'][:CO2_and_covs[0].size]
    #     CO2_and_covs[2] = PC.additional_graph['thetaCO'][:CO2_and_covs[0].size]
    #
    #     for j, v in enumerate(CO2_and_covs):
    #         CO2_and_covs[j] = np.mean(v[int(v.size * (1. - take_from_the_end)):])
    #
    #     return {'CO2': CO2_and_covs[0],
    #             'thetaO': CO2_and_covs[1],
    #             'thetaCO': CO2_and_covs[2], }
    #
    # def Ziff_summarize(foldapth):
    #     x_arr, avgs = [], []
    #     for fname in filter(lambda name: name.endswith('.json'), sorted(os.listdir(foldapth))):
    #         with open(f'{foldapth}/{fname}', 'r') as fread:
    #             d = json.load(fread)
    #             ret_d = d['return']
    #             # complicated lists expansion due to possible duplicates of x values
    #             if d['variables'].get('x', None) is not None:
    #                 x_ax_value = 10 * d['variables']['x']
    #             else:
    #                 x_ax_value = d['variables']['CO'] / pressure_unit
    #             idx = next(filter(lambda i: abs(x_ax_value - x_arr[i]) < 1.e-5, range(len(x_arr))), None)
    #             if idx is None:
    #                 avgs.append([[ret_d['CO2'], ret_d['thetaO'], ret_d['thetaCO']]])
    #                 x_arr.append(x_ax_value)
    #             else:
    #                 avgs[idx].append([ret_d['CO2'], ret_d['thetaO'], ret_d['thetaCO']])
    #
    #     for i, matr in enumerate(avgs):
    #         avgs[i] = np.mean(avgs[i], axis=0)
    #     x_arr, avgs = np.array(x_arr), np.array(avgs)
    #     idxs = np.argsort(x_arr)
    #     x_arr, avgs = x_arr[idxs], avgs[idxs]
    #     lib.plot_to_file(x_arr, avgs[:, 0], {'label': 'Average CO2 prod. rate', 'linestyle': 'solid',
    #                                          'marker': 'h', 'c': 'purple',
    #                                          'twin': True,
    #                                          },
    #                      x_arr, avgs[:, 1], {'label': 'Average O2 coverage', 'linestyle': (0, (1, 1)),
    #                                          'marker': 'x', 'c': 'blue'},
    #                      x_arr, avgs[:, 2], {'label': 'Average CO coverage', 'linestyle': (0, (5, 5)),
    #                                          'marker': '+', 'c': 'red'},
    #                      fileName=f'{foldapth}/Ziff_summarize_CO2.png',
    #                      xlabel=f'CO, {pressure_unit:.4g} Pa', ylabel='coverages', title='Summarize across benchmark',
    #                      twin_params={'ylabel': 'CO2 prod. rate'}, )
    #
    # return {'iteration_function': Ziff_iteration, 'summarize_function': Ziff_summarize}

    return get_for_steady_state_variations(episode_time, 'x_co', {'name': 'CO2', 'column': CO2_output_column},
                                           transform_params=lambda d: {
                                               'O2': 10 * pressure_unit * (1 - d['x_co']),
                                               'CO': 10 * pressure_unit * d['x_co'],
                                           },
                                           out_names_to_plot=out_names_to_plot,
                                           take_from_the_end=take_from_the_end)


def get_for_VanNeer_iterations(episode_time, CO2_output_column=3,
                               out_names_to_plot=('CO2',),
                               take_from_the_end=0.5):
    import lib
    import json
    import predefined_policies as policies

    def _iteration(PC, params: dict, foldpath, it_arg):
        CO2_and_covs = [0] * 3

        t = 1. / (10 ** params['log_omega'])
        T1, T2 = 500, 700
        pO2 = pCO = 1000
        R = PC.process_to_control['R']
        T_policy = policies.TwoStepPolicy({'1': T1, '2': T2, 't1': t / 2, 't2': t / 2})
        O2_policy = policies.TwoStepPolicy({'1': pO2 / T1 / R, '2': pO2 / T2 / R, 't1': t / 2, 't2': t / 2})
        CO_policy = policies.TwoStepPolicy({'1': pCO / T1 / R, '2': pCO / T2 / R, 't1': t / 2, 't2': t / 2})

        PC.reset()
        PC.process_by_policy_objs((O2_policy, CO_policy, T_policy), episode_time, t / 20)
        PC.get_and_plot(f'{foldpath}/VanNeer_omega({10 ** params["log_omega"]:.2f})_{it_arg}.png',
                        plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                     'plot_mode': 'separately', 'out_names': out_names_to_plot})
        CO2_and_covs[0] = PC.get_process_output()[1][:, CO2_output_column]  # should be CO2 output column
        CO2_and_covs[1] = PC.additional_graph['thetaO'][:CO2_and_covs[0].size]
        CO2_and_covs[2] = PC.additional_graph['thetaCO'][:CO2_and_covs[0].size]

        for j, v in enumerate(CO2_and_covs):
            CO2_and_covs[j] = np.mean(v[int(v.size * (1. - take_from_the_end)):])

        return {'CO2': CO2_and_covs[0],
                'thetaO': CO2_and_covs[1],
                'thetaCO': CO2_and_covs[2], }

    def VanNeer_summarize(foldapth):
        x_arr, avgs = [], []
        for fname in filter(lambda name: name.endswith('.json'), sorted(os.listdir(foldapth))):
            with open(f'{foldapth}/{fname}', 'r') as fread:
                d = json.load(fread)
                ret_d = d['return']
                # complicated lists expansion due to possible duplicates of x values
                remember_name = None
                for in_name in ('O2', 'CO', 'T'):
                    if d['variables'].get(in_name, None) is not None:
                        x_ax_value = d['variables'][in_name]
                        remember_name = in_name
                        break  # only one name allowed
                idx = next(filter(lambda i: abs(x_ax_value - x_arr[i]) < 1.e-5, range(len(x_arr))), None)
                if idx is None:
                    avgs.append([[ret_d['CO2'], ret_d['thetaO'], ret_d['thetaCO']]])
                    x_arr.append(x_ax_value)
                else:
                    avgs[idx].append([ret_d['CO2'], ret_d['thetaO'], ret_d['thetaCO']])

        for i, _ in enumerate(avgs):
            avgs[i] = np.mean(avgs[i], axis=0)
        x_arr, avgs = np.array(x_arr), np.array(avgs)
        idxs = np.argsort(x_arr)
        x_arr, avgs = x_arr[idxs], avgs[idxs]
        lib.plot_to_file(x_arr, avgs[:, 0], {'label': 'Average CO2 prod. rate', 'linestyle': 'solid',
                                             'marker': 'h', 'c': 'purple',
                                             'twin': True,
                                             },
                         x_arr, avgs[:, 1], {'label': 'Average O2 coverage', 'linestyle': (0, (1, 1)),
                                             'marker': 'x', 'c': 'blue'},
                         x_arr, avgs[:, 2], {'label': 'Average CO coverage', 'linestyle': (0, (5, 5)),
                                             'marker': '+', 'c': 'red'},
                         fileName=f'{foldapth}/VanNeer_summarize_CO2.png',
                         xlabel=f'{remember_name}', ylabel='coverages', title='Summarize across multiple runs',
                         twin_params={'ylabel': 'CO2 prod. rate'}, )

    return {'iteration_function': _iteration, 'summarize_function': VanNeer_summarize}


# ZGBk_dynamic_advantage_run
def repeat_periods_calc_rate(PC_obj, params: dict, foldpath, it_arg):
    PC_obj.reset()
    parts_number = sum(1 for p in params if p.startswith('part'))
    for _ in range(params['periods_number']):
        for i in range(1, parts_number + 1):
            if f'part{i}' not in params:
                raise ValueError
            PC_obj.set_controlled(params[f'part{i}'])
            PC_obj.time_forward(params[f't{i}'])

    # print(MC_time_step)
    # print(PC_obj.analyser_dt)
    # print(PC_obj.process_to_control['k'])
    # print(PC_obj.controlling_signals_history[:10])
    # print(PC_obj.controlling_signals_history_dt[:10])
    # exit(0)

    rate_history = PC_obj.get_and_plot(f'{foldpath}/repeat_periods_{it_arg}.png',
                                       plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                                    'plot_mode': 'separately', 'out_names': 'CO2_prod_rate'})[1][:, 0]
    # R = PC_obj.integrate_along_history(out_name='CO2_prod_rate', time_segment=[750 * MC_time_step, None])
    rate_history = rate_history[2 * rate_history.size // 3:]

    with open(f'{foldpath}/rate_{it_arg}.txt', 'w') as fwrite:
        # mean
        rate_mean = np.mean(rate_history)
        fwrite.write(f'mean CO2 production rate: {rate_mean}\n')

        # std
        measurements_in_period = int(sum(params[k] for k in filter(lambda p: p.startswith('t'), params.keys())) // PC_obj.analyser_dt)
        mean_period_rates = [np.mean(rate_history[i * measurements_in_period: (i + 1) * measurements_in_period])
                             for i in range(rate_history.size // measurements_in_period)]
        rate_std = np.std(mean_period_rates)
        fwrite.write(f'rate standard deviation: {rate_std}\n')

    return {'rate_mean': rate_mean, 'rate_std': rate_std}
