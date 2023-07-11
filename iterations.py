import os
import numpy as np


def get_common_1d_summarize(variable_name, names_to_plot, x_tolerance=1.e-5,
                            **kwargs):
    import json
    import lib

    def common_summarize(foldapth):
        x_arr, data = [], []
        for fname in filter(lambda name: name.endswith('.json'), sorted(os.listdir(foldapth))):
            with open(f'{foldapth}/{fname}', 'r') as fread:
                d = json.load(fread)
                ret_d = d['return']
                # complicated lists expansion due to possible duplicates of x values
                x_ax_value = d['variables'][variable_name]
                idx = next(filter(lambda i: abs(x_ax_value - x_arr[i]) < x_tolerance, range(len(x_arr))), None)
                if idx is None:
                    data.append([[ret_d[name] for name in names_to_plot]])
                    x_arr.append(x_ax_value)
                else:
                    data[idx].append([ret_d[name] for name in names_to_plot])

        for i, matr in enumerate(data):
            data[i] = np.mean(data[i], axis=0)
        x_arr, data = np.array(x_arr), np.array(data)
        idxs = np.argsort(x_arr)
        x_arr, data = x_arr[idxs], data[idxs]

        styles = [
            {'linestyle': 'solid', 'marker': 'h', 'c': 'purple', 'twin': True, },
            {'linestyle': (0, (1, 1)), 'marker': 'x', 'c': 'blue'},
            {'linestyle': (0, (5, 5)), 'marker': '+', 'c': 'red'},
        ]

        to_plot = []
        for i, name in enumerate(names_to_plot):
            styles[i].update({'label': name})
            to_plot += [x_arr, data[:, i], styles[i]]

        lib.plot_to_file(*to_plot,
                         fileName=f'{foldapth}/{kwargs.get("filename", "summarize.png")}',
                         xlabel=f'{variable_name}', ylabel=kwargs.get('ylabel', '?'), title=kwargs.get('title', 'summary'),
                         ylim=kwargs.get('ylim', (0, None)),
                         twin_params={'ylabel': names_to_plot[0], 'ylim': kwargs.get('twin_ylim', (0, None))})

    return common_summarize


def get_for_SB2_iteration(episode_time, target_name, names_to_plot: dict, additional_names=('thetaCO', 'thetaO'),):

    def switch_between_two_iteration(PC, params: dict, foldpath, it_arg):
        if ('t1' in params) and ('t2' in params):
            t1, t2 = params['t1'], params['t2']
        else:
            t1, t2 = params['first_part'] * params['total'], (1 - params['first_part']) * params['total']

        PC.reset()
        while PC.get_current_time() < episode_time:
            PC.set_controlled(params['part1'])
            PC.time_forward(t1)
            PC.set_controlled(params['part2'])
            PC.time_forward(t2)

        PC.get_and_plot(f'{foldpath}/SBP_{it_arg}_t1({t1:.4g})_t2({t2:.4g}).png',
                        plot_params={'time_segment': [0, None], 'additional_plot': additional_names,
                                     'plot_mode': 'separately',
                                     'input_names': names_to_plot['input'], 'out_names': names_to_plot['output']})
        R = PC.integrate_along_history(out_name=target_name)

        return {target_name: R}

    return {'iteration_function': switch_between_two_iteration}


def get_for_SB2_iteration_flexible(periods_number, target: dict, names_to_plot: dict, calc_analyser_dt=None, additional_names=('thetaCO', 'thetaO'), ):
    # TODO merge with get_for_SB2 function

    def switch_between_two_iteration(PC, params: dict, foldpath, it_arg):
        if ('t1' in params) and ('t2' in params):
            t1, t2 = params['t1'], params['t2']
        else:
            t1, t2 = params['first_part'] * params['total'], (1 - params['first_part']) * params['total']

        total = t1 + t2

        original_analyser_dt = PC.analyser_dt
        if calc_analyser_dt is not None:
            PC.analyser_dt = calc_analyser_dt(total)

        PC.reset()
        while PC.get_current_time() < periods_number * total:
            PC.set_controlled(params['part1'])
            PC.time_forward(t1)
            PC.set_controlled(params['part2'])
            PC.time_forward(t2)

        _, output = PC.get_and_plot(f'{foldpath}/SBP_{it_arg}_t1({t1:.4g})_t2({t2:.4g}).png',
                                    plot_params={'time_segment': [0, None], 'additional_plot': additional_names,
                                                 'plot_mode': 'separately',
                                                 'input_names': names_to_plot['input'], 'out_names': names_to_plot['output']})
        output = output[:, target['column']]

        PC.analyser_dt = original_analyser_dt

        return {target['name']: np.mean(output[output.size//2:])}

    return {'iteration_function': switch_between_two_iteration}


def get_for_steady_state_variations(episode_time,
                                    name_to_variate,
                                    target: dict,
                                    additional_names=('thetaCO', 'thetaO'),
                                    transform_params=lambda x: x,
                                    names_to_plot=None,
                                    take_from_the_end=0.5):

    target_name = target["name"]

    def variate_iteration(PC, params: dict, foldpath, it_arg):
        ret = [0] * (1 + len(additional_names))
        PC.reset()
        PC.set_controlled(transform_params(params))
        PC.time_forward(episode_time)
        PC.get_and_plot(f'{foldpath}/steady_state_variations_{name_to_variate}({params[name_to_variate]:.2f})_{it_arg}.png',
                        plot_params={'time_segment': [0, None], 'additional_plot': additional_names,
                                     'plot_mode': 'separately', 'input_names': names_to_plot['input'], 'out_names': names_to_plot['output'], })
        ret[0] = PC.get_process_output()[1][:, target['column']]  # should be CO2 output column
        for i in range(1, len(ret)):
            ret[i] = PC.additional_graph[additional_names[i - 1]][:ret[0].size]

        for j, v in enumerate(ret):
            ret[j] = np.mean(v[int(v.size * (1. - take_from_the_end)):])

        ret_d = {target_name: ret[0]}
        ret_d.update({name: ret[i + 1] for i, name in enumerate(additional_names)})
        return ret_d

    return {'iteration_function': variate_iteration,
            'summarize_function': get_common_1d_summarize(name_to_variate, [target_name] + [name for name in additional_names]),
            }


def get_for_common_variations(policies_dict,
                              name_to_variate,
                              target: dict,
                              transform_params=lambda x: x,
                              names_to_sum_plot=None,
                              kwargs_to_sum_plot=None,
                              additional_names=('thetaCO', 'thetaO'),
                              take_from_the_end=0.5):

    def _iteration(PC, params: dict, foldpath, it_arg):
        transform_params(params)
        ret = [0] * (1 + len(additional_names))

        for name, policy in policies_dict.items():
            policy.set_policy({p.replace(f'{name}_', ''): v for p, v in params.items() if p.startswith(f'{name}_')})

        episode_time = params['episode_time']
        calc_dt = params.get('calc_dt', False)
        old_analyser_dt = False
        if calc_dt:
            old_analyser_dt = PC.analyser_dt
            PC.analyser_dt = policy_step = calc_dt(episode_time)
        else:
            policy_step = params['policy_step']

        PC.reset()
        PC.process_by_policy_objs([policies_dict[name] for name in PC.controlled_names],
                                  episode_time, policy_step)
        PC.get_and_plot(f'{foldpath}/common_variations_{name_to_variate}({params[name_to_variate]:.2f})_{it_arg}.png')

        ret[0] = PC.get_process_output()[1][:, target['column']]  # should be CO2 output column
        for i in range(1, len(ret)):
            ret[i] = PC.additional_graph[additional_names[i - 1]][:ret[0].size]
        for j, v in enumerate(ret):
            ret[j] = np.mean(v[int(v.size * (1. - take_from_the_end)):])

        if old_analyser_dt:
            PC.analyser_dt = old_analyser_dt

        ret_d = {target['name']: ret[0]}
        ret_d.update({name: ret[i + 1] for i, name in enumerate(additional_names)})
        return ret_d

    if names_to_sum_plot is None:
        names_to_sum_plot = (target['name'], ) + tuple(additional_names)

    if kwargs_to_sum_plot is None:
        kwargs_to_sum_plot = {}

    return {'iteration_function': _iteration,
            'summarize_function': get_common_1d_summarize(name_to_variate, names_to_sum_plot, 1.e-5,
                                                          filename='common_var_summary_1d.png',
                                                          **kwargs_to_sum_plot)}


def get_for_Ziff_iterations(pressure_unit: float, episode_time, CO2_output_column=3,
                            out_names_to_plot=('CO2_count',),
                            take_from_the_end=0.5):

    return get_for_steady_state_variations(episode_time, 'x_co', {'name': 'CO2', 'column': CO2_output_column},
                                           transform_params=lambda d: {
                                               'O2': 10 * pressure_unit * (1 - d['x_co']),
                                               'CO': 10 * pressure_unit * d['x_co'],
                                           },
                                           names_to_plot={'input': None, 'output': out_names_to_plot},
                                           take_from_the_end=take_from_the_end)


# ZGBk_dynamic_advantage_run
def repeat_periods_calc_rate(PC, params: dict, foldpath, it_arg):
    PC.reset()
    parts_number = sum(1 for p in params if p.startswith('part'))
    for _ in range(params['periods_number']):
        for i in range(1, parts_number + 1):
            if f'part{i}' not in params:
                raise ValueError
            PC.set_controlled(params[f'part{i}'])
            PC.time_forward(params[f't{i}'])

    # print(MC_time_step)
    # print(PC_obj.analyser_dt)
    # print(PC_obj.process_to_control['k'])
    # print(PC_obj.controlling_signals_history[:10])
    # print(PC_obj.controlling_signals_history_dt[:10])
    # exit(0)

    rate_history = PC.get_and_plot(f'{foldpath}/repeat_periods_{it_arg}.png',
                                   plot_params={'time_segment': [0, None], 'additional_plot': ['thetaCO', 'thetaO'],
                                                    'plot_mode': 'separately', 'out_names': 'CO2_prod_rate'})[1][:, 0]
    # R = PC_obj.integrate_along_history(out_name='CO2_prod_rate', time_segment=[750 * MC_time_step, None])
    rate_history = rate_history[2 * rate_history.size // 3:]

    with open(f'{foldpath}/rate_{it_arg}.txt', 'w') as fwrite:
        # mean
        rate_mean = np.mean(rate_history)
        fwrite.write(f'mean CO2 production rate: {rate_mean}\n')

        # std
        measurements_in_period = int(sum(params[k] for k in filter(lambda p: p.startswith('t'), params.keys())) // PC.analyser_dt)
        mean_period_rates = [np.mean(rate_history[i * measurements_in_period: (i + 1) * measurements_in_period])
                             for i in range(rate_history.size // measurements_in_period)]
        rate_std = np.std(mean_period_rates)
        fwrite.write(f'rate standard deviation: {rate_std}\n')

    return {'rate_mean': rate_mean, 'rate_std': rate_std}