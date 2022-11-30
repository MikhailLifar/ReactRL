# import \
#     os
# import \
#     pandas as pd

from time import sleep

import \
    pandas as pd

from lib import plot_to_file
from run_RL import *


def train_list_parallel(
        params_variants: list,
        names: tuple,
        repeat: int = 1,
        out_path: str = None,
        const_params: dict = None,
        controller: ProcessController = None,
        unique_folder=True,
        n_episodes=12000,
        python_interpreter='venv/bin/python',
        file_to_execute_path='repos/parallel_trainRL.py',
        on_cluster=False,
        at_same_time: int = 20):
    assert len(names) == len(params_variants[0]), 'Error: lengths mismatch'

    if repeat > 1:
        assert isinstance(params_variants, list), f'params_variants should be list,' \
                                                  f'got {type(params_variants)} instead'
        new_list = []
        for values_set in params_variants:
            new_list += [values_set] * repeat
        params_variants = new_list

    for subd_name in ('env', 'model', 'agent'):
        if subd_name not in const_params:
            const_params[subd_name] = dict()

    if unique_folder:
        out_path = make_subdir_return_path(out_path, with_date=True, unique=True)

    # get arguments from the command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('iter', type=int, help='rl training iteration')
    args = parser.parse_args()

    try_num = len(params_variants)
    iter_arg = args.iter
    if iter_arg == -1:
        assert try_num <= at_same_time  # TODO TODO TODO
        for i in range(try_num):
            # ../RL_10_21/venv/bin/python
            if on_cluster:
                os.system(f'run-cluster -m 3000 -n 1 "{python_interpreter} {file_to_execute_path} {i}"')
            else:
                os.system(f'{python_interpreter} {file_to_execute_path} {i}')  # just put here path to the script
        if on_cluster:
            os.system(f'run-cluster -m 2000 -n 1 "{python_interpreter} {file_to_execute_path} -2"')
        else:
            os.system(f'{python_interpreter} {file_to_execute_path} -2')

    elif iter_arg == -2:
        # Cycle which run so far so not all the iterations are completed
        # and not all the files are created. Cycle does nothing
        file_not_found = True
        while file_not_found:
            file_not_found = False
            for i in range(try_num):
                if not os.path.exists(f'{out_path}/iter{i}.csv'):
                    # print(f'It is bad! Bad index: {i}')
                    file_not_found = True
        # delay after cycle exit
        sleep(10)
        # summarize outputs of different iterations in one file
        # and remove individual file for each iteration
        plot_list = []
        for i in range(try_num):
            df = pd.read_csv(f'{out_path}/iter{i}.csv', sep=' ')
            plot_list += [df['x'].to_numpy(), df['y'].to_numpy(), df.loc[0, 'label']]
            os.remove(f'{out_path}/iter{i}.csv')
        if len(plot_list) < 5 * 3:
            plot_to_file(*plot_list, fileName=f'{out_path}/graph/summary_graph.png')
        else:
            list_len = len(plot_list)
            for init in range(0, list_len, 3 * 4):
                plot_to_file(*(plot_list[init:min(init + 3 * 4, list_len)]),
                             fileName=f'{out_path}/graph/'
                                      f'summary_graph{(init + 1) // (3 * 4)}.png')
    else:
        model_obj = controller.process_to_control
        set_values = params_variants[iter_arg]
        variable_params = fill_variable_dict(set_values, names)
        # delete parameters that cannot be passed in one iteration,
        # byt need to be passed in other iteration
        # discarding is performing with usage of '#exclude' - special parameter value
        for name in ('env', 'model', 'agent'):
            for sub_name in list(variable_params[name].keys()):
                if variable_params[name][sub_name] == '#exclude':
                    del variable_params[name][sub_name]
        # print(set_values)

        model_obj.reset()
        if (len(variable_params['model'])) or (len(const_params['model'])):
            model_obj.assign_and_eval_values(**(variable_params['model']),
                                             **(const_params['model']))

        for d in [variable_params, const_params]:
            for attr_name in ['target_func', 'long_term_target']:
                if attr_name in d:
                    setattr(controller, attr_name, d[attr_name])
                    controller.target_func_name = d['target_func_name']

        limit_names = [name for name in variable_params['model'] if '_top' in name]
        max_top = max([variable_params['model'][name] for name in limit_names])
        limit_names = [name for name in const_params['model'] if '_top' in name]
        max_top = max([const_params['model'][name] for name in limit_names] + [max_top])
        if max_top > 0:
            controller.set_plot_params(input_lims=[-1.e-1 * max_top, 1.1 * max_top])
        else:
            # PC_obj.set_plot_params(input_lims=None)
            raise NotImplementedError

        env_obj = Environment.create(
            environment=RL2207_Environment(controller,
                                           **(const_params['env']), **(variable_params['env'])),
            max_episode_timesteps=6000)
        if 'agent_name' in const_params:
            agent_name = const_params['agent_name']
        else:
            agent_name = variable_params['agent_name']
        agent_rl = create_tforce_agent(env_obj, agent_name,
                                       **(const_params['agent']), **(variable_params['agent']))
        the_folder = make_subdir_return_path(out_path, name=f'_{iter_arg}', with_date=False, unique=False)
        # describe the agent to file
        with open(f'{the_folder}/_info.txt', 'a') as f:
            f.write(f'----Agent_information----\n')
            f.write(f'agent: {agent_name}\n')
            agent_params = {**const_params['agent'], **variable_params['agent']}
            for p in agent_params:
                f.write(f'{p}: {agent_params[p]}\n')
        run(env_obj, agent_rl,
            out_folder=the_folder,
            n_episodes=n_episodes, create_unique_folder=False)

        # individual iteration file
        x_vector = np.arange(env_obj.stored_integral_data['integral'][:env_obj.count_episodes].size)[::20]
        this_train_ress = env_obj.stored_integral_data['smooth_1000_step']
        if x_vector.size > this_train_ress.size:
            x_vector = x_vector[:this_train_ress.size]
        label = ''
        for name in variable_params:
            label += f'{variable_params[name]}, '
        df = pd.DataFrame(columns=['x', 'y', 'label'])
        df['x'] = x_vector
        df['y'] = this_train_ress
        df.loc[0, 'label'] = label
        df.to_csv(f'{out_path}/iter{iter_arg}.csv', sep=' ', index=False)


def train_grid_parallel(*value_sets,
                        names: tuple,
                        **train_list_args):

    assert len(names) == len(value_sets), 'Error: lengths mismatch'
    params_variants = list(itertools.product(*value_sets))
    contains_tuple = False
    # if tuple names contains subtuple of names
    for it in names:
        if isinstance(it, tuple):
            contains_tuple = True
            break
    # if contains...
    if contains_tuple:
        # realisation of grid not for the single parameter,
        # but for the sets of parameters,
        # i. e. creation grid of the form
        # [
        #  [a11, a12, a13..., b11, b12..., ...], [a11, a12, a13..., b21, b22..., ...], [a11, a12, a13..., b31, b32.., ...],
        #  [a21, a22, a23..., b11, b12..., ...], [a21, a22, a23..., b21, b22..., ...], [a21, a22, a23..., b31, b32.., ...],
        #  ]
        for i, _ in enumerate(params_variants):
            new_params_set = []
            for j, it in enumerate(names):
                if isinstance(it, tuple):
                    for k, _ in enumerate(it):
                        new_params_set.append(params_variants[i][j][k])
                else:
                    new_params_set.append(params_variants[i][j])
            params_variants[i] = new_params_set
        new_names = []
        for it in names:
            if isinstance(it, tuple):
                for name in it:
                    new_names.append(name)
            else:
                new_names.append(it)
        names = tuple(new_names)
    train_list_parallel(params_variants, names, **train_list_args)


if __name__ == '__main__':

    # episode_time = 500
    # time_step = 10

    # train_list_parallel(params_variants=[['each_step_base', 'each_step_base', 'each_step_base'],
    #                                      [3e-2, 5e-2, 5., 'each_step_new'],
    #                                      [3e-3, 5e-3, 4., 'full_ep_1'],
    #                                      [3e-2, 5e-2, 5., 'full_ep_1'],
    #                                      [3e-3, 5e-3, 4., 'full_ep_mean'],
    #                                      [3e-3, 5e-3, 4., 'full_ep_mean']],
    #                     names=('model:v_d', 'model:v_r', 'model:border', 'env:reward_spec'),
    #                     const_params={
    #                          'env': {'model_type': 'continuous',
    #                                  'state_spec': {
    #                                      'shape': (3, 3),
    #                                      'use_differences': False,
    #                                      },
    #                                  'episode_time': 500},
    #                          'model': {'CTs': 0.3, 'thetaCO_max': 0.5},
    #                          'agent_name': 'vpg',
    #                      },
    #                     out_path='run_RL_out/train_greed/high_border_diff_rewards',
    #                     controller=ProcessController(TestModel(), target_func_to_maximize=CO2_value),
    #                     n_episodes=30000,
    #                     unique_folder=False,
    #                     at_same_time=30)

    # train_grid_parallel(['each_step_base', 'each_step_base', 'full_ep_mean', 'full_ep_mean'],
    #                  names=('env:reward_spec', ),
    #                  const_params={
    #                      'env': {'model_type': 'continuous',
    #                              'state_spec': {
    #                                  'rows': 1,
    #                                  'use_differences': False,
    #                                  },
    #                              'episode_time': 500},
    #                      'agent_name': 'vpg',
    #                  },
    #                  out_path='run_RL_out/current_training/diff_rewards',
    #                  python_interpreter='../RL_10_21/venv/bin/python',
    #                  on_cluster=False,
    #                  controller=ProcessController(TestModel(), target_func_to_maximize=CO2_value,
    #                                               supposed_step_count=2 * episode_time // time_step,  # memory controlling parameters
    #                                               supposed_exp_time=2 * episode_time),
    #                  n_episodes=10_000,
    #                  unique_folder=False,)

    pass
