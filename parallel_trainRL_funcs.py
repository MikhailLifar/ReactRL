import os

import pandas as pd

from lib import plot_to_file
from run_RL import *


def get_for_RL_iterations():

    def RL_iteration(PC, params: dict, foldpath, it_arg):
        model_obj = PC.process_to_control

        # delete parameters that cannot be passed in one iteration,
        # byt need to be passed in other iteration
        # discarding is performing with usage of '#exclude' - special parameter value
        for name in ('env', 'model', 'agent'):
            for sub_name in list(params[name].keys()):
                if params[name][sub_name] == '#exclude':
                    del params[name][sub_name]

        model_obj.reset()
        if len(params['model']):
            model_obj.assign_and_eval_values(**(params['model']),)

        for attr_name in ['target_func', 'long_term_target']:
            if attr_name in params:
                setattr(PC, attr_name, params[attr_name])
                PC.target_func_name = params['target_func_name']

        max_top = None
        limit_names = [name for name in params['model'] if '_top' in name]
        if limit_names:
            max_top = max([params['model'][name] for name in limit_names])
        if max_top is not None:
            if max_top > 0.:
                PC.set_plot_params(input_lims=[-1.e-1 * max_top, 1.1 * max_top])
            else:
                # PC_obj.set_plot_params(input_lims=None)
                raise NotImplementedError

        env_obj = Environment.create(
            environment=RL2207_Environment(PC, **(params['env'])),
            max_episode_timesteps=6000)
        agent_rl = create_tforce_agent(env_obj, params['agent_name'], **(params['agent']))
        # describe the agent to file
        with open(f'{foldpath}/_info.txt', 'a') as f:
            f.write(f'----Agent_information----\n')
            f.write(f'agent: {params["agent_name"]}\n')
            agent_params = params['agent']
            for p in agent_params:
                f.write(f'{p}: {agent_params[p]}\n')
        ret = run(env_obj, agent_rl,
                  out_folder=foldpath,
                  n_episodes=params['n_episodes'], create_unique_folder=False)

        # individual iteration file
        x_vector = np.arange(env_obj.stored_integral_data['integral'][:env_obj.count_episodes].size)[::20]
        this_train_ress = env_obj.stored_integral_data['smooth_1000_step']
        if x_vector.size > this_train_ress.size:
            x_vector = x_vector[:this_train_ress.size]
        label = ''
        for name in params:
            label += f'{params[name]}, '
        df = pd.DataFrame(columns=['x', 'y', 'label'])
        df['x'] = x_vector
        df['y'] = this_train_ress
        df.loc[0, 'label'] = label
        df.to_csv(f'{foldpath}/iter{it_arg}.csv', sep=' ', index=False)

        return ret

    def RL_summarize(foldpath):
        iterations_number = sum((fold[0] == '_') and (fold[1:].isnumeric())
                                for fold in os.listdir(foldpath))
        plot_list = []
        for i in range(iterations_number):
            filepath = f'{foldpath}/_{i}/iter{i}.csv'
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, sep=' ')
                plot_list += [df['x'].to_numpy(), df['y'].to_numpy(), df.loc[0, 'label']]
                os.remove(f'{foldpath}/_{i}/iter{i}.csv')
        if len(plot_list) < 5 * 3:
            plot_to_file(*plot_list, fileName=f'{foldpath}/graph/summary_graph.png')
        else:
            list_len = len(plot_list)
            for init in range(0, list_len, 3 * 4):
                plot_to_file(*(plot_list[init:min(init + 3 * 4, list_len)]),
                             fileName=f'{foldpath}/graph/'
                                      f'summary_graph{(init + 1) // (3 * 4)}.png')

    return {'iteration_function': RL_iteration, 'summarize_function': RL_summarize,
            'separate_folds': True,
            'names_groups': ('env', 'model', 'agent')}


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
