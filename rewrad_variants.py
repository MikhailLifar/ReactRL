import numpy as np
import lib


def get_reward_func(params: dict):
    if params['name'] == 'each_step_base':
        bias = params['bias']

        def each_step_base(env_obj) -> float:
            time_step = env_obj.last_actual_time_step
            time_history, target_history = env_obj.controller.get_target_for_passed()
            count_step_measurements = int(np.floor(time_step / env_obj.controller.analyser_dt))
            integral_target_last_step = lib.integral(time_history[-count_step_measurements:], target_history[-count_step_measurements:])
            # mean_integral_part = np.mean(target_history[:-1]) * count_step_measurements  # this didn't work well
            # rew = (target_history[-1] - mean_integral_part * part) / mean_integral_part / part
            rew = integral_target_last_step * env_obj.normalize_coef / env_obj.episode_time -\
                  bias * time_step / env_obj.episode_time
            return rew

        out_func = each_step_base

    elif params['name'] == 'full_ep_base':
        bias = params['bias']

        def full_ep_base(env_obj) -> float:
            if env_obj.end_episode:
                return env_obj.cumm_episode_target * env_obj.normalize_coef / env_obj.episode_time - bias
            return 0.

        out_func = full_ep_base

    elif params['name'] == 'full_ep_2':
        subtype = params['subtype']
        depth = params['depth']

        def full_ep_2(env_obj) -> float:
            """

            Reward depends on the value of integral for the entire episode
            and the values of integral for the last [depth] episodes.
            Reward is got at the end of episode.

            :param env_obj:
            :return:
            """

            if env_obj.end_episode:
                # print('Something went wrong!')
                int_arr = env_obj.stored_integral_data['integral'][
                          :max(env_obj.count_episodes - 1, 1)]
                std_lasts = np.std(int_arr[-depth:])
                std_lasts = max(std_lasts, 5.e-3 / env_obj.normalize_coef)
                if subtype == 'mean_mode':
                    rew = (env_obj.cumm_episode_target - np.mean(int_arr[-depth:])) / std_lasts
                elif subtype == 'median_mode':
                    rew = (env_obj.cumm_episode_target - np.median(int_arr[-depth:])) / std_lasts
                elif subtype == 'max_mode':
                    rew = (env_obj.cumm_episode_target - np.max(int_arr[-depth:])) / std_lasts
                else:
                    rew = None
                if rew == 0:
                    rew = 0.5 - 1. * (env_obj.count_episodes % 2)
                return rew
            return 0.

        out_func = full_ep_2

    elif params['name'] == 'each_step_normalize':

        def each_step_normalize(env_obj) -> float:
            raise NotImplementedError

        out_func = each_step_normalize

    elif params['name'] == 'full_ep_normalize':
        def full_ep_normalize(env_obj) -> float:
            raise NotImplementedError

        out_func = full_ep_normalize

    elif params['name'] == 'each_step_new':

        def each_step_new(env_obj) -> float:
            raise NotImplementedError

        out_func = each_step_new

    elif params['name'] == 'full_ep_new':

        def full_ep_new(env_obj) -> float:
            raise NotImplementedError

        out_func = full_ep_new

    elif params['name'] == 'hybrid':
        subtype = params['subtype']
        depth = params['depth']

        def hybrid_reward(env_obj) -> float:
            """

            Reward combines full_ep_[mode] and each_step_base rewards

            :param env_obj:
            :return:
            """

            # if env_obj.end_episode:
            #     # print('Something went wrong!')
            #     int_arr = env_obj.stored_integral_data['integral'][:env_obj.count_episodes]
            #     std_lasts = np.std(int_arr[-depth:])
            #     if std_lasts < 1.e-2 * env_obj.integral:
            #         std_lasts = 1.e-2 * env_obj.integral
            #     if subtype == 'mean_mode':
            #         rew = (env_obj.integral - np.mean(int_arr[-depth-1:-1])) / std_lasts
            #     elif subtype == 'median_mode':
            #         rew = (env_obj.integral - np.median(int_arr[-depth-1:-1])) / std_lasts
            #     elif subtype == 'max_mode':
            #         rew = (env_obj.integral - np.max(int_arr[-depth-1:-1])) / std_lasts
            #     else:
            #         rew = None
            #     # rew = min(rew, 2.)
            #     # rew = max(-2., rew)
            #     if rew == 0:
            #         rew = 0.5 - 1. * (env_obj.count_episodes % 2)
            #     return rew
            # else:
            raise NotImplementedError

        out_func = hybrid_reward

    else:
        assert False, f'Error! Invalid type of reward function: {params["name"]}'

    return out_func
