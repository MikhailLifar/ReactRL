import numpy as np


def get_reward_func(params: dict):
    if params['name'] == 'each_step_base':
        bias = -0.3

        def each_step_base(env_obj) -> float:
            target_history = env_obj.controller.get_target_for_passed()
            count_step_measurements = int(np.floor(env_obj.time_step / env_obj.controller.analyser_dt))
            integral_target_last_step = np.sum(target_history[-count_step_measurements:])
            # mean_integral_part = np.mean(target_history[:-1]) * count_step_measurements  # this didn't work well
            # rew = (target_history[-1] - mean_integral_part * part) / mean_integral_part / part
            rew = integral_target_last_step * env_obj.normalize_coef -\
                  bias * env_obj.time_step / env_obj.episode_time
            return rew

        out_func = each_step_base

    elif params['name'] == 'full_ep_base':

        def full_ep_base(env_obj) -> float:
            if env_obj.end_episode:
                return env_obj.integral * env_obj.normalize_coef - 0.02
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
                    rew = (env_obj.integral - np.mean(int_arr[-depth:])) / std_lasts
                elif subtype == 'median_mode':
                    rew = (env_obj.integral - np.median(int_arr[-depth:])) / std_lasts
                elif subtype == 'max_mode':
                    rew = (env_obj.integral - np.max(int_arr[-depth:])) / std_lasts
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


# def reward_each_step_1(obj):
#     """
#
#     Reward depends on the current value of CO2
#     and the mean value of CO2 for previous steps.
#     Reward is got each step.
#
#     :param obj:
#     :return:
#     """
#
#     conversion = obj.hc.conversion[obj.hc.conversion >= 0]
#     # alpha = 0
#     # beta = 1
#     # rew = alpha * (conversion[-1] - np.mean(conversion[-21:-1]) / 2) + beta * (conversion[-1] - np.mean(conversion[:-1]) / 2)
#     mean_prev_conversion = np.mean(conversion[:-1])
#     rew = (conversion[-1] - mean_prev_conversion * 0.9) / mean_prev_conversion / 0.9
#     if rew == 0:
#         rew = 0.5 - 1. * (obj.count_episodes % 2)
#     return rew


# def reward_full_episode_1(obj):
#     """
#
#     Reward depends on the value of integral for the entire episode
#     and the best value of integral for the previous episodes.
#     Reward is got in the end of episode.
#
#     :param obj:
#     :return:
#     """
#
#     if obj.end_episode:
#         # print('Something went wrong!')
#         integral = obj.integral
#         # average_last_20 = np.mean(int_list[-20:])
#         # median_last_20 = np.median(int_list[-20:])
#         # std_last_20 = np.std(int_list[-20:])
#         # std_last_20 = np.sqrt(np.median((np.array(int_list[-20:]) - median_last_20) ** 2))
#         # if std_last_20 == 0:
#         #     std_last_20 = 50
#         # print(integral)
#         # rew = (integral - median_last_20) / 50
#         # rew = (integral - median_last_20) / std_last_20
#         # rew = (integral - np.max(int_list)) / 50
#         rew = (integral - obj.best_integral) / 50
#         # std_last_20 = np.sqrt(np.max((np.array(int_list[-21:-1]) - np.max(int_list[-21:-1])) ** 2))
#         # if std_last_20 == 0:
#         #     std_last_20 = 10
#         # rew = (integral - np.max(int_list[-21:-1])) / std_last_20
#         if rew == 0:
#             rew = 0.5 - 1. * (obj.count_episodes % 2)
#         return rew
#     return 0.


# def reward(obj):
#     if obj.end_episode:
#         # print('Something went wrong!')
#         integral = obj.integral
#         # last_integral = self.last_integral
#         # self.last_integral = integral
#         obj.last_integrals[:-1] = obj.last_integrals[1:]
#         obj.last_integrals[-1] = integral
#         # print(integral)
#         if integral > obj.best_integral:
#             return 1.
#         elif integral > 1.01 * np.mean(obj.last_integrals):
#             print('ATTENTION: reward > 0')
#             return 0.1
#         # elif integral > last_integral:
#         #     print('ATTENTION: reward > 0')
#         #     return 1.
#         else:
#             obj.success = False
#             one_plus_curiosity = 1. + 0.005
#             return -1. * (one_plus_curiosity * obj.best_integral - integral) / (one_plus_curiosity * obj.best_integral - obj.lowest_integral)
#     return 0.
