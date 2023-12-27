import numpy as np
import lib


def get_reward_func(params: dict):
    if params['name'] == 'each_step_base':

        def each_step_base(env_obj) -> float:
            time_step = env_obj.last_actual_time_step
            time_history, target_history = env_obj.controller.get_target_for_passed()
            count_step_measurements = int(np.floor(time_step / env_obj.controller.analyser_dt))
            integral_target_last_step = lib.integral(time_history[-count_step_measurements:], target_history[-count_step_measurements:])
            rew = integral_target_last_step / env_obj.rate_estimate / env_obj.episode_time
            return rew

        out_func = each_step_base

    elif params['name'] == 'full_ep_base':
        def full_ep_base(env_obj) -> float:
            if env_obj.end_episode:
                return env_obj.cumm_episode_target / env_obj.rate_estimate / env_obj.episode_time
            return 0.

        out_func = full_ep_base

    else:
        assert False, f'Error! Invalid type of reward function: {params["name"]}'

    return out_func
