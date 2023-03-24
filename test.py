# import matplotlib
# matplotlib.use('Agg')
# import itertools
import os

import matplotlib.pyplot as plt

# import numpy as np
import cv2

from test_models import TestModel, generate_max_rank_matr

from ProcessController import ProcessController
import test_models as models
from targets_metrics import *


def read_and_show():
    # reading
    # img1 = cv2.imread('imgs/1.jpg', cv2.IMREAD_COLOR)
    # print(img1.shape)

    # # show image with cv2
    # # I have a problem with imshow method
    # cv2.imshow('1st try', img1)
    # cv2.resizeWindow('1st try', 340, 602)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # without convertation image would have wrong colors
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # # show image with pyplot
    # plt.imshow(img1)
    # plt.show()  # not presented in GFG tutorial!

    pass


def increase_contrast():
    img1 = cv2.imread('imgs/1.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # normal color

    contrast_alpha = 1.5
    new_img = cv2.convertScaleAbs(img1, alpha=contrast_alpha, beta=-100)
    res = np.hstack((img1, new_img))
    plt.imshow(res)
    plt.show()


def show_with_pyplolt(img: np.ndarray):
    # show image with pyplot
    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.imshow(img)
    plt.show()  # not presented in GFG tutorial!
    plt.close(fig)


def show_with_resize(img: np.ndarray, name: str):
    # show image with pyplot
    h = img.shape[0] // 4
    w = img.shape[1] // 4
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized)


def practice_in_cv():
    # # reading
    # img = cv2.imread('imgs/5.jpg', cv2.IMREAD_COLOR)
    # # to grayscale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # read to grayscale
    img = cv2.imread('imgs/5.jpg', 0)

    # resize to some standard
    coef = img.shape[0] // 750
    h = img.shape[0] // coef
    w = img.shape[1] // coef
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    img = cv2.equalizeHist(img)

    # show_with_resize(img, 'gray')
    cv2.imshow('gray', img)

    ksize = 7
    # median = cv2.medianBlur(img, ksize)
    # show_with_resize(median, 'median')
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    # cv2.imshow('Gauss', img)

    # img = cv2.erode(img, None, iterations=5)
    # show_with_resize(img, 'erode')
    it = 3
    img = cv2.dilate(img, None, iterations=it)
    img = cv2.erode(img, None, iterations=it)
    # show_with_resize(img, 'erode+dilate')

    # cv2.imshow('intermediate', img)

    min_thresh = 30
    max_thresh = min_thresh + 100
    img = cv2.Canny(image=img, threshold1=min_thresh, threshold2=max_thresh, L2gradient=True)
    #
    cv2.imshow('final', img)
    cv2.waitKey(0)


def take_from_video(source_path, dest_folder_path, from_video, skip_frames=0, freqency=30, first_number=0):
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
    else:

        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)

        prob = 1. / freqency

        ret = True
        i1 = 0
        while ret and (i1 < skip_frames + from_video):
            ret, frame = cap.read()
            if i1 < skip_frames:
                i1 += 1
                continue
            if ret and (np.random.random() < prob):

                # coef = frame.shape[0] // 650
                # h = frame.shape[0] // coef
                # w = frame.shape[1] // coef
                # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                #
                # cv2.imshow('Frame', frame)
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break

                # cut frame to be quadratic
                h = frame.shape[0]
                frame = frame[:, -h:]  # cut left side
                # frame = frame[:, :h]  # cut right side

                frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

                cv2.imwrite(f'{dest_folder_path}/{first_number + (i1 - skip_frames)}.png', frame)
                i1 += 1


def frames_from_video_to_imgs(dest_folder_path, skip_frames=0, from_video=115, take_frequency=15):
    # images for testing
    take_from_video(f'data/video/220822.mp4', 'data/imgs/test', 115, freqency=10)

    # images for training
    # for i in range(1, 10):
    #     take_from_video(f'data/video/diff_shapes/{i}.MP4', dest_folder_path, from_video, skip_frames, take_frequency,
    #                     (i - 1) * from_video)


def run_test_model():
    model_obj = TestModel()
    vector_to_apply = np.random.randint(-5, 5, 5)
    print(f'input-vector: {vector_to_apply}')
    for _ in range(20):
        print(f'out vector: {model_obj.update(vector_to_apply, 1.)}')


def show_droplet_color(filepath, box):
    img = cv2.imread(filepath)

    img_part = img[box[1]:box[3], box[0]:box[2]]

    average_color = np.mean(img_part[img_part.shape[0]//3: 2 * img_part.shape[0]//3, img_part.shape[1]//3: 2 * img_part.shape[1]//3],
                            axis=(0, 1))
    average_color = average_color.astype('uint8')

    cv2.imshow('droplet', img_part)

    color_img = np.tile(np.array(average_color), (640, 640, 1))
    cv2.imshow('color', color_img)

    cv2.waitKey(0)


def benchmark_RL_agents():
    import tensorforce
    import tensorforce.environments
    # import tensorforce.agents
    import run_RL
    # import tensorforce.execution as tf_exec

    agent_name = 'vpg'
    env_name = 'CartPole'

    # env = tensorforce.environments.Environment.create(
    #     environment='gym', level=env_name, max_episode_timesteps=500,
    # )
    env = tensorforce.environments.OpenAIGym(f'{env_name}-v0', visualize=False)
    agent = run_RL.create_tforce_agent(env, agent_name)

    # runner = tf_exec.Runner(agent=agent, environment=env, max_episode_timesteps=500)
    #
    # runner.run(num_episodes=300)
    # runner.run(num_episodes=20, evaluation=True)
    # runner.close()

    folder = f'./benchmark_RL/{agent_name}_{env_name}'
    os.makedirs(folder, exist_ok=False)

    num_episodes = 2000
    cum_rewards = np.zeros(num_episodes)
    for i in range(num_episodes):
        states = env.reset()
        terminal = False
        sum_reward = 0.
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            sum_reward += reward
            if terminal:
                cum_rewards[i] = sum_reward

    def plot_rews(fname: str):
        fig, ax = plt.subplots()
        ax.plot(np.arange(num_episodes) + 1, cum_rewards, 'b-d')
        ax.set_xlabel('Episode number')
        ax.set_ylabel('Cumulative reward')
        fig.savefig(f'{folder}/{fname}')
        plt.close(fig)

    plot_rews('training_rews.png')

    env = tensorforce.environments.OpenAIGym(f'{env_name}-v0', visualize=True)
    num_episodes = 30
    cum_rewards = np.zeros(num_episodes)
    for i in range(num_episodes):
        states = env.reset()
        terminal = False
        sum_reward = 0.
        while not terminal:
            actions = agent.act(states=states, independent=True, deterministic=True)
            states, terminal, reward = env.execute(actions=actions)
            # agent.observe(terminal=terminal, reward=reward)
            sum_reward += reward
            if terminal:
                cum_rewards[i] = sum_reward

    plot_rews('eval_rews.png')


def working_with_csv_test():
    import lib

    csv_path = './PC_plots/Ziff/Ziff_2023_3_10__2/Ziff_summarize_CO2.csv'
    ops, df = lib.read_plottof_csv(csv_path, ret_ops=True)
    lib.plot_from_file({'label': 'Average CO2 prod. rate', 'linestyle': 'solid',
                        'marker': 'h', 'c': 'purple',
                        'twin': True, },
                       {'label': 'Average O2 coverage', 'linestyle': (0, (1, 1)),
                        'marker': 'x', 'c': 'blue'},
                       {'label': 'Average CO coverage', 'linestyle': (0, (5, 5)),
                        'marker': '+', 'c': 'red'},
                       csvFileName=csv_path,
                       fileName='./test.png',
                       xlabel=f'O2, {1.e4:.4g} Pa', ylabel='coverages', title='Summarize across benchmark',
                       twin_params={'ylabel': 'CO2 prod. rate'},
                       )


def test_policy():
    import predefined_policies as pp

    nsteps = 5
    policy = pp.AnyStepPolicy(nsteps, dict())

    params = {str(i): 10 + i for i in range(1, nsteps + 1)}
    tparams = {f't{i}': i for i in range(1, nsteps + 1)}
    policy.set_policy({**params, **tparams})
    policy(np.linspace(0, 50, 51))


def test_jobs_functions():
    import multiple_jobs_functions as jobfuncs

    size = [10, 10]
    PC_obj = ProcessController(models.KMC_CO_O2_Pt_Model((*size, 1), log_on=False,
                                                          O2_top=1.1e5, CO_top=1.1e5,
                                                          CO2_rate_top=1.4e6, CO2_count_top=1.e4,
                                                          T=373.),
                               analyser_dt=0.25e-7,
                               target_func_to_maximize=get_target_func('CO2_count'),
                               target_func_name='CO2_count',
                               target_int_or_sum='sum',
                               RESOLUTION=1,  # always should be 1 if we use KMC, otherwise we will get wrong results!
                               supposed_step_count=100,  # memory controlling parameters
                               supposed_exp_time=1.e-5)
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

    jobfuncs.run_jobs_list(
        jobfuncs.one_turn_search_iteration,
        **(jobfuncs.jobs_list_from_grid(
            (0.25, 0.2, 0.1),
            (3.e-8, 5.e-8, 1.e-7, 2.e-7),
            names=('x1', 't0')
        )),
        names_groups=(),
        const_params={'x0': 0.3, 't1': 2.e-7},
        sort_iterations_by='Total_CO2_Count',
        PC=PC_obj,
        python_interpreter='../RL_10_21/venv/bin/python',
        out_fold_path='PC_plots/220324_one_turn_search',
        separate_folds=False,
    )


if __name__ == '__main__':
    # working_with_csv_test()

    # test_policy()

    test_jobs_functions()

    # benchmark_RL_agents()

    # show_droplet_color('data/imgs/220909_from_vertical/right_side/0.png', [498, 464, 520, 485])

    # model testing
    # run_test_model()

    # # CV testing
    # increase_contrast()

    # M = generate_max_rank_matr(5, 5)
    # print(M)

    # practice_in_cv()

    # frames_from_video_to_imgs('data/imgs/220909_from_video')
    # take_from_video('./data/video/220909_vertical.mp4', './data/imgs/220909_from_vertical_right_side', 200, freqency=10)

    # d = {'CO_A': 0.0001, 'CO_bias_f': 0.0, 'CO_bias_t': 0.21801443183436786, 'CO_k': 0.6283185307179586, 'O2_A': 0.0001, 'O2_bias_f': 8.623641039884324e-05, 'O2_bias_t': 0.26438923328940805, 'O2_k': 0.3141592653589793}

    # A = 0.0001
    # k = 0.6283185307179586
    # b_t = 0.21801443183436786
    # b_f = 0.0

    # A = 2.e-5
    # k = 0.1 * np.pi
    # b_t = 0.0
    # b_f = 3.e-5
    #
    # def f(t):
    #     res = A * np.sin(k * t + b_t) + b_f
    #     res[res < 0.] = 0.
    #     res[res > 1.e-4] = 1.e-4
    #     return res
    #
    # x = np.linspace(0., 500., 5000)
    # y = f(x)
    # plt.plot(x, y)
    # plt.show()

    # M = generate_max_rank_matr(3, 3)
    # np.save('./M_3x3.npy', M, allow_pickle=False)

    pass
