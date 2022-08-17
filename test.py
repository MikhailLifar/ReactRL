# import matplotlib
# matplotlib.use('Agg')
# import \
#     itertools

import matplotlib.pyplot as plt

import numpy as np
import cv2

from test_models import TestModel, generate_max_rank_matr


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


def frames_from_video_to_imgs():
    for i in range(1, 10):

        cap = cv2.VideoCapture(f'data/video/diff_shapes/{i}.MP4')
        from_folder = 115

        if not cap.isOpened():
            print("Error opening video stream or file")
        else:
            ret = True
            i1 = 0
            while ret and i1 < from_folder:
                ret, frame = cap.read()
                if ret:

                    # coef = frame.shape[0] // 650
                    # h = frame.shape[0] // coef
                    # w = frame.shape[1] // coef
                    # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    #
                    # cv2.imshow('Frame', frame)
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     break

                    h = frame.shape[0]
                    frame = frame[:, :h]
                    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

                    cv2.imwrite(f'data/imgs/from_video/{(i - 1) * from_folder + i1}.png', frame)
                    # print(i)
                    i1 += 1


def run_test_model():
    model_obj = TestModel()
    vector_to_apply = np.random.randint(-5, 5, 5)
    print(f'input-vector: {vector_to_apply}')
    for _ in range(20):
        print(f'out vector: {model_obj.update(vector_to_apply, 1.)}')


if __name__ == '__main__':
    # model testing
    # run_test_model()

    # # CV testing
    # increase_contrast()

    # M = generate_max_rank_matr(5, 5)
    # print(M)

    # practice_in_cv()

    # frames_from_video_to_imgs()

    # d = {'CO_A': 0.0001, 'CO_bias_f': 0.0, 'CO_bias_t': 0.21801443183436786, 'CO_k': 0.6283185307179586, 'O2_A': 0.0001, 'O2_bias_f': 8.623641039884324e-05, 'O2_bias_t': 0.26438923328940805, 'O2_k': 0.3141592653589793}

    # A = 0.0001
    # k = 0.6283185307179586
    # b_t = 0.21801443183436786
    # b_f = 0.0

    A = 2.e-5
    k = 0.1 * np.pi
    b_t = 0.0
    b_f = 3.e-5

    def f(t):
        res = A * np.sin(k * t + b_t) + b_f
        res[res < 0.] = 0.
        res[res > 1.e-4] = 1.e-4
        return res

    x = np.linspace(0., 500., 5000)
    y = f(x)
    plt.plot(x, y)
    plt.show()

    # M = generate_max_rank_matr(3, 3)
    # np.save('./M_3x3.npy', M, allow_pickle=False)

    pass
