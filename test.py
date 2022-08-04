# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
import cv2

from test_models import TestModel, generate_max_rank_matr


def read_and_show():
    # reading
    img1 = cv2.imread('imgs/1.jpg', cv2.IMREAD_COLOR)
    # print(img1.shape)

    # # show image with cv2
    # # I have a problem with imshow method
    # cv2.imshow('1st try', img1)
    # cv2.resizeWindow('1st try', 340, 602)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # without convertation image would have wrong colors
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # # show image with pyplot
    # plt.imshow(img1)
    # plt.show()  # not presented in GFG tutorial!


def increase_contrast():
    img1 = cv2.imread('imgs/1.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # normal color

    contrast_alpha = 1.5
    new_img = cv2.convertScaleAbs(img1, alpha=contrast_alpha, beta=-100)
    res = np.hstack((img1, new_img))
    plt.imshow(res)
    plt.show()


def practice_in_cv():
    # reading
    img = cv2.imread('imgs/5.jpg', cv2.IMREAD_COLOR)

    # without convertation image would have wrong colors
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ksize = 7
    img = cv2.blur(img, (ksize, ksize))

    img = cv2.Canny(image=img, threshold1=20, threshold2=25)

    # show image with pyplot
    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.imshow(img)
    plt.show()  # not presented in GFG tutorial!
    plt.close(fig)


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

    M = generate_max_rank_matr(3, 3)
    np.save('./M_3x3.npy', M, allow_pickle=False)

    pass
