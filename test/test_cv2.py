import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # show_droplet_color('data/imgs/220909_from_vertical/right_side/0.png', [498, 464, 520, 485])
    # increase_contrast()
    # practice_in_cv()
    # frames_from_video_to_imgs('data/imgs/220909_from_video')
    # take_from_video('./data/video/220909_vertical.mp4', './data/imgs/220909_from_vertical_right_side', 200, freqency=10)

    pass
