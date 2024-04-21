import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
os.chdir(r"D:\Download\Năm 2 kì 2\Xử lý ảnh\BT\HW2")

def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    # dividing to get the size of padded part
    pad_size = int(filter_size/2)
    h, w = img.shape

    # create a temporary image with the size like that
    padded_img = np.zeros((h + 2 * pad_size, w + 2 * pad_size), dtype=img.dtype)

    # copying original image into padded_img
    padded_img[pad_size : -pad_size, pad_size : -pad_size] = img

    # pad rows
    padded_img[:pad_size, pad_size : - pad_size] = img[0,:]
    padded_img[- pad_size:, pad_size : - pad_size] = img[h - 1, :]

    # pad columns
    padded_img[pad_size : - pad_size, :pad_size] = img[:,0].reshape(-1, 1)
    padded_img[pad_size : - pad_size, - pad_size:] = img[:,w - 1].reshape(-1, 1)

    # pad 4 corners
    padded_img[:pad_size, :pad_size] = img[0,0]
    padded_img[:pad_size, - pad_size:] = img[0,-1]
    padded_img[- pad_size:, :pad_size] = img[-1,0]
    padded_img[- pad_size:, - pad_size:] = img[-1,-1]

    return padded_img

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    h, w = img.shape
    # create a temporary image with the size like that
    mean_img = np.zeros((h,w), dtype=img.dtype)
    padded_img = padding_img(img)
    # select filters
    pad_size = int(filter_size/2)
    mean_win = np.full((filter_size, filter_size), 1/(filter_size * filter_size), dtype=float)
    for i in range(pad_size, h + pad_size):
        for j in range(pad_size, w + pad_size):
            cur_sec = padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            res = np.multiply(cur_sec, mean_win)
            mean_img[i - pad_size, j - pad_size] = np.sum(res)

    return mean_img

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    h, w = img.shape
    # create a temporary image with the size like that
    med_img = np.zeros((h,w), dtype=img.dtype)
    padded_img = padding_img(img)
    pad_size = int(filter_size/2)
    for i in range(pad_size, h + pad_size):
        for j in range(pad_size, w + pad_size):
            cur_sec = padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            res = np.median(cur_sec)
            med_img[i - pad_size, j - pad_size] = res

    return med_img


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    m,n = gt_img.shape
    tmp = np.subtract(gt_img, smooth_img) * np.subtract(gt_img, smooth_img)
    mse = np.sum(tmp) * (1/(m*n))
    if (mse == 0):
        return np.Infinity
    psnr = 10 * np.log10((255 ** 2)/mse)
    return psnr


def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

