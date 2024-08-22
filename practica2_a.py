"""
    Adrián Riaño Martínez
    Tratamiento Digital de Imágenes
    Práctica 2. Apartado A
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math


def adjust_imag_to_kernel(img, kernel):
    height, width = img.shape[:2]
    height = height if (height % kernel) == 0 else math.trunc(height / kernel) * kernel
    width = width if (width % kernel) == 0 else math.trunc(width / kernel) * kernel
    return img[:height, :width].copy()


def compute_disparity_map(left, right, kernel=11, desp=50):
    def calculate_ssd(block1, block2):
        return 1 / 1 + np.sum(np.power(np.absolute(block1 - block2), 2))

    left = adjust_imag_to_kernel(left, kernel)
    right = adjust_imag_to_kernel(right, kernel)

    height, width = left.shape[:2]
    diparity_map = np.zeros_like(left, dtype="float64")
    for row in range(kernel, height - kernel):
        for column in range(kernel, width - kernel):
            ssd, disp_l = [65025], [0]  # 255^2 = 65025
            for distc in range(min(desp, column)):
                left_block = left[row:row + kernel, column:column + kernel]
                right_block = right[row:row + kernel, column - distc:column + kernel - distc]

                tmp_ssd = calculate_ssd(left_block, right_block)
                ssd.append(tmp_ssd)
                disp_l.append(distc + 1)

            diparity_map[row:row + kernel, column:column + kernel] = disp_l[ssd.index(min(ssd))]

    return diparity_map[kernel:-kernel, kernel:-kernel]


def save_fig(path, figure):
    plt.imshow(figure, 'gray')
    plt.savefig(f'{path}_gray_plt.jpg')
    cv.imwrite(f'{path}_gray_cv.jpg', figure)


def main():
    left = cv.imread("./inputs/teddy/im2.png")
    right = cv.imread("./inputs/teddy/im6.png")
    left = cv.imread("./inputs/mask/im2.png")
    right = cv.imread("./inputs/mask/im6.png")

    params = {
        'left': cv.cvtColor(left, cv.COLOR_BGR2GRAY),
        'right': cv.cvtColor(right, cv.COLOR_BGR2GRAY),
        'kernel': 11,
        'desp': 50
    }
    disparity_img = compute_disparity_map(**params)
    save_fig('./disparity_mask', disparity_img)


if __name__ == "__main__":
    main()
