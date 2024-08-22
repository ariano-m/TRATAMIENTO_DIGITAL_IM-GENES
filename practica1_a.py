"""
    Adrián Riaño Martínez
    Tratamiento Digital de Imágenes
    Práctica 1. Apartado A
"""
from numpy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math
import cv2


def highpassfilter(size, frec_threshold, filter_order):
    """
        H(u,v)_pa =  1 - H(u,v)_pb
        Filtro de Butterworth de orden n:
        frec_threshold:
            H(u,v) = 1 / 1 + [D(u,v) / D_0]^2n
            D(u,v) = [(u - N / 2)² + (v - M / 2)²]^(1/2)

            u, v = range(rows), range(columns)
            N = rows
            M = columns
            D_0 = frec_threshold
            n = filter_order


    :param size: size of filter in row & column number (coincide con el tamaño de la transformada de la imagen)
    :param frec_threshold: frequency threshold for filter [0, 0.5]
    :param filter_order: The order of the filter
    :return:
    """

    rows, columns = size
    n = filter_order
    D_0 = frec_threshold

    matrix = np.zeros((rows, columns))
    for u in range(rows):
        for v in range(columns):
            component_1 = (u - rows / 2) ** 2  # (u - N / 2)²
            component_2 = (v - columns / 2) ** 2  # (v - M / 2)²
            D_uv = np.sqrt(component_1 + component_2)  # [(u - N / 2)² + (v - M / 2)²]^(1/2)
            H = 1 / (1 + math.pow(D_uv / (D_0 * (rows / 2)), 2 * n))  # 1 / 1 + [D(u,v) / D_0]^2n
            matrix[u][v] = 1 - H
    return matrix


def apply_filter(img_fft, filter_):
    return img_fft * filter_


def imfft(image):
    f_transform = fft2(image)  # apply 2d FDT
    return fftshift(f_transform)  # center transform


def iimfft(image):
    return ifft2(image)
    #return numpy.fft.ifftshift(image)


def dump_images(img_l):
    fig, axarr = plt.subplots(1, 3, figsize=(12, 12))
    f, hist_axarr = plt.subplots(1, 3, figsize=(12, 4))
    for idx, (i, title) in enumerate(img_l):
        axarr[idx].imshow(i, cmap="gray")
        axarr[idx].set_title(title)
        hist_axarr[idx].hist(i.flatten(), bins=range(256), alpha=0.5, density=False)
        hist_axarr[idx].set_title(title)

    f.tight_layout()
    f.savefig("practica1_a_beatles_hist.png")


    fig.tight_layout()
    fig.savefig("practica1_beatles_a.png")


def main():
    image = cv.imread('inputs/beatles.jpg', cv.IMREAD_GRAYSCALE)
    size = image.shape
    frec_threshold, filter_order = 0.9, 6

    filter_ = highpassfilter(size, frec_threshold, filter_order)
    filtered_img = apply_filter(imfft(image), filter_)
    cv2.imshow("dd", filtered_img.astype("uint8"))
    cv2.waitKey(0)
    imgs = [(image, "Original"),
            (filtered_img.astype("uint8"), 'Highpass filtered'),
            (iimfft(filtered_img).astype("uint8"), "Filtered ifft2")]
    #dump_images(imgs)


if __name__ == "__main__":
    main()
