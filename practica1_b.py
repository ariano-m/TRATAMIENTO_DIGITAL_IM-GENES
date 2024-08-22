"""
    Adrián Riaño Martínez
    Tratamiento Digital de Imágenes
    Práctica 1. Apartado B
"""
from scipy.fftpack import dct, idct
import numpy as np
import cv2 as cv
import math

# matrix compression 1 / 64
one_comp_max = np.zeros((8, 8))
one_comp_max[0, 0] = 1

# matrix compression 10 / 64
ten_comp_max = np.zeros((8, 8))
for i in range(0, 4):
    ten_comp_max[i, : 4 - i] = 1

max = np.tri(8, 8, 1)

prueba3 = np.array([[1., 1., 0., 0., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])

prueba5 = np.array([[1., 1., 1., 0., 0., 0., 0., 0.],
                    [1., 1., 0., 0., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])

# quantization table
QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 58, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])

QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# https://towardsdatascience.com/image-compression-dct-method-f2bb79419587
# https://github.com/timmmGZ/JPEG-DCT-Compression/blob/master/JpegCompression.ipynb
# https://ocw.unican.es/pluginfile.php/2825/course/section/2775/Bloque_1._Tema_4._Codificacion_por_transformada_JPEG.pdf
# apuntes
def compress_dct(window_size, img, coef):
    height, width = img.shape
    ws_height, ws_width = window_size

    # ajustamos la imagen para que sea divisible entre bloques 8x8
    height = height if (height % ws_height) == 0 else math.trunc(height / ws_height) * ws_height
    width = width if (width % ws_width) == 0 else math.trunc(width / ws_width) * ws_width
    img = img[:height, :width].copy()

    new_image = np.zeros((height, width)).astype(float)
    new_image[:height, :width] = img

    for r in range(0, height, ws_height):
        for c in range(0, width, ws_width):
            block = img[r:r + ws_height, c:c + ws_width]
            block = dct2(block - 127)  # center intensity & apply dct & coef
            block = np.fix(np.divide(block, coef, out=np.zeros_like(block), where=coef != 0))  # normalization
            new_image[r:r + ws_height, c:c + ws_width] = block
    return new_image


def descompress_idct(window_size, img, coef):
    new_image = np.zeros(img.shape)
    w_height, w_width = window_size
    height, width = img.shape
    for r in range(0, height, w_height):
        for c in range(0, width, w_width):
            block = img[r:r + w_height, c:c + w_width]
            block = block * coef  # desnormalizar
            block = idct2(block)  # center intensity & apply dct & coef
            new_image[r:r + w_height, c:c + w_width] = block + 127
    return np.fix(new_image).astype(np.uint8)


def compress_and_descompress(img, window_size, coef):
    comp = compress_dct(window_size, img, coef)
    return comp, descompress_idct(window_size, comp, coef)


def main():
    img = cv.imread('inputs/lena.png', cv.IMREAD_GRAYSCALE)
    img = np.asarray(img.data, dtype=np.float32)
    window_size = (8, 8)

    comp, descmp = compress_and_descompress(img, window_size, one_comp_max)
    cv.imwrite("outputs/lena_1_64_dct.jpg", comp)
    cv.imwrite("outputs/lena_1_64_idct.jpg", descmp)

    comp, descmp = compress_and_descompress(img, window_size, ten_comp_max)
    cv.imwrite("outputs/lena_10_64_dct.jpg", comp)
    cv.imwrite("outputs/lena_10_64_idct.jpg", descmp)


if __name__ == "__main__":
    main()
