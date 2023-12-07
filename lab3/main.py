import cv2 as cv
import numpy as np
from enum import Enum

path = './lab3/'
image = cv.imread(path + 'images/text.png')

kernel_matrix = np.ones((2, 2))

class MorphologyMode(Enum):
    EROSION = 'erosion'
    DILATION = 'dilation'



def resizing_img(img: np.ndarray, percent: float) -> np.ndarray:
    return cv.resize(img, (np.array(img.shape[:2]) * percent)[::-1].astype(int))


def pading_img(img: np.ndarray, width: float) -> np.ndarray:
    return np.pad(img, ((width, width),
                        (width, width),
                        (0, 0)),
                  mode='edge')


def morphology_operation(img: np.ndarray, kernel: np.ndarray, mode='errosion') -> np.ndarray:
    kw = kernel.shape[0]  # kernel width
    img = (img + 1) * (-1) % 256  # inverse img
    padded_img = pading_img(img, int(kw / 2))
    result_img = np.empty(img.shape)
    # check mode
    if mode == MorphologyMode.EROSION:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if np.vdot(padded_img[x:x + kw, y:y + kw, 0], kernel) == kernel.sum() * 255:
                    result_img[x, y] = [255, 255, 255]
                else:
                    result_img[x, y] = [0, 0, 0]
    elif mode == MorphologyMode.DILATION:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if np.vdot(padded_img[x:x + kw, y:y + kw, 0], kernel) > 0:
                    result_img[x, y] = [255, 255, 255]
                else:
                    result_img[x, y] = [0, 0, 0]
    else:
        raise ValueError("Invalid morphology mode")
    return ((result_img + 1) * (-1) % 256).astype('uint8')  # inverse back



def closing(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return morphology_operation(morphology_operation(img, kernel, MorphologyMode.DILATION), kernel, MorphologyMode.EROSION)

def opening(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return morphology_operation(morphology_operation(img, kernel, MorphologyMode.EROSION), kernel, MorphologyMode.DILATION)

resized_img = resizing_img(image, 0.3)
# image = bw_method(image)
# image = binarization_on_bw_image(image)

cv.imwrite(path + 'output/img_e.jpg', morphology_operation(image, kernel_matrix, MorphologyMode.EROSION))
cv.imwrite(path + 'output/img_d.jpg', morphology_operation(image, kernel_matrix, MorphologyMode.DILATION))
cv.imwrite(path + 'output/img_closing.jpg', closing(image, kernel_matrix))
cv.imwrite(path + 'output/img_opening.jpg', opening(image, kernel_matrix))

cv.waitKey(0)
cv.destroyAllWindows()
