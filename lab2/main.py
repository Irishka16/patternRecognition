import numpy as np
import cv2
import time
from datetime import datetime
start_timer = datetime.now()
np.set_printoptions(threshold = np.inf)

def bw_method(image):
    height, width, _ = image.shape
    image1 = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            gray_value = int(0.36 * r + 0.53 * g + 0.11 * b)
            image1[y, x] = gray_value

    return image1

def filter_image(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    padded_image = np.pad(image, (padding_height, padding_width), mode='constant', constant_values=0)

    filtered_image = np.zeros_like(image, dtype=float)

    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            filtered_pixel = np.sum(patch * kernel)
            filtered_image[i, j] = filtered_pixel
    return filtered_image

def scalling(filtered_image):
    min_val = np.min(filtered_image)
    max_val = np.max(filtered_image)
    range_val = max_val - min_val
    scaled_image = 255 * (filtered_image - min_val) / range_val
    result_image = scaled_image.astype(np.uint8)
    return result_image

def shift_kernel_down():
    matrix = np.zeros((41, 41), dtype=np.float32)
    matrix[0, 9] = 1
    return matrix

def shift_image(image):
    kernel = shift_kernel_down()
    shifted_image = filter_image(image, kernel)
    cv2.imwrite('lab2/output/shift_10_20.jpg', shifted_image.astype(np.uint8))

def invert_cv(img, img_name):
    inverted_image = cv2.bitwise_not(img)
    cv2.imwrite(f'./lab2/output/cv_{img_name}_inverted.jpg', inverted_image)

def parse_image_with_kernels(image,kernels, use_cv=False):
    for i, kernel in enumerate(kernels):
        if use_cv:
            result = cv2.filter2D(image, -1, kernel)
        else:
            result = filter_image(image, kernel)
            if i == 0:
                result = scalling(result)  # for inversion

        prefix = 'cv' if use_cv else 'basic'
        cv2.imwrite(f'./lab2/output/{prefix}_convolved_image{i}.jpg', result)

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
        (size, size), dtype=np.float32
    )
    return kernel / np.sum(kernel)

image = cv2.imread("lab2/images/dog.jpg")
gray_image = bw_method(image)

kernels = [
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    
    gaussian_kernel(11, 1),

    (1/7) * np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32),

    np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32),

    np.array([
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]
    ], dtype=np.float32),

    np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32),

    np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
]
# parse_image_with_kernels(gray_image,kernels)
# parse_image_with_kernels(gray_image,kernels,True)
# shift_image(gray_image)
invert_cv(gray_image,"dog")
end = datetime.now() - start_timer
print(str(end) + " seconds spent")
