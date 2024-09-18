#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    image_H, image_W = image.shape
    kernel_H, kernel_W = kernel.shape
   
    pad_H = kernel_H // 2
    pad_W = kernel_W // 2

    padded_image = np.pad(image, ((pad_H, pad_H), (pad_W, pad_W)), mode='constant', constant_values=0)
    
    output = np.zeros_like(image)

    for i in range(image_H):
        for j in range(image_W):
            selected_region = padded_image[i:i + kernel_H, j:j + kernel_W]  # correlation
            output[i, j] = np.sum(selected_region * kernel)
        
    return output


def compute_image_gradients(image, kernel_='sobel'):
    if kernel_ == 'sobel':
        G_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    elif kernel_ == 'finite-difference':
        G_x = np.array([[-1, 1]])
    else:
        raise ValueError(f"{kernel_} not a recognized kernel")

    G_y = G_x.transpose()
    
    Ix = convolve(image, G_x)
    Iy = convolve(image, G_y)

    return Ix, Iy


def compute_harris_response(Ix, Iy, window_size, k):
    """
    Compute Harris corner response using the sliding window approach from the shared code
    """
    Ixx = Ix**2
    Ixy = Ix * Iy
    Iyy = Iy**2

    height, width = Ix.shape
    offset = int(window_size / 2)
    
    R = np.zeros_like(Ix, dtype=float)

    # Apply sliding window to compute corner response
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Sum of squares in the window
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]

            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Harris response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            R[y, x] = det - k * (trace**2)

    return R


def is_local_maxima(R, pixel, window_size=3):
    '''
    Check whether pixel at (x, y) is at a local maximum
    '''
    half_size = window_size // 2
    y, x = pixel  # Unpack pixel tuple

    # Ensure the window doesn't go out of bounds
    if y - half_size < 0 or y + half_size >= R.shape[0] or x - half_size < 0 or x + half_size >= R.shape[1]:
        return False  # If out of bounds, return False

    local_window = R[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]

    return R[y, x] == np.max(local_window)  # Compare the value of (y, x)



def extract_keypoints(R, threshold):
    """
    Extract keypoints that are above the threshold and are local maxima
    """
    keypoints = np.argwhere(R > threshold)

    keypoints_list = []
    for y, x in keypoints:
        if is_local_maxima(R, (x, y)):
            keypoints_list.append((x, y))
    
    return keypoints_list


def harris_corner_detector(image, window_size=3, k=0.04, threshold=10000):  # Adjusted threshold
    Ix, Iy = compute_image_gradients(image)
    R = compute_harris_response(Ix, Iy, window_size, k)

    # Visualize Harris response matrix before extracting keypoints
    plt.imshow(R, cmap='hot')
    plt.title("Harris Response Matrix")
    plt.colorbar()
    plt.show()

    keypoints = extract_keypoints(R, threshold)

    return keypoints


# Example usage:
# Assuming you have a way to load the image
import data_loader  # Assuming you have this module to load your images

gray, images = data_loader.load_images('data/Images/')

for img in gray:
    keypoints = harris_corner_detector(img, k=0.1, threshold=10000)  # Adjusted threshold
    
    print(keypoints)
    plt.imshow(img, cmap='gray')
    for y, x in keypoints:
        plt.plot(x, y, 'ro', markersize=3)
    plt.show()

    break
