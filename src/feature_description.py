import numpy as np

def extract_descriptors(image, keypoints, patch_size=9):
    descriptors = []
    offset = patch_size//2
    padded_img = np.pad(image,((offset,offset),(offset,offset)), mode='constant',constant_values=0)
    for (y,x) in keypoints:
        patch = padded_img[y:y+patch_size, x:x+patch_size]
        descriptor = patch.flatten()

        descriptor = (descriptor - np.mean(descriptor))/(np.std(descriptor) + 1e-10)
        descriptors.append(descriptor)
    
    return descriptors
