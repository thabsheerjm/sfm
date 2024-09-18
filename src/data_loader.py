import os 
from PIL import Image
import numpy as np

def load_images(image_dir, dataset='colmap'):
    if dataset == 'malaga':
        image_files = [file for file in os.listdir(image_dir) if file.endswith('right.jpg')]
        image_files.sort(key=lambda x:float(x.split('_')[2]))
    elif dataset == 'colmap':
        image_files = [file for file in os.listdir(image_dir) if file.endswith('.JPG')]
        image_files.sort()
    else:
        print('select a recognized dataset')

    images, gray_imgs  = [],[]
    for files in image_files:
        image_path = os.path.join(image_dir,files)
        image = Image.open(image_path)

        gray_image = np.array(image.convert('L'))
        color_image = np.array(image)

        images.append(color_image)
        gray_imgs.append(gray_image)
        
    return gray_imgs,images