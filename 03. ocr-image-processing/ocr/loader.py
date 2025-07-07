# loader.py - For loading images
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            images.append(os.path.join(folder, filename))
        return images