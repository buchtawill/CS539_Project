import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

image_dir = "../colorization_data/images/color"

color_tensors_list = []
bw_tensors_list = []

def preprocess_color_image(image):
    image = np.array(image) / 255.0
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    return image_tensor

def preprocess_bw_image(image):
    image = image.convert("L")
    image = np.expand_dims(np.array(image) / 255.0, axis=-1)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    return image_tensor

for filename in tqdm(os.listdir(image_dir)):
    color_image = Image.open(os.path.join(image_dir, filename))
    
    color_tensor = preprocess_color_image(color_image)
    bw_tensor = preprocess_bw_image(color_image)
    
    color_tensors_list.append(color_tensor)
    bw_tensors_list.append(bw_tensor)

color_tensors = tf.stack(color_tensors_list)
bw_tensors = tf.stack(bw_tensors_list)

print(color_tensors.shape)
print(bw_tensors.shape)

np.save('./color_tensors.npy', color_tensors.numpy())
np.save('./bw_tensors.npy', bw_tensors.numpy())