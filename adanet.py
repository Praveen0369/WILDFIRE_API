import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


class ADA:
    
    def adanet_con(input_image):
        
        generatorada = tf.keras.models.load_model('ADANET.h5')
        input_size = (256, 256)
        input_image = input_image.resize(input_size, Image.LANCZOS)
        input_image_array = np.array(input_image) / 255.0  # Normalize to [0, 1]
        input_image_array = np.expand_dims(input_image_array, axis=0)
        generated_imageada = generatorada(input_image_array, training=False)
        generated_imageada = (generated_imageada + 1) / 2.0
        brightness_factor = 2.2
        lower_red = [0.7, 0, 0]
        upper_red = [1, 0.7, 0]
        mask_red = tf.logical_and(generated_imageada >= lower_red, generated_imageada <= upper_red)
        brightened_image = tf.where(mask_red, generated_imageada * brightness_factor, generated_imageada)
        darkening_factor = 1.7
        output_image_ada = tf.where(mask_red, brightened_image * darkening_factor, brightened_image)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(input_image)

        plt.subplot(1, 2, 2)
        plt.title('AdaNet')
        plt.imshow(output_image_ada[0])
        plt.savefig('foio.png')
        return 'foio.png'

    

    