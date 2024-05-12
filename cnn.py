import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf



class CNNET:
    
    def cnet_con(input_image2):
        
        generatorc = tf.keras.models.load_model('CNET.h5',compile=False)
        input_size = (256, 256)
        input_image2 = input_image2.resize(input_size, Image.LANCZOS)
        input_image_array2 = np.array(input_image2) / 255.0
        input_image_array2 = np.expand_dims(input_image_array2, axis=0)
        generated_image = generatorc(input_image_array2, training=False)
        generated_image = (generated_image + 1) / 2.0
        

        # Display the input image, generated image, and color-mapped image side by side
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(input_image2)

        plt.subplot(1, 2, 2)
        plt.title('CNet')
        plt.imshow(generated_image[0])

        plt.savefig('fo.png')
        
        return 'fo.png'



    