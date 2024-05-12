import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # Add this import
import tensorflow as tf
import cv2

class OPTI:
    def color_map_images(self, images, target_size=(256, 256)):
        # Create a copy of the input images to store the mapped colors
        mapped_images = np.copy(images)

        # Define the color mappings with darker colors
        red_threshold = 0.6  # Adjust as needed
        yellow_threshold = 0.5  # Adjust as needed
        green_color = [0, 1.0, 0]  # Dark green
        red_color = [0.8, 0, 0]  # Darker red (adjust the first value)
        orange_color = [0.8, 0.4, 0]  # Dark orange
        black_threshold = 0.1

        # Resize the images to a larger target size
        mapped_images = zoom(mapped_images, (1, target_size[0] / mapped_images.shape[1], target_size[1] / mapped_images.shape[2], 1), order=5)

        # White to Red Mapping
        white_mask = np.all(mapped_images > red_threshold, axis=-1)
        mapped_images[white_mask] = red_color

        # Red, Yellow, Orange to Orange Mapping
        ryo_mask = np.any([
            (mapped_images[:, :, :, 0] >= red_threshold) & (mapped_images[:, :, :, 1] < yellow_threshold),  # Red
            (mapped_images[:, :, :, 0] >= red_threshold) & (mapped_images[:, :, :, 1] >= yellow_threshold),  # Orange
            (mapped_images[:, :, :, 1] >= yellow_threshold) & (mapped_images[:, :, :, 0] < red_threshold)  # Yellow
        ], axis=0)

        # Ensure that blue is not mapped to red
        ryo_mask &= mapped_images[:, :, :, 2] < red_threshold

        # Modify the condition to map orange
        orange_mask = ryo_mask & (mapped_images[:, :, :, 1] >= yellow_threshold)
        orange_color_array = np.array(orange_color)  # Convert the list to a NumPy array
        mapped_images[orange_mask] = orange_color_array

        # Handle black areas
        black_mask = np.all(mapped_images <= black_threshold, axis=-1)
        mapped_images[black_mask] = [0, 0, 0]

        # All Other Colors to Green Mapping
        other_colors_mask = np.logical_not(white_mask | ryo_mask | black_mask)
        mapped_images[other_colors_mask] = green_color

        return mapped_images
    
    def opti_con(self, input_image2):
        input_size = (128, 128)
        target_size = (256, 256)
        generatoropti = tf.keras.models.load_model('optinet.h5')
        input_image2 = input_image2.resize(input_size, Image.LANCZOS)
        input_image_array2 = np.array(input_image2) / 255.0
        input_image_array2 = np.expand_dims(input_image_array2, axis=0)
        #generated_imageopti = generatoropti(input_image_array2, training=False)
        #generated_imageopti = (generated_imageopti + 1) / 2.0
        output_image_opti = self.color_map_images(input_image_array2, target_size)

        # Convert the image to HSV
        original_image = output_image_opti[0]

        # Convert the image to 8-bit unsigned integer format
        original_image = cv2.convertScaleAbs(original_image)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Define a range for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Threshold the image to get a binary mask of red pixels
        red_mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

        # Define a second range for red color (due to hue wrapping around in the HSV color space)
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])

        # Threshold the image to get a binary mask of red pixels
        red_mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

        # Combine both red masks
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Calculate the percentage of red pixels
        total_pixels = original_image.size
        red_pixels = cv2.countNonZero(red_mask)
        percentage_red = (red_pixels / total_pixels) * 100

        print(f"Percentage of red pixels: {percentage_red:.2f}%")

        # Rescale the output image back to [0, 255] (uint8)
        output_image_opti = (output_image_opti * 255).astype(np.uint8)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(input_image2)

        plt.subplot(1, 2, 2)
        plt.title('Optinet')
        plt.imshow(output_image_opti[0])
        plt.savefig('foo.png')
        return 'foo.png'
