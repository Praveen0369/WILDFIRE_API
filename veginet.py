import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
class VGG:
    def con_veg(inputimage3):
        # Read the image
        #img = cv2.imread(inputimage3)
        #imga=Image.open(inputimage3)
        img_array = np.array(inputimage3)

        #if img is None:
           # raise ValueError("Image not found or unable to load the image.")

        # Increase brightness
        brightness_factor = 1.6  # You can adjust this value as needed
        brighter_img = np.clip(img_array* brightness_factor, 0, 255).astype(np.uint8)

        # Convert the brighter image to grayscale
        imgray = cv2.cvtColor(brighter_img, cv2.COLOR_BGR2GRAY)

        # Convert to HSV
        hsv = cv2.cvtColor(brighter_img, cv2.COLOR_BGR2HSV)

        # Mask of green (30, 40, 40) ~ (140, 255, 255)
        mask = cv2.inRange(hsv, (30, 40, 40), (80, 255, 255))

        # Slice the green
        imask = mask > 0
        green = np.zeros_like(brighter_img, np.uint8)
        green[imask] = brighter_img[imask]
        green_grayed = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
        dimensions = green_grayed.shape

        count = np.count_nonzero(green_grayed)

        print("Count is " + str(count))
        total_pixels = dimensions[0] * dimensions[1]

        # Write the brighter image
        cv2.imwrite("veg.png", green)

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(brighter_img, cv2.COLOR_BGR2HSV)

        # Define a range for green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Threshold the image to get a binary mask of green pixels
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Calculate the percentage of green pixels
        green_pixels = cv2.countNonZero(green_mask)

        percentage_green = (green_pixels / total_pixels) * 100

        print(f"Percentage of green pixels: {percentage_green:.2f}%")

        # Display the images
        image = cv2.imread("veg.png")
        width, height = 256, 256
        resized_img = cv2.resize(img_array, (width, height))
        resized_image = cv2.resize(image, (width, height))

        
        plt.figure(figsize=(7, 4))

        # Display input image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        #plt.title('Input Image')
        plt.text(0, -10, f"Count of green pixels: {count}", fontsize=10, ha='left')
        plt.text(0, -30, f"Percentage of green pixels: {percentage_green:.2f}%", fontsize=10, ha='left')

        # Display mask
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Vegitation Recovery')
        

        plt.tight_layout()
        plt.savefig('foii.png')

        return 'foii.png'
    
