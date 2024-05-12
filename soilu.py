import numpy as np
from PIL import Image
import os
import random


class SOIL:
    
    def con_soil(folder_path):

        files = os.listdir(folder_path)
        # Filter only the image files (assuming all are images)
        image_files = [file for file in files if file.endswith(".png")]
        # Select a random image filename
        random_image = random.choice(image_files)
        # Return the full path to the randomly selected image
        return os.path.join(folder_path, random_image)
        
        
        

