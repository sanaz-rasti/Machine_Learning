#!/usr/bin/env python

# This function is written to change RED color in an image into BLUE. 
# The attempt is to keep the shades of red. 

# Dependencies 
import cv2
import numpy as np 

def red_to_blue(image_path, output_path):
    # Reading image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print('Image is Null. ')
        return

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV
    bright_red_lower = np.array([0, 120, 70], dtype=np.uint8)
    bright_red_upper = np.array([10, 255, 255], dtype=np.uint8)
    dark_red_lower = np.array([170, 120, 70], dtype=np.uint8)
    dark_red_upper = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for both ranges of red
    mask1 = cv2.inRange(hsv_image, bright_red_lower, bright_red_upper)
    mask2 = cv2.inRange(hsv_image, dark_red_lower, dark_red_upper)

    # Combine the masks
    red_mask = mask1 + mask2

    # Create a copy of the image to modify
    modified_image = image.copy()

    # Iterate over each pixel detected by the mask and change red into blue
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if red_mask[i, j] > 0:  
                # Preserve the intensity but switch to blue shades
                b, g, r = image[i, j]
                # Use the red intensity as the blue channel
                blue_intensity = r  
                # Set blue with preserved shades
                modified_image[i, j] = [blue_intensity, g, 0]  

    # Save the new image
    cv2.imwrite(output_path, modified_image)

    print(f"Modified image is saved as: {output_path}")



if __name__ == "__main__":
    # Prompt input image
    image_path = input("Enter the path to the image: ").strip()

    # Prompt output image path
    output_path = "NEW_IMAGE.jpeg"

    # Call the function
    change_red_shades_to_blue(image_path, output_path)
