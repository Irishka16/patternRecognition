import cv2
import numpy as np

def convert_to_black_and_white_and_get_mask(input_image_path, output_bw_path, output_mask_path):
    # Load the image using OpenCV
    image = cv2.imread(input_image_path)

    # Check if the image was loaded successfully
    if image is not None:
    # Get the height and width of the image
        height, width, _ = image.shape

        # Create an empty black and white image of the same size
        black_white_image = np.zeros((height, width), dtype=np.uint8)

        # Iterate over each pixel in the image
        for y in range(height):
            for x in range(width):
                # Get the pixel value at the current position
                pixel = image[y, x]

                # Calculate the grayscale value using the formula: gray = 0.299*R + 0.587*G + 0.114*B
                gray = int(0.36 * pixel[2] + 0.53 * pixel[1] + 0.11 * pixel[0])

                # Set the grayscale value for the corresponding pixel in the black and white image
                black_white_image[y, x] = gray

        # Save the black and white image
        cv2.imwrite(output_bw_path, black_white_image)

        # Apply a global threshold to create a binary mask
        _, binary_mask = cv2.threshold(black_white_image, 248, 255, cv2.THRESH_BINARY_INV)

        # Save the binary mask
        cv2.imwrite(output_mask_path, binary_mask)

        print("Image converted to grayscale and saved as", output_bw_path)
        print("Binary mask saved as", output_mask_path)
    else:
        print("Failed to load the image.")
    return image,binary_mask,black_white_image

def cut_object_from_image(input_image_path, binary_mask_path, output_cut_path):
    # Load the input image and binary mask using OpenCV
    input_image = cv2.imread(input_image_path)
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

    # Check if the images were loaded successfully
    if input_image is not None and binary_mask is not None:
        # Create an empty image with the same dimensions as the input image
        output_cut_image = np.zeros_like(input_image)

        # Copy the object from the input image where the mask is white
        output_cut_image = cv2.bitwise_and(input_image, input_image, mask=binary_mask)

        # Save the result with the object extracted
        cv2.imwrite(output_cut_path, output_cut_image)

        print("Object cut and saved as", output_cut_path)
    else:
        print("Failed to load the input image or binary mask.")

for i in ['dog.jpg','loremipsum.png','pear.jpg','text.png']:
    image = 'lab1/input/'+i
    output_bw_path='lab1/output/black_white_'+i
    output_mask_path='lab1/output/binary_mask_'+i
    output_extracted_path='lab1/output/extracted_'+i

    _, masked, bw_image = convert_to_black_and_white_and_get_mask(image,output_bw_path,output_mask_path)
    cut_object_from_image(image,output_mask_path,output_extracted_path)