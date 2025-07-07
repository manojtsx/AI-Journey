# processor.py - For preprocessing images
import cv2
import numpy as np

def preprocess_image(image):
    # Convert the input image from BGR (Blue, Green, Red) color space to grayscale
    # Grayscale simplifies the image to one channel (intensity) for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the scaling factor to increase image size by 150% for better detail
    scale_percent = 150
    # Calculate the new width based on the original width and scale percent
    width = int(gray.shape[1] * scale_percent / 100)
    # Calculate the new height based on the original height and scale percent
    height = int(gray.shape[0] * scale_percent / 100)
    # Resize the grayscale image to the new dimensions using linear interpolation
    # This improves text visibility for OCR by adding more pixels
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian blur to the resized image to reduce noise
    # A 5x5 kernel size smooths out small speckles, helping with later thresholding
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Define a sharpening kernel to enhance image edges
    # The array [[0,-1,0], [-1,5,-1], [0,-1,0]] boosts the center pixel and reduces neighbors
    # This makes text boundaries clearer for better recognition
    kernel_sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    # Apply the sharpening kernel to the blurred image using filter2D
    # -1 indicates the depth is the same as the input image
    sharpened = cv2.filter2D(blur, -1, kernel_sharpen)

    # Apply adaptive thresholding to create a binary image
    # This converts the image to black and white based on local intensity
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C uses a Gaussian window to calculate thresholds
    # 31 is the block size, and 10 is the constant subtracted from the mean
    # Helps separate text from background effectively
    thresh = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # Find coordinates of non-zero (white) pixels in the thresholded image
    # np.column_stack combines x and y coordinates into a single array
    coords = np.column_stack(np.where(thresh > 0))
    # Calculate the rotation angle of the text using the minimum area rectangle
    # -1 indexes the angle from the minAreaRect output
    angle = cv2.minAreaRect(coords)[-1]
    # Adjust the angle if it's less than -45 degrees for proper rotation
    # This ensures the text is rotated correctly regardless of initial orientation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Get the height and width of the thresholded image
    (h, w) = thresh.shape[:2]
    # Calculate the center point of the image for rotation
    center = (w // 2, h // 2)
    # Create a rotation matrix using the center, calculated angle, and scale of 1.0
    # This defines how the image will be rotated
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Rotate the thresholded image using the rotation matrix
    # INTER_CUBIC provides high-quality interpolation, and BORDER_REPLICATE handles edges
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Return the final preprocessed (rotated) image
    return rotated