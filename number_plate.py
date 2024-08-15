import cv2
import numpy as np
import imutils
import pytesseract
import os
import json

# Path to the tesseract executable
# Change this path if tesseract is installed in a different location
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'



def remove_space(reading):
    text=reading.strip()
    
    newtext=''
    for i in text:
        if(i.isalnum()):
            newtext+=i
    if(newtext[2]=='G'):
        newtext=newtext[:2]+'0'+newtext[3:]
    return newtext


def process_image(image_path):
    """Process a single image for number plate detection and OCR."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filter and detect edges
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours and apply mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return {"number_plate": "Not detected", "cropped_image_path": ""}

    # Create a mask and crop the detected area
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Find the bounding box of the detected area
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Save the cropped image
    cropped_images_folder = 'croppedimages'
    os.makedirs(cropped_images_folder, exist_ok=True)
    cropped_image_path = os.path.join(cropped_images_folder, os.path.basename(image_path))
    cv2.imwrite(cropped_image_path, cropped_image)

    # Use Tesseract OCR to read the text
    text = pytesseract.image_to_string(cropped_image, config='--psm 8').strip()
    ftext=remove_space(text)
    # Draw a rectangle around the detected number plate
    cv2.drawContours(img, [location], -1, (0, 255, 0), 2)
    # Overlay the text on the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, ftext, (y1, x1 - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # Save the final image with text overlay
    result_images_folder = 'resultimages'
    os.makedirs(result_images_folder, exist_ok=True)
    result_image_path = os.path.join(result_images_folder, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(result_image_path, img)
    # return {
    #     "number_plate": text,
    #     "cropped_image_path": cropped_image_path,
    #     "result_image_path": result_image_path
    # }
    return f_text


