from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import json
import requests
import numpy as np
import imutils
import pytesseract
import sys

app = Flask(__name__)

# Configurations

UPLOAD_FOLDER = 'downloads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Folder to save uploaded files



pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
def remove_space(reading):
    text = reading.strip()
    newtext = ''.join(i for i in text if i.isalnum())
    if len(newtext) > 2 and newtext[2] == 'G':
        newtext = newtext[:2] + '0' + newtext[3:]
    return newtext

def image_processor(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
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
        return "Not detected"

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    cropped_images_folder = 'croppedimages'
    os.makedirs(cropped_images_folder, exist_ok=True)
    cropped_image_path = os.path.join(cropped_images_folder, os.path.basename(image_path))
    cv2.imwrite(cropped_image_path, cropped_image)

    text = pytesseract.image_to_string(cropped_image, config='--psm 8').strip()
    final_text = remove_space(text)
    
    # Draw a rectangle around the detected number plate
    cv2.drawContours(img, [location], -1, (0, 255, 0), 2)
    # Overlay the text on the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, final_text, (y1, x1 - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


    final_folder = "final_folder"
    os.makedirs(final_folder, exist_ok=True)
    final_path = os.path.join(final_folder, os.path.basename(image_path))
    cv2.imwrite(final_path, img)

    return final_text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'reg_no' not in request.files:
        return 'No file part', 400
    
    file = request.files['reg_no']
    
    if file.filename == '':
        return 'No selected file', 400

    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    
    result =image_processor(file_path)
    print(result)

    payload = {
            "reg_no": result,
            "consent": "Y",
            "consent_text": "I hereby declare my consent agreement for fetching my information via AITAN Labs API"
        }
    headers = {
	"x-rapidapi-key": "624a00a81fmsh5a70d17cc6d72a6p106a35jsn9ffdbe248289",
	"x-rapidapi-host": "rto-vehicle-information-verification-india.p.rapidapi.com",
	"Content-Type": "application/json"
}
        # Simulate processed data for the example

    try:
            response = requests.post(
                "https://rto-vehicle-information-verification-india.p.rapidapi.com/api/v1/rc/vehicleinfo",
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            if(len(result)==10 or response.status_code == 200):
                data2 = response.json()
                data=data2.get('result',{})

                permanent_address_line_1 = data.get("permanent_address_line1", "")
                permanent_address_line_2 = data.get("permanent_address_line2", "")
                permanent_address_line_3 = data.get("permanent_address_line3", "")
                district_name = data.get("permanent_district_name", "")



                for key, value in data2.items():
                    print(f"{key}: {value}")


                details = {
                    'messege':data2.get('message','error'),
                    'reg_no':result,
                    'owner_name':data.get('owner_name', 'Unknown'),
                    'purchase_date': data.get('purchase_date', 'Unknown'),
                    'permanent_full_address' : f"{permanent_address_line_1} {permanent_address_line_2} {permanent_address_line_3} {district_name}".strip(),
                    'model': data.get('model', 'Unknown'),
                    'manufacturing_year': data.get('manufacturing_yr', 'Unknown'),
                    'office_name': data.get('office_name', 'Unknown'),
                    'color': data.get('color', 'Unknown'),
                    'image_url': url_for('uploaded_file', filename=filename)
                    }
                    
                return render_template('car_details.html', **details)
            else:
                return render_template('error.html',image_url=url_for('uploaded_file', filename=filename), messege=str(e))

    except requests.exceptions.RequestException as e:
        return render_template('error.html',image_url=url_for('uploaded_file', filename=filename), messege=str(e))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
