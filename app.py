# Importing libraries
import streamlit as st

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import torch
from ultralytics import YOLO
import easyocr
from deep_translator import GoogleTranslator
from docx import Document
import urllib.request


MODEL_URL = "https://github.com/dummy-repo-1/vocabLearn-YOLO/releases/download/weights.v1.0/bestFinal.pt"
MODEL_PATH = "bestFinal.pt"

st.set_page_config(page_title="Image Input", layout="wide")

st.title("📸 Image Capture & Upload")
st.write("Select how you want to provide the document image.")

# Replaces your CLI input()
input_method = None
input_method = st.radio("Choose input method:", ("Upload Image", "Capture from Camera"))

# We will store the final image file here
raw_image_file = []
camera_input = None
image = None

# --- METHOD 1: UPLOAD ---
# Replaces tkinter filedialog
if input_method == "Upload Image":
    raw_image_file = st.file_uploader("Choose an image...", 
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=True)

# --- METHOD 2: CAMERA ---
# Replaces cv2.VideoCapture(), cv2.imshow(), and cv2.waitKey()
elif input_method == "Capture from Camera":
    camera_input = st.camera_input("Take a picture of the document")

# --- PROCESSING THE CAPTURE ---
if camera_input is not None:
    # 1. Display the image on the web app
    if input_method == "Capture from Camera" :
        st.image(camera_input, caption="Ready for processing", use_container_width=True)
        raw_image_file.append(camera_input)
        if st.button("Process Image", type="primary"):
            st.success("Image successfully loaded")


    # 2. Convert to OpenCV format (NumPy array) for your pipeline
    # Since your YOLO/EasyOCR pipeline likely expects a cv2 frame, we convert 
    # the uploaded in-memory file directly into an OpenCV-compatible array.
    # file_bytes = np.asarray(bytearray(raw_image_file.read()), dtype=np.uint8)
    # cv2_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 3. Trigger your pipeline
    
        
        # Here is where you will call your pipeline:
        # result = your_pipeline_function(cv2_frame)
@st.cache_resource
def load_model():
    # Check if the file exists locally
    # if not os.path.exists(MODEL_PATH):
    st.info("Downloading model weights for the first time. This may take a minute...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded successfully!")
        
    return YOLO(MODEL_PATH)

model = load_model()

def predict_boxes(raw_image_file) :

  real_img = "real.jpg"

  # taking sharpen kernel
  sharpen_kernel = np.array([
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0]
  ])

  # loading EasyOCR model
  reader = easyocr.Reader(['en'])

  # for saving ocr results
  ocr_results = {}

  # taking images from input folder
  for img_name in raw_image_file :
    # img_path = os.path.join(raw_image_file, img_name)
    # reading the image with cv2
    
  
    image = Image.open(img_name).convert('RGB')
    image = ImageOps.exif_transpose(image)
    image.save(real_img)
    img = cv2.imread(real_img)


    # img = cv2.imread(img_path)

    # getting RGB values from the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # predicting boxes of highlighted text on the image
    results = model.predict(real_img, save=False, show_labels=False, conf=0.5)

    # getting box coordinates
    boxes = results[0].boxes.xyxy

    # making sub-folder with image name
    # img_output_dir = os.path.join(output_folder, os.path.splitext(img_name)[0])
    # os.makedirs(img_output_dir, exist_ok=True)

    # collect ocr text for each image
    text_list = []

    # traversing each boxes
    for i, box in enumerate(boxes) :
      x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())

      # normalize coordinates clamp within image bounds
      h, w = img_rgb.shape[:2]
      x_min = max(0, min(x_min, w-1))
      x_max = max(0, min(x_max, w-1))
      y_min = max(0, min(y_min, h-1))
      y_max = max(0, min(y_max, h-1))

      # crop the portion
      crop = img_rgb[y_min : y_max, x_min : x_max]

      # applying sharpening
      sharp_crop = cv2.filter2D(crop, -1, sharpen_kernel)

      # applying bilateral filter
      final_crop = cv2.bilateralFilter(sharp_crop, d=9, sigmaColor=75, sigmaSpace=75)

    #   save_path = os.path.join(img_output_dir, f'box_{i+1}.jpg')
    #   cv2.imwrite(save_path, cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR))

      # applying easyocr on cropped portion
      ocr_output = reader.readtext(final_crop)

      for det in ocr_output :
        text_list.append(det[1])

    # save text results at image names
    ocr_results[img_name] = text_list

  return ocr_results


def translate(ocr_results) :
  translation = {}
  GTrans = GoogleTranslator(source='auto', target='hi')
  for key in ocr_results.keys() :
    translation[key] = {}
    for item in ocr_results[key] :
      meaning = GTrans.translate(item)
      translation[key][item] = meaning

  return translation


def write_doc(translation, new_doc=True) :
  if new_doc :
    # create a new document
    doc = Document()

    # add a title
    doc.add_heading("English-Bengali Word Translation", level=1)

    # creating a table with 2 columns
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'

    # Add heading
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "English"
    hdr_cells[1].text = "Bengali"

    for key in translation.keys() :

      for item in translation[key].keys() :
        row_cells = table.add_row().cells
        row_cells[0].text = item
        row_cells[1].text = translation[key][item]

    # Save the document
    doc.save('translations.docx')

# loading trained model
# model = YOLO("bestFinal.pt")

ocr_results = predict_boxes(raw_image_file)

translation = translate(ocr_results)

write_doc(translation, new_doc=True)
