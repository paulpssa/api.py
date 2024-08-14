from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import re
import easyocr
from ultralytics import YOLO

app = FastAPI()

def segment_image(model, img, conf=0.5):
    results = model.predict(img, conf=conf, verbose=False)
    segmented_images = []
    for result in results:
        for mask in result.masks.xy:
            points = np.int32([mask])
            mask_img = np.zeros_like(img)
            cv2.fillPoly(mask_img, points, (255, 255, 255))
            segmented_img = cv2.bitwise_and(img, mask_img)
            x, y, w, h = cv2.boundingRect(points)
            cropped_segmented_img = segmented_img[y:y+h, x:x+w]
            segmented_images.append(cropped_segmented_img)
    return segmented_images

# Inicializar el lector de EasyOCR
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

def detect_text(img):
    results = reader.readtext(img, paragraph=True)
    processed_text = []
    for res in results:
        pt0 = tuple(map(int, res[0][0]))
        pt2 = tuple(map(int, res[0][2]))
        cv2.rectangle(img, pt0, pt2, (166, 56, 242), 2)
        cv2.putText(img, res[1], (pt0[0], pt0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        for point in res[0]:
            cv2.circle(img, tuple(map(int, point)), 2, (0, 255, 0), 2)
        processed_text.append(res[1])
    return img, processed_text

def clean_text(text):
    # Convertir a mayúsculas y eliminar caracteres especiales y espacios
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text.upper())
    return cleaned_text

# Cargar el modelo YOLO
model = YOLO("best.pt")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Segmentar la imagen
    segmented_images = segment_image(model, img, conf=0.5)
    
    # Resultado final de texto
    final_texts = []

    # Detectar y mostrar el texto en cada segmento
    for segmented_img in segmented_images:
        img_hsv = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2HSV)
        brightness_increase1 = 50
        brightness_image1 = cv2.subtract(img_hsv, np.ones(img_hsv.shape, dtype=np.uint8) * brightness_increase1)
        _, text = detect_text(brightness_image1)
        cleaned_text_without_filters = [clean_text(t) for t in text]
        text_len_without_filters = sum(len(t) for t in cleaned_text_without_filters)
        
        brightness_increase = 50
        brightness_image = cv2.subtract(segmented_img, np.ones(segmented_img.shape, dtype=np.uint8) * brightness_increase)
        _, text_filtered = detect_text(brightness_image)
        cleaned_text_with_filters = [clean_text(t) for t in text_filtered]
        text_len_with_filters = sum(len(t) for t in cleaned_text_with_filters)
        
        # Comparar resultados y mostrar el mejor
        if text_len_without_filters >= text_len_with_filters:
            final_text = cleaned_text_without_filters
        else:
            if text_len_with_filters > text_len_without_filters and text_len_without_filters > 7:
                final_text = cleaned_text_without_filters
            else:
                final_text = cleaned_text_with_filters
        
        final_texts.extend(final_text)
    
    return {"text": final_texts}
