import cv2
import pytesseract
from .processor import preprocess_image

# Optional: specify path if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    preproccesed = preprocess_image(image)
    text = pytesseract.image_to_string(preproccesed)
    return text