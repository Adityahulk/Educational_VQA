from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import io
import os
import re

PDF_PATH = "./document_test/document_11.pdf"
TEMP_IMAGE_PATH = "./temp_image.jpg"

def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def compress_image(image, quality=95):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def enhance_image(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2.0)

def clean_text(text):
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_text_from_ocr(ocr_result):
    plain_text = []
    for page in ocr_result['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    plain_text.append(word['value'])
    return " ".join(plain_text)

def process_pdf(pdf_path):
    ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    images = convert_pdf_to_images(pdf_path)
    extracted_text = []
    for page_number, image in enumerate(images, start=1):
        image = enhance_image(image)
        compressed_image = compress_image(image)
        compressed_image.save(TEMP_IMAGE_PATH, "JPEG")
        document = DocumentFile.from_images(TEMP_IMAGE_PATH)
        ocr_result = ocr_model(document).export()
        plain_text = extract_text_from_ocr(ocr_result)
        cleaned_text = clean_text(plain_text)
        extracted_text.append(f"Page {page_number}:\n{cleaned_text}")
        os.remove(TEMP_IMAGE_PATH)
    return "\n\n".join(extracted_text)

if __name__ == "__main__":
    extracted_text = process_pdf(PDF_PATH)
    print(f"Extracted Text:\n{extracted_text}")
