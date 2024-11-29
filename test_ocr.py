from fastapi import FastAPI, HTTPException
import os
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import io
from pdf2image import convert_from_path

app = FastAPI()

# Set environment variables for library compatibility
os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths to the PDF files and mapping file
PDF_DIRECTORY = "./document_test"


def convert_pdf_to_images(pdf_path):
    """Converts a PDF file into a list of images."""
    return convert_from_path(pdf_path)


def compress_image(image, new_width=256, new_height=256, quality=75):
    """Compress and resize an image."""
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    buffer = io.BytesIO()
    resized_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def initialize_ocr_model():
    """Loads the OCR model."""
    return ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def process_ocr(image, save_path="./reference.jpg"):
    """Processes OCR on an image and extracts text."""
    image.save(save_path, "JPEG")
    ocr_model = initialize_ocr_model()
    document = DocumentFile.from_images(save_path)
    ocr_result = ocr_model(document)
    os.remove(save_path)  # Clean up saved file
    return ocr_result.export()


def process_query_across_pdfs(folder_path):
    """Processes OCR on all PDFs in a folder and extracts text."""
    ocr_texts = []
    # List all PDF files in the folder
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]

    for pdf_path in pdf_files:
        try:
            print(f"Processing: {pdf_path}")
            # Convert PDF pages to images
            images = convert_pdf_to_images(pdf_path)
            for page_number, image in enumerate(images):
                # Compress each image
                compressed_image = compress_image(image)
                # Perform OCR on the compressed image
                ocr_result = process_ocr(compressed_image)
                ocr_texts.append({
                    "pdf": pdf_path,
                    "page": page_number + 1,
                    "text": ocr_result
                })
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")

    return ocr_texts

if __name__ == "__main__":
    results = process_query_across_pdfs(PDF_DIRECTORY)
    for result in results:
        print(f"PDF: {result['pdf']} | Page: {result['page']}\nExtracted Text:\n{result['text']}\n{'-'*80}")
