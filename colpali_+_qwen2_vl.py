from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
import os
import json
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import io
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

app = FastAPI()

# Set environment variables for library compatibility
os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths to the PDF files and mapping file
PDF_DIRECTORY = "./document_test"
MAPPING_FILE = './doc_id_to_path.json'
INDEX_ROOT = "/home/ubuntu/Educational_VQA/.byaldi"
INDEX_NAME = "global_index"
overwrite= False

class QueryRequest(BaseModel):
    query: str

def convert_pdf_to_images(pdf_path):
    """Converts a PDF file into a list of images."""
    return convert_from_path(pdf_path)

def initialize_rag_model(overwrite=False, device="cuda", verbose=1):
    index_path = os.path.join(INDEX_ROOT, INDEX_NAME)
    print("overwrite-----", overwrite)
    print('index_path---', index_path)
    print(f"Checking index path: {index_path}")
    if os.path.exists(index_path) and not overwrite:
        print("Index exists and overwrite=False. Loading existing index.")
        RAG = RAGMultiModalModel.from_index(
            index_path=index_path,
            index_root=INDEX_ROOT,
            device=device,
            verbose=verbose
        )
        print("Loaded existing index.")
    else:
        if os.path.exists(index_path) and overwrite:
            print("Index exists and overwrite=True. Deleting existing index.")
            shutil.rmtree(index_path)
        # Initialize RAG from pretrained
        print("Initializing RAG from pretrained.")
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    return RAG

'''
def initialize_rag_model(overwrite=False, device="cuda", verbose=1):
    index_path = os.path.join(INDEX_ROOT, INDEX_NAME)
    if os.path.exists(index_path) and not overwrite:
        RAG = RAGMultiModalModel.from_index(
            index_path=index_path,
            index_root=INDEX_ROOT,
            device=device,
            verbose=verbose
        )
        print("Loaded existing index.")
    else:
        if os.path.exists(index_path) and overwrite:
            shutil.rmtree(index_path)
        # Initialize RAG from pretrained
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
        # Index documents
        index_documents_in_folder(RAG, PDF_DIRECTORY, INDEX_NAME, overwrite)
    return RAG

'''
def index_documents_in_folder(RAG, folder_path, index_name, overwrite=False):
    """Indexes all documents in a folder and creates a doc_id-to-path mapping."""
    print(f"Indexing documents in {folder_path} with index_name {index_name}, overwrite={overwrite}")
    # Get list of PDF files
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    # Create mapping between doc_id and file paths
    doc_id_to_path = {i: pdf_path for i, pdf_path in enumerate(pdf_files)}
    # Write mapping file only if overwrite is True or it doesn't exist
    if overwrite or not os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "w") as f:
            json.dump(doc_id_to_path, f)
    # Index documents
    RAG.index(
        input_path=folder_path,
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=overwrite,
    )

def add_index_documents_in_folder(RAG, folder_path, index_name, overwrite=False):
    """
    Indexes all documents in a folder and creates a doc_id-to-path mapping 
    using the `add_to_index` method.
    """
    print(f"Indexing documents in {folder_path} with index_name {index_name}, overwrite={overwrite}")
    
    # Get list of PDF files
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not overwrite and os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r") as f:
            doc_id_to_path = json.load(f)
        # Ensure no duplicate entries by finding the max doc_id
        existing_doc_ids = set(doc_id_to_path.keys())
        max_doc_id = max(map(int, existing_doc_ids)) if existing_doc_ids else 0
    else:
        doc_id_to_path = {}
        max_doc_id = 0

    # Add new documents to the mapping
    new_doc_id_to_path = {
        max_doc_id + i + 1: pdf_path for i, pdf_path in enumerate(pdf_files) 
        if pdf_path not in doc_id_to_path.values()  # Avoid duplicate file entries
    }
    doc_id_to_path.update(new_doc_id_to_path)

    with open(MAPPING_FILE, "w") as f:
        json.dump(doc_id_to_path, f, indent=4)
    
    # Add documents to the index individually
    for doc_id, pdf_path in new_doc_id_to_path.items():
        print(f"Adding document {pdf_path} to index with doc_id {doc_id}")
        RAG.add_to_index(
            input_item=pdf_path,
            store_collection_with_index=False,  # Adjust based on your requirement
            doc_id=doc_id,
            metadata={"filename": os.path.basename(pdf_path)}  # Optional metadata
        )
    print(f"Indexing completed for {len(pdf_files)} documents.")

def search_query_with_rag(RAG, query, k=10):
    """Performs a search query on the indexed data."""
    results = RAG.search(query, k=k)
    return results

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

def is_same_line(box1, box2):
    """Determines if two boxes are on the same line."""
    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2
    return box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]

def union_box(box1, box2):
    """Combines two bounding boxes."""
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    ]

def extract_text_and_boxes(ocr_data):
    """Extracts text and bounding boxes from OCR data."""
    texts, boxes = [], []
    for page in ocr_data['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    texts.append(word['value'])
                    boxes.append([
                        word['geometry'][0][0], word['geometry'][0][1],
                        word['geometry'][1][0], word['geometry'][1][1]
                    ])
    return {"text": texts, "boxes": boxes}

def layout_text_with_spaces(texts, boxes):
    """Arranges text spatially based on bounding boxes."""
    line_boxes, line_texts = [], []
    max_line_char_num, line_width = 0, 0

    while boxes:
        line_box, line_text = [boxes.pop(0)], [texts.pop(0)]
        char_num = len(line_text[-1])
        line_union_box = line_box[-1]

        while boxes and is_same_line(line_box[-1], boxes[0]):
            line_box.append(boxes.pop(0))
            line_text.append(texts.pop(0))
            char_num += len(line_text[-1])
            line_union_box = union_box(line_union_box, line_box[-1])

        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num, line_width = char_num, line_union_box[2] - line_union_box[0]

    char_width = line_width / max_line_char_num if max_line_char_num else 1
    space_line_texts = []

    for line_box, line_text in zip(line_boxes, line_texts):
        space_line_text = ""
        for box, text in zip(line_box, line_text):
            left_char_num = int(box[0] / char_width)
            space_line_text += " " * (left_char_num - len(space_line_text)) + text
        space_line_texts.append(space_line_text)

    return space_line_texts

def initialize_llm():
    """Loads the LLM and its processor with proper GPU support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:- ", device)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        trust_remote_code=True
    )
    return model, processor, device

def generate_answer_with_llm(model, processor, text, images=None, videos=None, device="cuda"):
    """
    Generates an answer using the LLM with proper GPU handling.
    :param model: Preloaded model moved to the correct device
    :param processor: Processor to handle inputs
    :param text: Query text
    :param images: List of images (optional, as tensors)
    :param videos: List of videos (optional, as tensors)
    :param device: The device ('cuda' or 'cpu') to perform computations
    :return: Decoded output
    """
    # Prepare inputs on the appropriate device
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # Ensure model inference uses the correct device
    with torch.no_grad():  # Disable gradient calculation for inference
        generated_ids = model.generate(**inputs, max_new_tokens=10000)
    
    # Decode the outputs, handling sequences properly
    results = processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return results

# @app.post("/query")

def process_query_across_pdfs(query, use_index_documents: bool):
    # query = request.query
    """Processes a query across multiple PDFs."""
    RAG = initialize_rag_model(overwrite=False, device="cuda", verbose=1)
    
    # Load index and document mapping
    if use_index_documents:
        # Use the specified indexing function
        index_documents_in_folder(RAG, folder_path=PDF_DIRECTORY, index_name="global_index")
    else:
        add_index_documents_in_folder(RAG, folder_path=PDF_DIRECTORY, index_name="global_index")
    
    with open(MAPPING_FILE, "r") as f:
        doc_id_to_path = json.load(f)

    # Search for the query
    search_results = search_query_with_rag(RAG, query, k=10)
    ocr_texts = []

    # Process OCR on top results
    for result in search_results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        pdf_path = doc_id_to_path[str(doc_id)]
        images = convert_pdf_to_images(pdf_path)
        compressed_image = compress_image(images[page_num - 1])
        ocr_result = process_ocr(compressed_image)
        extracted_data = extract_text_and_boxes(ocr_result)
        space_line_texts = layout_text_with_spaces(extracted_data["text"], extracted_data["boxes"])
        ocr_texts.append("\n".join(space_line_texts))

    # Combine texts from all documents
    combined_text = "\n\n".join(ocr_texts)

    # Prepare LLM and processor
    model, processor, device = initialize_llm()

    # Prepare input for the LLM
    prompt = (
        "You are asked to answer questions asked on a document image.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n\n"
        f"Document:\n{combined_text}\n\nQuestion: {query}\n\nAnswer:"
    )

    # Generate output
    output = generate_answer_with_llm(model, processor, prompt)
    print("answer:- " + output)

if __name__ == "__main__":
    process_query_across_pdfs("Explain power of accomoddation", False)
