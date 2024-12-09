import streamlit as st
import argparse
from pydantic import BaseModel
import os
import json
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image, ImageEnhance
import io
import re
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

TEMP_IMAGE_PATH = "./temp_image.jpg"
TEMP_IMAGE_DIR = "./temp_images/"

# Set environment variables for library compatibility
os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")
    return RAG

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

def search_query_with_rag(RAG, query, k=2):
    """Performs a search query on the indexed data."""
    results = RAG.search(query, k=k)
    return results

def initialize_llm():
    """Loads the LLM and its processor with proper GPU support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:- ", device)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", 
        trust_remote_code=True
    )
    return model, processor, device

def generate_answer_with_llm(model, processor, text, images=None, videos=None, device="cuda"):
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
        # Generate with more diverse sampling parameters
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=3000,  # Limit the output length
            do_sample=True,       # Use sampling for diversity
            top_p=0.95,           # Nucleus sampling
            top_k=50,             # Limit candidate tokens
            temperature=0.7       # Control randomness
        )
    
    # Decode the outputs, handling sequences properly
    results = processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return results

def extract_pages_as_images(pdf_path, page_num, temp_image_dir, MAPPING_FILE):
    """Extract the first three pages from a PDF, save them as images, and resize them to 768x768."""
    os.makedirs(temp_image_dir, exist_ok=True)
    image_paths = []
    #pdf_path = MAPPING_FILE[]
    
    # Convert specific PDF page to an image
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
    print('images---', images)
    
    # Define the path for saving the resized image
    temp_image_path = os.path.join(temp_image_dir, f"page_{page_num}.jpg")
    
    # Resize the image to 768x768 and save it
    resized_image = images[0].resize((1024, 1024), Image.Resampling.LANCZOS)
    resized_image.save(temp_image_path, "JPEG")
    
    # Add the saved image path to the list
    image_paths.append(temp_image_path)
    #print('image_paths', image_paths)
    
    return image_paths

def prepare_vlm_input(image_paths, prompt_text):
    """Prepare inputs for the Vision-Language Model."""
    # Load Qwen2VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", device_map="auto")
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", min_pixels=min_pixels, max_pixels=max_pixels)
    print(len(image_paths))
    # Prepare messages for the VLM
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},  # Shared query text
            ] + [
                {"type": "image", "image": image_path} for image_path in image_paths  # Images from the list
            ]
        }
    ]

    # Prepare inputs for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    return model, processor, inputs

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
    search_results = search_query_with_rag(RAG, query, k=2)
    print('search_results----',search_results)
    
    image_paths = []
    for result in search_results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        #filename = result["metadata"]["filename"]
        #pdf_path = os.path.join(PDF_DIRECTORY, filename)

        with open('doc_id_to_path.json', 'r') as file:
            pdf_paths = json.load(file)
        pdf_path = pdf_paths.get(str(doc_id+1))
        print(f"pdf_path: {pdf_path}, page_num: {page_num}, doc_id: {doc_id}")
        # image_paths += extract_pages_as_images(PDF_DIRECTORY, doc_id, TEMP_IMAGE_DIR, MAPPING_FILE)
        image_paths += extract_pages_as_images(pdf_path, page_num, TEMP_IMAGE_DIR, MAPPING_FILE)

    print('image_paths after adding----', image_paths)
    image_paths = image_paths[0:2]
    # Run VLM inference across all images
    model, processor, inputs = prepare_vlm_input(image_paths, query)
    torch.cuda.empty_cache()
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    torch.cuda.empty_cache()
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Combine the results
    combined_output = "\n".join(output_texts)
    print(f"Combined Output for Query '{query}':\n{combined_output}")

def main():
    st.title("PDF Query Processor")
    st.write("Enter a query to search across the indexed PDF documents.")

    # User input for query
    query = st.text_area("Query", placeholder="Type your query here...")
    
    # Option to choose whether to index new documents
    use_index_documents = st.checkbox("Index new documents before querying", value=False)
    
    # Process the query on button click
    if st.button("Process Query"):
        if query.strip():
            st.info("Processing your query, please wait...")
            try:
                output = process_query_across_pdfs(query, use_index_documents)
                st.success("Query processed successfully!")
                st.write("### Output")
                st.text(output)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid query.")

if __name__ == "__main__":
    main()