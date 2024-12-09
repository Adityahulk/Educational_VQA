from fastapi import FastAPI, HTTPException
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

app = FastAPI()
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

def search_query_with_rag(RAG, query, k=2):
    """Performs a search query on the indexed data."""
    results = RAG.search(query, k=k)
    return results

def initialize_ocr_model():
    """Loads the OCR model."""
    return ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

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
        os.remove(TEMP_IMAGE_PATH)
    return "".join(cleaned_text)

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

# def extract_pages_as_images(PDF_DIRECTORY, doc_id, temp_image_dir, MAPPING_FILE):
#     """Extract the first three pages from a PDF, save them as images, and resize them to 768x768."""
#     os.makedirs(temp_image_dir, exist_ok=True)
#     #image_paths = []
#     #pdf_path = MAPPING_FILE[]
#     for page_num in page_numbers:
        
#         # Convert specific PDF page to an image
#         images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
#         #print('images---', images)
        
#         # Define the path for saving the resized image
#         temp_image_path = os.path.join(temp_image_dir, f"page_{page_num}.jpg")
        
#         # Resize the image to 768x768 and save it
#         resized_image = images[0].resize((1024, 1024), Image.Resampling.LANCZOS)
#         resized_image.save(temp_image_path, "JPEG")
        
#         # Add the saved image path to the list
#         image_paths.append(temp_image_path)
#         #print('image_paths', image_paths)
        
#     return image_paths

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
    # ocr_texts = []
    # # # Process OCR on top results
    # for result in search_results:
    #     doc_id = result["doc_id"]
    #     print(doc_id)
    #     pdf_path = doc_id_to_path[str(doc_id)]
    #     print(pdf_path)
    #     ocr_texts = process_pdf(pdf_path)

    # # # Combine texts from all documents
    # combined_text = "".join(ocr_texts)
    # print(combined_text)

    # # Prepare LLM and processor
    # model, processor, device = initialize_llm()

    # # # Prepare input for the LLM
    # prompt = (
    #     "You are asked to answer questions asked on a document image.\n"
    #     "The answers to questions are short text spans taken verbatim from the document. "
    #     "This means that the answers comprise a set of contiguous text tokens present in the document.\n\n"
    #     f"Document: {combined_text} and the Question is: {query}, provide  the Answer:"
    # )

    # # # Generate output
    # output = generate_answer_with_llm(model, processor, prompt)
    # print("answer:- " + str(output))

'''
document_101
1. (page: 311–313, 315–317) Explain how the concept of binding energy per nucleon helps us understand nuclear stability and its implications for fission and fusion reactions.
2. (314–317) Compare and contrast the processes of nuclear fission and nuclear fusion, emphasizing the energy release mechanisms in both.
3. (308–309) What role do isotopes, isobars, and isotones play in understanding nuclear structure and reactions? Illustrate with examples.
4. (310–311) How does Einstein’s mass-energy equivalence principle apply to nuclear reactions, and how does it differ from its application in chemical reactions?
5. (310-311) How does Einstein’s mass-energy equivalence principle apply to nuclear reactions and how it explain mass defect?

document_79
1. (125–127) Differentiate between geometrical and optical isomerism in coordination compounds with examples.
2. (128–130) Discuss the differences between the Valence Bond Theory (VBT) and Crystal Field Theory (CFT) in explaining the bonding in coordination compounds.
3. (131–133) What is the crystal field splitting energy, and how does it determine the magnetic properties of octahedral complexes?
4. (135-136) Explain Bonding in Metal Carbonyls.

document_86
1. (46–49) Describe the methods and challenges historians face in deciphering and interpreting ancient inscriptions.
2. (38-40)Evaluate the impact of agricultural innovations, such as the use of iron ploughs and transplantation, on rural society and economy.
'''

'''
Including images and equation:
document_8
1. (131-132) Using Mendel’s dihybrid cross experiment, calculate the expected phenotypic ratio for a cross between heterozygous tall plants with round seeds (TtRr). Describe how independent assortment influences this ratio. -----> Wrong answer
2. (130-131) Based on Mendel’s monohybrid cross, design an experiment to confirm the 1:2:1 genotypic ratio in the F2 generation for tall and short plants. Include the method and calculations for phenotypic and genotypic ratios. -----> Wrong answer

document_86

'''
if __name__ == "__main__":
    # process_query_across_pdfs("How is sodium chloride formulated?", False)
    process_query_across_pdfs("- Mention the equation of photosynthesis and explain the graph showing the absorption spectrum of chlorophyll a, b and the carotenoids.", False)
