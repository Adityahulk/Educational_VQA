import os
import json
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import io
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Set environment variables for library compatibility
os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths to the PDF files and mapping file
PDF_DIRECTORY = "./documents"
MAPPING_FILE = "doc_id_to_path.json"

def convert_pdf_to_images(pdf_path):
    """Converts a PDF file into a list of images."""
    return convert_from_path(pdf_path)

def initialize_rag_model(index_name="global_index"):
    """Initializes the RAG MultiModal model."""
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    return RAG

def index_documents_in_folder(RAG, folder_path, index_name):
    """Indexes all documents in a folder and creates a doc_id-to-path mapping."""
    # Get list of PDF files
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

    # Create mapping between doc_id and file paths
    doc_id_to_path = {i: pdf_path for i, pdf_path in enumerate(pdf_files)}
    with open(MAPPING_FILE, "w") as f:
        json.dump(doc_id_to_path, f)

    # Index documents
    RAG.index(
        input_path=folder_path,
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=True,
    )

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

def process_ocr(image):
    """Processes OCR on an image and extracts text."""
    ocr_model = initialize_ocr_model()
    document = DocumentFile.from_images([image])
    ocr_result = ocr_model(document)
    return ocr_result.export()

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
    """Loads the LLM and its processor."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return model, processor

def generate_answer_with_llm(model, processor, text, images=None, videos=None):
    """Generates an answer using the LLM."""
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=10000)
    return processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

def process_query_across_pdfs(query, k=10):
    """Processes a query across multiple PDFs."""
    RAG = initialize_rag_model()
    
    # Load index and document mapping
    if not os.path.exists(MAPPING_FILE):
        index_documents_in_folder(RAG, folder_path=PDF_DIRECTORY, index_name="global_index")
    with open(MAPPING_FILE, "r") as f:
        doc_id_to_path = json.load(f)

    # Search for the query
    search_results = search_query_with_rag(RAG, query, k=k)
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
    model, processor = initialize_llm()

    # Prepare input for the LLM
    prompt = (
        "You are asked to answer questions asked on a document image.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n\n"
        f"Document:\n{combined_text}\n\nQuestion: {query}\n\nAnswer:"
    )

    # Generate output
    output = generate_answer_with_llm(model, processor, prompt)
    print("Generated Answer:", output)

# Example usage
if __name__ == "__main__":
    user_query = "Which model gave best result?"
    process_query_across_pdfs(user_query, k=20)
