# -*- coding: utf-8 -*-
"""ColPali_+_Qwen2_VL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb

# Multimodal RAG using ColPali (with Byaldi) and Qwen2-VL

[ColPali](https://huggingface.co/blog/manu/colpali) is a multimodal retriever that removes the need for hefty and brittle document processors. It natively handles images and processes and encodes image patches to be compatible with text, thus removing need to do OCR, or image captioning.

![ColPali](https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/La8vRJ_dtobqs6WQGKTzB.png)

After indexing data, we will use [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) to do generation part in RAG.

[Byaldi](https://github.com/AnswerDotAI/byaldi) is a new library by answer.ai to easily use ColPali. This library is in a very early stage, so this notebook will likely be updated soon with API changes.

"""

#installing dependencies
"""
!pip install --upgrade byaldi

!sudo apt-get install -y poppler-utils

!pip install -q pdf2image

!pip install -q transformers==4.46.1

!pip install -q qwen-vl-utils

!git clone https://github.com/Dao-AILab/flash-attention && (cd flash-attention && pip install .)
"""

# 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
# from huggingface_hub import notebook_login
# notebook_login()

from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image
import io

"""Let's see how the PDF looks like."""
# Define the path to the PDF file
pdf_path = "./pdfs/pdfs/visual incontext learning.pdf"

# Convert PDF pages to images
images = convert_from_path(pdf_path)
images[0].show()  # Display the 7th page

"""We should initialize `RAGMultiModalModel` object with a ColPali model from Hugging Face. By default this model uses GPU but we are going to have Qwen2-VL in the same GPU so we are loading this in CPU for now."""
torch.cuda.empty_cache()
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")

"""We can directly index our document using RAG, simply passing pdf file path is enough."""
RAG.index(
    input_path="./pfds/pfds/",
    index_name="image_index", # index will be saved at index_root/index_name/
    store_collection_with_index=False,
    overwrite=True
)

"""
Importing model
"""
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

"""Now let's retrieve!"""
text_query = "What this should be learned before to learn LVLM?"
results = RAG.search(text_query, k=1)
print(results)

"""
Now we can actually build a RAG pipeline. For this tutorial we will use Qwen2-VL-7B model.
"""
image_index = results[0]["page_num"] - 1
images[image_index]

# Load the image
image = images[image_index]

# Resize the image
# You can adjust the (width, height) tuple below as needed
new_width, new_height = 256, 256  # Example target size
resized_image = image.resize((new_width, new_height), Image.LANCZOS)

# Save as JPEG to reduce file size further, and adjust quality
buffer = io.BytesIO()
resized_image.save(buffer, format="JPEG", quality=75)  # Adjust quality as needed (0-100)
buffer.seek(0)

# Load the compressed image back for the LVLM
compressed_image = Image.open(buffer)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": compressed_image,
            },
            {"type": "text", "text": text_query},
        ],
    }
]

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

generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)