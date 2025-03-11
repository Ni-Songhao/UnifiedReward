# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import tqdm
import warnings
import os
from datasets import load_dataset
import random
warnings.filterwarnings("ignore")
pretrained = "CodeGoat24/UnifiedReward-7b"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

image_path = '/path/to/image'

Query = '' # Question

R1 = '' # Response1
R2 = '' # Response2

image = Image.open(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"

question = f'<image>\nYou are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better\nQuestion: {Query}\nAnswer 1: {R1}\nAnswer 2: {R2}\nPlease evaluate both answers based on the following criteria:\n1. Accuracy: How well does the answer align with the visual information in the image?\n2. Completeness: Does the answer fully address all aspects of the question?\n3. Clarity: Is the answer easy to understand and well-articulated?\n4. Relevance: Does the answer directly relate to the question and the image?\nAfter your evaluation, please:\n1. Explain your reasoning for each criterion.\n2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: \"Overall Judgment: Answer X is better.\"\nYour response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task.'


conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

print(text_outputs[0])
