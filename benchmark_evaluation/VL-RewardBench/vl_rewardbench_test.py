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

random.seed(0)

import json
with open('./data.json', 'r') as file:
    dataset = json.load(file)

group_correct = {}
group_total = {}
group_mapping = {
        "vlfeedback": "general",
        "povid": "hallucination",
        "reasoning_tasks": "reasoning",
        "rlhf-v": "hallucination",
        "rlaif-v": "hallucination",
        "wildvision-battle": "general"
    }
group_total['general'] = 0
group_total['hallucination'] = 0
group_total['reasoning'] = 0
group_correct['general'] = 0
group_correct['hallucination'] = 0
group_correct['reasoning'] = 0
correct = 0
for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    image = Image.open(os.path.join(data['image']))
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"

    R1 = data['response'][0]
    R2 = data['response'][1]

    if data['human_ranking'][0] == 0:
        if random.choice([True, False]):
            R1 = data['response'][0]
            R2 = data['response'][1]
            answer = 'Answer 1 is better'
        else:
            R1 = data['response'][1]
            R2 = data['response'][0]
            answer = 'Answer 2 is better'
    else:
        if random.choice([True, False]):
            R1 = data['response'][0]
            R2 = data['response'][1]
            answer = 'Answer 2 is better'
        else:
            R1 = data['response'][1]
            R2 = data['response'][0]
            answer = 'Answer 1 is better'

    Query = data['query']
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

    if 'Overall Judgment' in text_outputs[0]:
        output = text_outputs[0].split('Overall Judgment')[-1].strip().split('.')[0].strip()
    else:
        output = text_outputs[0].split('.')
        if len(output) > 1:
            output = output[-2].strip()
        else:
            output = output[0]

    id_value = data['id']
    split_index = min((id_value.find('_'), id_value.find('-')), key=lambda x: x if x != -1 else float('inf'))
    if split_index != -1:
        id_prefix = id_value[:split_index]
    else:
        id_prefix = id_value
    
    if id_prefix=="RLAIF":
        dtype = "rlaif-v"
    elif id_prefix=="RLHF":
        dtype = "rlhf-v"
    elif id_prefix=="mathverse" or id_prefix=="mmmu":
        dtype = "reasoning_tasks"
    elif id_prefix=="wildvision":
        dtype = "wildvision-battle"
    else:
        dtype = "vlfeedback"
    
    group_total[group_mapping[dtype]] += 1

    if answer in output:
        correct += 1
        group_correct[group_mapping[dtype]] += 1
        
accuracy = correct / len(dataset)
print(f"Acc.: {correct} / {len(dataset)} = {accuracy}")

task_list =  ['reasoning', 'hallucination', 'general']

macro_average = sum(group_correct[k]/group_total[k] for k in task_list) / 3

reasoning_acc = group_correct['reasoning']/group_total['reasoning']
hallucination_acc = group_correct['hallucination']/group_total['hallucination']
general_acc = group_correct['general']/group_total['general']

print(f'reasoning: {reasoning_acc}')
print(f'hallucination: {hallucination_acc}')
print(f'general: {general_acc}')
print(f'macro: {macro_average}')
