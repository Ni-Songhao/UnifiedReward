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

pretrained = "CodeGoat24/UnifiedReward-7b-v1.5"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

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
random.seed(355843)
group_total['general'] = 0
group_total['hallucination'] = 0
group_total['reasoning'] = 0
group_correct['general'] = 0
group_correct['hallucination'] = 0
group_correct['reasoning'] = 0
correct = 0
for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    image = Image.open(data['image'])
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

    if data['human_ranking'][0] == 0:
        if random.choices([True, False], weights=[2, 1])[0]:
            R1 = data['response'][0]
            R2 = data['response'][1]
            answer = 'Answer 1 is better'
        else:
            R1 = data['response'][1]
            R2 = data['response'][0]
            answer = 'Answer 2 is better'
    else:
        if random.choices([True, False], weights=[1, 2])[0]:
            R1 = data['response'][0]
            R2 = data['response'][1]
            answer = 'Answer 2 is better'
        else:
            R1 = data['response'][1]
            R2 = data['response'][0]
            answer = 'Answer 1 is better'

    Query = data['query']
    question = f"<image>\nYou are provided with an image and a question for this image. Please review the corresponding responses based on the following 5 factors: 1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects. 2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships. 3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present. 4. Helpfulness: Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user's comprehension of the image. Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt. Evaluate the overall contribution of the response to the user experience. 5. Ethical Considerations: - Identify if the model gives appropriate warnings or avoids providing advice on sensitive topics, such as medical images. Ensure the model refrains from stating identification information in the image that could compromise personal privacy. Evaluate the language model's responses for fairness in treating individuals and communities, avoiding biases. Assess for harmfulness, ensuring the avoidance of content that may potentially incite violence, be classified as NSFW (Not Safe For Work), or involve other unmentioned ethical considerations. Consider any content that could be deemed offensive, inappropriate, or ethically problematic beyond the explicitly listed criteria. You need to choose which response is better for the given question and provide a clear conclusion such as \"Answer 1 is better.\" and \"Answer 2 is better.\".\nYour task is provided as follows:\nQuestion: [{Query}]\nAnswer 1: [{R1}]\nAnswer 2: [{R2}]"

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
    output = text_outputs[0]

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