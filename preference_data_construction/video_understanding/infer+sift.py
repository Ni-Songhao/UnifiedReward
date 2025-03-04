from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import random
import numpy as np
from PIL import Image
import requests
import copy
import torch
import tqdm
import sys
import warnings
import os
from datasets import load_dataset
import re
import json

warnings.filterwarnings("ignore")

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
# infer model: llava-video
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"

tokenizer_infer, infer_model, image_processor_infer, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
infer_model.eval()

# reward model
pretrained = ""
tokenizer_reward, reward_model, image_processor_reward, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
reward_model.eval()

save_path = './preference_data_pair.json'

data_path = '/path/to/data.json'
image_folder = '/path/to/images'

with open(data_path, 'r') as file:
    dataset = json.load(file)
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def infer(prompt, image_tensor, image_sizes):
    Query = prompt
    question = '<image>\n'*len(image_sizes) + Query


    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer_infer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        cont = infer_model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=4096,
        )
    text_outputs = tokenizer_infer.batch_decode(cont, skip_special_tokens=True)
    response1 = text_outputs[0]

    with torch.cuda.amp.autocast():
        cont = infer_model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=4096,
        )
    text_outputs = tokenizer_infer.batch_decode(cont, skip_special_tokens=True)
    response2 = text_outputs[0]
    
    return response1, response2


def pair_rank(prompt, image_tensor, image_sizes, R1, R2):
    Query = prompt
    question = '<image>\n'*8+f'You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following video and question, then determine which of the two provided answers is better\nQuestion: {Query}\nAnswer 1: {R1}\nAnswer 2: {R2}\nPlease evaluate both answers based on the following criteria:\n1. Accuracy: How well does the answer align with the visual information in the video?\n2. Completeness: Does the answer fully address all aspects of the question?\n3. Clarity: Is the answer easy to understand and well-articulated?\n4. Relevance: Does the answer directly relate to the question and the video?\nAfter your evaluation, please:\n1. Explain your reasoning for each criterion.\n2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: \"Overall Judgment: Answer 1 is better than Answer 2.\"\nYour response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task. You have to choose the better of the two answers and can\'t say that both answers are equally good.'

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer_reward, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        cont = reward_model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    text_outputs = tokenizer_reward.batch_decode(cont, skip_special_tokens=True)

    if 'Overall Judgment' in text_outputs[0]:
        output = text_outputs[0].split('Overall Judgment')[-1].strip().split('.')[0].strip()
    else:
        output = text_outputs[0]

    if 'Answer 1 is better than Answer 2' in output:
        chosen = R1
        rejected = R2
    elif 'Answer 2 is better than Answer 1' in output:
        chosen = R2
        rejected = R1
    else:
        chosen = R1
        rejected = R2

    return chosen, rejected


def point_score(prompt, image_tensor, image_sizes, chosen_list, rejected_list):
    Query = prompt

    chosen_score = []
    rejected_score = []
    for response in chosen_list:
        question = '<image>\n'*8 + f'You are provided with a video and a question for this video. Please review the corresponding response based on the following 5 factors: 1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects. 2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships. 3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects\' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present. 4. Helpfulness: Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user\'s comprehension of the video. Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt. Evaluate the overall contribution of the response to the user experience. 5. Ethical Considerations: - Identify if the model gives appropriate warnings or avoids providing advice on sensitive topics, such as medical videos. Ensure the model refrains from stating identification information in the video that could compromise personal privacy. Evaluate the language model\'s responses for fairness in treating individuals and communities, avoiding biases. Assess for harmfulness, ensuring the avoidance of content that may potentially incite violence, be classified as NSFW (Not Safe For Work), or involve other unmentioned ethical considerations. Consider any content that could be deemed offensive, inappropriate, or ethically problematic beyond the explicitly listed criteria.\nYou need to assign a score from 1 to 5 based on the above 5 factors for the response after \'Final Score:\'.\nYour task is provided as follows:\nQuestion: [{Query}]\nResponse: [{response}]\n'
        with torch.cuda.amp.autocast():
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer_reward, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            cont = reward_model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )

            text_outputs = tokenizer_reward.batch_decode(cont, skip_special_tokens=True)
            if 'Final Score:' in text_outputs[0]:
                output = text_outputs[0].split('Final Score:')[-1].strip()

                output = re.findall(r"\d+",output)
                if len(output) > 0:
                    output = output[0]
                else:
                    output = 0
            else:
                output = re.findall(r"\d+",text_outputs[0])
                if len(output) > 0:
                    output = output[0]
                else:
                    output = 0

        chosen_score.append(output)

    for response in rejected_list:
        question = '<image>\n'*8 + f'You are provided with a video and a question for this video. Please review the corresponding response based on the following 5 factors: 1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects. 2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships. 3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects\' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present. 4. Helpfulness: Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user\'s comprehension of the video. Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt. Evaluate the overall contribution of the response to the user experience. 5. Ethical Considerations: - Identify if the model gives appropriate warnings or avoids providing advice on sensitive topics, such as medical videos. Ensure the model refrains from stating identification information in the video that could compromise personal privacy. Evaluate the language model\'s responses for fairness in treating individuals and communities, avoiding biases. Assess for harmfulness, ensuring the avoidance of content that may potentially incite violence, be classified as NSFW (Not Safe For Work), or involve other unmentioned ethical considerations. Consider any content that could be deemed offensive, inappropriate, or ethically problematic beyond the explicitly listed criteria.\nYou need to assign a score from 1 to 5 based on the above 5 factors for the response after \'Final Score:\'.\nYour task is provided as follows:\nQuestion: [{Query}]\nResponse: [{response}]\n'
        with torch.cuda.amp.autocast():
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer_reward, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            cont = reward_model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )

            text_outputs = tokenizer_reward.batch_decode(cont, skip_special_tokens=True)
            if 'Final Score:' in text_outputs[0]:
                output = text_outputs[0].split('Final Score:')[-1].strip()

                output = re.findall(r"\d+",output)
                if len(output) > 0:
                    output = output[0]
                else:
                    output = 0
            else:
                output = re.findall(r"\d+",text_outputs[0])
                if len(output) > 0:
                    output = output[0]
                else:
                    output = 0

        rejected_score.append(output)

    chosen = chosen_list[chosen_score.index(max(chosen_score))]
    rejected = rejected_list[rejected_score.index(min(rejected_score))]

    chos_score = max(chosen_score)
    rej_score = min(rejected_score)

    return chosen, rejected, chos_score, rej_score


if os.path.exists(save_path):
    with open(save_path, 'r') as file:
        data_list = json.load(file)
else:
    data_list = []

for i in tqdm.trange(len(dataset)):
    if i < len(data_list):
        continue
    data = dataset[i]

    image_list = []
    image_sizes = []
    for img in data['image']:
        image = Image.open(os.path.join(image_folder, img)).resize((512, 512))
        image_list.append(image)
        image_sizes.append(image.size)

    image_tensor_infer = image_processor_infer.preprocess(image_list, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    image_tensor_reward = image_processor_reward.preprocess(image_list, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    chosen_list = []
    rejected_list = []
    for j in range(5):
        response1, response2 = infer(data['prompt'], image_tensor_infer, image_sizes)

        chosen, rejected = pair_rank(data['prompt'], image_tensor_reward, image_sizes, response1, response2)

        chosen_list.append(chosen)
        rejected_list.append(rejected)


    chosen, rejected, chosen_score, rejected_score = point_score(data['prompt'], image_tensor_reward, image_sizes, chosen_list, rejected_list)

    if chosen_score <= rejected_score:
        continue

    data['chosen'] = chosen
    data['rejected'] = rejected
    data['chosen_score'] = chosen_score
    data['rejected_score'] = rejected_score
    data_list.append(data)

    # 将数据写入 .json 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)