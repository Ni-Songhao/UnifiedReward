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
from random import sample

warnings.filterwarnings("ignore")

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):
        from torchvision import transforms

        from llava.mm_utils import opencv_extract_frames

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded


model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

# reward model
pretrained = "/path/to/unified_reward_model"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

save_path = './turbo_dpo_dataset/dpo_data.json'

video_path = './turbo_dpo_dataset/videos'

with open('./turbo_dpo_dataset/data.json', 'r') as file:
    dataset = json.load(file)

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def pair_rank(prompt, video1, video2, images_1, images_2):

    image_sizes = []
    images_1.extend(images_2)
    images = images_1
    for img in images:
        img.resize((512, 512))
        image_sizes.append(img.size)

    image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    conv_template = "qwen_1_5"
    # pairwise ranking
    
    question = f'<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n <image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\nSuppose you are an expert in judging and evaluating the quality of AI-generated videos. You are given a text caption and the frames of two generated videos based on that caption. Your task is to evaluate and compare two videos based on two key criteria:\n1. Alignment with the Caption: Assess how well each video aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Video Quality: Examine the visual quality of each video, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both videos using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Video 1 is better than Video 2.\", \"Video 2 is better than Video 1.\" and \"Both videos are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]'

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    with torch.cuda.amp.autocast():
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    output = text_outputs[0]

    if 'Video 2 is better than Video 1' in output:
        chosen = video2
        chosen_images = images_2
        rejected = video1
        rejected_images = images_1

    else:
        chosen = video1
        chosen_images = images_1
        rejected = video2
        rejected_images = images_2
    
    return chosen, chosen_images, rejected, rejected_images


def point_score(prompt, chosen_list, rejected_list, chosen_images_list, rejected_images_list):
    chosen_score = []
    rejected_score = []

    for images in chosen_images_list:
        image_sizes = []
        for img in images:
            img.resize((512, 512))
            image_sizes.append(img.size)

        image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

        # print(image_tensor[0].shape)
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

        # pairwise ranking
        question = '<image>\n'*8 + f'Suppose you are an expert in judging and evaluating the quality of AI-generated videos. You are given a text caption and the frames of a generated video based on that caption. Your task is to evaluate this videos based on two key criteria:\n1. Alignment with the Caption: Assess how well each video aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Video Quality: Examine the visual quality of each video, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nEvaluate this video using the above criteria and assign a score from 1 to 10 after \'Final Score:\'\nYour task is provided as follows:\nText Caption: [{prompt}]'
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        with torch.cuda.amp.autocast():
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096
            )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        output = text_outputs[0].split('Final Score:')[-1].strip()

        import re
        # 获取所有数字
        output = re.findall(r"\d+",output)
        if len(output) > 0:
            output = output[0]
        else:
            output = 0

        chosen_score.append(output)

    for images in rejected_images_list:
        image_sizes = []
        for img in images:
            img.resize((512, 512))
            image_sizes.append(img.size)

        image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

        # print(image_tensor[0].shape)
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

        # pairwise ranking
        question = '<image>\n'*8 + f'Suppose you are an expert in judging and evaluating the quality of AI-generated videos. You are given a text caption and the frames of a generated video based on that caption. Your task is to evaluate this videos based on two key criteria:\n1. Alignment with the Caption: Assess how well each video aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Video Quality: Examine the visual quality of each video, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nEvaluate this video using the above criteria and assign a score from 1 to 10 after \'Final Score:\'\nYour task is provided as follows:\nText Caption: [{prompt}]'

        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        with torch.cuda.amp.autocast():
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096
            )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        output = text_outputs[0].split('Final Score:')[-1].strip()

        import re
        # 获取所有数字
        output = re.findall(r"\d+",output)
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
    prompt = data['caption']

    chosen_list = []
    chosen_images_list = []
    rejected_list = []
    rejected_images_list = []
    for j in range(5):
        video_1 = os.path.join(video_path, data['videos'][2*j].split('/')[-1])
        video_2 = os.path.join(video_path, data['videos'][2*j+1].split('/')[-1])

        num_video_frames = 8
        loader_fps = 0.0
        fps = None
        frame_count = None

        images_1, frames_loaded = _load_video(
            video_1, num_video_frames, loader_fps, fps=fps, frame_count=frame_count
        )
        images_2, frames_loaded = _load_video(
            video_2, num_video_frames, loader_fps, fps=fps, frame_count=frame_count
        )

        chosen, chosen_images, rejected, rejected_images = pair_rank(prompt, video_1, video_2, images_1, images_2)

        chosen_list.append(chosen)
        chosen_images_list.append(chosen_images)
        rejected_list.append(rejected)
        rejected_images_list.append(rejected_images)

    chosen, rejected, chosen_score, rejected_score = point_score(prompt, chosen_list, rejected_list, chosen_images_list, rejected_images_list)

    if int(chosen_score) <= int(rejected_score):
        continue

    data['chosen'] = '/'.join(chosen.split('/')[-2:])
    data['rejected'] = '/'.join(rejected.split('/')[-2:])
    data['chosen_score'] = chosen_score
    data['rejected_score'] = rejected_score

    del data['videos']

    data_list.append(data)

    # 将数据写入 .json 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)