from datasets import load_dataset
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel
import torch
import os
import random
import json

prompt = 'a colossal metallic robot with an angular design stands atop a modern glass building nestled on a mountain peak. the robot dons a pair of gigantic headphones with neon lights, while a vivid sunset paints the sky and surrounding mountains with hues of orange and purple.'

pipe_path = "stabilityai/sdxl-turbo"

pipe = AutoPipelineForText2Image.from_pretrained(
    pipe_path, 
    torch_dtype=torch.float16, 
    variant="fp16",
    low_cpu_mem_usage=False
)

save_path = './inference_results'

if not os.path.exists(save_path):
    os.makedirs(save_path)

unet_id = "./turbo-trial-beta5k-lr1e-8-bs32-accu16-warmup5"

unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)

pipe.unet = unet
pipe = pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(3407 + i) 
if 'sdxl-turbo' not in pipe_path:
    image = pipe(prompt=prompt, generator=generator).images[0]
else:
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=generator).images[0]

image.save(os.path.join(save_path, f"turbo_{prompt}.png"))




