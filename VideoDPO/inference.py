from __future__ import annotations
import argparse
import os
import random
import time
from omegaconf import OmegaConf
import numpy as np
import json
try:
    import intel_extension_for_pytorch as ipex
except:
    pass
from lvdm.models.turbo_utils.lora import collapse_lora, monkeypatch_remove_lora
from lvdm.models.turbo_utils.lora_handler import LoraHandler
from utils.common_utils import load_model_checkpoint
from utils.common_utils import instantiate_from_config
from lvdm.models.turbo_utils.t2v_turbo_scheduler import T2VTurboScheduler
from lvdm.models.turbo_utils.t2v_turbo_pipeline import T2VTurboVC2Pipeline
from diffusers.utils import export_to_video
import torch
import torchvision
from concurrent.futures import ThreadPoolExecutor
import uuid


MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"

device = "cuda"  # Linux & Windows
DTYPE = (
    torch.float16
)  # torch.float16 works as well, but pictures seem to be a bit worse


def save_video(vid_tensor, output_path, fps=16):

    # Convert the video tensor from [C, T, H, W] to [T, C, H, W]
    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)  # t,c,h,w
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

    # Save the video using torchvision.io.write_video
    torchvision.io.write_video(
        output_path, video, fps=fps, video_codec="h264", options={"crf": "10"}
    )
    return output_path


def generate_videos_from_prompts(prompt, output_dir, base_model_dir):

    os.makedirs(output_dir, exist_ok=True)
    seed = 123
    guidance_scale = 7.5
    num_inference_steps = 4
    num_frames = 16
    fps = 8
    randomize_seed = True
    param_dtype = "torch.float16"
    
    config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, base_model_dir)

    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
    pipeline.to(device)
    pipeline.to(
        torch_dtype=torch.float16 if param_dtype == "torch.float16" else torch.float32,
    )

    result = pipeline(
        prompt=prompt,
        frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_videos_per_prompt=1,
    )[0]


    save_video(result, f'{output_dir}/{prompt}.mp4', fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video generation from prompts.")
    parser.add_argument(
        "--prompt",
        default='Vbench_info.json',
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='vbench/turbo_dpo_bs16_acc4_lr2e-5_epoch3',
        help="Directory to save generated videos.",
    )
    parser.add_argument(
        "--base_model_dir",
        default="/path/to/model.ckpt",
        type=str,
        help="Directory of the checkpoint.",
    )

    args = parser.parse_args()
    generate_videos_from_prompts(args.prompt, args.output_dir, args.base_model_dir)