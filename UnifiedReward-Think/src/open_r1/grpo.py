# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using temporal GRPO"},
    )


def extract_answer(text):
    """Extract the answer part from text with <answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def accuracy_reward(completions, solution, **kwargs):
    rewards = []
    
    for i, completion in enumerate(completions):
        reward = 0.1
        
        if not completion or len(completion) == 0:
            rewards.append(0.0)
            continue
            
        answer_text = completion[0].get("content", "") if isinstance(completion, list) else completion
     
        extracted_answer = extract_answer(answer_text)
        
        if solution and i < len(solution):
            sol = solution[i]

            if sol == extracted_answer:
                reward = 1
            else:
                reward = 0
            
        rewards.append(reward)
        
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        list_data_dict = []
        import yaml
        import random
        import json

        with open(script_args.dataset_name, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")

            script_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
            for dataset in datasets:
                json_path = dataset.get("json_path")
                image_folder = dataset.get("image_folder")
                sampling_strategy = dataset.get("sampling_strategy", "all")
                sampling_number = None

                print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                    for data in cur_data_dict:
                        if 'video' in data:
                            data['video'] = os.path.join(image_folder, data['video'])
                        if 'image' in data:
                            data['images'] = [os.path.join(image_folder, data['image'])]
                            del data['image']
                        if 'images' in data:
                            for i in range(len(data['images'])):
                                data['images'][i] = os.path.join(image_folder, data['images'][i])
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]
                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")

                list_data_dict.extend(cur_data_dict)

        from datasets import DatasetDict, Dataset
        dataset = DatasetDict({
            "train": Dataset.from_list(list_data_dict),
        })

    train_dataset = dataset.get('train', None)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }
        
    def make_conversation_image_and_video(example):
        content = []

        if len(example['images']) > 8:
            frame_num = len(example['images'])
            first_half = example['images'][:int(frame_num/2)]
            second_half = example['images'][int(frame_num/2):]
            step = len(first_half) / 4
            first_half = [first_half[int(i * step)] for i in range(4)]
            second_half = [second_half[int(i * step)] for i in range(4)]
            example['images'] = first_half + second_half

        if 'video' in example:
            example["prompt"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ]
        else:
            for i in range(len(example["images"])):
                content.append({"type": "image"})
            
            content.append({"type": "text", "text": example["problem"]})

            example["prompt"] = [
                    {
                        "role": "user",
                        "content": content,
                    },
            ]

        return example

    
    train_dataset = train_dataset.map(make_conversation_image_and_video)

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
