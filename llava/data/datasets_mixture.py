# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    HPD = Dataset(
        dataset_name="HPD",
        dataset_type="torch",
        data_path="dataset/HPD/train_data.json",
        image_path="dataset/HPD",
        description="",
    )
    add_dataset(HPD)

    LiFT_HRA = Dataset(
        dataset_name="LiFT-HRA",
        dataset_type="torch",
        data_path="dataset/LiFT-HRA/train_data.json",
        image_path="dataset/LiFT-HRA",
        description="",
    )
    add_dataset(LiFT_HRA)

    llava_critic_pairwise = Dataset(
        dataset_name="llava-critic-pairwise",
        dataset_type="torch",
        data_path="dataset/llava-critic/pairwise/train_data_25k.json",
        image_path="dataset/llava-critic/pairwise",
        description="",
    )
    add_dataset(llava_critic_pairwise)

    llava_critic_pointwise = Dataset(
        dataset_name="llava-critic-pointwise",
        dataset_type="torch",
        data_path="dataset/llava-critic/pointwise/train_data_25k.json",
        image_path="dataset/llava-critic/pointwise",
        description="",
    )
    add_dataset(llava_critic_pointwise)

    OIP = Dataset(
        dataset_name="OIP",
        dataset_type="torch",
        data_path="dataset/OIP/train_data.json",
        image_path="dataset/OIP",
        description="",
    )
    add_dataset(OIP)

    videofeedback = Dataset(
        dataset_name="videofeedback",
        dataset_type="torch",
        data_path="dataset/videofeedback/train_data.json",
        image_path="dataset/videofeedback/frames",
        description="",
    )
    add_dataset(videofeedback)

    VideoDPO = Dataset(
        dataset_name="videodpo",
        dataset_type="torch",
        data_path="dataset/VideoDPO/train_data.json",
        image_path="dataset/VideoDPO",
        description="",
    )
    add_dataset(VideoDPO)

    EvalMuse_pairwise = Dataset(
        dataset_name="evalmuse-pairwise",
        dataset_type="torch",
        data_path="dataset/EvalMuse/pairwise/train_data.json",
        image_path="dataset/EvalMuse/images",
        description="",
    )
    add_dataset(EvalMuse_pairwise)
    
    EvalMuse_pointwise = Dataset(
        dataset_name="evalmuse-pointwise",
        dataset_type="torch",
        data_path="dataset/EvalMuse/pointwise/train_data.json",
        image_path="dataset/EvalMuse/images",
        description="",
    )
    add_dataset(EvalMuse_pointwise)

    ShareGPTVideo_pairwise = Dataset(
        dataset_name="sharegptvideo-pairwise",
        dataset_type="torch",
        data_path="dataset/ShareGPTVideo/pairwise/train_data.json",
        image_path="dataset/ShareGPTVideo/videos1",
        description="",
    )
    add_dataset(ShareGPTVideo_pairwise)
    
    ShareGPTVideo_pointwise = Dataset(
        dataset_name="sharegptvideo-pointwise",
        dataset_type="torch",
        data_path="dataset/ShareGPTVideo/pointwise/train_data.json",
        image_path="dataset/ShareGPTVideo/videos1",
        description="",
    )
    add_dataset(ShareGPTVideo_pointwise)

    GenAI_Bench_baiqi = Dataset(
        dataset_name="GenAI-Bench_baiqi",
        dataset_type="torch",
        data_path="dataset/GenAI-Bench_baiqi/train_data.json",
        image_path="dataset/GenAI-Bench_baiqi",
        description="",
    )
    add_dataset(GenAI_Bench_baiqi)

    GenAI_Bench_tiger_video = Dataset(
        dataset_name="GenAI-Bench_tiger_video",
        dataset_type="torch",
        data_path="dataset/GenAI-Bench_tiger_video/train_data.json",
        image_path="dataset/GenAI-Bench_tiger_video/images",
        description="",
    )
    add_dataset(GenAI_Bench_tiger_video)

    GenAI_Bench_tiger_image = Dataset(
        dataset_name="GenAI-Bench_tiger_image",
        dataset_type="torch",
        data_path="dataset/GenAI-Bench_tiger_image/train_data_2k.json",
        image_path="dataset/GenAI-Bench_tiger_image",
        description="",
    )
    add_dataset(GenAI_Bench_tiger_image)