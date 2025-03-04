<div align="center">
    <h1 align="center"> Unified Reward Model for Multimodal Understanding and Generation
    </h1>
</div>


## 📖 Introduction

This repository release the **UnifiedReward** -- the first unified reward model for multimodal understanding and generation assessment, enabling both pairwise ranking and pointwise scoring, which can be employed for vision model preference alignment. 

<img src=./docs/images/pipeline.png width="100%"/>


## 🏁 Compared with Current Reward Models

|  Reward Model | Method| Image Generation | Image Understanding | Video Generation | Video Understanding
| :-----: | :-----: |:-----: |:-----: | :-----: | :-----: |
|  [PickScore](https://github.com/yuvalkirstain/PickScore) |Point | √ |  | ||
|  [HPS](https://github.com/tgxs002/HPSv2) | Point | √ |  |||
|  [ImageReward](https://github.com/THUDM/ImageReward) |  Point| √|  |||
|  [LLaVA-Critic](https://huggingface.co/lmms-lab/llava-critic-7b) | Pair/Point | | √  |||
|  [VideoScore](https://github.com/TIGER-AI-Lab/VideoScore) | Point |  |  |√ ||
|  [LiFT](https://github.com/CodeGoat24/LiFT) | Point |  |  |√| |
|  [VisionReward](https://github.com/THUDM/VisionReward) | Point |√  | |√||
|  [VideoReward](https://github.com/KwaiVGI/VideoAlign) | Point |  |  |√ ||
|  UnifiedReward (Ours) | Pair/Point | √ | √ |√|√|

##  🔧 Environment Set Up
Install the inference package:
```bash
conda create -n unifiedreward python=3.10 -y
conda activate unifiedreward
pip install --upgrade pip  
pip install -e ".[train]"
```



## 💻 Training UnifiedReward
### 1. Unified Preference Training Dataset Preparation
Please download our constructed unified preference dataset put it in `./dataset/`.

```
dataset
├── EvalMuse                  
    ├── pairwise            
    └── pointwise
    └── ...            
└── HPD                   
└── LiFT-HRA
└── LLaVA-Critic 
    ├── pairwise            
    └── pointwise
    └── ...
└── OIP
└── ShareGPTVideo
    ├── pairwise            
    └── pointwise
    └── ...      
└── VideoDPO 
└── VideoFeedback
└── train_data.yaml
```
### 2. Training
```bash
bash train.sh
```

## ✨ Direct Preference Optimization 
### 🎨 Image and Video Understanding DPO
#### 1. Construct Preference data

The data for preference data construction should adhere to the following structure:
```bash
[
    {
    "prompt": "",
    "image": "",
    },
    ...
]
```
Then 
```bash
# image understanding 
cd preference_data_construction/image_understanding
python infer+sift.py # you need to fill the 'image_folder' and 'data_path' in this file

# video understanding 
cd preference_data_construction/video_understanding
python infer+sift.py # you need to fill the 'image_folder' and 'data_path' in this file
```

#### 2. Training
The training data format in `data.json` should adhere to the following structure:
```bash
[
    {
    "id": "",
    "image": "",
    "prompt": "",
    "chosen": "",
    "rejected": ""
    },
    ...
]
```
Then start training:
```bash
# image understanding 
bash dpo_image_understand_ov7b.sh 

# video understanding 
bash dpo_video_understand_llava_video_7b.sh
```

### 🖼️ Image Generation DPO
Prepare Environments
```bash
cd DiffusionDPO
conda create -n diffdpo python=3.10 -y
conda activate diffdpo
pip install -r requirements.txt
```

#### 1. Construct Preference data
Image Generation

The data for preference data construction should adhere to the following structure:
```bash
[
    {
    "prompt": "",
    },
    ...
]
```
Then 
```bash
python data_generation.py # you need to fill the 'data_path' in this file
```

Preference Pair Data Construction
```bash
python sift_dpo_data.py
```

#### 2. Training
The training data format in `data.json` should adhere to the following structure:
```bash
[
    {
        "id": "",
        "caption": "",
        "jpg_0": "", #chosen image path
        "jpg_1": "", #rejected image path
        "label_0": 1,
    },
    ...
]
 ```
Then start training:
```bash
bash launchers/turbo_dpo.sh
```

### 🎬 Video Generation DPO
Prepare Environments
```bash
cd VideoDPO
conda create -n videodpo python=3.10 -y
conda activate videodpo
pip install -r requirements.txt
```

Please download the our constructed T2V-Turbo model and put it in `./checkpoints/t2v-turbo`.

#### 1. Construct Preference data
Video Generation

The data for preference data construction should adhere to the following structure:
```bash
[
    {
    "prompt": "",
    },
    ...
]
```
Then
```bash
bash data_generation.sh # you need to fill '--prompts_file' in this file
```

Preference Pair Data Construction

```bash
python sift_dpo_data.py
```

#### 2. Training
The training data format in `data.json` should adhere to the following structure:
```bash
[
    {
        "id": "",
        "caption": "",
        "chosen": "", # chosen video path
        "rejected": "", # rejected video path
    },
    ...
]
```
Then start training:
```bash
bash run.sh
```

## 🚀 Evaluation
We provide several evaluation code in `./benchmark_evaluation` directory. 
### Reward model
We provide evaluation code for [GenAI-Bench-Video](https://github.com/TIGER-AI-Lab/GenAI-Bench), [GenAI-Bench-Image](https://github.com/TIGER-AI-Lab/GenAI-Bench), [VideoGen-RewardBench](https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench) and [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench) benchmarks.

### Video Understanding
We provide evaluation code for [MSRVTT](https://github.com/xudejing/video-question-answering), [MSVD](https://github.com/xudejing/video-question-answering), and [TGIF](https://github.com/YunseokJANG/tgif-qa) benchmarks while using the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) toolkit for evaluating LongVideoBench, MLVU, and Video-MME benchmarks.

### Image Understanding
We use [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit to evaluate LLaVABench, WildVision, LLaVABench-Wilder, LiveBench, and MMHal benchmarks.

### Image Generation
We utilize the image reward model,
i.e., [PickScore](https://github.com/yuvalkirstain/PickScore), [HPS](https://github.com/tgxs002/HPSv2) and [ImageReward](https://github.com/THUDM/ImageReward) for
quality assessment. 

### Video Generation
[VBench](https://github.com/Vchitect/VBench) is used for video generation assessment.

## 🤗 Acknowledgments
In this work, reward model and image/video understanding DPO code is based on [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT). while image and video generation DPO is based on [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) and [VideoDPO](https://github.com/CIntellifusion/VideoDPO), 

We also utilize [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) toolkits for evaluation.

Thanks to all the contributors!