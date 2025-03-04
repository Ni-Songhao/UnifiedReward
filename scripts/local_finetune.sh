LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

export HF_HOME='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/wjq/.cache/huggingface/hub'
# export NCCL_P2P_DISABLE=1

BASE_RUN_NAME="llava-critic"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="test" 
PREV_STAGE_CHECKPOINT="lmms-lab/llava-critic-7b" # replace it with your last checkpoint training from single image collection
# PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

# HPD+llava-critic-pointwise+llava-critic-pairwise+OIP+evalmuse-pairwise+evalmuse-pointwise+sharegptvideo-pointwise+sharegptvideo-pairwise+LiFT-HRA+videodpo+videofeedback

MASTER_ADDR="localhost"
MASTER_PORT="6668"
NNODES=1
NODE_RANK=0
python3 -m torch.distributed.run --nnodes=$NNODES \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_mixture HPD+llava-critic-pointwise+llava-critic-pairwise+OIP+evalmuse-pairwise+evalmuse-pointwise+sharegptvideo-pointwise+sharegptvideo-pairwise+LiFT-HRA+videodpo+videofeedback+GenAI-Bench_baiqi+GenAI-Bench_tiger_image+GenAI-Bench_tiger_video \
    --mm_tunable_parts="mm_language_model," \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/$RUN_NAME \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 4 \
    --learning_rate 2.5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 13000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
