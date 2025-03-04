VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# DPO Stage
PROMPT_VERSION="qwen_1_5"
SFT_MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"
EPOCH=3
beta=0.1

DPO_RUN_NAME="llava_ov_qwen2-unified_reward-dpo"
DPO_CLEAN_NAME="${DPO_RUN_NAME##*/}"
OUTPUT_DIR="checkpoints_dpo_image/${DPO_CLEAN_NAME}"

DATA_PATH="./preference_data_construction/image_understanding/preference_data_pair.json"
IMAGE_FOLDER="/path/to/images"

echo $DPO_RUN_NAME
MASTER_ADDR="localhost"
MASTER_PORT="6667"
NNODES=1
NODE_RANK=0

DS_SKIP_CUDA_CHECK=1 python3 -m torch.distributed.run --nnodes=$NNODES \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    llava/train/train_dpo.py \
    --deepspeed scripts/zero3_offload.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1.0 \
    --beta=${beta} \
    --gamma=0 \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $DPO_CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 4 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 10000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --dataloader_drop_last True 

