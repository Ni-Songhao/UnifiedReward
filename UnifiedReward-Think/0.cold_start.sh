VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

export NCCL_P2P_DISABLE=1

BASE_RUN_NAME="UnifiedReward-Think-cold_start"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="qwen_1_5"
RUN_NAME="UnifiedReward-Think-cold_start" 
PREV_STAGE_CHECKPOINT="CodeGoat24/UnifiedReward-7b" # replace it with your last checkpoint training from single image collection

DATA_PATH="dataset/cold_start.yaml"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

MASTER_ADDR="localhost"
MASTER_PORT="6668"
NNODES=1
NODE_RANK=0
DS_SKIP_CUDA_CHECK=1 python3 -m torch.distributed.run --nnodes=$NNODES \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --mm_tunable_parts="mm_language_model" \
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
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 8 \
    --learning_rate 2.5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 13000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
