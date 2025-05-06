VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
BASE_RUN_NAME="UnifiedReward-Think-GRPO"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"


PROMPT_VERSION="qwen_1_5"
RUN_NAME="UnifiedReward-Think-GRPO" 
PREV_STAGE_CHECKPOINT="your/rejection_sampling_model/path"
DATA_PATH="dataset/grpo.yaml"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

MASTER_ADDR="localhost"
MASTER_PORT="6668"
NNODES=1
NODE_RANK=0
# Launch training
DS_SKIP_CUDA_CHECK=1 python3 -m torch.distributed.run --nnodes=$NNODES \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    llava/train/train_grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir ./checkpoints/$RUN_NAME  \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --data_path $DATA_PATH \
    --dataset_name xxx \
    --max_prompt_length 2048 \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --beta 0.04 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_total_limit 10